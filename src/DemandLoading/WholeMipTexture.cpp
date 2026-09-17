// SPDX-License-Identifier: MIT
#include <hip/hip_runtime.h>
#include <DemandLoading/WholeMipTexture.h>
#include <DemandLoading/ContractState.h>
#include <DemandLoading/Logging.h>
#include "Internal/HipCalls.h"
#include "Internal/MipSuffixData.h"
#include "Internal/TextureMetadata.h"
#include "Internal/TextureRuntime.h"
#include "Internal/CubicRuntime.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <mutex>

namespace hip_demand { namespace whole_mip_v1 {
namespace cv = contract_v1;
namespace cap = capability_v1;
using cv::Outcome;
using internal::HipOperation;

namespace {
cv::SamplerDesc contractDescriptor(const TextureDesc& desc, cv::SamplingPolicy sampling,
                                  cap::MipPolicy policy) {
    cv::SamplerDesc result;
    result.addressMode[0] = internal::cubicAddress(desc.addressMode[0]);
    result.addressMode[1] = internal::cubicAddress(desc.addressMode[1]);
    result.spatialFilter = desc.filterMode == hipFilterModePoint ? cv::FilterMode::Point : cv::FilterMode::Linear;
    result.mipFilter = desc.mipmapFilterMode == hipFilterModePoint ? cv::FilterMode::Point : cv::FilterMode::Linear;
    result.normalizedCoords = desc.normalizedCoords;
    result.sRGB = desc.sRGB;
    result.generateMipmaps = desc.generateMipmaps;
    result.maxMipLevels = std::min(desc.maxMipLevel, 32u);
    result.priority = static_cast<cv::Priority>(desc.evictionPriority);
    result.mipPolicy = policy == cap::MipPolicy::Disabled ? cv::MipPolicy::Disabled : cv::MipPolicy::Required;
    result.samplingPolicy = sampling;
    return result;
}
}

class Texture::Impl {
public:
    Impl(std::shared_ptr<ImageSource> source, const TextureDesc& desc, const Options& options)
        : source_(std::move(source)), storageDesc_(desc), options_(options), ledger_(options.maxManagedBytes) {}

    struct Backing {
        internal::ImageStorage storage;
        cv::MipLayout mips;
        std::vector<hipTextureObject_t> samplers;
        std::vector<cubic_v1::Entry> cubic;
        std::vector<hipTextureDesc> submitted, returned;
        hipTextureDesc submittedSampler{}, returnedSampler{};
        cv::ResourceLifecycle lifetime;
        Backing(cv::BudgetLedger& ledger, cv::GpuKey key, size_t count)
            : samplers(count, 0), cubic(count), submitted(count), returned(count), lifetime(ledger, key, 1) {}
    };

    struct CandidateOwner {
        Impl& owner;
        std::unique_ptr<Backing>& candidate;
        ~CandidateOwner() {
            if (!candidate) return;
            // Unexpected exceptions still leave an owning, charged retirement
            // record. The explicit collect operation performs fallible HIP cleanup.
            const auto retired = candidate->lifetime.retire();
            const auto invalidated = candidate->lifetime.invalidateMapping();
            if (retired != Outcome::Success || invalidated != Outcome::Success)
                logMessage(LogLevel::Error, "Whole-mip exception retirement invariant failed");
            owner.retired_ = std::move(candidate);
            owner.refresh();
            std::lock_guard<std::mutex> lock(owner.statusMutex_);
            owner.status_.operation = Outcome::RuntimeFailure;
            if (owner.status_.primary.outcome == Outcome::Success)
                owner.status_.primary = {Outcome::RuntimeFailure, cap::Operation::None, 0};
        }
    };

    struct ReadStatsOwner {
        Impl& owner;
        const internal::MipSuffixReadStats& stats;
        ~ReadStatsOwner() {
            std::lock_guard<std::mutex> lock(owner.statusMutex_);
            owner.status_.decodedPeakBytes = std::max(owner.status_.decodedPeakBytes, stats.decodedPeakBytes);
            owner.status_.sourceBytes += stats.sourceBytes;
            owner.status_.authoredReads += stats.authoredReads;
            owner.status_.generatedLevels += stats.generatedLevels;
        }
    };

    Outcome initialize() {
        if (options_.abi.version != Version || options_.abi.byteSize != sizeof(Options))
            return failInitialization(Outcome::AbiMismatch);
        if (!source_ || !internal::validDescriptor(storageDesc_) || !options_.maxSamplers ||
            options_.maxSamplers > 4096 || !options_.maxRequests || options_.maxRequests > 1048576 ||
            !options_.maxManagedBytes || !options_.maxDecodedBytes || !options_.maxPinnedBytes ||
            options_.reserved != 0)
            return failInitialization(Outcome::InvalidInput);
        if (options_.mipPolicy != cap::MipPolicy::Required && options_.mipPolicy != cap::MipPolicy::Disabled)
            return failInitialization(Outcome::Unsupported);
        try {
            if (!source_->isOpen())
                source_->open(&sourceInfo_);
            else
                sourceInfo_ = source_->getInfo();
            if (!source_->isOpen() || !sourceInfo_.isValid)
                return failInitialization(Outcome::SourceFailure);
            internal::imageByteSize(sourceInfo_.width, sourceInfo_.height, sourceInfo_.numChannels,
                                   getBytesPerChannel(sourceInfo_.format));
            internal::imageByteSize(sourceInfo_.width, sourceInfo_.height, 4,
                                   sourceInfo_.format == HIP_AD_FORMAT_UNSIGNED_INT8 ? 1 : sizeof(float));
            if (!sourceInfo_.numMipLevels ||
                sourceInfo_.numMipLevels > cv::fullMipCount(sourceInfo_.width, sourceInfo_.height))
                return failInitialization(Outcome::InvalidInput);
        } catch (const std::bad_alloc&) {
            return failInitialization(Outcome::HostOutOfMemory);
        } catch (const std::exception& error) {
            logMessage(LogLevel::Error, "Whole-mip source metadata: %s", error.what());
            return failInitialization(Outcome::SourceFailure);
        }
        baseLayout_.originalWidth = sourceInfo_.width;
        baseLayout_.originalHeight = sourceInfo_.height;
        baseLayout_.originalLevels = options_.mipPolicy == cap::MipPolicy::Disabled ? 1 :
            cv::fullMipCount(sourceInfo_.width, sourceInfo_.height);
        if (storageDesc_.maxMipLevel)
            baseLayout_.originalLevels = std::min(baseLayout_.originalLevels, storageDesc_.maxMipLevel);
        auto outcome = cv::allocateLoaderIncarnation(incarnation_);
        if (outcome != Outcome::Success)
            return failInitialization(outcome);
        entries_.resize(options_.maxSamplers);
        descriptors_.reserve(options_.maxSamplers);
        anisotropy_.reserve(options_.maxSamplers);
        anisoStatus_.reserve(options_.maxSamplers);
        cubicEnabled_.resize(options_.maxSamplers,false);
        requests_.resize(options_.maxRequests);
        hipError_t error = calls_.call(HipOperation::GetDevice, [&] { return hipGetDevice(&device_); });
        if (error != hipSuccess)
            return failInitialization(hipResult(cap::Operation::SelectDevice, error));
        deviceKnown_ = true;
        error = calls_.call(HipOperation::Initialize, [] { return hipFree(nullptr); });
        if (error == hipSuccess)
            error = calls_.call(HipOperation::GetContext, [&] { return hipCtxGetCurrent(&ownerContext_); });
        if (error != hipSuccess)
            return failInitialization(hipResult(cap::Operation::SelectDevice, error));
        hipDeviceProp_t properties{};
        error = hipGetDeviceProperties(&properties, device_);
        if (error == hipSuccess) error = hipRuntimeGetVersion(&status_.device.runtimeVersion);
        if (error == hipSuccess) error = hipDriverGetVersion(&status_.device.driverVersion);
        if (error != hipSuccess)
            return failInitialization(hipResult(cap::Operation::SelectDevice, error));
        status_.device.device = device_;
        status_.device.ownerContext = reinterpret_cast<uint64_t>(ownerContext_);
        std::memcpy(status_.device.deviceName, properties.name, sizeof(status_.device.deviceName));
        std::memcpy(status_.device.architecture, properties.gcnArchName, sizeof(status_.device.architecture));
        const uint64_t overhead = 2 * tableBytes() + requestBytes() + sizeof(stats_);
        outcome = ledger_.reserve(cv::Charge::Overhead, overhead);
        if (outcome != Outcome::Success)
            return failInitialization(outcome);
        error = calls_.call(HipOperation::DeviceAllocation, [&] { return hipMalloc(&tables_[0], tableBytes()); });
        if (error == hipSuccess)
            error = calls_.call(HipOperation::DeviceAllocation, [&] { return hipMalloc(&tables_[1], tableBytes()); });
        if (error == hipSuccess)
            error = calls_.call(HipOperation::DeviceAllocation, [&] { return hipMalloc(&deviceRequests_, requestBytes()); });
        if (error == hipSuccess)
            error = calls_.call(HipOperation::DeviceAllocation, [&] { return hipMalloc(&deviceStats_, sizeof(stats_)); });
        if (error == hipSuccess)
            error = calls_.call(HipOperation::Initialize, [&] { return hipMemset(deviceStats_, 0, sizeof(stats_)); });
        if (error != hipSuccess)
            return failInitialization(hipResult(cap::Operation::AllocateArray, error));
        status_.initialization = Outcome::Success;
        status_.operation = Outcome::Success;
        refresh();
        return Outcome::Success;
    }

    Outcome failInitialization(Outcome outcome) {
        status_.initialization = outcome;
        return finish(outcome);
    }

    cv::RegistrationResult addSampler(const TextureDesc& desc, cv::SamplingPolicy sampling,
                                      const anisotropy_v1::Request& anisotropy) {
        std::lock_guard<std::mutex> operation(operationMutex_);
        begin();
        auto reject = [&](Outcome outcome) { finish(outcome); return cv::RegistrationResult{{}, outcome}; };
        if (status_.initialization != Outcome::Success) return reject(status_.initialization);
        if (cancelled_) return reject(Outcome::Cancelled);
        if (!internal::validDescriptor(desc) || sampling > cv::SamplingPolicy::AllowCoarsePreview)
            return reject(Outcome::InvalidInput);
        if (anisotropy.abi.version != anisotropy_v1::Version ||
            anisotropy.abi.byteSize != sizeof(anisotropy))
            return reject(Outcome::AbiMismatch);
        if (!internal::validAnisotropy(anisotropy)) return reject(Outcome::InvalidInput);
        if (anisotropy.requirement == anisotropy_v1::Requirement::RequireQualified)
            return reject(Outcome::Unsupported);
        if (desc.sRGB != storageDesc_.sRGB || desc.generateMipmaps != storageDesc_.generateMipmaps ||
            desc.maxMipLevel != storageDesc_.maxMipLevel ||
            (!desc.normalizedCoords && baseLayout_.originalLevels != 1))
            return reject(Outcome::Unsupported);
        auto deviceDesc = contractDescriptor(desc, sampling, options_.mipPolicy);
        deviceDesc.maxAnisotropy = anisotropy.maxAnisotropy ? anisotropy.maxAnisotropy : 1;
        for (size_t id = 0; id < descriptors_.size(); ++id) {
            if (descriptors_[id] == desc && entries_[id].descriptor == deviceDesc && anisotropy_[id] == anisotropy) {
                finish(Outcome::Success);
                return {entries_[id].texture.key, Outcome::Success};
            }
        }
        if (current_ || launchActive_ || retired_) return reject(Outcome::InvalidTransition);
        if (descriptors_.size() == options_.maxSamplers) return reject(Outcome::CapacityExhausted);
        const uint32_t id = static_cast<uint32_t>(descriptors_.size());
        descriptors_.push_back(desc);
        anisotropy_.push_back(anisotropy);
        anisoStatus_.emplace_back();
        Entry entry;
        entry.texture.key = {id, 1, incarnation_};
        entry.texture.revision = 1;
        entry.texture.mips = baseLayout_;
        entry.texture.state = cv::RegistrationState::Live;
        entry.descriptor = deviceDesc;
        entries_[id] = entry;
        finish(Outcome::Success);
        return {entry.texture.key, Outcome::Success};
    }

    Outcome enableCubicV1(cv::GpuKey key) {
        std::lock_guard<std::mutex> operation(operationMutex_);
        if (key.slot >= descriptors_.size() || !(entries_[key.slot].texture.key == key))
            return Outcome::InvalidKey;
        if (current_ || retired_ || launchActive_) return Outcome::InvalidTransition;
        if (!descriptors_[key.slot].normalizedCoords || descriptors_[key.slot].filterMode != hipFilterModeLinear)
            return Outcome::Unsupported;
        auto outcome = usable();
        if (outcome != Outcome::Success) return outcome;
        if (cubicEntries_.empty()) {
            try { cubicEntries_.resize(options_.maxSamplers); }
            catch (const std::bad_alloc&) { return Outcome::HostOutOfMemory; }
            outcome = ledger_.reserve(cv::Charge::Overhead,2*cubicTableBytes());
            if (outcome != Outcome::Success) { cubicEntries_.clear(); return outcome; }
        }
        for (auto& table : cubicTables_) {
            if (table) continue;
            const auto error = calls_.call(HipOperation::DeviceAllocation, [&] {
                return hipMalloc(&table,cubicTableBytes());
            });
            if (error != hipSuccess) return hipResult(cap::Operation::Publish,error);
        }
        cubicEnabled_[key.slot] = true;
        return Outcome::Success;
    }

    Outcome prepareCubicV1(hipStream_t stream, cubic_v1::DeviceContext& context) {
        if (context.abi.version != cubic_v1::Version || context.abi.byteSize != sizeof(context))
            return Outcome::AbiMismatch;
        if (!cubicTables_[0] || !cubicTables_[1]) return Outcome::InvalidTransition;
        DeviceContext whole;
        const auto outcome = prepare(stream,whole);
        if (outcome != Outcome::Success) return outcome;
        context = {};
        context.backing = cubic_v1::Backing::WholeMip;
        context.whole = whole;
        context.incarnation = incarnation_;
        context.entries = cubicTables_[activeTable_];
        context.count = static_cast<uint32_t>(descriptors_.size());
        return Outcome::Success;
    }

    Outcome getAnisotropyStatusV1(cv::GpuKey key, anisotropy_v1::Status& result) const {
        if (result.abi.version != anisotropy_v1::Version || result.abi.byteSize != sizeof(result))
            return Outcome::AbiMismatch;
        std::lock_guard<std::mutex> operation(operationMutex_);
        if (key.slot >= descriptors_.size() || !(entries_[key.slot].texture.key == key))
            return Outcome::InvalidKey;
        result = {};
        result.requested = anisotropy_[key.slot];
        {
            std::lock_guard<std::mutex> lock(statusMutex_);
            result.texture = status_.device;
        }
        result.texture.textureId = key.slot;
        result.texture.requested = descriptors_[key.slot];
        result.texture.submittedSampler = current_ ? current_->submitted[key.slot] :
            anisoStatus_[key.slot].submittedSampler;
        result.texture.returnedSampler = current_ ? current_->returned[key.slot] :
            anisoStatus_[key.slot].returnedSampler;
        result.texture.submitted = current_ ? 1 : anisoStatus_[key.slot].submitted;
        result.texture.returned = current_ ? 1 : anisoStatus_[key.slot].returned;
        result.limitations = anisotropy_v1::UnqualifiedBehavior;
        if (!result.requested.maxAnisotropy) result.limitations |= anisotropy_v1::LegacySetting;
        if (result.texture.returned &&
            result.texture.returnedSampler.maxAnisotropy != result.requested.maxAnisotropy)
            result.limitations |= anisotropy_v1::DescriptorMismatch;
        result.samplerSupport = current_ ? cap::Support::OperationSupported : cap::Support::Unknown;
        return Outcome::Success;
    }

    Outcome resize(uint32_t first) {
        std::lock_guard<std::mutex> operation(operationMutex_);
        begin();
        return finish(resizeLocked(first));
    }

    Outcome prepare(hipStream_t stream, DeviceContext& context) {
        std::lock_guard<std::mutex> operation(operationMutex_);
        begin();
        auto outcome = usable();
        if (outcome != Outcome::Success) return finish(outcome);
        if (context.abi.version != Version || context.abi.byteSize != sizeof(DeviceContext))
            return finish(Outcome::AbiMismatch);
        if (launchActive_ || descriptors_.empty()) return finish(Outcome::InvalidTransition);
        hipDevice_t streamDevice = device_;
        if (stream) {
            const auto streamError = hipStreamGetDevice(stream, &streamDevice);
            if (streamError != hipSuccess) return finish(hipResult(cap::Operation::SelectDevice, streamError));
            if (streamDevice != device_) return finish(Outcome::InvalidInput);
        }
        outcome = fence();
        if (outcome != Outcome::Success) return finish(outcome);
        // Publish/reset synchronously: no queued transfer borrows mutable host data.
        outcome = uploadTable(current_.get(), emptyOutcome_);
        if (outcome != Outcome::Success) return finish(outcome);
        auto error = calls_.call(HipOperation::Initialize, [&] {
            return hipMemset(deviceStats_, 0, sizeof(stats_));
        });
        // hipMemset may return before its default-stream work completes.
        // A nonblocking consuming stream does not inherit that dependency.
        if (error == hipSuccess)
            error = calls_.call(HipOperation::Initialize, [] { return hipStreamSynchronize(nullptr); });
        if (error != hipSuccess) return finish(hipResult(cap::Operation::Publish, error));
        if (current_) {
            try {
                outcome = current_->lifetime.beginConsumer(consumer_);
            } catch (const std::bad_alloc&) {
                consumer_ = 0;
                return finish(Outcome::HostOutOfMemory);
            }
            if (outcome != Outcome::Success) return finish(outcome);
        }
        launchActive_ = true;
        context = {{Version, sizeof(DeviceContext)}, incarnation_, tables_[activeTable_], deviceRequests_,
                   &deviceStats_->count, &deviceStats_->overflow,
                   static_cast<uint32_t>(descriptors_.size()), options_.maxRequests};
        return finish(Outcome::Success);
    }

    Outcome processRequests() {
        std::lock_guard<std::mutex> operation(operationMutex_);
        begin();
        auto outcome = usable();
        if (outcome != Outcome::Success) return finish(outcome);
        if (!launchActive_) return finish(Outcome::InvalidTransition);
        outcome = fence();
        if (outcome != Outcome::Success) return finish(outcome);
        outcome = readRequestStats();
        if (outcome != Outcome::Success) return finish(outcome);
        launchActive_ = false;
        if (stats_.overflow || stats_.count > options_.maxRequests)
            return finish(Outcome::RequestOverflow);
        if (!stats_.count) return finish(Outcome::Success);
        const auto error = hipMemcpy(requests_.data(), deviceRequests_, stats_.count * sizeof(cv::RequestKey),
                                     hipMemcpyDeviceToHost);
        if (error != hipSuccess) return finish(hipResult(cap::Operation::Publish, error));
        uint32_t first = baseLayout_.originalLevels;
        uint32_t rejected = 0;
        for (uint32_t index = 0; index < stats_.count; ++index) {
            const auto& request = requests_[index];
            if (request.texture.slot >= descriptors_.size() ||
                !(request.texture == entries_[request.texture.slot].texture.key) ||
                request.revision != 1 || request.reserved != 0 ||
                request.originalMip >= baseLayout_.originalLevels) {
                ++rejected;
                continue;
            }
            first = std::min(first, request.originalMip);
        }
        {
            std::lock_guard<std::mutex> lock(statusMutex_);
            status_.rejectedRequests = rejected;
        }
        if (rejected) return finish(Outcome::InvalidKey);
        if (current_ && current_->mips.firstResidentMip <= first)
            return finish(Outcome::Success);
        return finish(resizeLocked(first));
    }

    Outcome unload() {
        std::lock_guard<std::mutex> operation(operationMutex_);
        begin();
        const auto outcome = usable();
        return finish(outcome == Outcome::Success ? unloadLocked(Outcome::Pending) : outcome);
    }

    Outcome cancel() {
        {
            std::lock_guard<std::mutex> lock(statusMutex_);
            cancelled_ = true;
        }
        std::lock_guard<std::mutex> operation(operationMutex_);
        return finish(unloadLocked(Outcome::Cancelled));
    }

    Outcome collectRetired() {
        std::lock_guard<std::mutex> operation(operationMutex_);
        auto outcome = selectDevice();
        if (outcome == Outcome::Success) outcome = collect();
        return finish(outcome);
    }

    Outcome getStatus(Status& result) const {
        if (result.abi.version != Version || result.abi.byteSize != sizeof(Status))
            return Outcome::AbiMismatch;
        std::lock_guard<std::mutex> lock(statusMutex_);
        result = status_;
        return Outcome::Success;
    }

    void shutdown() {
        cancelled_ = true;
        if (!deviceKnown_) return;
        // Public destruction requires the caller to have stopped submitting work.
        if (selectDevice() != Outcome::Success || fence() != Outcome::Success) {
            logMessage(LogLevel::Error, "Whole-mip destruction could not fence owning device; resources retained");
            return;
        }
        for (int attempt = 0; attempt < 2; ++attempt) {
            if (!retired_ && current_) retireCurrent();
            collect();
        }
        if (current_ || retired_)
            logMessage(LogLevel::Error, "Whole-mip destruction retained backing after cleanup failure");
        freeStaging();
        const auto freeDevice = [&](auto& pointer) {
            if (!pointer) return;
            const auto error = calls_.call(HipOperation::FreeDevice, [&] { return hipFree(pointer); });
            if (error == hipSuccess) pointer = nullptr;
            else cleanupError(cap::Operation::FreeDevice, error);
        };
        for (int attempt = 0; attempt < 2; ++attempt) {
            freeDevice(tables_[0]);
            freeDevice(tables_[1]);
            freeDevice(cubicTables_[0]);
            freeDevice(cubicTables_[1]);
            freeDevice(deviceRequests_);
            freeDevice(deviceStats_);
        }
        const bool allFreed = !tables_[0] && !tables_[1] && !cubicTables_[0] && !cubicTables_[1] &&
            !deviceRequests_ && !deviceStats_;
        if (allFreed && ledger_.charged(cv::Charge::Overhead))
            check(ledger_.release(cv::Charge::Overhead, ledger_.charged(cv::Charge::Overhead)));
        refresh();
    }

private:
    size_t tableBytes() const { return options_.maxSamplers * sizeof(Entry); }
    size_t cubicTableBytes() const { return options_.maxSamplers * sizeof(cubic_v1::Entry); }
    size_t requestBytes() const { return options_.maxRequests * sizeof(cv::RequestKey); }

    Outcome readRequestStats() {
        const auto error = hipMemcpy(&stats_, deviceStats_, sizeof(stats_), hipMemcpyDeviceToHost);
        if (error != hipSuccess) return hipResult(cap::Operation::Publish, error);
        std::lock_guard<std::mutex> lock(statusMutex_);
        status_.requestCount = stats_.count;
        status_.requestOverflow = stats_.overflow;
        status_.rejectedRequests = 0;
        return Outcome::Success;
    }

    void check(Outcome outcome) const {
        if (outcome != Outcome::Success)
            throw std::logic_error("Whole-mip internal lifetime/accounting invariant failed");
    }
    void begin() {
        std::lock_guard<std::mutex> lock(statusMutex_);
        status_.operation = Outcome::Pending;
        status_.primary = {};
        status_.cleanup = {};
    }
    Outcome finish(Outcome outcome) {
        refresh();
        std::lock_guard<std::mutex> lock(statusMutex_);
        status_.operation = outcome;
        if (outcome != Outcome::Success)
            logMessage(LogLevel::Error, "Whole-mip operation outcome %u", static_cast<unsigned>(outcome));
        return outcome;
    }
    Outcome hipResult(cap::Operation operation, hipError_t error) {
        const auto failure = internal::hipFailure(operation, error);
        std::lock_guard<std::mutex> lock(statusMutex_);
        if (status_.primary.outcome == Outcome::Success) status_.primary = failure;
        return failure.outcome;
    }
    Outcome cleanupError(cap::Operation operation, hipError_t error) {
        auto failure = internal::hipFailure(operation, error);
        if (operation == cap::Operation::FreeHost && error == hipErrorOutOfMemory)
            failure.outcome = Outcome::HostOutOfMemory;
        std::lock_guard<std::mutex> lock(statusMutex_);
        if (status_.cleanup.outcome == Outcome::Success) status_.cleanup = failure;
        return failure.outcome;
    }
    Outcome usable() {
        if (status_.initialization != Outcome::Success) return status_.initialization;
        if (cancelled_) return Outcome::Cancelled;
        return selectDevice();
    }
    Outcome selectDevice() {
        if (!deviceKnown_) return Outcome::RuntimeFailure;
        auto error = calls_.call(HipOperation::SelectDevice, [&] { return hipSetDevice(device_); });
        if (error == hipSuccess && ownerContext_) {
            hipCtx_t context = nullptr;
            error = calls_.call(HipOperation::GetContext, [&] { return hipCtxGetCurrent(&context); });
            if (error == hipSuccess && context != ownerContext_) error = hipErrorInvalidContext;
        }
        return error == hipSuccess ? Outcome::Success : hipResult(cap::Operation::SelectDevice, error);
    }
    Outcome fence() {
        auto outcome = selectDevice();
        if (outcome != Outcome::Success) return outcome;
        auto error = calls_.call(HipOperation::SynchronizeConsumers, [] { return hipDeviceSynchronize(); });
        if (error != hipSuccess) return hipResult(cap::Operation::Publish, error);
        if (consumer_) {
            check(current_->lifetime.completeConsumer(consumer_));
            consumer_ = 0;
        }
        return Outcome::Success;
    }
    void refresh() {
        std::lock_guard<std::mutex> lock(statusMutex_);
        status_.mips = current_ ? current_->mips : baseLayout_;
        status_.numSamplers = static_cast<uint32_t>(descriptors_.size());
        status_.residentBytes = ledger_.charged(cv::Charge::Resident);
        status_.pendingBytes = ledger_.charged(cv::Charge::Pending);
        status_.retiringBytes = ledger_.charged(cv::Charge::Retiring);
        status_.overheadBytes = ledger_.charged(cv::Charge::Overhead);
        status_.managedPeakBytes = ledger_.peak();
        status_.pinnedBytes = stagingBytes_;
        status_.pinnedPeakBytes = std::max(status_.pinnedPeakBytes, stagingBytes_);
        auto& device = status_.device;
        device.originalWidth = baseLayout_.originalWidth;
        device.originalHeight = baseLayout_.originalHeight;
        device.originalLevels = baseLayout_.originalLevels;
        device.firstResidentMip = status_.mips.firstResidentMip;
        device.lastResidentMip = current_ ? baseLayout_.originalLevels - 1 : UINT32_MAX;
        device.resourceWidth = status_.mips.resourceWidth;
        device.resourceHeight = status_.mips.resourceHeight;
        device.resourceLevels = status_.mips.resourceLevels;
        device.resource = !current_ ? cap::Resource::None :
            (current_->storage.mipmapArray ? cap::Resource::MipmappedArray : cap::Resource::Array);
        device.reason = options_.mipPolicy == cap::MipPolicy::Disabled ? cap::Reason::Disabled :
            (baseLayout_.originalLevels == 1 ? (storageDesc_.maxMipLevel == 1 ?
                cap::Reason::LevelLimit : cap::Reason::Singleton) : cap::Reason::None);
        device.payloadBytes = status_.residentBytes;
        device.requested = descriptors_.empty() ? storageDesc_ : descriptors_.front();
        device.submitted = device.returned = current_ ? 1 : 0;
        device.submittedSampler = current_ ? current_->submittedSampler : hipTextureDesc{};
        device.returnedSampler = current_ ? current_->returnedSampler : hipTextureDesc{};
        status_.submittedMaxAnisotropy = device.submittedSampler.maxAnisotropy;
        device.capability = current_ ? cap::Support::OperationSupported : cap::Support::Unknown;
        device.primary = status_.primary;
        device.cleanup = status_.cleanup;
        device.published = current_ ? 1 : 0;
        device.state = cancelled_ ? cap::State::Cancelled :
            (current_ ? cap::State::Resident : cap::State::Unloaded);
        device.policy = options_.mipPolicy;
    }
    Outcome uploadTable(const Backing* backing, Outcome empty, bool activate = true, bool cleanup = false) {
        for (size_t id = 0; id < descriptors_.size(); ++id) {
            auto& texture = entries_[id].texture;
            texture.mips = backing ? backing->mips : baseLayout_;
            texture.textureObject = backing ? reinterpret_cast<uint64_t>(backing->samplers[id]) : 0;
            texture.residency = backing ? Outcome::Success : empty;
        }
        auto error = calls_.call(HipOperation::PublishMappings, [&] {
            return hipMemcpy(tables_[1 - activeTable_], entries_.data(), tableBytes(), hipMemcpyHostToDevice);
        });
        if (error == hipSuccess && cubicTables_[0] && cubicTables_[1]) {
            for (size_t id=0; id<descriptors_.size(); ++id) {
                auto& entry = cubicEntries_[id];
                entry = {};
                if (!cubicEnabled_[id]) continue;
                if (backing) entry = backing->cubic[id];
                entry.texture = entries_[id].texture;
                entry.descriptor = entries_[id].descriptor;
            }
            error = calls_.call(HipOperation::PublishMappings, [&] {
                return hipMemcpy(cubicTables_[1-activeTable_],cubicEntries_.data(),
                    cubicTableBytes(),hipMemcpyHostToDevice);
            });
        }
        if (error != hipSuccess) return cleanup ? cleanupError(cap::Operation::Publish, error) :
                                                 hipResult(cap::Operation::Publish, error);
        if (activate) activeTable_ = 1 - activeTable_;
        return Outcome::Success;
    }
    Outcome freeStaging() {
        if (!staging_) return Outcome::Success;
        const auto error = calls_.call(HipOperation::FreeHost, [&] { return hipHostFree(staging_); });
        if (error != hipSuccess) return cleanupError(cap::Operation::FreeHost, error);
        staging_ = nullptr;
        stagingBytes_ = 0;
        return Outcome::Success;
    }
    Outcome collect() {
        if (staging_ || retired_) {
            // Also fence failed transfers before releasing their host storage.
            // A synchronization failure retains both handles and charges.
            const auto error = calls_.call(HipOperation::SynchronizeConsumers, [] {
                return hipDeviceSynchronize();
            });
            if (error != hipSuccess) return cleanupError(cap::Operation::Publish, error);
        }
        const auto stagingResult = freeStaging();
        if (!retired_) return stagingResult;
        auto& backing = *retired_;
        auto outcome = backing.lifetime.canDestroy();
        if (outcome != Outcome::Success) return outcome;
        for (auto& entry : backing.cubic) {
            const auto error = internal::destroyCubicPoints(calls_,entry.points);
            if (error != hipSuccess) return cleanupError(cap::Operation::DestroySampler,error);
        }
        for (auto& sampler : backing.samplers) {
            if (!sampler) continue;
            const auto error = calls_.call(HipOperation::DestroySampler, [&] { return hipDestroyTextureObject(sampler); });
            if (error != hipSuccess) {
                const auto failure = internal::hipFailure(cap::Operation::DestroySampler, error);
                backing.lifetime.destroyed({failure.outcome, cv::Operation::Destroy, failure.rawHipError});
                return cleanupError(cap::Operation::DestroySampler, error);
            }
            sampler = 0;
        }
        hipError_t error = hipSuccess;
        auto operation = cap::Operation::FreeMipmapped;
        if (backing.storage.mipmapArray) {
            error = calls_.call(HipOperation::FreeMipmapped, [&] { return hipFreeMipmappedArray(backing.storage.mipmapArray); });
            if (error == hipSuccess) backing.storage.mipmapArray = nullptr;
        }
        if (error == hipSuccess && backing.storage.array) {
            operation = cap::Operation::FreeArray;
            error = calls_.call(HipOperation::FreeArray, [&] { return hipFreeArray(backing.storage.array); });
            if (error == hipSuccess) backing.storage.array = nullptr;
        }
        if (error != hipSuccess) {
            const auto failure = internal::hipFailure(operation, error);
            backing.lifetime.destroyed({failure.outcome, cv::Operation::Destroy, failure.rawHipError});
            return cleanupError(operation, error);
        }
        check(backing.lifetime.destroyed());
        retired_.reset();
        return stagingResult;
    }
    void retireCurrent() {
        check(current_->lifetime.retire());
        check(current_->lifetime.invalidateMapping());
        retired_ = std::move(current_);
    }
    Outcome unloadLocked(Outcome empty) {
        auto outcome = fence();
        if (outcome != Outcome::Success) return outcome;
        outcome = collect();
        if (outcome != Outcome::Success) return outcome;
        if (tables_[0] && tables_[1]) {
            outcome = uploadTable(nullptr, empty);
            if (outcome != Outcome::Success) return outcome;
        }
        emptyOutcome_ = empty;
        launchActive_ = false;
        if (current_) retireCurrent();
        return collect();
    }

    Outcome missingFailure(Outcome failure) {
        {
            std::lock_guard<std::mutex> lock(statusMutex_);
            if (status_.primary.outcome == Outcome::Success && status_.cleanup.outcome == Outcome::Success)
                status_.primary = {failure, cap::Operation::None, 0};
        }
        if (!current_) {
            emptyOutcome_ = failure;
            // The caller has fenced consumers. Publication failure is recorded
            // separately; the host's original terminal outcome is not lost.
            uploadTable(nullptr, failure, true, true);
        }
        return failure;
    }

    Outcome resizeLocked(uint32_t first) {
        auto outcome = usable();
        if (outcome != Outcome::Success) return outcome;
        if (descriptors_.empty()) return Outcome::InvalidTransition;
        if (first >= baseLayout_.originalLevels) return Outcome::InvalidInput;
        {
            std::lock_guard<std::mutex> lock(statusMutex_);
            status_.desiredFirstMip = first;
        }
        outcome = fence();
        if (outcome != Outcome::Success) {
            if (!current_) emptyOutcome_ = outcome;
            return outcome;
        }
        if (launchActive_) {
            outcome = readRequestStats();
            if (outcome != Outcome::Success) return outcome;
            if (stats_.overflow || stats_.count > options_.maxRequests) return Outcome::RequestOverflow;
            if (stats_.count != 0) return Outcome::InvalidTransition;
            launchActive_ = false;
        }
        outcome = collect();
        if (outcome != Outcome::Success) return missingFailure(outcome);
        if (current_ && current_->mips.firstResidentMip == first) return Outcome::Success;
        std::unique_ptr<Backing> candidate;
        try {
            candidate = std::make_unique<Backing>(ledger_, entries_[0].texture.key, descriptors_.size());
        } catch (const std::bad_alloc&) {
            return missingFailure(Outcome::HostOutOfMemory);
        }
        candidate->mips = baseLayout_;
        candidate->mips.firstResidentMip = first;
        candidate->mips.resourceWidth = cv::mipDimension(sourceInfo_.width, first);
        candidate->mips.resourceHeight = cv::mipDimension(sourceInfo_.height, first);
        candidate->mips.resourceLevels = baseLayout_.originalLevels - first;
        const bool floating = sourceInfo_.format != HIP_AD_FORMAT_UNSIGNED_INT8;
        auto& storage = candidate->storage;
        storage.memoryUsage = internal::mipImageByteSize(candidate->mips.resourceWidth,
            candidate->mips.resourceHeight, floating ? sizeof(float) : 1, candidate->mips.resourceLevels);
        if (storage.memoryUsage > options_.maxManagedBytes - ledger_.charged(cv::Charge::Overhead))
            return missingFailure(Outcome::DemandTooLarge);
        const auto pinnedBytes = internal::imageByteSize(candidate->mips.resourceWidth,
            candidate->mips.resourceHeight, 4, floating ? sizeof(float) : 1);
        if (pinnedBytes > options_.maxPinnedBytes) return missingFailure(Outcome::DemandTooLarge);
        outcome = candidate->lifetime.reserve(storage.memoryUsage);
        if (outcome != Outcome::Success) return missingFailure(outcome);
        CandidateOwner candidateOwner{*this, candidate};
        refresh();
        auto rollback = [&](Outcome failure) {
            if (failure == Outcome::Cancelled) check(candidate->lifetime.cancel());
            else check(candidate->lifetime.retire());
            check(candidate->lifetime.invalidateMapping());
            retired_ = std::move(candidate);
            collect();
            return missingFailure(failure);
        };
        auto error = calls_.call(HipOperation::HostAllocation, [&] { return hipHostMalloc(&staging_, pinnedBytes); });
        if (error != hipSuccess) {
            outcome = hipResult(cap::Operation::AllocateHost, error);
            if (error == hipErrorOutOfMemory) {
                outcome = Outcome::HostOutOfMemory;
                std::lock_guard<std::mutex> lock(statusMutex_);
                status_.primary.outcome = outcome;
            }
            return rollback(outcome);
        }
        stagingBytes_ = pinnedBytes;
        refresh();
        const auto channel = floating ? hipCreateChannelDesc<float4>() : hipCreateChannelDesc<uchar4>();
        if (candidate->mips.resourceLevels > 1) {
            error = calls_.call(HipOperation::AllocateMipmapped, [&] {
                return hipMallocMipmappedArray(&storage.mipmapArray, &channel,
                    make_hipExtent(candidate->mips.resourceWidth, candidate->mips.resourceHeight, 0),
                    candidate->mips.resourceLevels);
            });
        } else {
            error = calls_.call(HipOperation::AllocateArray, [&] {
                return hipMallocArray(&storage.array, &channel, candidate->mips.resourceWidth,
                                      candidate->mips.resourceHeight);
            });
        }
        if (error != hipSuccess)
            return rollback(hipResult(storage.mipmapArray || candidate->mips.resourceLevels > 1 ?
                cap::Operation::AllocateMipmapped : cap::Operation::AllocateArray, error));
        check(candidate->lifetime.allocated());
        internal::MipSuffixReadStats readStats;
        ReadStatsOwner readStatsOwner{*this, readStats};
        outcome = internal::visitMipSuffix(*source_, sourceInfo_, storageDesc_.sRGB,
            storageDesc_.generateMipmaps, first, baseLayout_.originalLevels, options_.maxDecodedBytes, readStats,
            [&](uint32_t originalMip, const internal::ImageData& image) {
                if (cancelled_) return Outcome::Cancelled;
                hipArray_t array = storage.array;
                if (storage.mipmapArray) {
                    const auto getError = calls_.call(HipOperation::GetLevel, [&] {
                        return hipGetMipmappedArrayLevel(&array, storage.mipmapArray, originalMip - first);
                    });
                    if (getError != hipSuccess) return hipResult(cap::Operation::GetLevel, getError);
                }
                std::memcpy(staging_, image.data(), image.sizeBytes());
                const auto uploadError = calls_.call(HipOperation::Upload, [&] {
                    return hipMemcpy2DToArray(array, 0, 0, staging_, image.rowBytes(),
                                             image.rowBytes(), image.height, hipMemcpyHostToDevice);
                });
                if (uploadError != hipSuccess) return hipResult(cap::Operation::Upload, uploadError);
                std::lock_guard<std::mutex> lock(statusMutex_);
                status_.uploadedBytes += image.sizeBytes();
                return Outcome::Success;
            });
        {
            std::lock_guard<std::mutex> lock(statusMutex_);
            if (outcome != Outcome::Success && status_.primary.outcome == Outcome::Success)
                status_.primary = {outcome, cap::Operation::SourceRead, 0};
        }
        if (outcome != Outcome::Success) return rollback(outcome);
        check(candidate->lifetime.uploaded());
        hipResourceDesc resource{};
        if (storage.mipmapArray) {
            resource.resType = hipResourceTypeMipmappedArray;
            resource.res.mipmap.mipmap = storage.mipmapArray;
        } else {
            resource.resType = hipResourceTypeArray;
            resource.res.array.array = storage.array;
        }
        for (size_t id = 0; id < descriptors_.size(); ++id) {
            const auto sampler = internal::makeSampler(descriptors_[id], floating,
                storage.mipmapArray != nullptr, candidate->mips.resourceLevels, anisotropy_[id].maxAnisotropy);
            candidate->submitted[id] = sampler;
            anisoStatus_[id].submittedSampler = sampler;
            anisoStatus_[id].submitted = 1;
            anisoStatus_[id].returned = 0;
            error = calls_.call(HipOperation::CreateSampler, [&] {
                return hipCreateTextureObject(&candidate->samplers[id], &resource, &sampler, nullptr);
            });
            if (error != hipSuccess) return rollback(hipResult(cap::Operation::CreateSampler, error));
            hipTextureDesc returned{};
            error = calls_.call(HipOperation::ReadSampler, [&] {
                return hipGetTextureObjectTextureDesc(&returned, candidate->samplers[id]);
            });
            if (error != hipSuccess) return rollback(hipResult(cap::Operation::ReadSampler, error));
            calls_.observeSampler(returned);
            candidate->returned[id] = returned;
            anisoStatus_[id].returnedSampler = returned;
            anisoStatus_[id].returned = 1;
            if (returned.maxAnisotropy != sampler.maxAnisotropy) {
                {
                    std::lock_guard<std::mutex> lock(statusMutex_);
                    status_.primary = {Outcome::Unsupported, cap::Operation::ReadSampler, 0};
                }
                return rollback(Outcome::Unsupported);
            }
            if (cubicEnabled_[id]) {
                candidate->cubic[id].pointSRGB = descriptors_[id].sRGB && !floating;
                error = internal::createCubicPoints(calls_,storage.array,storage.mipmapArray,
                    candidate->mips.resourceLevels,descriptors_[id],floating,candidate->cubic[id].points);
                if (error != hipSuccess) return rollback(hipResult(cap::Operation::CreateSampler,error));
            }
            if (id == 0) {
                candidate->submittedSampler = sampler;
                candidate->returnedSampler = returned;
            }
        }
        check(candidate->lifetime.samplersCreated());
        outcome = freeStaging();
        if (outcome != Outcome::Success) return rollback(outcome);
        // cancel and the commit have one ordering point; source I/O and fences
        // never hold this mutex. No caller may submit consumers during resize.
        {
            std::unique_lock<std::mutex> lock(statusMutex_);
            if (cancelled_) {
                lock.unlock();
                return rollback(Outcome::Cancelled);
            }
            lock.unlock();
            outcome = uploadTable(candidate.get(), Outcome::Pending, false);
            if (outcome != Outcome::Success) return rollback(outcome);
            lock.lock();
            if (cancelled_) {
                lock.unlock();
                return rollback(Outcome::Cancelled);
            }
            activeTable_ = 1 - activeTable_;
            check(candidate->lifetime.publish(entries_[0].texture.key, 1));
            if (current_) {
                retireCurrent();
                ++status_.replacements;
            }
            current_ = std::move(candidate);
            emptyOutcome_ = Outcome::Pending;
        }
        return collect();
    }

    std::shared_ptr<ImageSource> source_;
    TextureInfo sourceInfo_{};
    TextureDesc storageDesc_;
    Options options_;
    internal::HipCalls calls_;
    cv::BudgetLedger ledger_;
    cv::MipLayout baseLayout_{};
    std::vector<Entry> entries_;
    std::vector<TextureDesc> descriptors_;
    std::vector<anisotropy_v1::Request> anisotropy_;
    std::vector<cap::Status> anisoStatus_;
    std::vector<bool> cubicEnabled_;
    std::vector<cubic_v1::Entry> cubicEntries_;
    cubic_v1::Entry* cubicTables_[2]{};
    std::vector<cv::RequestKey> requests_;
    Entry* tables_[2]{};
    unsigned int activeTable_ = 0;
    cv::RequestKey* deviceRequests_ = nullptr;
    internal::RequestStats* deviceStats_ = nullptr;
    internal::RequestStats stats_{};
    std::unique_ptr<Backing> current_, retired_;
    void* staging_ = nullptr;
    uint64_t stagingBytes_ = 0;
    int device_ = 0;
    bool deviceKnown_ = false;
    hipCtx_t ownerContext_ = nullptr;
    uint64_t incarnation_ = 0, consumer_ = 0;
    bool launchActive_ = false;
    Outcome emptyOutcome_ = Outcome::Pending;
    std::atomic<bool> cancelled_{false};
    mutable std::mutex operationMutex_;
    mutable std::mutex statusMutex_;
    Status status_{};
};

Texture::Texture(std::shared_ptr<ImageSource> source, const TextureDesc& descriptor, const Options& options)
    : impl_(std::make_unique<Impl>(std::move(source), descriptor, options)) {
    try {
        impl_->initialize();
    } catch (const std::bad_alloc&) {
        impl_->failInitialization(Outcome::HostOutOfMemory);
    }
}
Texture::~Texture() { impl_->shutdown(); }
cv::RegistrationResult Texture::addSampler(const TextureDesc& desc, cv::SamplingPolicy sampling,
                                           const anisotropy_v1::Request& anisotropy) {
    return impl_->addSampler(desc, sampling, anisotropy);
}
Outcome Texture::resize(uint32_t first) { return impl_->resize(first); }
Outcome Texture::unload() { return impl_->unload(); }
Outcome Texture::prepare(hipStream_t stream, DeviceContext& context) { return impl_->prepare(stream, context); }
Outcome Texture::processRequests() { return impl_->processRequests(); }
Outcome Texture::cancel() { return impl_->cancel(); }
Outcome Texture::collectRetired() { return impl_->collectRetired(); }
Outcome Texture::getStatus(Status& status) const { return impl_->getStatus(status); }
Outcome Texture::enableCubicV1(cv::GpuKey key) { return impl_->enableCubicV1(key); }
Outcome Texture::prepareCubicV1(hipStream_t stream, cubic_v1::DeviceContext& context) {
    return impl_->prepareCubicV1(stream,context);
}
Outcome Texture::getAnisotropyStatusV1(cv::GpuKey key, anisotropy_v1::Status& status) const {
    return impl_->getAnisotropyStatusV1(key,status);
}

} }
