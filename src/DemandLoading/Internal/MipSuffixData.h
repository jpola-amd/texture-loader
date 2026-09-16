// SPDX-License-Identifier: MIT
#pragma once

#include "ImageData.h"
#include <DemandLoading/Contracts.h>
#include <DemandLoading/Logging.h>

#include <new>
#include <utility>

namespace hip_demand {
namespace internal {

struct MipSuffixReadStats {
    uint64_t decodedPeakBytes = 0;
    uint64_t sourceBytes = 0;
    uint32_t authoredReads = 0;
    uint32_t generatedLevels = 0;
};

namespace mip_suffix_detail {

using contract_v1::Outcome;

inline uint64_t addBytes(uint64_t a, uint64_t b) {
    if (b > std::numeric_limits<uint64_t>::max() - a)
        throw std::overflow_error("Mip suffix byte accounting overflow");
    return a + b;
}

struct LevelBytes {
    uint64_t decoded;
    uint64_t native;
    uint64_t readPeak;
};

inline LevelBytes levelBytes(const TextureInfo& info, uint32_t level) {
    const auto width = contract_v1::mipDimension(info.width, level);
    const auto height = contract_v1::mipDimension(info.height, level);
    const bool floating = info.format != HIP_AD_FORMAT_UNSIGNED_INT8;
    const uint64_t decoded = imageByteSize(width, height, 4, floating ? sizeof(float) : 1);
    const uint64_t native = imageByteSize(width, height, info.numChannels,
                                         getBytesPerChannel(info.format));
    if (native > std::vector<unsigned char>{}.max_size() ||
        (floating ? decoded / sizeof(float) > std::vector<float>{}.max_size()
                  : decoded > std::vector<unsigned char>{}.max_size()))
        throw std::overflow_error("Mip suffix exceeds host container capacity");
    const bool direct = info.numChannels == 4 &&
        (info.format == HIP_AD_FORMAT_UNSIGNED_INT8 || info.format == HIP_AD_FORMAT_FLOAT);
    return {decoded, native, direct ? decoded : addBytes(decoded, native)};
}

// Freeze the allocation metadata used by readImageSource. A changed source
// header must not make that helper allocate outside the admitted reservation.
// This adapter neither owns nor closes the caller-owned source.
class RegisteredSource final : public ImageSource {
public:
    RegisteredSource(ImageSource& source, TextureInfo info, MipSuffixReadStats& stats)
        : source_(source), info_(info), stats_(stats) {}

    bool check(bool allowOpen) {
        if (!source_.isOpen() && allowOpen) {
            TextureInfo opened;
            source_.open(&opened);
            if (!(opened == info_)) return changed();
        }
        if (!source_.isOpen() || !(source_.getInfo() == info_)) return changed();
        return true;
    }

    void open(TextureInfo* result) override { if (result) *result = info_; }
    void close() override {}
    bool isOpen() const override { return true; }
    const TextureInfo& getInfo() const override { return info_; }
    bool readMipLevel(char* destination, unsigned int level, unsigned int width,
                      unsigned int height, hipStream_t stream = 0) override {
        if (!check(false)) return false;
        if (!source_.readMipLevel(destination, level, width, height, stream)) {
            logMessage(LogLevel::Error, "Whole-mip source read failed at original mip %u", level);
            return false;
        }
        // Count successful native input even if later validation/conversion fails.
        ++stats_.authoredReads;
        stats_.sourceBytes += imageByteSize(width, height, info_.numChannels,
                                            getBytesPerChannel(info_.format));
        return check(false);
    }
    bool readBaseColor(float4&) override { return false; }
    unsigned long long getNumBytesRead() const override { return stats_.sourceBytes; }
    double getTotalReadTime() const override { return 0; }

private:
    bool changed() const {
        logMessage(LogLevel::Error, "Whole-mip source metadata changed or source closed");
        return false;
    }

    ImageSource& source_;
    const TextureInfo info_;
    MipSuffixReadStats& stats_;
};

// Classify standard source/decode failures only. Caller callbacks and
// non-standard source exceptions remain the operation owner's responsibility.
template<class Work>
Outcome sourceWork(Work&& work) {
    try {
        return work() ? Outcome::Success : Outcome::SourceFailure;
    } catch (const std::bad_alloc&) {
        logMessage(LogLevel::Error, "Whole-mip source/decode host allocation failed");
        return Outcome::HostOutOfMemory;
    } catch (const std::exception& error) {
        logMessage(LogLevel::Error, "Whole-mip source/decode exception: %s", error.what());
        return Outcome::SourceFailure;
    }
}

} // namespace mip_suffix_detail

// Visits [firstMip, originalLevels) in original mip numbering. The callback is
// synchronous and must finish using image storage before returning. The caller
// keeps the source alive and immutable, and budgets its upload buffers separately.
//
// Stats reset per invocation. Peak counts reserved image/native payload (not
// allocator overhead or source-owned caches); rejected admission has peak zero.
// Authored reads/bytes count successful native reads, including dependencies.
// Generated levels count every completed reduction, including discarded levels.
template<class Visitor>
contract_v1::Outcome visitMipSuffix(ImageSource& source, const TextureInfo& registeredInfo,
                                   bool srgb, bool generateMissing, uint32_t firstMip,
                                   uint32_t originalLevels, uint64_t maxDecodedBytes,
                                   MipSuffixReadStats& stats, Visitor&& visitor) {
    using contract_v1::Outcome;
    using mip_suffix_detail::levelBytes;
    stats = {};
    const TextureInfo info = registeredInfo;
    uint32_t startMip = 0;
    uint64_t requiredPeak = 0;
    try {
        const uint32_t fullLevels = contract_v1::fullMipCount(info.width, info.height);
        if (!info.isValid || !info.numMipLevels || info.numMipLevels > fullLevels ||
            !originalLevels || originalLevels > fullLevels || firstMip >= originalLevels)
            return Outcome::InvalidInput;
        imageByteSize(info.width, info.height, info.numChannels, getBytesPerChannel(info.format));
        imageByteSize(info.width, info.height, 4,
                      info.format == HIP_AD_FORMAT_UNSIGNED_INT8 ? 1 : sizeof(float));
        startMip = std::min(firstMip, info.numMipLevels - 1);
        uint64_t previousBytes = 0;
        uint64_t sourceBytes = 0;
        // Preflight the entire dependency path without allocating or touching
        // the source, so even a later oversized reduction performs no I/O.
        for (uint32_t level = startMip; level < originalLevels; ++level) {
            const auto bytes = levelBytes(info, level);
            uint64_t live;
            if (level < info.numMipLevels) {
                live = bytes.readPeak;
                sourceBytes = mip_suffix_detail::addBytes(sourceBytes, bytes.native);
            } else {
                if (!srgb && info.format == HIP_AD_FORMAT_UNSIGNED_INT8) {
                    const uint64_t weight =
                        uint64_t{contract_v1::mipDimension(info.width, level - 1)} *
                        contract_v1::mipDimension(info.height, level - 1);
                    if (weight > std::numeric_limits<uint64_t>::max() / 255)
                        return Outcome::InvalidInput;
                }
                live = mip_suffix_detail::addBytes(previousBytes, bytes.decoded);
            }
            requiredPeak = std::max(requiredPeak, live);
            previousBytes = bytes.decoded;
        }
    } catch (const std::bad_alloc&) {
        return Outcome::HostOutOfMemory;
    } catch (const std::invalid_argument&) {
        return Outcome::InvalidInput;
    } catch (const std::overflow_error&) {
        return Outcome::InvalidInput;
    }
    if (!generateMissing && originalLevels > info.numMipLevels) return Outcome::Unsupported;
    if (requiredPeak > maxDecodedBytes) return Outcome::DemandTooLarge;

    mip_suffix_detail::RegisteredSource registered(source, info, stats);
    auto outcome = mip_suffix_detail::sourceWork([&] { return registered.check(true); });
    if (outcome != Outcome::Success) return outcome;

    ImageData image;
    for (uint32_t level = startMip; level < originalLevels; ++level) {
        const auto bytes = levelBytes(info, level);
        if (level < info.numMipLevels) {
            // clear() retains capacity: destroy the old payload before reserving
            // an independent authored read, including native/float conversion.
            image = ImageData{};
            stats.decodedPeakBytes = std::max(stats.decodedPeakBytes, bytes.readPeak);
            outcome = mip_suffix_detail::sourceWork([&] {
                if (!readImageSource(registered, image, level)) return false;
                if (srgb) linearizeFloatSRGB(image);
                return true;
            });
            if (outcome != Outcome::Success) return outcome;
        } else {
            outcome = mip_suffix_detail::sourceWork([&] { return registered.check(false); });
            if (outcome != Outcome::Success) return outcome;
            const uint64_t live = mip_suffix_detail::addBytes(image.sizeBytes(), bytes.decoded);
            stats.decodedPeakBytes = std::max(stats.decodedPeakBytes, live);
            try {
                ImageData next = downsampleImage(image, srgb);
                image = std::move(next);
            } catch (const std::bad_alloc&) {
                return Outcome::HostOutOfMemory;
            }
            ++stats.generatedLevels;
        }
        if (level >= firstMip) {
            outcome = visitor(level, static_cast<const ImageData&>(image));
            if (outcome != Outcome::Success) return outcome;
        }
    }
    return Outcome::Success;
}

} // namespace internal
} // namespace hip_demand
