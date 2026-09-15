// SPDX-License-Identifier: MIT
#pragma once

#include "DemandLoading/Contracts.h"

#include <atomic>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <utility>

namespace hip_demand {
class ImageSource;

namespace contract_v1 {

class NonWrappingCounter {
public:
    explicit NonWrappingCounter(uint64_t lastIssued = 0) : last_(lastIssued) {}

    Outcome next(uint64_t& value) {
        uint64_t previous = last_.load(std::memory_order_relaxed);
        while (previous != UINT64_MAX) {
            if (last_.compare_exchange_weak(previous, previous + 1, std::memory_order_relaxed)) {
                value = previous + 1;
                return Outcome::Success;
            }
        }
        value = 0;
        return Outcome::IdentityExhausted;
    }

private:
    std::atomic<uint64_t> last_;
};

// Process-local, library-owned incarnation issuer. No slot recycling is provided.
Outcome allocateLoaderIncarnation(uint64_t& incarnation);

inline Outcome advanceGeneration(uint32_t& generation) {
    if (generation == UINT32_MAX)
        return Outcome::IdentityExhausted;
    ++generation;
    return Outcome::Success;
}

namespace detail {
struct RegistrationControl {
    mutable std::mutex mutex;
    GpuKey key;
    ImageIdentity image;
    SamplerDesc descriptor;
    std::shared_ptr<ImageSource> source;
    uint64_t owners = 0;
    RegistrationState state = RegistrationState::Live;

    RegistrationControl(GpuKey gpuKey, ImageIdentity identity, SamplerDesc desc,
                        std::shared_ptr<ImageSource> imageSource)
        : key(gpuKey), image(identity), descriptor(desc), source(std::move(imageSource)) {}
};
} // namespace detail

class Registration;

class RegistrationLease {
public:
    RegistrationLease() = default;
    RegistrationLease(const RegistrationLease&) = delete;
    RegistrationLease& operator=(const RegistrationLease&) = delete;
    RegistrationLease(RegistrationLease&& other) noexcept : control_(std::move(other.control_)) {}
    RegistrationLease& operator=(RegistrationLease&& other) noexcept {
        if (this != &other) {
            if (control_)
                release();
            control_ = std::move(other.control_);
        }
        return *this;
    }
    ~RegistrationLease() {
        if (control_)
            release();
    }

    GpuKey key() const { return control_ ? control_->key : GpuKey{}; }
    Outcome release() {
        if (!control_)
            return Outcome::InvalidKey;
        auto control = std::move(control_);
        std::lock_guard<std::mutex> lock(control->mutex);
        if (--control->owners == 0)
            control->state = RegistrationState::Retiring;
        return Outcome::Success;
    }

private:
    explicit RegistrationLease(std::shared_ptr<detail::RegistrationControl> control)
        : control_(std::move(control)) {}
    std::shared_ptr<detail::RegistrationControl> control_;
    friend class Registration;
};

struct Acquisition {
    Outcome outcome = Outcome::InvalidKey;
    RegistrationLease lease{};
};

// The future registration registry owns this record; cache clients own leases.
// GPU keys and registry references do not count as registration acquisitions.
class Registration {
public:
    Registration(GpuKey key, ImageIdentity image, SamplerDesc desc,
                 std::shared_ptr<ImageSource> source = {}) {
        if (!valid(key) || image.token == 0 || image.revision == 0 || image.reserved != 0 ||
            image.nameSpace > ImageNamespace::Memory || validate(desc) != Outcome::Success)
            throw std::invalid_argument("Invalid contract registration identity or descriptor");
        control_ = std::make_shared<detail::RegistrationControl>(key, image, desc, std::move(source));
    }
    Registration(const Registration&) = delete;
    Registration& operator=(const Registration&) = delete;

    Acquisition acquire() {
        std::lock_guard<std::mutex> lock(control_->mutex);
        if (control_->state != RegistrationState::Live)
            return {Outcome::InvalidKey, {}};
        if (control_->owners == UINT64_MAX)
            return {Outcome::IdentityExhausted, {}};
        ++control_->owners;
        return {Outcome::Success, RegistrationLease(control_)};
    }
    uint64_t ownerCount() const {
        std::lock_guard<std::mutex> lock(control_->mutex);
        return control_->owners;
    }
    RegistrationState state() const {
        std::lock_guard<std::mutex> lock(control_->mutex);
        return control_->state;
    }
    bool matches(ImageIdentity image, const SamplerDesc& desc) const {
        return control_->image == image && control_->descriptor == desc;
    }

private:
    std::shared_ptr<detail::RegistrationControl> control_;
};

enum class Charge : uint32_t { Resident, Pending, Retiring, Temporary, Overhead, Count };

// Serialized primitives: the owning coordinator supplies exclusion, never a worker-held wait.
// Use independent ledgers for device payload, source cache, decoded data and pinned staging.
class BudgetLedger {
public:
    explicit BudgetLedger(uint64_t limit) : limit_(limit) {}
    BudgetLedger(const BudgetLedger&) = delete;
    BudgetLedger& operator=(const BudgetLedger&) = delete;

    Outcome reserve(Charge charge, uint64_t bytes) {
        if (charge >= Charge::Count || bytes == 0)
            return Outcome::InvalidInput;
        if (bytes > limit_)
            return Outcome::DemandTooLarge;
        if (bytes > limit_ - total_)
            return Outcome::Deferred;
        charges_[static_cast<uint32_t>(charge)] += bytes;
        total_ += bytes;
        if (total_ > peak_)
            peak_ = total_;
        return Outcome::Success;
    }
    Outcome transfer(Charge from, Charge to, uint64_t bytes) {
        if (from >= Charge::Count || to >= Charge::Count || from == to || bytes == 0 ||
            bytes > charged(from))
            return Outcome::InvalidInput;
        charges_[static_cast<uint32_t>(from)] -= bytes;
        charges_[static_cast<uint32_t>(to)] += bytes;
        return Outcome::Success;
    }
    Outcome release(Charge charge, uint64_t bytes) {
        if (charge >= Charge::Count || bytes == 0 || bytes > charged(charge))
            return Outcome::InvalidInput;
        charges_[static_cast<uint32_t>(charge)] -= bytes;
        total_ -= bytes;
        return Outcome::Success;
    }
    uint64_t charged(Charge charge) const {
        if (charge >= Charge::Count)
            throw std::invalid_argument("Invalid contract accounting charge");
        return charges_[static_cast<uint32_t>(charge)];
    }
    uint64_t total() const { return total_; }
    uint64_t peak() const { return peak_; }

private:
    uint64_t limit_;
    uint64_t charges_[static_cast<uint32_t>(Charge::Count)]{};
    uint64_t total_ = 0;
    uint64_t peak_ = 0;
};

enum class ResourceState : uint32_t {
    Empty, Reserved, Allocated, Uploaded, Ready, Published, Retiring, Destroyed
};

// One instance per backing allocation, shared by all sampler variants. This guards
// transitions; the runtime adapter must supply actual HIP completion/destruction evidence.
class ResourceLifecycle {
public:
    ResourceLifecycle(BudgetLedger& ledger, GpuKey key, uint64_t revision)
        : ledger_(ledger), key_(key), revision_(revision) {
        if (!valid(key) || revision == 0)
            throw std::invalid_argument("Invalid contract operation identity");
    }
    ResourceLifecycle(const ResourceLifecycle&) = delete;
    ResourceLifecycle& operator=(const ResourceLifecycle&) = delete;

    Outcome reserve(uint64_t bytes) {
        if (state_ != ResourceState::Empty)
            return Outcome::InvalidTransition;
        const Outcome outcome = ledger_.reserve(Charge::Pending, bytes);
        if (outcome == Outcome::Success) {
            bytes_ = bytes;
            state_ = ResourceState::Reserved;
        }
        return outcome;
    }
    Outcome allocated() { return advance(ResourceState::Reserved, ResourceState::Allocated); }
    Outcome uploaded() { return advance(ResourceState::Allocated, ResourceState::Uploaded); }
    Outcome samplersCreated() { return advance(ResourceState::Uploaded, ResourceState::Ready); }

    Outcome publish(GpuKey currentKey, uint64_t currentRevision) {
        if (!(currentKey == key_) || currentRevision != revision_)
            return Outcome::InvalidKey;
        if (state_ != ResourceState::Ready || !workers_.empty())
            return Outcome::InvalidTransition;
        const Outcome outcome = ledger_.transfer(Charge::Pending, Charge::Resident, bytes_);
        if (outcome == Outcome::Success) {
            state_ = ResourceState::Published;
            mappingInvalidated_ = false;
        }
        return outcome;
    }

    Outcome beginConsumer(uint64_t& token) {
        if (state_ != ResourceState::Published) {
            token = 0;
            return Outcome::InvalidTransition;
        }
        return begin(consumers_, token);
    }
    Outcome completeConsumer(uint64_t token) {
        return consumers_.erase(token) == 1 ? Outcome::Success : Outcome::InvalidInput;
    }
    Outcome beginWorker(uint64_t& token) {
        if (state_ < ResourceState::Reserved || state_ > ResourceState::Ready) {
            token = 0;
            return Outcome::InvalidTransition;
        }
        return begin(workers_, token);
    }
    Outcome completeWorker(uint64_t token) {
        // Completion includes closure, source/staging, event and request-readback use.
        return workers_.erase(token) == 1 ? Outcome::Success : Outcome::InvalidInput;
    }

    Outcome retire() {
        if (state_ < ResourceState::Reserved || state_ > ResourceState::Published)
            return Outcome::InvalidTransition;
        const Charge from = state_ == ResourceState::Published ? Charge::Resident : Charge::Pending;
        const Outcome outcome = ledger_.transfer(from, Charge::Retiring, bytes_);
        if (outcome == Outcome::Success)
            state_ = ResourceState::Retiring;
        return outcome;
    }
    Outcome cancel() {
        if (state_ < ResourceState::Reserved || state_ > ResourceState::Published)
            return Outcome::InvalidTransition;
        status_.recordPrimary({Outcome::Cancelled, Operation::Publish, 0});
        return retire();
    }
    Outcome fail(Failure failure) {
        if (failure.outcome == Outcome::Success || retryable(failure.outcome))
            return Outcome::InvalidInput;
        if (state_ < ResourceState::Reserved || state_ > ResourceState::Published)
            return Outcome::InvalidTransition;
        status_.recordPrimary(failure);
        return retire();
    }
    Outcome invalidateMapping() {
        if (state_ != ResourceState::Retiring)
            return Outcome::InvalidTransition;
        mappingInvalidated_ = true;
        return Outcome::Success;
    }
    Outcome canDestroy() const {
        if (state_ != ResourceState::Retiring)
            return Outcome::InvalidTransition;
        if (!mappingInvalidated_ || !consumers_.empty() || !workers_.empty())
            return Outcome::Pending;
        return Outcome::Success;
    }
    Outcome destroyed(Failure cleanup = {}) {
        const Outcome ready = canDestroy();
        if (ready != Outcome::Success)
            return ready;
        if (cleanup.outcome != Outcome::Success) {
            if (retryable(cleanup.outcome))
                return Outcome::InvalidInput;
            status_.recordCleanup(cleanup);
            return cleanup.outcome;
        }
        const Outcome outcome = ledger_.release(Charge::Retiring, bytes_);
        if (outcome == Outcome::Success)
            state_ = ResourceState::Destroyed;
        return outcome;
    }
    ResourceState state() const { return state_; }
    const OperationStatus& status() const { return status_; }

private:
    Outcome advance(ResourceState from, ResourceState to) {
        if (state_ != from)
            return Outcome::InvalidTransition;
        state_ = to;
        return Outcome::Success;
    }
    Outcome begin(std::set<uint64_t>& active, uint64_t& token) {
        const Outcome outcome = tokens_.next(token);
        if (outcome == Outcome::Success)
            active.insert(token);
        return outcome;
    }
    BudgetLedger& ledger_;
    GpuKey key_;
    uint64_t revision_;
    uint64_t bytes_ = 0;
    ResourceState state_ = ResourceState::Empty;
    bool mappingInvalidated_ = true;
    NonWrappingCounter tokens_;
    std::set<uint64_t> consumers_;
    std::set<uint64_t> workers_;
    OperationStatus status_{};
};

} // namespace contract_v1
} // namespace hip_demand
