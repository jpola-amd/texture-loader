// SPDX-License-Identifier: MIT
#include "HipCalls.h"
#include <stdexcept>

#ifdef HIP_DEMAND_TEST_HOOKS
namespace hip_demand {
namespace internal {
namespace {
thread_local std::shared_ptr<HipFaultState> faultState;
}

std::shared_ptr<HipFaultState> currentHipFaultState() { return faultState; }
void setHipFaultState(std::shared_ptr<HipFaultState> state) { faultState = std::move(state); }

void HipFaultState::fail(HipOperation operation, hipError_t error, size_t invocation) {
    if (error == hipSuccess || invocation == 0)
        throw std::invalid_argument("A fault requires a non-success error and positive invocation");
    std::lock_guard<std::mutex> lock(mutex_);
    rules_.push_back({operation, error, invocation});
}

hipError_t HipFaultState::before(HipOperation operation) {
    std::lock_guard<std::mutex> lock(mutex_);
    hipError_t result = hipSuccess;
    for (auto& rule : rules_) {
        if (rule.operation == operation && rule.remaining && --rule.remaining == 0 && result == hipSuccess)
            result = rule.error;
    }
    return result;
}

void HipFaultState::record(HipOperation operation, hipError_t error, bool injected) {
    std::lock_guard<std::mutex> lock(mutex_);
    records_.push_back({operation, error, injected});
}

std::vector<HipCallRecord> HipFaultState::records() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return records_;
}

void HipFaultState::overrideReturnedAnisotropy(unsigned int value) {
    std::lock_guard<std::mutex> lock(mutex_);
    overrideAnisotropy_ = true;
    returnedAnisotropy_ = value;
}

void HipFaultState::observeSampler(hipTextureDesc& sampler) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (overrideAnisotropy_)
        sampler.maxAnisotropy = returnedAnisotropy_;
}
} // namespace internal
} // namespace hip_demand
#endif
