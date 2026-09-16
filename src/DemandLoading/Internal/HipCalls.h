// SPDX-License-Identifier: MIT
#pragma once

#include <hip/hip_runtime.h>
#include <DemandLoading/Logging.h>
#include <memory>
#include <new>

#ifdef HIP_DEMAND_TEST_HOOKS
#include <mutex>
#include <vector>
#endif

namespace hip_demand {
namespace internal {

enum class HipOperation {
    GetDevice, DeviceAllocation, HostAllocation, Initialize,
    AllocateArray, AllocateMipmapped, Upload, CreateSampler,
    FreeArray, FreeMipmapped, DestroySampler,
    CacheData, FilenameMap, SourceMap, ContentMap,
    InvalidateMappings, SynchronizeConsumers, PublishMappings,
    SelectDevice, GetContext, ProbeAllocate, ProbeGetLevel, ProbeUpload,
    ProbeCreateSampler, ProbeDestroySampler, ProbeFree, GetLevel, ReadSampler, FreeHost, FreeDevice
};

#ifdef HIP_DEMAND_TEST_HOOKS
struct HipCallRecord {
    HipOperation operation;
    hipError_t error;
    bool injected;
};

// A loader captures this state at construction, so its workers use the same
// seam without affecting other loaders or relying on worker thread-local state.
class HipFaultState {
public:
    void fail(HipOperation operation, hipError_t error, size_t invocation = 1);
    hipError_t before(HipOperation operation);
    void record(HipOperation operation, hipError_t error, bool injected);
    std::vector<HipCallRecord> records() const;
    void overrideReturnedAnisotropy(unsigned int value);
    void observeSampler(hipTextureDesc& sampler);
private:
    struct Rule { HipOperation operation; hipError_t error; size_t remaining; };
    mutable std::mutex mutex_;
    std::vector<Rule> rules_;
    std::vector<HipCallRecord> records_;
    bool overrideAnisotropy_ = false;
    unsigned int returnedAnisotropy_ = 0;
};

std::shared_ptr<HipFaultState> currentHipFaultState();
void setHipFaultState(std::shared_ptr<HipFaultState> state);
#endif

class HipCalls {
public:
    void observeSampler(hipTextureDesc& sampler) const {
#ifdef HIP_DEMAND_TEST_HOOKS
        if (state_)
            state_->observeSampler(sampler);
#else
        (void)sampler;
#endif
    }

    template<class F> hipError_t call(HipOperation operation, F&& realCall) const {
        hipError_t error = hipSuccess;
        bool injected = false;
#ifdef HIP_DEMAND_TEST_HOOKS
        if (state_) {
            error = state_->before(operation);
            injected = error != hipSuccess;
        }
#endif
        if (!injected)
            error = realCall();
#ifdef HIP_DEMAND_TEST_HOOKS
        if (state_)
            state_->record(operation, error, injected);
#endif
        if (error != hipSuccess)
            logMessage(LogLevel::Error, "HIP operation %u failed: %s (%d)",
                       static_cast<unsigned>(operation), hipGetErrorString(error), static_cast<int>(error));
        return error;
    }

    void registrationCheckpoint(HipOperation operation) const {
#ifdef HIP_DEMAND_TEST_HOOKS
        if (call(operation, [] { return hipSuccess; }) != hipSuccess)
            throw std::bad_alloc();
#else
        (void)operation;
#endif
    }
private:
#ifdef HIP_DEMAND_TEST_HOOKS
    std::shared_ptr<HipFaultState> state_ = currentHipFaultState();
#endif
};

} // namespace internal
} // namespace hip_demand
