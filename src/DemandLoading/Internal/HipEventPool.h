// SPDX-License-Identifier: MIT
// Pool for reusable HIP events

#pragma once

#include <hip/hip_runtime.h>

#include <mutex>
#include <vector>

namespace hip_demand {
namespace internal {

/// A pool of reusable HIP events.
/// hipEventCreate/Destroy are expensive syscalls - pooling amortizes cost.
class HipEventPool {
public:
    explicit HipEventPool(size_t initialSize = 4) {
        events_.reserve(initialSize);
        for (size_t i = 0; i < initialSize; ++i) {
            hipEvent_t event{};
            if (hipEventCreateWithFlags(&event, hipEventDisableTiming) == hipSuccess) {
                events_.push_back(event);
            }
        }
    }
    
    ~HipEventPool() {
        std::lock_guard<std::mutex> lock(mutex_);
        for (hipEvent_t event : events_) {
            hipEventDestroy(event);
        }
        events_.clear();
    }
    
    // Non-copyable, non-movable
    HipEventPool(const HipEventPool&) = delete;
    HipEventPool& operator=(const HipEventPool&) = delete;
    
    /// Acquire an event from the pool, or create a new one if empty.
    hipEvent_t acquire() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!events_.empty()) {
                hipEvent_t event = events_.back();
                events_.pop_back();
                return event;
            }
        }
        
        // Pool empty - create new event
        hipEvent_t event{};
        if (hipEventCreateWithFlags(&event, hipEventDisableTiming) != hipSuccess) {
            return nullptr;
        }
        return event;
    }
    
    /// Return an event to the pool for reuse.
    void release(hipEvent_t event) {
        if (!event) return;
        
        std::lock_guard<std::mutex> lock(mutex_);
        events_.push_back(event);
    }
    
    /// Get current pool size
    size_t pooledCount() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return events_.size();
    }
    
private:
    mutable std::mutex mutex_;
    std::vector<hipEvent_t> events_;
};

/// RAII wrapper for pooled HIP events
class PooledEvent {
public:
    PooledEvent() = default;
    PooledEvent(HipEventPool* pool, hipEvent_t event) : pool_(pool), event_(event) {}
    
    ~PooledEvent() {
        if (pool_ && event_) {
            pool_->release(event_);
        }
    }
    
    // Move-only
    PooledEvent(PooledEvent&& other) noexcept
        : pool_(other.pool_), event_(other.event_) {
        other.pool_ = nullptr;
        other.event_ = nullptr;
    }
    
    PooledEvent& operator=(PooledEvent&& other) noexcept {
        if (this != &other) {
            if (pool_ && event_) {
                pool_->release(event_);
            }
            pool_ = other.pool_;
            event_ = other.event_;
            other.pool_ = nullptr;
            other.event_ = nullptr;
        }
        return *this;
    }
    
    PooledEvent(const PooledEvent&) = delete;
    PooledEvent& operator=(const PooledEvent&) = delete;
    
    hipEvent_t get() const { return event_; }
    explicit operator bool() const { return event_ != nullptr; }
    
private:
    HipEventPool* pool_ = nullptr;
    hipEvent_t event_ = nullptr;
};

} // namespace internal
} // namespace hip_demand
