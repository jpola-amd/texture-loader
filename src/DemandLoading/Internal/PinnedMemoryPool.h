// SPDX-License-Identifier: MIT
// Pool for reusable pinned (page-locked) host memory buffers

#pragma once

#include <hip/hip_runtime.h>

#include <cstddef>
#include <memory>
#include <mutex>
#include <vector>

namespace hip_demand {
namespace internal {

/// A pool of reusable pinned memory buffers.
/// Reduces hipHostMalloc/hipHostFree overhead for frequently used temporary buffers.
class PinnedMemoryPool {
public:
    /// A handle to a pinned buffer that returns it to the pool on destruction.
    class BufferHandle {
    public:
        BufferHandle() = default;
        BufferHandle(PinnedMemoryPool* pool, void* ptr, size_t size)
            : pool_(pool), ptr_(ptr), size_(size) {}
        
        ~BufferHandle() {
            if (pool_ && ptr_) {
                pool_->release(ptr_, size_);
            }
        }
        
        // Move-only
        BufferHandle(BufferHandle&& other) noexcept
            : pool_(other.pool_), ptr_(other.ptr_), size_(other.size_) {
            other.pool_ = nullptr;
            other.ptr_ = nullptr;
            other.size_ = 0;
        }
        
        BufferHandle& operator=(BufferHandle&& other) noexcept {
            if (this != &other) {
                if (pool_ && ptr_) {
                    pool_->release(ptr_, size_);
                }
                pool_ = other.pool_;
                ptr_ = other.ptr_;
                size_ = other.size_;
                other.pool_ = nullptr;
                other.ptr_ = nullptr;
                other.size_ = 0;
            }
            return *this;
        }
        
        BufferHandle(const BufferHandle&) = delete;
        BufferHandle& operator=(const BufferHandle&) = delete;
        
        void* get() const { return ptr_; }
        size_t size() const { return size_; }
        explicit operator bool() const { return ptr_ != nullptr; }
        
        template<typename T>
        T* as() const { return static_cast<T*>(ptr_); }
        
    private:
        PinnedMemoryPool* pool_ = nullptr;
        void* ptr_ = nullptr;
        size_t size_ = 0;
    };

    explicit PinnedMemoryPool(size_t maxPooledBuffers = 8)
        : maxPooledBuffers_(maxPooledBuffers) {}
    
    ~PinnedMemoryPool() {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto& entry : pool_) {
            hipHostFree(entry.ptr);
        }
        pool_.clear();
    }
    
    // Non-copyable, non-movable
    PinnedMemoryPool(const PinnedMemoryPool&) = delete;
    PinnedMemoryPool& operator=(const PinnedMemoryPool&) = delete;
    
    /// Acquire a pinned buffer of at least `size` bytes.
    /// Returns a handle that automatically releases the buffer back to the pool.
    BufferHandle acquire(size_t size) {
        void* ptr = nullptr;
        
        {
            std::lock_guard<std::mutex> lock(mutex_);
            
            // Look for an existing buffer that's large enough
            for (auto it = pool_.begin(); it != pool_.end(); ++it) {
                if (it->size >= size) {
                    ptr = it->ptr;
                    size = it->size;  // Use actual buffer size for proper return
                    pool_.erase(it);
                    return BufferHandle(this, ptr, size);
                }
            }
        }
        
        // No suitable buffer found, allocate new one
        if (hipHostMalloc(&ptr, size) != hipSuccess) {
            return BufferHandle();  // Return invalid handle on failure
        }
        
        return BufferHandle(this, ptr, size);
    }
    
    /// Get current number of pooled buffers
    size_t pooledCount() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return pool_.size();
    }
    
private:
    void release(void* ptr, size_t size) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        // If pool is full, free the buffer; otherwise add to pool
        if (pool_.size() >= maxPooledBuffers_) {
            hipHostFree(ptr);
        } else {
            pool_.push_back({ptr, size});
        }
    }
    
    struct PoolEntry {
        void* ptr;
        size_t size;
    };
    
    mutable std::mutex mutex_;
    std::vector<PoolEntry> pool_;
    size_t maxPooledBuffers_;
};

} // namespace internal
} // namespace hip_demand
