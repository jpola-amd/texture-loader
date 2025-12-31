// SPDX-License-Identifier: MIT
// Simple thread pool for parallel texture loading

#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace hip_demand {
namespace internal {

/// A simple thread pool for parallel I/O operations.
/// Uses a fixed number of worker threads that process tasks from a queue.
class ThreadPool {
public:
    explicit ThreadPool(unsigned int numThreads = 0) {
        if (numThreads == 0) {
            numThreads = std::max(1u, std::thread::hardware_concurrency());
        }
        // Cap at reasonable maximum for I/O work
        numThreads = std::min(numThreads, 16u);
        
        for (unsigned int i = 0; i < numThreads; ++i) {
            workers_.emplace_back([this] { workerLoop(); });
        }
    }
    
    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
        }
        cv_.notify_all();
        for (auto& worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }
    
    // Non-copyable, non-movable
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;
    ThreadPool(ThreadPool&&) = delete;
    ThreadPool& operator=(ThreadPool&&) = delete;
    
    /// Submit a task to the pool
    void submit(std::function<void()> task) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            tasks_.push(std::move(task));
        }
        cv_.notify_one();
    }
    
    /// Wait for all currently submitted tasks to complete
    void waitAll() {
        std::unique_lock<std::mutex> lock(mutex_);
        completionCv_.wait(lock, [this] {
            return tasks_.empty() && activeWorkers_ == 0;
        });
    }
    
    /// Get number of worker threads
    unsigned int size() const { return static_cast<unsigned int>(workers_.size()); }

private:
    void workerLoop() {
        while (true) {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait(lock, [this] { return stopping_ || !tasks_.empty(); });
                
                if (stopping_ && tasks_.empty()) {
                    return;
                }
                
                task = std::move(tasks_.front());
                tasks_.pop();
                ++activeWorkers_;
            }
            
            task();
            
            {
                std::lock_guard<std::mutex> lock(mutex_);
                --activeWorkers_;
            }
            completionCv_.notify_all();
        }
    }
    
    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::condition_variable completionCv_;
    unsigned int activeWorkers_ = 0;
    bool stopping_ = false;
};

} // namespace internal
} // namespace hip_demand
