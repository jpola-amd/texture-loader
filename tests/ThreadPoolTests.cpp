// SPDX-License-Identifier: MIT
// Unit tests for ThreadPool

#include <gtest/gtest.h>
#include "DemandLoading/Internal/ThreadPool.h"

#include <atomic>
#include <chrono>
#include <thread>
#include <vector>

namespace hip_demand {
namespace test {

// ============================================================================
// Construction Tests
// ============================================================================

TEST(ThreadPoolTest, DefaultConstruction) {
    internal::ThreadPool pool;
    EXPECT_GT(pool.size(), 0u);
}

TEST(ThreadPoolTest, SpecificThreadCount) {
    internal::ThreadPool pool(4);
    EXPECT_EQ(pool.size(), 4u);
}

TEST(ThreadPoolTest, SingleThread) {
    internal::ThreadPool pool(1);
    EXPECT_EQ(pool.size(), 1u);
}

// ============================================================================
// Task Execution Tests
// ============================================================================

TEST(ThreadPoolTest, ExecuteSingleTask) {
    internal::ThreadPool pool(2);
    
    std::atomic<bool> executed{false};
    
    pool.submit([&executed]() {
        executed.store(true, std::memory_order_release);
    });
    
    pool.waitAll();
    EXPECT_TRUE(executed.load(std::memory_order_acquire));
}

TEST(ThreadPoolTest, ExecuteMultipleTasks) {
    internal::ThreadPool pool(4);
    
    constexpr int numTasks = 100;
    std::atomic<int> counter{0};
    
    for (int i = 0; i < numTasks; ++i) {
        pool.submit([&counter]() {
            counter.fetch_add(1, std::memory_order_relaxed);
        });
    }
    
    pool.waitAll();
    EXPECT_EQ(counter.load(std::memory_order_acquire), numTasks);
}

TEST(ThreadPoolTest, TasksRunConcurrently) {
    internal::ThreadPool pool(4);
    
    std::atomic<int> concurrent{0};
    std::atomic<int> maxConcurrent{0};
    
    for (int i = 0; i < 8; ++i) {
        pool.submit([&concurrent, &maxConcurrent]() {
            int current = concurrent.fetch_add(1, std::memory_order_acq_rel) + 1;
            
            // Update max observed concurrency
            int expected = maxConcurrent.load(std::memory_order_acquire);
            while (current > expected && 
                   !maxConcurrent.compare_exchange_weak(expected, current,
                       std::memory_order_acq_rel, std::memory_order_acquire)) {
            }
            
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            concurrent.fetch_sub(1, std::memory_order_release);
        });
    }
    
    pool.waitAll();
    
    // With 4 threads and 50ms tasks, we should see > 1 concurrent execution
    EXPECT_GT(maxConcurrent.load(), 1);
}

// ============================================================================
// WaitAll Tests
// ============================================================================

TEST(ThreadPoolTest, WaitAllBlocksUntilComplete) {
    internal::ThreadPool pool(2);
    
    std::atomic<int> counter{0};
    
    for (int i = 0; i < 5; ++i) {
        pool.submit([&counter]() {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            counter.fetch_add(1, std::memory_order_relaxed);
        });
    }
    
    pool.waitAll();
    
    // All tasks should be complete after waitAll returns
    EXPECT_EQ(counter.load(std::memory_order_acquire), 5);
}

// ============================================================================
// Destruction Tests
// ============================================================================

TEST(ThreadPoolTest, DestructorWaitsForTasks) {
    std::atomic<int> counter{0};
    
    {
        internal::ThreadPool pool(2);
        
        for (int i = 0; i < 10; ++i) {
            pool.submit([&counter]() {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                counter.fetch_add(1, std::memory_order_relaxed);
            });
        }
        // Pool destructor should wait for all tasks
    }
    
    EXPECT_EQ(counter.load(std::memory_order_acquire), 10);
}

// ============================================================================
// Stress Tests
// ============================================================================

TEST(ThreadPoolTest, StressTest) {
    internal::ThreadPool pool(8);
    
    constexpr int numTasks = 1000;
    std::atomic<int> counter{0};
    
    for (int i = 0; i < numTasks; ++i) {
        pool.submit([&counter, i]() {
            // Do some trivial work
            volatile int x = i * 2;
            (void)x;
            counter.fetch_add(1, std::memory_order_relaxed);
        });
    }
    
    pool.waitAll();
    EXPECT_EQ(counter.load(std::memory_order_acquire), numTasks);
}

}  // namespace test
}  // namespace hip_demand
