// SPDX-License-Identifier: MIT
// Unit tests for PinnedMemoryPool and HipEventPool

#include "TestUtils.h"
#include "DemandLoading/Internal/PinnedMemoryPool.h"
#include "DemandLoading/Internal/HipEventPool.h"

namespace hip_demand {
namespace test {

// ============================================================================
// PinnedMemoryPool Tests
// ============================================================================

TEST_F(HipTestFixture, PinnedPoolConstruction) {
    internal::PinnedMemoryPool pool(4);
    // Should not throw
}

TEST_F(HipTestFixture, PinnedPoolAcquire) {
    internal::PinnedMemoryPool pool(4);
    
    auto buffer = pool.acquire(1024);
    ASSERT_TRUE(buffer);
    EXPECT_NE(buffer.get(), nullptr);
    EXPECT_GE(buffer.size(), 1024u);
}

TEST_F(HipTestFixture, PinnedPoolMultipleAcquire) {
    internal::PinnedMemoryPool pool(4);
    
    auto buffer1 = pool.acquire(512);
    auto buffer2 = pool.acquire(512);
    auto buffer3 = pool.acquire(1024);
    
    EXPECT_TRUE(buffer1);
    EXPECT_TRUE(buffer2);
    EXPECT_TRUE(buffer3);
    
    // All should be different pointers
    EXPECT_NE(buffer1.get(), buffer2.get());
    EXPECT_NE(buffer2.get(), buffer3.get());
    EXPECT_NE(buffer1.get(), buffer3.get());
}

TEST_F(HipTestFixture, PinnedPoolReuse) {
    internal::PinnedMemoryPool pool(4);
    
    void* ptr1;
    {
        auto buffer = pool.acquire(512);
        ptr1 = buffer.get();
        ASSERT_NE(ptr1, nullptr);
        // buffer released on scope exit
    }
    
    // Acquire again - might reuse the same buffer
    auto buffer2 = pool.acquire(512);
    // The pool might reuse the buffer, but this is implementation-defined
    EXPECT_TRUE(buffer2);
}

TEST_F(HipTestFixture, PinnedPoolTypedAccess) {
    internal::PinnedMemoryPool pool(4);
    
    auto buffer = pool.acquire(sizeof(int) * 100);
    ASSERT_TRUE(buffer);
    
    int* data = buffer.as<int>();
    ASSERT_NE(data, nullptr);
    
    // Should be able to write/read
    for (int i = 0; i < 100; ++i) {
        data[i] = i * 2;
    }
    
    for (int i = 0; i < 100; ++i) {
        EXPECT_EQ(data[i], i * 2);
    }
}

// ============================================================================
// HipEventPool Tests
// ============================================================================

TEST_F(HipTestFixture, EventPoolConstruction) {
    internal::HipEventPool pool(4);
    // Should not throw
}

TEST_F(HipTestFixture, EventPoolAcquire) {
    internal::HipEventPool pool(4);
    
    hipEvent_t event = pool.acquire();
    EXPECT_NE(event, nullptr);
    
    pool.release(event);
}

TEST_F(HipTestFixture, EventPoolMultipleAcquire) {
    internal::HipEventPool pool(4);
    
    std::vector<hipEvent_t> events;
    for (int i = 0; i < 8; ++i) {
        hipEvent_t event = pool.acquire();
        EXPECT_NE(event, nullptr);
        events.push_back(event);
    }
    
    // Release all
    for (hipEvent_t event : events) {
        pool.release(event);
    }
}

TEST_F(HipTestFixture, EventPoolEventWorks) {
    internal::HipEventPool pool(4);
    
    hipStream_t stream;
    ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);
    
    hipEvent_t event = pool.acquire();
    ASSERT_NE(event, nullptr);
    
    // Record and synchronize should work
    EXPECT_EQ(hipEventRecord(event, stream), hipSuccess);
    EXPECT_EQ(hipEventSynchronize(event), hipSuccess);
    
    pool.release(event);
    hipStreamDestroy(stream);
}

TEST_F(HipTestFixture, EventPoolReuse) {
    internal::HipEventPool pool(2);
    
    // Exhaust initial pool
    hipEvent_t e1 = pool.acquire();
    hipEvent_t e2 = pool.acquire();
    
    // Return one
    pool.release(e1);
    
    // Acquire again - should reuse e1
    hipEvent_t e3 = pool.acquire();
    EXPECT_EQ(e3, e1);  // Should be the same event
    
    pool.release(e2);
    pool.release(e3);
}

}  // namespace test
}  // namespace hip_demand
