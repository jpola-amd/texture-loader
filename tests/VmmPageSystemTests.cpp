// SPDX-License-Identifier: MIT
// Tests for the VMM loader's page system: shared physical pages and the remapping of virtual pages.

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <DemandLoading/vmm/PageSystem.h>
#include <cstdint>
#include <vector>

namespace hip_demand::vmm::test {

namespace {

constexpr size_t pageBytes = PageSystem::PAGE_BYTES;

class VmmPageSystemTest : public ::testing::Test
{
  protected:
    void SetUp() override
    {
        int deviceCount = 0;
        if( hipGetDeviceCount( &deviceCount ) != hipSuccess || deviceCount == 0 )
            GTEST_SKIP() << "No HIP device";
        ASSERT_EQ( hipSetDevice( 0 ), hipSuccess );
        int vmmSupported = 0;
        ASSERT_EQ( hipDeviceGetAttribute( &vmmSupported, hipDeviceAttributeVirtualMemoryManagementSupported, 0 ), hipSuccess );
        if( !vmmSupported )
            GTEST_SKIP() << "No virtual memory management";
    }
};

void fillPage( const DeviceSpan<uint8_t>& page, uint8_t value )
{
    ASSERT_EQ( hipMemset( page.ptr, value, page.len ), hipSuccess );
    ASSERT_EQ( hipDeviceSynchronize(), hipSuccess );
}

// The value the page holds, or -1 when it holds more than one.
int pageValue( const DeviceSpan<uint8_t>& page )
{
    std::vector<uint8_t> host( page.len );
    if( hipMemcpy( host.data(), page.ptr, page.len, hipMemcpyDeviceToHost ) != hipSuccess )
        return -2;
    for( uint8_t byte : host )
    {
        if( byte != host[0] )
            return -1;
    }
    return host[0];
}

void evict( PageSystem& pages, std::vector<uint32_t> pageIds, uint32_t reusablePageCount )
{
    std::vector<EvictionCandidate> candidates;
    for( uint32_t pageId : pageIds )
        candidates.push_back( { pageId, 0 } );
    pages.enqueueEvictedPages( HostSpan<EvictionCandidate>( candidates.data(), candidates.size() ), reusablePageCount );
}

}  // namespace

TEST_F( VmmPageSystemTest, AliasesShareOnePhysicalPage )
{
    PageSystem     pages( 16, 16 );
    const uint32_t shared = pages.map( 0 );
    pages.mapAlias( 1, shared );
    pages.mapAlias( 2, shared );
    EXPECT_EQ( pages.physicalPageCount(), 1u );
    EXPECT_EQ( pages.mappedPageCount(), 3u );

    fillPage( pages.page( 0 ), 0xAB );
    EXPECT_EQ( pageValue( pages.page( 1 ) ), 0xAB );
    EXPECT_EQ( pageValue( pages.page( 2 ) ), 0xAB );

    // Mapping a mapped page again, as an alias or on its own, changes nothing.
    pages.mapAlias( 0, shared );
    EXPECT_EQ( pages.map( 1 ), shared );
    EXPECT_EQ( pages.physicalPageCount(), 1u );
    EXPECT_EQ( pages.mappedPageCount(), 3u );
}

TEST_F( VmmPageSystemTest, EvictedAliasLeavesThePageToItsOtherUsers )
{
    PageSystem     pages( 16, 16 );
    const uint32_t shared = pages.map( 0 );
    pages.mapAlias( 1, shared );
    pages.mapAlias( 2, shared );
    fillPage( pages.page( 0 ), 0xAB );

    // Evicting an alias frees no physical page, so the next page is a new one, whatever the eviction asked to reuse.
    evict( pages, { 1 }, 1 );
    const uint32_t own = pages.map( 3 );
    EXPECT_NE( own, shared );
    EXPECT_EQ( pages.physicalPageCount(), 2u );
    EXPECT_EQ( pages.mappedPageCount(), 3u );
    fillPage( pages.page( 3 ), 0x11 );
    EXPECT_EQ( pageValue( pages.page( 0 ) ), 0xAB );
    EXPECT_EQ( pageValue( pages.page( 2 ) ), 0xAB );

    // The evicted alias gets a page of its own next and no longer follows the shared page.
    EXPECT_NE( pages.map( 1 ), shared );
    fillPage( pages.page( 1 ), 0x22 );
    EXPECT_EQ( pageValue( pages.page( 0 ) ), 0xAB );
    EXPECT_EQ( pageValue( pages.page( 3 ) ), 0x11 );

    // The last user releases it.
    evict( pages, { 0, 2 }, 0 );
    pages.map( 4 );
    EXPECT_EQ( pages.physicalPageCount(), 3u );
    EXPECT_EQ( pages.mappedPageCount(), 3u );
}

TEST_F( VmmPageSystemTest, RetainedPageOutlivesItsMappings )
{
    PageSystem     pages( 16, 16 );
    const uint32_t shared = pages.map( 0 );
    pages.retain( shared );
    pages.retain( shared );  // Idempotent.
    fillPage( pages.page( 0 ), 0xAB );

    evict( pages, { 0 }, 0 );
    pages.map( 1 );  // Processes the eviction; the retained page is not released or reused.
    EXPECT_EQ( pages.physicalPageCount(), 2u );
    EXPECT_EQ( pages.mappedPageCount(), 1u );

    pages.mapAlias( 2, shared );
    EXPECT_EQ( pageValue( pages.page( 2 ) ), 0xAB );
    EXPECT_EQ( pages.physicalPageCount(), 2u );
}

TEST_F( VmmPageSystemTest, UnusedPageCannotBeAliased )
{
    PageSystem     pages( 16, 16 );
    const uint32_t released = pages.map( 0 );
    evict( pages, { 0 }, 0 );
    EXPECT_THROW( pages.mapAlias( 2, released ), std::logic_error );
    EXPECT_THROW( pages.mapAlias( 2, 7 ), std::out_of_range );
}

TEST_F( VmmPageSystemTest, EvictedPagesAreRemappedToOtherPhysicalPages )
{
    // The content a page shows after it is unmapped and mapped to another physical page: ROCm's hipMemMap() leaves
    // stale translations in the device TLB, which PageSystem::invalidateDeviceTlb() clears after unmapping.
    PageSystem pages( 16, 16 );
    pages.map( 0 );
    pages.map( 1 );
    fillPage( pages.page( 0 ), 0xAA );
    fillPage( pages.page( 1 ), 0xBB );

    // Both physical pages stay reusable; page 2 and 3 get them in reverse order of eviction.
    evict( pages, { 0, 1 }, 2 );
    pages.map( 2 );
    pages.map( 3 );
    EXPECT_EQ( pages.physicalPageCount(), 2u );
    EXPECT_EQ( pages.unmappedPageCount(), 2u );
    EXPECT_EQ( pageValue( pages.page( 2 ) ), 0xBB );
    EXPECT_EQ( pageValue( pages.page( 3 ) ), 0xAA );

    // Pages 0 and 1 get their physical pages back, swapped. A write through page 0 must land in its new physical
    // page, not in the one a stale translation would still point to, which is now behind page 1.
    evict( pages, { 3, 2 }, 2 );
    pages.map( 0 );
    pages.map( 1 );
    EXPECT_EQ( pages.unmappedPageCount(), 4u );
    EXPECT_EQ( pageValue( pages.page( 0 ) ), 0xBB );
    EXPECT_EQ( pageValue( pages.page( 1 ) ), 0xAA );
    fillPage( pages.page( 0 ), 0x01 );
    EXPECT_EQ( pageValue( pages.page( 0 ) ), 0x01 );
    EXPECT_EQ( pageValue( pages.page( 1 ) ), 0xAA );
}

TEST_F( VmmPageSystemTest, RemappedPagesShowTheirNewPhysicalPageAfterInvalidation )
{
    // Documents the driver behaviour the page system works around, with raw VMM calls.
    int dev = 0;
    ASSERT_EQ( hipGetDevice( &dev ), hipSuccess );
    hipMemAllocationProp prop{};
    prop.type          = hipMemAllocationTypePinned;
    prop.location.type = hipMemLocationTypeDevice;
    prop.location.id   = dev;
    hipMemAccessDesc access{};
    access.location.type = hipMemLocationTypeDevice;
    access.location.id   = dev;
    access.flags         = hipMemAccessFlagsProtReadWrite;

    uint8_t* va = nullptr;
    ASSERT_EQ( hipMemAddressReserve( reinterpret_cast<void**>( &va ), 2 * pageBytes, pageBytes, nullptr, 0 ), hipSuccess );
    hipMemGenericAllocationHandle_t a = nullptr, b = nullptr;
    ASSERT_EQ( hipMemCreate( &a, pageBytes, &prop, 0 ), hipSuccess );
    ASSERT_EQ( hipMemCreate( &b, pageBytes, &prop, 0 ), hipSuccess );
    const auto mapAt = [&]( uint32_t page, hipMemGenericAllocationHandle_t handle ) {
        ASSERT_EQ( hipMemMap( va + page * pageBytes, pageBytes, 0, handle, 0 ), hipSuccess );
        ASSERT_EQ( hipMemSetAccess( va + page * pageBytes, pageBytes, &access, 1 ), hipSuccess );
    };
    const DeviceSpan<uint8_t> page0( va, pageBytes ), page1( va + pageBytes, pageBytes );

    mapAt( 0, a );
    mapAt( 1, b );
    fillPage( page0, 0xAA );
    fillPage( page1, 0xBB );
    ASSERT_EQ( hipMemUnmap( va, pageBytes ), hipSuccess );
    ASSERT_EQ( hipMemUnmap( va + pageBytes, pageBytes ), hipSuccess );
    mapAt( 0, b );
    mapAt( 1, a );
    PageSystem::invalidateDeviceTlb();
    EXPECT_EQ( pageValue( page0 ), 0xBB );
    EXPECT_EQ( pageValue( page1 ), 0xAA );

    ASSERT_EQ( hipMemUnmap( va, 2 * pageBytes ), hipSuccess );
    PageSystem::invalidateDeviceTlb();  // Or the next user of the address range reads these pages.
    ASSERT_EQ( hipMemRelease( a ), hipSuccess );
    ASSERT_EQ( hipMemRelease( b ), hipSuccess );
    ASSERT_EQ( hipMemAddressFree( va, 2 * pageBytes ), hipSuccess );
}

TEST_F( VmmPageSystemTest, PhysicalPagesAreLimited )
{
    PageSystem pages( 16, 2 );
    pages.map( 0 );
    pages.map( 1 );
    EXPECT_THROW( pages.map( 2 ), std::runtime_error );
    EXPECT_EQ( pages.physicalPageCount(), 2u );
    EXPECT_EQ( pages.mappedPageCount(), 2u );

    // An alias costs no physical page, and a released page makes room again.
    pages.mapAlias( 2, pages.map( 0 ) );
    EXPECT_EQ( pages.mappedPageCount(), 3u );
    evict( pages, { 1 }, 0 );
    EXPECT_NO_THROW( pages.map( 3 ) );
    EXPECT_EQ( pages.physicalPageCount(), 2u );
}

TEST_F( VmmPageSystemTest, VirtualPagesOutsideTheAddressSpaceAreRejected )
{
    PageSystem pages( 16, 16 );
    EXPECT_THROW( pages.map( 16 ), std::out_of_range );
    EXPECT_THROW( pages.map( ~0u ), std::out_of_range );
    const uint32_t physical = pages.map( 0 );
    EXPECT_THROW( pages.mapAlias( 16, physical ), std::out_of_range );
    EXPECT_EQ( pages.mappedPageCount(), 1u );
    EXPECT_EQ( pages.page( 15 ).ptr, pages.virtualAddressSpace().ptr + 15 * pageBytes );
}

TEST_F( VmmPageSystemTest, PagesBeyondTheReusableCountAreReleased )
{
    PageSystem pages( 16, 16 );
    for( uint32_t pageId = 0; pageId < 4; ++pageId )
        pages.map( pageId );
    ASSERT_EQ( pages.physicalPageCount(), 4u );

    // One of the four is kept for reuse, the other three are released; the next mapping reuses it.
    evict( pages, { 0, 1, 2, 3 }, 1 );
    pages.map( 4 );
    EXPECT_EQ( pages.physicalPageCount(), 1u );
    EXPECT_EQ( pages.mappedPageCount(), 1u );
    EXPECT_EQ( pages.unmappedPageCount(), 4u );

    // Released pages get allocated again under their old ids.
    const uint32_t reallocated = pages.map( 5 );
    EXPECT_LT( reallocated, 4u );
    EXPECT_EQ( pages.physicalPageCount(), 2u );
    fillPage( pages.page( 5 ), 0x55 );
    EXPECT_EQ( pageValue( pages.page( 5 ) ), 0x55 );

    // Asking to reuse more pages than are evicted keeps them all.
    evict( pages, { 4, 5 }, 8 );
    pages.map( 6 );
    pages.map( 7 );
    EXPECT_EQ( pages.physicalPageCount(), 2u );
    EXPECT_EQ( pages.unmappedPageCount(), 6u );
}

TEST_F( VmmPageSystemTest, EvictingAnUnmappedPageChangesNothing )
{
    PageSystem pages( 16, 16 );
    pages.map( 0 );
    evict( pages, { 5, 0, 5 }, 2 );
    pages.map( 1 );
    EXPECT_EQ( pages.physicalPageCount(), 1u );
    EXPECT_EQ( pages.mappedPageCount(), 1u );
    EXPECT_EQ( pages.unmappedPageCount(), 1u );

    // Nothing was reserved for the pages that were not mapped.
    pages.map( 2 );
    EXPECT_EQ( pages.physicalPageCount(), 2u );
}

TEST_F( VmmPageSystemTest, RetainedPageIsReleasedWithTheSystem )
{
    // A page retained and never mapped again is freed by the destructor, not leaked: a second system of the same
    // size can allocate its maximum afterwards.
    {
        PageSystem pages( 16, 4 );
        pages.retain( pages.map( 0 ) );
        evict( pages, { 0 }, 0 );
        pages.map( 1 );
        EXPECT_EQ( pages.physicalPageCount(), 2u );
    }
    PageSystem pages( 16, 4 );
    for( uint32_t pageId = 0; pageId < 4; ++pageId )
        EXPECT_NO_THROW( pages.map( pageId ) );
}

}  // namespace hip_demand::vmm::test
