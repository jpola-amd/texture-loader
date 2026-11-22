#include "PageSystem.h"
#include "../Internal/HipCheck.h"
#include <stdexcept>
#include <string>

namespace hip_demand::vmm {

PageSystem::PageSystem( uint32_t maxVirtualPages, uint32_t maxPhysicalPages )
{
    virtual_.maxPages  = maxVirtualPages;
    physical_.maxPages = maxPhysicalPages;

    HIP_CHECK( hipGetDevice( &device_ ) );
    int vmmSupported = 0;
    HIP_CHECK( hipDeviceGetAttribute( &vmmSupported, hipDeviceAttributeVirtualMemoryManagementSupported, device_ ) );
    if( vmmSupported == 0 )
        throw std::runtime_error( "The active HIP device does not support virtual memory management" );

    allocationProp_.type          = hipMemAllocationTypePinned;
    allocationProp_.location.type = hipMemLocationTypeDevice;
    allocationProp_.location.id   = device_;
    
    size_t granularity = 0;
    HIP_CHECK( hipMemGetAllocationGranularity( &granularity, &allocationProp_, hipMemAllocationGranularityMinimum ) );
    // Each page is backed by its own physical allocation, so the page size must be a multiple of the granularity.
    if( granularity == 0 || PAGE_BYTES % granularity != 0 )
        throw std::runtime_error( "Page size is not a multiple of hipMemAllocationGranularityMinimum" );

    pageBytes_                = PAGE_BYTES;
    virtual_.addressSpace.len = static_cast<size_t>( virtual_.maxPages ) * pageBytes_;
    HIP_CHECK( hipMemAddressReserve( reinterpret_cast<void**>( &virtual_.addressSpace.ptr ), virtual_.addressSpace.len,
                                     pageBytes_, nullptr, 0 ) );
}

PageSystem::~PageSystem()
{
    for( const auto& [virtualPageId, physicalPageId] : virtualIdToPhysicalId_ )
    {
        DeviceSpan<uint8_t> virtualPage = page( virtualPageId );
        HIP_WARN( hipMemUnmap( virtualPage.ptr, virtualPage.len ) );
    }
    // The address space may be reserved again, by another page system, while the device still holds translations of
    // these mappings.
    if( !virtualIdToPhysicalId_.empty() )
        invalidateDeviceTlb();

    for( PhysicalPage& physicalPage : physical_.pages )
    {
        if( physicalPage.handle )
            HIP_WARN( hipMemRelease( physicalPage.handle ) );
    }

    if( virtual_.addressSpace.ptr )
        HIP_WARN( hipMemAddressFree( virtual_.addressSpace.ptr, virtual_.addressSpace.len ) );
}

uint32_t PageSystem::map( uint32_t pageId )
{
    if( pageId >= virtual_.maxPages )
        throw std::out_of_range( "Virtual page id " + std::to_string( pageId ) + " is outside the reserved address space" );

    std::lock_guard<std::mutex> lock( mutex_ );

    processPendingEvictedPages();

    const auto mapping = virtualIdToPhysicalId_.find( pageId );
    if( mapping != virtualIdToPhysicalId_.end() )
        return mapping->second;

    uint32_t physicalPageId = INVALID_PAGE;
    if( !physical_.reusablePages.empty() )
    {
        physicalPageId = physical_.reusablePages.back();
        physical_.reusablePages.pop_back();
    }
    else
    {
        if( !physical_.unallocatedPages.empty() )
        {
            physicalPageId = physical_.unallocatedPages.back();
            physical_.unallocatedPages.pop_back();
        }
        else
        {
            if( physical_.pages.size() >= physical_.maxPages )
                throw std::runtime_error( "Maximum demand physical page count exceeded" );

            physicalPageId = static_cast<uint32_t>( physical_.pages.size() );
            physical_.pages.push_back( {} );
        }

        HIP_CHECK( hipMemCreate( &physical_.pages.at( physicalPageId ).handle, pageBytes_, &allocationProp_, 0 ) );
        physical_.allocatedCount++;
    }

    PhysicalPage&       physicalPage = physical_.pages.at( physicalPageId );
    DeviceSpan<uint8_t> virtualPage  = page( pageId );
    HIP_CHECK( hipMemMap( virtualPage.ptr, virtualPage.len, 0, physicalPage.handle, 0 ) );

    hipMemAccessDesc access{};
    access.location.type = hipMemLocationTypeDevice;
    access.location.id   = device_;
    access.flags         = hipMemAccessFlagsProtReadWrite;
    HIP_CHECK( hipMemSetAccess( virtualPage.ptr, virtualPage.len, &access, 1 ) );

    virtualIdToPhysicalId_[pageId] = physicalPageId;
    physicalPage.useCount++;
    return physicalPageId;
}

void PageSystem::mapAlias( uint32_t pageId, uint32_t physicalPageId )
{
    if( pageId >= virtual_.maxPages )
        throw std::out_of_range( "Virtual page id " + std::to_string( pageId ) + " is outside the reserved address space" );

    std::lock_guard<std::mutex> lock( mutex_ );

    processPendingEvictedPages();

    if( virtualIdToPhysicalId_.find( pageId ) != virtualIdToPhysicalId_.end() )
        return;

    PhysicalPage& physicalPage = physical_.pages.at( physicalPageId );
    if( !physicalPage.handle || physicalPage.useCount == 0 )
        throw std::logic_error( "Physical page " + std::to_string( physicalPageId ) + " is not in use and cannot be aliased" );

    DeviceSpan<uint8_t> virtualPage = page( pageId );
    HIP_CHECK( hipMemMap( virtualPage.ptr, virtualPage.len, 0, physicalPage.handle, 0 ) );

    hipMemAccessDesc access{};
    access.location.type = hipMemLocationTypeDevice;
    access.location.id   = device_;
    access.flags         = hipMemAccessFlagsProtReadWrite;
    HIP_CHECK( hipMemSetAccess( virtualPage.ptr, virtualPage.len, &access, 1 ) );

    virtualIdToPhysicalId_[pageId] = physicalPageId;
    physicalPage.useCount++;
}

void PageSystem::retain( uint32_t physicalPageId )
{
    std::lock_guard<std::mutex> lock( mutex_ );

    PhysicalPage& physicalPage = physical_.pages.at( physicalPageId );
    if( !physicalPage.handle || physicalPage.useCount == 0 )
        throw std::logic_error( "Physical page " + std::to_string( physicalPageId ) + " is not in use and cannot be retained" );
    if( physicalPage.retained )
        return;

    physicalPage.retained = true;
    physicalPage.useCount++;
}

void PageSystem::enqueueEvictedPages( const HostSpan<EvictionCandidate>& pages, uint32_t reusablePageCount )
{
    std::lock_guard<std::mutex> lock( mutex_ );

    virtual_.pendingReusableEvictedPageCount += std::min( static_cast<uint32_t>( pages.len ), reusablePageCount );
    for( size_t i = 0; i < pages.len; i++ )
        virtual_.pendingEvictedPages.push_back( pages.ptr[i].pageId );
}

uint32_t PageSystem::physicalPageCount() const
{
    std::lock_guard<std::mutex> lock( mutex_ );
    return physical_.allocatedCount;
}

uint32_t PageSystem::mappedPageCount() const
{
    std::lock_guard<std::mutex> lock( mutex_ );
    return static_cast<uint32_t>( virtualIdToPhysicalId_.size() );
}

uint32_t PageSystem::unmappedPageCount() const
{
    std::lock_guard<std::mutex> lock( mutex_ );
    return virtual_.unmappedCount;
}

// TODO_BS: remove this workaround (and its calls) once ROCm/rocm-systems#13106 is fixed and hipMemUnmap() invalidates the GPU TLB itself.
void PageSystem::invalidateDeviceTlb()
{
#ifdef __linux__
    void* hostMemory = nullptr;
    HIP_CHECK( hipHostMalloc( &hostMemory, 4096 ) );
    HIP_CHECK( hipHostFree( hostMemory ) );
#endif
}

void PageSystem::processPendingEvictedPages()
{
    if( virtual_.pendingEvictedPages.empty() )
        return;

    const size_t requestedReusablePages = virtual_.pendingReusableEvictedPageCount;
    const size_t availableReusablePages = physical_.reusablePages.size();
    const size_t pagesToReuse =
        virtual_.pendingReusableEvictedPageCount - std::min( requestedReusablePages, availableReusablePages );
    size_t freedPages    = 0;
    bool   unmappedPages = false;
    for( const uint32_t virtualPageId : virtual_.pendingEvictedPages )
    {
        const auto mapping = virtualIdToPhysicalId_.find( virtualPageId );
        if( mapping == virtualIdToPhysicalId_.end() )
            continue;
        const uint32_t physicalPageId = mapping->second;

        const DeviceSpan<uint8_t> virtualPage = page( virtualPageId );
        HIP_CHECK( hipMemUnmap( virtualPage.ptr, virtualPage.len ) );
        virtualIdToPhysicalId_.erase( mapping );
        virtual_.unmappedCount++;
        unmappedPages = true;

        PhysicalPage& physicalPage = physical_.pages.at( physicalPageId );
        physicalPage.useCount--;
        // Shared and retained pages stay allocated for their other users.
        if( physicalPage.useCount > 0 )
            continue;

        if( freedPages < pagesToReuse )
        {
            physical_.reusablePages.push_back( physicalPageId );
        }
        else
        {
            HIP_CHECK( hipMemRelease( physicalPage.handle ) );
            physicalPage.handle = nullptr;
            physical_.allocatedCount--;
            physical_.unallocatedPages.push_back( physicalPageId );
        }
        freedPages++;
    }

    virtual_.pendingReusableEvictedPageCount = 0;
    virtual_.pendingEvictedPages.clear();

    // The unmapped pages are mapped again, to other physical pages, before the device touches them next.
    if( unmappedPages )
        invalidateDeviceTlb();
}

}  // namespace hip_demand::vmm
