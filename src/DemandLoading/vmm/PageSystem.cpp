#include "PageSystem.h"
#include "../Internal/HipCheck.h"

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
    HIP_CHECK( hipMemGetAllocationGranularity( &granularity_, &allocationProp_, hipMemAllocationGranularityMinimum ) );
    if( granularity_ == 0 )
        throw std::runtime_error( "hipMemAllocationGranularityMinimum is zero" );

    pageBytes_                = granularity_;
    virtual_.addressSpace.len = virtual_.maxPages * pageBytes_;
    HIP_CHECK( hipMemAddressReserve( reinterpret_cast<void**>( &virtual_.addressSpace.ptr ), virtual_.addressSpace.len,
                                     granularity_, nullptr, 0 ) );

    virtualIdToPhysicalId_.assign( virtual_.maxPages, INVALID_PAGE );
}

PageSystem::~PageSystem()
{
    for( PhysicalPage& physicalPage : physical_.pages )
    {
        if( physicalPage.virtualPageId != INVALID_PAGE )
        {
            DeviceSpan<uint8_t> virtualPage = page( physicalPage.virtualPageId );
            HIP_WARN( hipMemUnmap( virtualPage.ptr, virtualPage.len ) );
        }

        if( physicalPage.handle )
            HIP_WARN( hipMemRelease( physicalPage.handle ) );
    }

    if( virtual_.addressSpace.ptr )
        HIP_WARN( hipMemAddressFree( virtual_.addressSpace.ptr, virtual_.addressSpace.len ) );
}

void PageSystem::map( uint32_t pageId )
{
    std::lock_guard<std::mutex> lock( mutex_ );

    processPendingEvictedPages();

    if( virtualIdToPhysicalId_.at( pageId ) != INVALID_PAGE )
        return;

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
            physical_.pages.push_back( { nullptr, INVALID_PAGE } );
        }

        HIP_CHECK( hipMemCreate( &physical_.pages.at( physicalPageId ).handle, pageBytes_, &allocationProp_, 0 ) );
    }

    PhysicalPage&       physicalPage = physical_.pages.at( physicalPageId );
    DeviceSpan<uint8_t> virtualPage  = page( pageId );
    HIP_CHECK( hipMemMap( virtualPage.ptr, virtualPage.len, 0, physicalPage.handle, 0 ) );

    hipMemAccessDesc access{};
    access.location.type = hipMemLocationTypeDevice;
    access.location.id   = device_;
    access.flags         = hipMemAccessFlagsProtReadWrite;
    HIP_CHECK( hipMemSetAccess( virtualPage.ptr, virtualPage.len, &access, 1 ) );

    virtualIdToPhysicalId_.at( pageId ) = physicalPageId;
    physicalPage.virtualPageId          = pageId;
}

void PageSystem::enqueueEvictedPages( const HostSpan<EvictionCandidate>& pages,
                                      uint32_t                           reusablePageCount,
                                      std::function<void( uint32_t )>    callback )
{
    std::lock_guard<std::mutex> lock( mutex_ );

    virtual_.pendingReusableEvictedPageCount += std::min( static_cast<uint32_t>( pages.len ), reusablePageCount );
    for( size_t i = 0; i < pages.len; i++ )
    {
        const uint32_t pageId = pages.ptr[i].pageId;
        virtual_.pendingEvictedPages.push_back( pageId );
        callback( pageId );
    }
}

void PageSystem::processPendingEvictedPages()
{
    if( !virtual_.pendingEvictedPages.empty() )
    {
        const size_t requestedReusablePages = virtual_.pendingReusableEvictedPageCount;
        const size_t availableReusablePages = physical_.reusablePages.size();
        const size_t pagesToReuse = virtual_.pendingReusableEvictedPageCount - std::min( requestedReusablePages, availableReusablePages );
        for( size_t i = 0; i < virtual_.pendingEvictedPages.size(); i++ )
        {
            const uint32_t virtualPageId  = virtual_.pendingEvictedPages.at( i );
            const uint32_t physicalPageId = virtualIdToPhysicalId_.at( virtualPageId );
            if( physicalPageId == INVALID_PAGE )
                continue;

            const DeviceSpan<uint8_t> virtualPage = page( virtualPageId );
            HIP_CHECK( hipMemUnmap( virtualPage.ptr, virtualPage.len ) );
            virtualIdToPhysicalId_.at( virtualPageId ) = INVALID_PAGE;

            PhysicalPage& physicalPage = physical_.pages.at( physicalPageId );
            physicalPage.virtualPageId = INVALID_PAGE;

            if( i < pagesToReuse )
            {
                physical_.reusablePages.push_back( physicalPageId );
            }
            else
            {
                HIP_CHECK( hipMemRelease( physicalPage.handle ) );
                physicalPage.handle = nullptr;
                physical_.unallocatedPages.push_back( physicalPageId );
            }
        }

        virtual_.pendingReusableEvictedPageCount = 0;
        virtual_.pendingEvictedPages.clear();
    }
}

}  // namespace hip_demand::vmm
