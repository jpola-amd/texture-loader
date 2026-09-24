#pragma once

#include "../Internal/Utils.h"
#include <DemandLoading/DeviceContext.h>
#include <cstdint>
#include <hip/hip_runtime.h>
#include <mutex>
#include <functional>

namespace hip_demand::vmm {

struct PageTable
{
    struct Range
    {
        uint32_t startPage         = 0;
        uint32_t pageCount         = 0;
        uint32_t nextAvailablePage = 0;

        Range() {}
        Range( uint32_t start, uint32_t count )
            : startPage( start )
            , pageCount( count )
            , nextAvailablePage( start )
        {
        }
    };

    Range textureInfos{};
    Range textureTiles{};
};

class PageSystem : NonCopyble
{
  public:
    explicit PageSystem( uint32_t maxVirtualPages, uint32_t maxPhysicalPages );
    ~PageSystem();
    void map( uint32_t pageId );
    void enqueueEvictedPages( const HostSpan<EvictionCandidate>& pages, uint32_t reusablePageCount, std::function<void( uint32_t )> callback );

    DeviceSpan<uint8_t> page( uint32_t pageId ) const
    {
        return DeviceSpan<uint8_t>( virtual_.addressSpace.ptr + pageId * pageBytes_, pageBytes_ );
    }
    DeviceSpan<uint8_t> virtualAddressSpace() const { return virtual_.addressSpace; }
    size_t              granularity() const { return granularity_; }
    size_t              pageBytes() const { return pageBytes_; }
    int                 device() const { return device_; }

  private:
    void processPendingEvictedPages();

    struct PhysicalPage
    {
        hipMemGenericAllocationHandle_t handle        = nullptr;
        uint32_t                        virtualPageId = INVALID_PAGE;
    };

    int                   device_      = 0;
    size_t                granularity_ = 0;
    size_t                pageBytes_   = 0;
    hipMemAllocationProp  allocationProp_{};
    std::vector<uint32_t> virtualIdToPhysicalId_{};
    struct
    {
        uint32_t                  maxPages;
        std::vector<PhysicalPage> pages;
        std::vector<uint32_t>     reusablePages;
        std::vector<uint32_t>     unallocatedPages;
    } physical_{};
    struct
    {
        uint32_t              maxPages;
        uint32_t              pendingReusableEvictedPageCount;
        std::vector<uint32_t> pendingEvictedPages;
        DeviceSpan<uint8_t>   addressSpace;
    } virtual_{};
    mutable std::mutex mutex_;
};

}  // namespace hip_demand::vmm