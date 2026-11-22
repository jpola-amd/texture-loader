#pragma once

#include "../Internal/Utils.h"
#include <DemandLoading/DeviceContext.h>
#include <cstdint>
#include <hip/hip_runtime.h>
#include <mutex>
#include <unordered_map>
#include <vector>

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

    Range textureTiles{};
};

class PageSystem : NonCopyble
{
  public:
    // Fixed page size, matching the OptiX Toolkit tile size. Must be a power of two:
    // the sampler computes page addresses with shifts.
    static constexpr size_t PAGE_BYTES = 64 * 1024;
    static_assert( ( PAGE_BYTES & ( PAGE_BYTES - 1 ) ) == 0, "PAGE_BYTES must be a power of two" );

    explicit PageSystem( uint32_t maxVirtualPages, uint32_t maxPhysicalPages );
    ~PageSystem();

    uint32_t map( uint32_t pageId );
    void mapAlias( uint32_t pageId, uint32_t physicalPageId );
    void retain( uint32_t physicalPageId );
    void enqueueEvictedPages( const HostSpan<EvictionCandidate>& pages, uint32_t reusablePageCount );

    DeviceSpan<uint8_t> page( uint32_t pageId ) const
    {
        return DeviceSpan<uint8_t>( virtual_.addressSpace.ptr + pageId * pageBytes_, pageBytes_ );
    }
    DeviceSpan<uint8_t> virtualAddressSpace() const { return virtual_.addressSpace; }
    size_t              pageBytes() const { return pageBytes_; }
    int                 device() const { return device_; }
    uint32_t physicalPageCount() const;
    uint32_t mappedPageCount() const;
    uint32_t unmappedPageCount() const;

    /// Makes the device pick up changed page mappings. On Linux, ROCm's hipMemMap()/hipMemUnmap() do not invalidate
    /// the GPU TLB (the amdgpu kernel driver skips it, see ROCm/rocm-systems#13106), so a virtual page that is unmapped
    /// and then mapped to another physical page keeps reading and writing the old one until its translation happens
    /// to be evicted. Mapping host memory for the GPU goes through the kernel driver, which does invalidate the TLB.
    /// Windows is not affected, and the call does nothing there.
    static void invalidateDeviceTlb();

  private:
    void processPendingEvictedPages();

    struct PhysicalPage
    {
        hipMemGenericAllocationHandle_t handle   = nullptr;
        uint32_t                        useCount = 0;  // Virtual pages mapped to the page, plus one if retained.
        bool                            retained = false;
    };

    int                   device_      = 0;
    size_t                pageBytes_   = 0;
    hipMemAllocationProp  allocationProp_{};
    std::unordered_map<uint32_t, uint32_t> virtualIdToPhysicalId_{};
    struct
    {
        uint32_t                  maxPages;
        uint32_t                  allocatedCount;
        std::vector<PhysicalPage> pages;
        std::vector<uint32_t>     reusablePages;
        std::vector<uint32_t>     unallocatedPages;
    } physical_{};
    struct
    {
        uint32_t              maxPages;
        uint32_t              pendingReusableEvictedPageCount;
        uint32_t              unmappedCount;
        std::vector<uint32_t> pendingEvictedPages;
        DeviceSpan<uint8_t>   addressSpace;
    } virtual_{};
    mutable std::mutex mutex_;
};

}  // namespace hip_demand::vmm
