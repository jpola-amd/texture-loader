#pragma once

#include "../Internal/Utils.h"
#include "PageSystem.h"
#include <DemandLoading/DeviceContext.h>
#include <cstdint>
#include <hip/hip_runtime.h>

namespace hip_demand::vmm {


template <typename T>
class Allocator : NonCopyble
{
  public:
    explicit Allocator( PageSystem& pageSystem )
        : pageSystem_( pageSystem )
    {
    }

    DevicePtr<T> alloc() { return reinterpret_cast<DevicePtr<T>>( alloc( sizeof( T ) ) ); }

    void setRange( const PageTable::Range& pageRange )
    {
        pageRange_          = pageRange;
        availablePageBytes_ = pageSystem_.pageBytes();
    }

  private:
    hipDeviceptr_t alloc( size_t size )
    {
        assert( size > 0 );

        if( pageRange_.nextAvailablePage < pageRange_.startPage )
            throw std::logic_error( "VmmAllocator page range has an invalid next available page" );

        const uint32_t usedPages = pageRange_.nextAvailablePage - pageRange_.startPage;
        if( usedPages >= pageRange_.pageCount )
            throw std::bad_alloc{};

        const size_t   pageBytes      = pageSystem_.pageBytes();
        const uint32_t remainingPages = pageRange_.pageCount - usedPages;

        size_t bytesAfterCurrentPage = 0;
        size_t additionalPages       = 0;
        if( size > availablePageBytes_ )
        {
            bytesAfterCurrentPage = size - availablePageBytes_;
            additionalPages       = 1 + ( bytesAfterCurrentPage - 1 ) / pageBytes;
        }

        if( additionalPages > remainingPages - 1 )
            throw std::bad_alloc{};

        for( size_t pageOffset = 0; pageOffset < additionalPages + 1; ++pageOffset )
            pageSystem_.map( static_cast<uint32_t>( pageRange_.nextAvailablePage + pageOffset ) );

        hipDeviceptr_t result = pageSystem_.page( pageRange_.nextAvailablePage ).ptr + pageBytes - availablePageBytes_;

        if( size < availablePageBytes_ )
        {
            availablePageBytes_ -= size;
        }
        else
        {
            const uint32_t pagesAdvanced = static_cast<uint32_t>( 1 + bytesAfterCurrentPage / pageBytes );
            pageRange_.nextAvailablePage += pagesAdvanced;
            availablePageBytes_ = pageBytes - ( bytesAfterCurrentPage % pageBytes );
        }

        return result;
    }

    PageSystem&      pageSystem_;
    PageTable::Range pageRange_{};
    size_t           availablePageBytes_ = 0;
};

}  // namespace hip_demand::vmm