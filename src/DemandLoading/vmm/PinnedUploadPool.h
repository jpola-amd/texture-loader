#pragma once

#include "../Internal/HipCheck.h"
#include "../Internal/Utils.h"
#include "HipEventPool.h"
#include <DemandLoading/DeviceContext.h>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <map>
#include <mutex>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace hip_demand::vmm {

class PinnedUploadPool : NonCopyble
{
  public:
    PinnedUploadPool( size_t maxPendingUploads )
        : maxPendingUploads_( std::max<size_t>( maxPendingUploads, 1 ) )
        , eventPool_( 10 )
    {
    }

    ~PinnedUploadPool()
    {
        drain();

        for( auto& [size, buffers] : available_ )
        {
            for( HostSpan<uint8_t> buffer : buffers )
                HIP_WARN( hipHostFree( buffer.ptr ) );
        }
    }

    template <typename T>
    class Buffer
    {
      public:
        explicit Buffer( PinnedUploadPool* pool, HostSpan<T> memory, hipStream_t stream )
            : pool_( pool )
            , memory_( memory )
            , stream_( stream )
        {
        }
        ~Buffer() { reset(); }

        Buffer( const Buffer& )            = delete;
        Buffer& operator=( const Buffer& ) = delete;

        Buffer( Buffer&& other ) noexcept
            : pool_( std::exchange( other.pool_, nullptr ) )
            , memory_( std::exchange( other.memory_, {} ) )
            , stream_( std::exchange( other.stream_, nullptr ) )
        {
        }

        Buffer& operator=( Buffer&& other ) noexcept
        {
            if( this != &other )
            {
                reset();
                pool_   = std::exchange( other.pool_, nullptr );
                memory_ = std::exchange( other.memory_, {} );
                stream_ = std::exchange( other.stream_, nullptr );
            }
            return *this;
        }

        HostSpan<T> span() const noexcept { return memory_; }
        T*          operator->() noexcept { return memory_.ptr; }
        T&          operator*() noexcept { return *memory_.ptr; }

      private:

        void reset()
        {
            if( pool_ && memory_.ptr )
                pool_->retire( memory_, stream_ );
            pool_   = nullptr;
            memory_ = {};
            stream_ = nullptr;
        }

        PinnedUploadPool* pool_{};
        HostSpan<T>       memory_{};
        hipStream_t       stream_{};
    };

    template <typename T>
    Buffer<T> acquire( size_t count, hipStream_t stream )
    {
        static_assert( std::is_trivially_copyable<T>::value, "Pinned uploads require trivially copyable elements" );

        HostSpan<uint8_t> bytes = acquireBytes( count * sizeof( T ) );
        return Buffer<T>( this, HostSpan<T>( reinterpret_cast<T*>( bytes.ptr ), count ), stream );
    }

    template <typename T>
    void retire( HostSpan<T>& span, hipStream_t stream )
    {
        assert( span.ptr );

        std::lock_guard<std::mutex> lock( mutex_ );

        Pending async = {
            HostSpan<uint8_t>( reinterpret_cast<uint8_t*>( span.ptr ), span.sizeInBytes() ),
            eventPool_.acquire()
        };
        HIP_CHECK( hipEventRecord( async.event, stream ) );
        pendingQueue_.push_back( async );

        span = {};
    }

    void drain()
    {
        std::lock_guard<std::mutex> lock( mutex_ );
        collectCompleted( true );
    }

  private:
    struct Pending
    {
        HostSpan<uint8_t> memory{};
        hipEvent_t        event{};
    };

    HostSpan<uint8_t> acquireBytes( size_t size )
    {
        assert( size > 0 );

        {
            std::lock_guard<std::mutex> lock( mutex_ );

            collectCompleted( false );

            auto& allocations = available_[size];
            if( !allocations.empty() )
            {
                HostSpan<uint8_t> result = allocations.back();
                allocations.pop_back();
                return result;
            }
        }

        return hostAllocArray<uint8_t>( size );
    }

    void collectCompleted( bool wait )
    {
        while( !pendingQueue_.empty() )
        {
            const Pending& pendingRelease   = pendingQueue_.front();
            hipError_t     eventQueryStatus = hipEventQuery( pendingRelease.event );
            if( eventQueryStatus == hipErrorNotReady )
            {
                if( wait || pendingQueue_.size() > maxPendingUploads_ )
                {
                    HIP_WARN( hipEventSynchronize( pendingRelease.event ) );
                }
                else
                {
                    break;
                }
            }
            else
            {
                HIP_WARN( eventQueryStatus );
            }

            eventPool_.release( pendingRelease.event );
            available_.at( pendingRelease.memory.len ).push_back( pendingRelease.memory );
            pendingQueue_.pop_front();
        }
    }

    const size_t                                     maxPendingUploads_;
    std::mutex                                       mutex_;
    HipEventPool                                     eventPool_;
    std::deque<Pending>                              pendingQueue_;
    std::map<size_t, std::vector<HostSpan<uint8_t>>> available_;
};

}  // namespace hip_demand::vmm
