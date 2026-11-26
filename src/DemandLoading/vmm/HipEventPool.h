#pragma once

#include "../Internal/HipCheck.h"
#include "../Internal/Utils.h"
#include <mutex>

namespace hip_demand::vmm {

class HipEventPool : NonCopyble
{
  public:
    explicit HipEventPool( size_t initialSize )
    {
        events_.reserve( initialSize );
        for( size_t i = 0; i < initialSize; ++i )
        {
            hipEvent_t event{};
            HIP_CHECK( hipEventCreateWithFlags( &event, hipEventDisableTiming ) );
            events_.push_back( event );
        }
    }

    ~HipEventPool()
    {
        for( hipEvent_t event : events_ )
            HIP_WARN(hipEventDestroy( event ));

        events_.clear();
    }

    hipEvent_t acquire()
    {
        {
            std::lock_guard<std::mutex> lock( mutex_ );
            if( !events_.empty() )
            {
                hipEvent_t event = events_.back();
                events_.pop_back();
                return event;
            }
        }

        hipEvent_t event{};
        HIP_CHECK( hipEventCreateWithFlags( &event, hipEventDisableTiming ) );
        return event;
    }

    void release( hipEvent_t event )
    {
        assert( event );
        std::lock_guard<std::mutex> lock( mutex_ );
        events_.push_back( event );
    }

  private:
    mutable std::mutex      mutex_;
    std::vector<hipEvent_t> events_;
};


}  // namespace hip_demand::vmm
