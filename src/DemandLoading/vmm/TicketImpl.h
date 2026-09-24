#pragma once

#include "../Internal/HipCheck.h"
#include <DemandLoading/VmmDemandTextureLoader.h>
#include <cassert>
#include <condition_variable>
#include <exception>
#include <mutex>
#include <utility>

namespace hip_demand::vmm {

class TicketImpl
{
  public:
    static Ticket create( hipStream_t stream ) { return Ticket( std::make_shared<TicketImpl>( stream ) ); }
    static std::shared_ptr<TicketImpl>& getImpl( Ticket& ticket ) { return ticket.impl_; }

    TicketImpl( hipStream_t stream )
        : stream_( stream )
    {
    }

    void initialize( unsigned int numTasks )
    {
        {
            std::unique_lock<std::mutex> lock( mutex_ );
            numTasksTotal_     = numTasks;
            numTasksRemaining_ = numTasks;
        }
        if( numTasks == 0 )
            isDone_.notify_all();
    }

    hipStream_t getStream() const { return stream_; }

    int numTasksTotal() const
    {
        std::unique_lock<std::mutex> lock( mutex_ );
        return numTasksTotal_;
    }

    int numTasksRemaining() const
    {
        std::unique_lock<std::mutex> lock( mutex_ );
        return numTasksRemaining_;
    }

    void wait( hipEvent_t* event = nullptr )
    {
        std::unique_lock<std::mutex> lock( mutex_ );
        isDone_.wait( lock, [this] { return numTasksRemaining_ == 0; } );
        if( event )
        {
            HIP_CHECK( hipEventRecord( *event, stream_ ) );
        }
    }

    bool finishTaskAndClaimFinalization()
    {
        std::lock_guard lock( mutex_ );
        assert( numTasksRemaining_ > 0 );

        if( numTasksRemaining_ > 1 )
        {
            --numTasksRemaining_;
            return false;
        }

        return true;
    }

    void publishCompletion()
    {
        std::lock_guard lock( mutex_ );
        assert( numTasksRemaining_ == 1 );
        numTasksRemaining_ = 0;
        isDone_.notify_all();
    }

  private:
    const hipStream_t       stream_            = nullptr;
    int                     numTasksTotal_     = -1;
    int                     numTasksRemaining_ = -1;
    mutable std::mutex      mutex_;
    std::condition_variable isDone_;
};

}  // namespace hip_demand::vmm
