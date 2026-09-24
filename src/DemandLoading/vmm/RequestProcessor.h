#pragma once

#include "../Internal/Utils.h"
#include "HipEventPool.h"
#include "TicketImpl.h"
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <queue>

namespace hip_demand::vmm {

struct ResourceRequest
{
    uint32_t resourceId = 0;
    Ticket   ticket;

    explicit ResourceRequest( uint32_t resourceId_, Ticket ticket_ )
        : resourceId( resourceId_ )
        , ticket( ticket_ )
    {
    }

    ResourceRequest() {}
};

class RequestQueue : NonCopyble
{
  public:
    explicit RequestQueue( size_t maxQueueSize )
        : maxQueueSize_( maxQueueSize )
    {
        if( maxQueueSize_ == 0 )
            throw std::invalid_argument( "RequestQueue maxQueueSize must be greater than zero" );
    }

    void push( const uint32_t* resourceIds, uint32_t count, Ticket ticket )
    {
        std::unique_lock<std::mutex> lock( mutex_ );

        if( requests_.size() >= maxQueueSize_ )
            count = 0;
        else if( count + requests_.size() > maxQueueSize_ )
            count = static_cast<uint32_t>( maxQueueSize_ - requests_.size() );

        TicketImpl::getImpl( ticket )->initialize( count );
        if( count == 0 )
            return;

        if( closed_ )
            return;

        for( uint32_t i = 0; i < count; i++ )
            requests_.emplace( resourceIds[i], ticket );

        notEmpty_.notify_all();
    }

    bool pop( ResourceRequest& value )
    {
        std::unique_lock<std::mutex> lock( mutex_ );
        notEmpty_.wait( lock, [this] { return closed_ || !requests_.empty(); } );
        if( requests_.empty() )
            return false;

        value = std::move( requests_.front() );
        requests_.pop();
        return true;
    }

    void close()
    {
        {
            std::lock_guard<std::mutex> lock( mutex_ );
            closed_ = true;
        }
        notEmpty_.notify_all();
    }

  private:
    const size_t                maxQueueSize_;
    mutable std::mutex          mutex_;
    std::condition_variable     notEmpty_;
    std::queue<ResourceRequest> requests_;
    bool                        closed_{ false };
};

class DemandTextureLoaderImpl;

class RequestProcessor : NonCopyble
{
  public:
    explicit RequestProcessor( DemandTextureLoaderImpl* loader, uint32_t maxThreads, uint32_t maxQueueSize );
    ~RequestProcessor();

    void submit( const uint32_t* resourceIds, uint32_t count, Ticket ticket );
    void stop();

    uint32_t threadCount() const noexcept { return workers_.size(); }

  private:
    void workerLoop();

    mutable std::mutex       mutex_;
    RequestQueue             queue_;
    std::vector<std::thread> workers_;
    bool                     stopped_                    = false;
    DemandTextureLoaderImpl* loader_                     = nullptr;
};

}  // namespace hip_demand::vmm
