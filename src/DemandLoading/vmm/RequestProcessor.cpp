#pragma once

#include "RequestProcessor.h"
#include "DemandTextureLoaderImpl.h"

namespace hip_demand::vmm {

RequestProcessor::RequestProcessor( DemandTextureLoaderImpl* loader, uint32_t maxThreads, uint32_t maxQueueSize )
    : queue_( maxQueueSize )
    , loader_( loader )
{
    if( maxThreads == 0 )
        maxThreads = std::thread::hardware_concurrency();

    try
    {
        workers_.reserve( maxThreads );
        for( uint32_t i = 0; i < maxThreads; ++i )
            workers_.emplace_back( [this] { workerLoop(); } );
    }
    catch( ... )
    {
        queue_.close();
        for( std::thread& worker : workers_ )
            worker.join();
        throw;
    }
}

RequestProcessor::~RequestProcessor()
{
    stop();
}

void RequestProcessor::stop()
{
    {
        std::lock_guard<std::mutex> lock( mutex_ );
        stopped_ = true;
    }
    queue_.close();

    for( std::thread& worker : workers_ )
    {
        if( worker.joinable() )
            worker.join();
    }
}

void RequestProcessor::submit( const uint32_t* resourceIds, uint32_t count, Ticket ticket )
{
    std::lock_guard<std::mutex> lock( mutex_ );
    if( stopped_ )
    {
        TicketImpl::getImpl( ticket )->initialize( 0 );
        return;
    }

    queue_.push( resourceIds, count, ticket );
}

void RequestProcessor::workerLoop()
{
    try
    {
        HIP_CHECK( hipSetDevice( loader_->device() ) );
        while( true )
        {
            ResourceRequest request;
            if( !queue_.pop( request ) )
                break;

            std::shared_ptr<TicketImpl>& ticket = TicketImpl::getImpl( request.ticket );
            loader_->processRequest( ticket->getStream(), request.resourceId );
            if (ticket->finishTaskAndClaimFinalization())
            {
                loader_->updateProccedResources( ticket->getStream() );
                ticket->publishCompletion();
            }
        }
    }
    catch( const std::exception& e )
    {
        std::cerr << "Error: " << e.what() << std::endl;
    }
}

}  // namespace hip_demand::vmm
