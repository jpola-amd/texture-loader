#include "DemandTextureLoaderImpl.h"
#include <ImageSource/TextureInfo.h>

#include <filesystem>

#if defined( _WIN32 )
#define NOMINMAX
#include <Windows.h>
#else
#include <dlfcn.h>
#endif

namespace hip_demand::vmm {

static std::filesystem::path getLibraryDirectory()
{
#if defined( _WIN32 )
    HMODULE module = nullptr;
    if( !GetModuleHandleExW( GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                             reinterpret_cast<LPCWSTR>( &getLibraryDirectory ), &module ) )
    {
        throw std::runtime_error( "Failed to locate hip_demand_texture library" );
    }

    std::vector<wchar_t> path( MAX_PATH );
    DWORD                pathLength = GetModuleFileNameW( module, path.data(), static_cast<DWORD>( path.size() ) );
    if( pathLength == path.size() )
    {
        path.resize( 32768 );
        pathLength = GetModuleFileNameW( module, path.data(), static_cast<DWORD>( path.size() ) );
    }

    if( pathLength == 0 || pathLength == path.size() )
        throw std::runtime_error( "Failed to get hip_demand_texture library path" );

    return std::filesystem::path( path.data(), path.data() + pathLength ).parent_path();
#else
    Dl_info libraryInfo{};
    if( dladdr( reinterpret_cast<const void*>( &getLibraryDirectory ), &libraryInfo ) == 0 || !libraryInfo.dli_fname )
        throw std::runtime_error( "Failed to locate hip_demand_texture library" );

    return std::filesystem::path( libraryInfo.dli_fname ).parent_path();
#endif
}

class HipCallback
{
  public:
    virtual ~HipCallback() = default;
    static void enqueue( hipStream_t stream, HipCallback* callback )
    {
        HIP_CHECK( hipLaunchHostFunc( stream, &staticCallback, callback ) );
    }

    static void staticCallback( void* arg )
    {
        std::unique_ptr<HipCallback> callback( static_cast<HipCallback*>( arg ) );
        callback->callback();
    }

  protected:
    virtual void callback() = 0;
};

class ProcessRequestCallback : public HipCallback
{
  public:
    explicit ProcessRequestCallback( DemandTextureLoaderImpl& loader, Ticket ticket )
        : loader_( loader )
        , ticket_( std::move( ticket ) )
    {
    }

  protected:
    void callback() { loader_.processRequestsCallback( std::move( ticket_ ) ); }

  private:
    DemandTextureLoaderImpl& loader_;
    Ticket                   ticket_;
};

inline ResourceTable createResourceTable( const Options& options, size_t pageSize )
{
    const size_t   maxTextureInfoSize  = options.maxTextures * sizeof( DeviceTextureInfo );
    const uint32_t maxTextureInfoPages = static_cast<uint32_t>( ceilDiv( maxTextureInfoSize, pageSize ) );
    if( maxTextureInfoPages > options.maxVirtualPages )
        throw std::invalid_argument( "maxVirtualPages is too small to store texture metadata" );

    const uint32_t maxTextureTilePages = options.maxVirtualPages - maxTextureInfoPages;

    ResourceTable table{};
    table.textureTiles = ResourceTable::Range( 0, maxTextureTilePages );
    table.textureInfos = ResourceTable::Range( table.textureTiles.end(), options.maxTextures );

    uint32_t resourceCount = 0;
    if( !safeAdd( table.textureTiles.count, table.textureInfos.count, resourceCount ) )
        throw std::overflow_error(
            "Cannot create demand texture loader: total resource count exceeds the uint32_t limit" );

    return table;
}

inline PageTable createPageTable( const ResourceTable& resourceTable, const Options& options, size_t pageSize )
{
    const size_t   maxTextureInfoSize  = resourceTable.textureInfos.count * sizeof( DeviceTextureInfo );
    const uint32_t maxTextureInfoPages = static_cast<uint32_t>( ceilDiv( maxTextureInfoSize, pageSize ) );
    if( maxTextureInfoPages > options.maxVirtualPages )
        throw std::invalid_argument( "maxVirtualPages is too small to store texture metadata" );
    const uint32_t maxTextureTilePages = options.maxVirtualPages - maxTextureInfoPages;

    PageTable pageTable{};
    pageTable.textureTiles = PageTable::Range( 0, maxTextureTilePages );
    pageTable.textureInfos = PageTable::Range( pageTable.textureTiles.pageCount, maxTextureInfoPages );
    return pageTable;
}

DemandTextureLoaderImpl::DemandTextureLoaderImpl( const Options& options )
    : options_( options )
    , pageSystem_( options_.maxVirtualPages, options_.maxPhysicalPages )
    , textureInfoAllocator_( pageSystem_ )
    , resourceTable_( createResourceTable( options_, pageSystem_.pageBytes() ) )
    , pageTable_( createPageTable( resourceTable_, options_, pageSystem_.pageBytes() ) )
    , requestProcessor_( this, options_.maxThreads, options_.maxRequests )
    , uploadPool_( requestProcessor_.threadCount() * 3 )
    , bits_( resourceTable_.count() )
{
    if( options_.maxRequests == 0 )
        throw std::invalid_argument( "maxRequests cannot be 0" );

    textureInfoAllocator_.setRange( pageTable_.textureInfos );

    const std::filesystem::path kernelsPath = getLibraryDirectory() / "hip_demand_texture_kernels.co";
    HIP_CHECK( hipModuleLoad( &kernels_.module, kernelsPath.string().c_str() ) );

    options_.enableEviction = options_.enableEviction && options_.maxEvictedPages > 0;
    HIP_CHECK( hipModuleGetFunction( &kernels_.collectRequests, kernels_.module, "collectRequestsKernel" ) );
    HIP_CHECK(
        hipModuleGetFunction( &kernels_.updateProcessedResources, kernels_.module, "updateProcessedResourcesKernel" ) );
    HIP_CHECK( hipModuleGetFunction( &kernels_.updateEvictedPages, kernels_.module, "updateEvictedPagesKernel" ) );

    hipStream_t stream                = nullptr;
    deviceContext_.resourceTable      = resourceTable_;
    deviceContext_.pageSize           = pageSystem_.pageBytes();
    deviceContext_.pageMemory         = pageSystem_.virtualAddressSpace();
    deviceContext_.residenceBits      = allocArray<uint32_t>( bits_.wordCount(), true, stream );
    deviceContext_.textureInfos       = allocArray<DeviceTextureInfo*>( options_.maxTextures, true, stream );
    deviceContext_.referenceBits      = allocArray<uint32_t>( bits_.wordCount(), true, stream );
    deviceContext_.requestedResources = allocArray<uint32_t>( options_.maxRequests, false, stream );
    deviceContext_.counters = allocArray<uint32_t>( static_cast<size_t>( CounterIndex::NumCounters ), true, stream );
    deviceContext_.processedResources = allocArray<ProcessedResource>( options_.maxRequests, false, stream );
    if( options_.enableEviction )
    {
        deviceContext_.requestIfResident  = true;
        deviceContext_.evictionCandidates = allocArray<EvictionCandidate>( options_.maxEvictedPages, false, stream );
        // 4 bits per resource, 8 values per uint32_t
        deviceContext_.lru = allocArray<uint32_t>( ceilDiv( resourceTable_.textureTiles.count, 8u ), true, stream );
    }
    HIP_CHECK( hipStreamSynchronize( stream ) );

    // set host transfer buffers
    requestedResources_  = hostAllocArray<uint32_t>( deviceContext_.requestedResources.len );
    counters_            = hostAllocArray<uint32_t>( deviceContext_.counters.len, true );
    eviction_.candidates = hostAllocArray<EvictionCandidate>( deviceContext_.evictionCandidates.len );
    processed_.resources = hostAllocArray<ProcessedResource>( options_.maxRequests );
}

DemandTextureLoaderImpl::~DemandTextureLoaderImpl()
{
    requestProcessor_.stop();
    uploadPool_.drain();

    if( kernels_.module )
    {
        HIP_WARN( hipModuleUnload( kernels_.module ) );
        kernels_ = {};
    }

    freeArray( deviceContext_.residenceBits );
    freeArray( deviceContext_.lru );
    freeArray( deviceContext_.textureInfos );
    freeArray( deviceContext_.referenceBits );
    freeArray( deviceContext_.requestedResources );
    freeArray( deviceContext_.evictionCandidates );
    freeArray( deviceContext_.processedResources );
    freeArray( deviceContext_.counters );

    hostFreeArray( requestedResources_ );
    hostFreeArray( counters_ );
    hostFreeArray( eviction_.candidates );
    hostFreeArray( processed_.resources );
}

const DemandTexture& DemandTextureLoaderImpl::createTexture( std::shared_ptr<ImageSource> imageSource, const TextureDescriptor& descriptor )
{
    std::lock_guard lock( mutex_ );

    if( !imageSource )
        throw std::invalid_argument( "Image source is null" );

    if( textures_.size() >= options_.maxTextures )
        throw std::runtime_error( "Maximum demand texture count exceeded" );

    const uint32_t textureId = static_cast<uint32_t>( textures_.size() );
    textures_.emplace_back( std::make_unique<DemandTextureImpl>( textureId, imageSource, descriptor ) );
    return *textures_.back();
}

void DemandTextureLoaderImpl::launchPrepare( hipStream_t stream, DeviceContext& deviceContext )
{
    std::lock_guard lock( mutex_ );

    deviceContext              = deviceContext_;
    deviceContext.textureInfos = deviceContext.textureInfos.subspan( 0, textures_.size() );
    clearEvictedPages( stream );
    memset( deviceContext.referenceBits, 0, stream );
}

inline uint32_t roundUpToMultipleOf32( uint32_t value )
{
    return ( value + 31 ) & 0xFFFFFFE0;  // Round up to a multiple of 32
}

Ticket DemandTextureLoaderImpl::processRequests( hipStream_t stream, const DeviceContext& deviceContext )
{
    std::lock_guard lock( mutex_ );

    Ticket ticket = TicketImpl::create( stream );

    memset( deviceContext.counters, 0, stream );

    const uint32_t resourcesPerThread = std::max( 32U, roundUpToMultipleOf32( resourceTable_.count() / 65536U ) );
    const uint32_t blockSize          = 256;
    const uint32_t resourcesPerBlock  = resourcesPerThread * blockSize;
    const uint32_t gridSize           = ceilDiv( resourceTable_.count(), resourcesPerBlock );

    launchNum_++;
    void* arguments[] = { &deviceContext_, &launchNum_, &lruThreshold_ };
    clearEvictedPages( stream );
    memcpyDtoH( eviction_.candidates, deviceContext.evictionCandidates, stream );
    HIP_CHECK( hipModuleLaunchKernel( kernels_.collectRequests, gridSize, 1, 1, blockSize, 1, 1, 0, stream, arguments, nullptr ) );
    memcpyDtoH( requestedResources_, deviceContext.requestedResources, stream );
    memcpyDtoH( counters_, deviceContext.counters, stream );
    ProcessRequestCallback::enqueue( stream, new ProcessRequestCallback( *this, ticket ) );
    return ticket;
}

void DemandTextureLoaderImpl::processRequestsCallback( Ticket ticket )
{
    uint32_t requestCount  = counters_.ptr[static_cast<uint32_t>( CounterIndex::RequestedResources )];
    uint32_t evictionCount = counters_.ptr[static_cast<uint32_t>( CounterIndex::EvictionCandidates )];

    // clamp here in order to avoid atomicMin on gpu
    requestCount  = std::min( static_cast<uint32_t>( requestedResources_.len ), requestCount );
    evictionCount = std::min( static_cast<uint32_t>( eviction_.candidates.len ), evictionCount );

    if( options_.enableEviction )
    {
        std::lock_guard lock( eviction_.mutex );

        // here we are sure that residence bits are cleared on device for candidates from previous run
        // so we can safely start unmapping them
        if( eviction_.pendingUnmapCount > 0 )
        {
            std::lock_guard<std::mutex> lock( bits_.mutex );
            auto candidates = eviction_.candidates.subspan( 0, eviction_.pendingUnmapCount );
            auto callback   = [this]( uint32_t pageId ) {
                const uint32_t resourceId = resourceTable_.textureTiles.getResourceId( pageId );
                bits_.residence.set( resourceId, false );
            };
            pageSystem_.enqueueEvictedPages( candidates, requestCount, callback );
        }

        if( eviction_.pendingUnmapCount < options_.maxEvictedPages / 2 )
            lruThreshold_ -= std::min( lruThreshold_ - lruThresholdMin, 4u );
        else if( eviction_.pendingUnmapCount < options_.maxEvictedPages )
            lruThreshold_ -= std::min( lruThreshold_ - lruThresholdMin, 2u );
        else if( eviction_.pendingUnmapCount > 0 )  // only if eviction_.candidates is full
        {
            auto pred = []( EvictionCandidate a, EvictionCandidate b ) { return a.lru < b.lru; };
            std::sort( eviction_.candidates.ptr, eviction_.candidates.ptr + eviction_.pendingUnmapCount, pred );

            const uint32_t medianLru = eviction_.candidates.ptr[eviction_.pendingUnmapCount / 2].lru;
            if( medianLru > lruThreshold_ )
            {
                lruThreshold_++;
            }
        }

        eviction_.pendingUnmapCount = 0;
        eviction_.clearCount        = evictionCount;
    }

    requestProcessor_.submit( requestedResources_.ptr, requestCount, std::move( ticket ) );
}

void DemandTextureLoaderImpl::clearEvictedPages( hipStream_t stream )
{
    if( !options_.enableEviction )
        return;

    std::lock_guard lock( eviction_.mutex );
    if( eviction_.clearCount == 0 )
        return;

    assert( eviction_.pendingUnmapCount == 0 );

    const uint32_t resourcesPerThread = 2;
    const uint32_t blockSize          = 256;
    const uint32_t resourcesPerBlock  = resourcesPerThread * blockSize;
    const uint32_t gridSize           = ceilDiv( eviction_.clearCount, resourcesPerBlock );
    DeviceContext  deviceContext      = deviceContext_;
    void*          args[]             = { &deviceContext, &eviction_.clearCount };
    HIP_CHECK( hipModuleLaunchKernel( kernels_.updateEvictedPages, gridSize, 1, 1, blockSize, 1, 1, 0, stream, args, nullptr ) );

    eviction_.pendingUnmapCount = eviction_.clearCount;
    eviction_.clearCount        = 0;
}

void DemandTextureLoaderImpl::processRequest( hipStream_t stream, uint32_t resourceId )
{
    {
        std::lock_guard lock( bits_.mutex );
        if( bits_.residence.test( resourceId ) || bits_.loading.test( resourceId ) )
            return;

        bits_.loading.set( resourceId, true );
    }

    bool success = false;
    try
    {
        ProcessedResource processed{};
        processed.resourceId = resourceId;

        const Resource resource = decode( resourceId );
        switch( resource.type )
        {
            case Resource::Type::TextureId: {
                processed.textureInfo.ptr = processTextureInfo( stream, resource.textureId );
                break;
            }
            case Resource::Type::Tile: {
                processTile( stream, resource.tile );
                break;
            }
            case Resource::Type::MipTail: {
                processMipTail( stream, resource.mipTail );
                break;
            }
            default: {
                throw std::logic_error( "Unhandled resource type: " + std::to_string( static_cast<uint32_t>( resource.type ) ) );
            }
        }

        {
            std::lock_guard lock( processed_.mutex );
            assert( processed_.count < options_.maxRequests );
            processed_.resources.ptr[processed_.count++] = processed;
        }
        success = true;
    }
    catch( const std::exception& e )
    {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    {
        std::lock_guard lock( bits_.mutex );
        bits_.residence.set( resourceId, success );
        bits_.loading.set( resourceId, false );
    }
}

DevicePtr<DeviceTextureInfo> DemandTextureLoaderImpl::processTextureInfo( hipStream_t stream, const uint32_t textureId )
{
    DemandTextureImpl* texture = nullptr;
    {
        std::lock_guard lock( mutex_ );
        texture = textures_.at( textureId ).get();
    }
    TextureInfo imageInfo{};
    texture->image->open( &imageInfo );
    imageInfo.numMipLevels = std::min( imageInfo.numMipLevels, MAX_TEXTURE_MIP_LEVELS );
    assert( imageInfo.isValid );

    const uint32_t bytesPerChannel = getBytesPerChannel( imageInfo.format );
    // TODO_BS: handle images with width = 0, height = 0, mips = 0, unsupported format
    if( bytesPerChannel == 0 )
        throw std::invalid_argument( "Unsupported hipArray_Format" );
    if( imageInfo.width == 0 )
        throw std::runtime_error( "Unsupported width: 0" );
    if( imageInfo.height == 0 )
        throw std::runtime_error( "Unsupported height: 0" );
    if( imageInfo.numMipLevels == 0 )
        throw std::runtime_error( "Unsupported mip count: " + std::to_string( imageInfo.numMipLevels ) );
    if( imageInfo.numChannels == 0 )
        throw std::runtime_error( "Unsupported channel count: 0" );

    const uint32_t bytesPerTexel = bytesPerChannel * imageInfo.numChannels;
    const uint2    tileShape     = tileShapeForGranularity( pageSystem_.granularity(), bytesPerTexel );

    std::array<size_t, MAX_TEXTURE_MIP_LEVELS> mipTailOffsets{};

    uint32_t mipTailFirstLevel = imageInfo.numMipLevels;
    size_t   tailBytes         = 0;
    for( uint32_t mipLevel = imageInfo.numMipLevels; mipLevel-- > 0; )
    {
        const uint2  dimensions = mipDimensions( make_uint2( imageInfo.width, imageInfo.height ), mipLevel );
        const size_t levelBytes = static_cast<size_t>( bytesPerTexel ) * dimensions.x * dimensions.y;

        if( levelBytes > pageSystem_.pageBytes() - tailBytes )
            break;

        tailBytes += levelBytes;
        mipTailFirstLevel = mipLevel;
    }

    size_t mipTailOffset = 0;
    for( uint32_t mipLevel = mipTailFirstLevel; mipLevel < imageInfo.numMipLevels; ++mipLevel )
    {
        const uint2  dimensions = mipDimensions( make_uint2( imageInfo.width, imageInfo.height ), mipLevel );
        const size_t levelBytes = static_cast<size_t>( bytesPerTexel ) * dimensions.x * dimensions.y;

        mipTailOffsets[mipLevel] = mipTailOffset;
        mipTailOffset += levelBytes;
    }
    size_t mipTailSize = mipTailOffset;

    auto h_info              = uploadPool_.acquire<DeviceTextureInfo>( 1, stream );
    *h_info                  = {};
    h_info->textureId        = textureId;
    h_info->addressMode[0]   = texture->descriptor.addressMode[0];
    h_info->addressMode[1]   = texture->descriptor.addressMode[1];
    h_info->filterMode       = texture->descriptor.filterMode;
    h_info->mipmapFilterMode = texture->descriptor.mipmapFilterMode;
    h_info->normalizedCoords = texture->descriptor.normalizedCoords ? 1u : 0u;
    h_info->format           = imageInfo.format;
    h_info->numChannels      = imageInfo.numChannels;
    h_info->tileWidth        = tileShape.x;
    h_info->tileHeight       = tileShape.y;
    h_info->tileWidthShift   = powerOfTwoExponent( tileShape.x );
    h_info->tileHeightShift  = powerOfTwoExponent( tileShape.y );
    h_info->tileWidthMask    = tileShape.x - 1;
    h_info->tileHeightMask   = tileShape.y - 1;
    h_info->bytesPerTexel    = bytesPerTexel;
    h_info->width            = imageInfo.width;
    h_info->height           = imageInfo.height;

    h_info->mipCount          = imageInfo.numMipLevels;
    h_info->mipTailFirstLevel = mipTailFirstLevel;
    h_info->mipTailSize       = static_cast<uint32_t>( mipTailSize );

    {
        std::lock_guard lock( metadataMutex_ );

        h_info->startPage = pageTable_.textureTiles.nextAvailablePage;

        uint32_t pageCount = 0;
        for( uint32_t mip = 0; mip < h_info->mipTailFirstLevel; ++mip )
        {
            auto& level     = h_info->mips[mip];
            level.startPage = pageTable_.textureTiles.nextAvailablePage + pageCount;
            // we can call getMipLevel after we init pinnedDeviceInfo->mips[mip]
            pageCount += h_info->getMipLevel( mip ).pageCount();
        }

        if( h_info->mipTailFirstLevel < h_info->mipCount )
        {
            h_info->mipTailPage = pageTable_.textureTiles.nextAvailablePage + pageCount;
            pageCount++;

            for( uint32_t mip = h_info->mipTailFirstLevel; mip < h_info->mipCount; ++mip )
            {
                auto& level         = h_info->mips[mip];
                level.startPage     = h_info->mipTailPage;
                level.mipTailOffset = static_cast<uint32_t>( mipTailOffsets[mip] );
            }
        }

        // TODO_BS: how to handle it?
        const uint32_t end = pageTable_.textureTiles.startPage + pageTable_.textureTiles.pageCount;
        if( pageTable_.textureTiles.nextAvailablePage > end || pageCount > end - pageTable_.textureTiles.nextAvailablePage )
            throw std::runtime_error( "Maximum demand virtual page count exceeded" );

        texture->loadedTextureInfoId = static_cast<uint32_t>( loadedTextureInfos_.size() );
        loadedTextureInfos_.push_back( *h_info );

        pageTable_.textureTiles.nextAvailablePage += pageCount;

        DevicePtr<DeviceTextureInfo> d_info = textureInfoAllocator_.alloc();
        memcpyHtoD( DeviceSpan<DeviceTextureInfo>( d_info, 1 ), h_info.span(), stream );
        return d_info;
    }
}

void DemandTextureLoaderImpl::processTile( hipStream_t stream, const Resource::Tile& tile )
{
    std::shared_ptr<ImageSource> image;
    DeviceTextureInfo            info{};
    {
        std::scoped_lock         lock( mutex_, metadataMutex_ );
        const DemandTextureImpl& texture = *textures_.at( tile.textureId );
        image                            = texture.image;
        info                             = loadedTextureInfos_.at( texture.loadedTextureInfoId );
    }

    Tile t{};
    t.x      = tile.tileX;
    t.y      = tile.tileY;
    t.width  = info.tileWidth;
    t.height = info.tileHeight;
    assert( static_cast<size_t>( t.width ) * t.height * info.bytesPerTexel <= pageSystem_.pageBytes() );
    auto pageBuffer = uploadPool_.acquire<uint8_t>( pageSystem_.pageBytes(), stream );
    if( !image->readTile( reinterpret_cast<char*>( pageBuffer.span().ptr ), tile.mipLevel, t, stream ) )
    {
        throw std::runtime_error( "Failed to read texture tile for texture " + std::to_string( tile.textureId )
                                  + ", mip " + std::to_string( tile.mipLevel ) + ", tile ("
                                  + std::to_string( tile.tileX ) + ", " + std::to_string( tile.tileY ) + ")" );
    }

    pageSystem_.map( tile.pageId );
    memcpyHtoD( pageSystem_.page( tile.pageId ), pageBuffer.span(), stream );
}

void DemandTextureLoaderImpl::processMipTail( hipStream_t stream, const Resource::MipTail& mipTail )
{
    std::shared_ptr<ImageSource> image;
    DeviceTextureInfo            info{};
    {
        std::scoped_lock         lock( mutex_, metadataMutex_ );
        const DemandTextureImpl& texture = *textures_.at( mipTail.textureId );
        image                            = texture.image;
        info                             = loadedTextureInfos_.at( texture.loadedTextureInfoId );
    }

    assert( info.mipTailFirstLevel < info.mipCount );

    auto pageBuffer = uploadPool_.acquire<uint8_t>( pageSystem_.pageBytes(), stream );
    std::memset( pageBuffer.span().ptr, 0, pageBuffer.span().sizeInBytes() );
    for( uint32_t mipLevel = info.mipTailFirstLevel; mipLevel < info.mipCount; ++mipLevel )
    {
        const DeviceMipLevel mip = info.getMipLevel( mipLevel );
        char*                dst = reinterpret_cast<char*>( pageBuffer.span().ptr + mip.mipTailOffset );
        if( !image->readMipLevel( dst, mipLevel, mip.width, mip.height, stream ) )
        {
            throw std::runtime_error( "Failed to read mip tail level " + std::to_string( mipLevel ) + " for texture "
                                      + std::to_string( mipTail.textureId ) );
        }
    }

    pageSystem_.map( mipTail.pageId );
    memcpyHtoD( pageSystem_.page( mipTail.pageId ), pageBuffer.span(), stream );
}

Resource DemandTextureLoaderImpl::decode( uint32_t resourceId )
{
    std::lock_guard lock( metadataMutex_ );

    if( resourceTable_.textureInfos.contains( resourceId ) )
    {
        return Resource( resourceTable_.textureInfos.getLocalId( resourceId ) );
    }
    else
    {
        const uint32_t pageId = resourceTable_.textureTiles.getLocalId( resourceId );
        const auto     it =
            std::upper_bound( loadedTextureInfos_.cbegin(), loadedTextureInfos_.cend(), pageId,
                              []( uint32_t page, const DeviceTextureInfo& info ) { return page < info.startPage; } );

        if( it == loadedTextureInfos_.begin() )
            throw std::out_of_range( "Cannot decode resourceId " + std::to_string( resourceId )
                                     + ": it does not belong to any registered texture" );

        const auto infoIt = std::prev( it );

        const DeviceTextureInfo& info = *infoIt;
        if( pageId == info.mipTailPage )
            return Resource( Resource::MipTail{ info.textureId, pageId } );

        for( uint32_t mipLevel = 0; mipLevel < info.mipTailFirstLevel; ++mipLevel )
        {
            const DeviceMipLevel level = info.getMipLevel( mipLevel );
            if( level.startPage <= pageId && pageId < level.startPage + level.pageCount() )
            {
                const uint32_t pageInLevel = pageId - level.startPage;
                const uint32_t tileX       = pageInLevel % level.tilesX;
                const uint32_t tileY       = pageInLevel / level.tilesX;

                return Resource( Resource::Tile{ info.textureId, mipLevel, tileX, tileY, pageId } );
            }
        }

        // we should never be here!
        throw std::out_of_range( "Cannot decode resourceId " + std::to_string( resourceId )
                                 + ": it is outside the mip ranges of texture " + std::to_string( info.textureId ) );
    }
}

void DemandTextureLoaderImpl::updateProccedResources( hipStream_t stream )
{
    std::lock_guard lock( processed_.mutex );
    if( processed_.count > 0 )
    {
        const uint32_t resourcesPerThread = 2;
        const uint32_t blockSize          = 256;
        const uint32_t resourcesPerBlock  = resourcesPerThread * blockSize;
        const uint32_t gridSize           = ceilDiv( processed_.count, resourcesPerBlock );
        void*          args[]             = { &deviceContext_, &processed_.count };

        memcpyHtoD( deviceContext_.processedResources, processed_.resources, processed_.count, stream );
        HIP_CHECK( hipModuleLaunchKernel( kernels_.updateProcessedResources, gridSize, 1, 1, blockSize, 1, 1, 0, stream,
                                          args, nullptr ) );
        processed_.count = 0;
    }
}

std::unique_ptr<DemandTextureLoader> createDemandTextureLoader( const Options& options )
{
    return std::make_unique<DemandTextureLoaderImpl>( options );
}


}  // namespace hip_demand::vmm
