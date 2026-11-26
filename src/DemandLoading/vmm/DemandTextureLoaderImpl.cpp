#include "DemandTextureLoaderImpl.h"
#include "WhiteBlackTile.h"
#include <ImageSource/TextureInfo.h>

#include <algorithm>
#include <array>
#include <filesystem>
#include <set>

#if defined( _WIN32 )
#define NOMINMAX
#include <Windows.h>
#else
#include <dlfcn.h>
#endif

namespace hip_demand::vmm {

namespace {

std::filesystem::path getLibraryDirectory()
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
    void callback() override { loader_.processRequestsCallback( std::move( ticket_ ) ); }

  private:
    DemandTextureLoaderImpl& loader_;
    Ticket                   ticket_;
};

constexpr uint32_t maxTexturesLimit = 1024 * 1024;
constexpr uint32_t maxMipLevels = 32;

ResourceTable createResourceTable( const Options& options )
{
    if( options.maxTextures == 0 || options.maxTextures > maxTexturesLimit )
        throw std::invalid_argument( "maxTextures must be between 1 and " + std::to_string( maxTexturesLimit ) );

    ResourceTable table{};
    table.textureTiles = ResourceTable::Range( 0, options.maxVirtualPages );
    table.textureInfos = ResourceTable::Range( table.textureTiles.end(), options.maxTextures );
    const uint32_t cascadeCount = options.enableCascades ? options.maxTextures * MAX_TILED_MIP_LEVELS : 0;
    table.cascades              = ResourceTable::Range( table.textureInfos.end(), cascadeCount );

    uint32_t resourceCount = 0;
    if( !safeAdd( table.textureTiles.count, table.textureInfos.count, resourceCount )
        || !safeAdd( resourceCount, table.cascades.count, resourceCount ) )
        throw std::overflow_error(
            "Cannot create demand texture loader: total resource count exceeds the uint32_t limit" );

    return table;
}

PageTable createPageTable( const Options& options )
{
    PageTable pageTable{};
    pageTable.textureTiles = PageTable::Range( 0, options.maxVirtualPages );
    return pageTable;
}

uint32_t roundUpToMultipleOf32( uint32_t value )
{
    return ( value + 31 ) & 0xFFFFFFE0;
}

void validateImage( const TextureInfo& image )
{
    if( !image.isValid )
        throw std::runtime_error( "Invalid image" );
    if( getBytesPerChannel( image.format ) == 0 )
        throw std::invalid_argument( "Unsupported hipArray_Format" );
    if( image.width == 0 )
        throw std::runtime_error( "Unsupported width: 0" );
    if( image.height == 0 )
        throw std::runtime_error( "Unsupported height: 0" );
    if( image.numMipLevels == 0 )
        throw std::runtime_error( "Unsupported mip count: 0" );
    if( image.numChannels == 0 || image.numChannels > 4 )
        throw std::runtime_error( "Unsupported channel count: " + std::to_string( image.numChannels ) );
}

uint32_t computeMipTailFirstLevel( const TextureInfo& image, uint32_t bytesPerTexel, size_t pageBytes )
{
    uint32_t firstLevel = image.numMipLevels;
    size_t   tailBytes  = 0;
    for( uint32_t level = image.numMipLevels; level-- > 0; )
    {
        const uint2  dimensions = mipDimensions( make_uint2( image.width, image.height ), level );
        const size_t bytes      = static_cast<size_t>( bytesPerTexel ) * dimensions.x * dimensions.y;
        if( bytes > pageBytes - tailBytes )
            break;

        tailBytes += bytes;
        firstLevel = level;
    }
    return firstLevel;
}

}  // namespace

DemandTextureLoaderImpl::DemandTextureLoaderImpl( const Options& options )
    : options_( options )
    , pageSystem_( options_.maxVirtualPages, options_.maxPhysicalPages )
    , resourceTable_( createResourceTable( options_ ) )
    , pageTable_( createPageTable( options_ ) )
    , requestProcessor_( this, options_.maxThreads, options_.maxRequests )
    , uploadPool_( requestProcessor_.threadCount() * 3 )
    , bits_( resourceTable_.count() )
{
}

void DemandTextureLoaderImpl::init()
{
    assert( !kernels_.module );

    if( options_.maxRequests == 0 )
        throw std::invalid_argument( "maxRequests cannot be 0" );

    const std::filesystem::path kernelsPath = getLibraryDirectory() / "hip_demand_texture_kernels.co";
    HIP_CHECK( hipModuleLoad( &kernels_.module, kernelsPath.string().c_str() ) );
    HIP_CHECK( hipModuleGetFunction( &kernels_.collectRequests, kernels_.module, "collectRequestsAndEvictionCandidates" ) );
    HIP_CHECK( hipModuleGetFunction( &kernels_.updateProcessedResources, kernels_.module, "updateProcessedResources" ) );
    HIP_CHECK( hipModuleGetFunction( &kernels_.updateEvictedPages, kernels_.module, "updateEvictedPages" ) );

    options_.enableEviction = options_.enableEviction && options_.maxEvictedPages > 0;

    hipStream_t stream                = nullptr;
    deviceContext_.resourceTable      = resourceTable_;
    deviceContext_.pageSize           = pageSystem_.pageBytes();
    deviceContext_.pageMemory         = pageSystem_.virtualAddressSpace();
    deviceContext_.residenceBits      = deviceAllocArray<uint32_t>( bits_.wordCount(), true, stream );
    deviceContext_.textureInfos       = deviceAllocArray<DeviceTextureInfo>( options_.maxTextures, true, stream );
    deviceContext_.referenceBits      = deviceAllocArray<uint32_t>( bits_.wordCount(), true, stream );
    deviceContext_.requestedResources = deviceAllocArray<uint32_t>( options_.maxRequests, false, stream );
    deviceContext_.counters = deviceAllocArray<uint32_t>( static_cast<size_t>( CounterIndex::NumCounters ), true, stream );
    deviceContext_.processedResources = deviceAllocArray<uint32_t>( options_.maxRequests, false, stream );
    if( options_.enableEviction )
    {
        deviceContext_.requestIfResident  = true;
        deviceContext_.evictionCandidates = deviceAllocArray<EvictionCandidate>( options_.maxEvictedPages, false, stream );
        // 4 bits per resource, 8 values per uint32_t
        deviceContext_.lru = deviceAllocArray<uint32_t>( ceilDiv( resourceTable_.textureTiles.count, 8u ), true, stream );
    }
    HIP_CHECK( hipStreamSynchronize( stream ) );

    // Host transfer buffers
    requestedResources_  = hostAllocArray<uint32_t>( deviceContext_.requestedResources.len );
    counters_            = hostAllocArray<uint32_t>( deviceContext_.counters.len, true );
    eviction_.candidates = hostAllocArray<EvictionCandidate>( deviceContext_.evictionCandidates.len );
    processed_.resources = hostAllocArray<uint32_t>( options_.maxRequests );
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

    deviceFreeArray( deviceContext_.residenceBits );
    deviceFreeArray( deviceContext_.lru );
    deviceFreeArray( deviceContext_.textureInfos );
    deviceFreeArray( deviceContext_.referenceBits );
    deviceFreeArray( deviceContext_.requestedResources );
    deviceFreeArray( deviceContext_.evictionCandidates );
    deviceFreeArray( deviceContext_.processedResources );
    deviceFreeArray( deviceContext_.counters );

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

const DemandTexture& DemandTextureLoaderImpl::createUdimTexture( const std::vector<std::shared_ptr<ImageSource>>& imageSources,
                                                                 const std::vector<TextureDescriptor>& textureDescs,
                                                                 uint32_t                              udim,
                                                                 uint32_t                              vdim,
                                                                 uint32_t                              numChannels )
{
    std::scoped_lock lock( mutex_, metadataMutex_ );

    constexpr uint32_t maxGridDimension = 0xFFFF;  // DeviceTextureInfo::udimGrid packs udim and vdim into 16 bits each.
    constexpr uint32_t maxChannels      = 0xFF;    // DeviceTextureInfo packs the channel count and index into 8 bits each.
    if( udim == 0 || vdim == 0 )
        throw std::invalid_argument( "udim and vdim must both be positive" );
    if( udim > maxGridDimension || vdim > maxGridDimension )
        throw std::invalid_argument( "udim and vdim must not exceed " + std::to_string( maxGridDimension ) );
    if( numChannels == 0 || numChannels > maxChannels )
        throw std::invalid_argument( "numChannels must be between 1 and " + std::to_string( maxChannels ) );

    const size_t textureCount = static_cast<size_t>( udim ) * vdim * numChannels;
    if( imageSources.size() != textureCount )
        throw std::invalid_argument( "Expected one image source per UDIM grid cell and channel: "
                                     + std::to_string( textureCount ) + ", got " + std::to_string( imageSources.size() ) );
    if( textureDescs.size() != textureCount )
        throw std::invalid_argument( "Expected one texture descriptor per UDIM grid cell and channel: "
                                     + std::to_string( textureCount ) + ", got " + std::to_string( textureDescs.size() ) );
    if( textureCount > options_.maxTextures - textures_.size() )
        throw std::runtime_error( "Maximum demand texture count exceeded" );

    // tex2DGradUdim() maps grid coordinates onto subtextures as fractions of their extent.
    for( size_t i = 0; i < textureCount; ++i )
    {
        if( imageSources[i] && !textureDescs[i].normalizedCoords )
            throw std::invalid_argument( "UDIM subtextures must use normalized coordinates" );
    }

    const uint32_t                    startTextureId = static_cast<uint32_t>( textures_.size() );
    const DemandTextureImpl::UdimGrid grid{ startTextureId, udim, vdim, numChannels };
    for( size_t i = 0; i < textureCount; ++i )
    {
        const uint32_t textureId = startTextureId + static_cast<uint32_t>( i );
        textures_.emplace_back( std::make_unique<DemandTextureImpl>( textureId, imageSources[i], textureDescs[i] ) );
        textures_.back()->udim = grid;
    }

    return *textures_[startTextureId];
}

void DemandTextureLoaderImpl::setUdimFields( DeviceTextureInfo& info, const DemandTextureImpl::UdimGrid& udim ) const
{
    if( udim.isUdim() )
        info.setUdimGrid( udim.startTexture, udim.udim, udim.vdim, udim.channels );
}

void DemandTextureLoaderImpl::launchPrepare( hipStream_t stream, DeviceContext& deviceContext )
{
    std::lock_guard lock( mutex_ );

    deviceContext              = deviceContext_;
    deviceContext.textureInfos = deviceContext.textureInfos.subspan( 0, textures_.size() );
    clearEvictedPages( stream );
    memset( deviceContext.referenceBits, 0, stream );
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

    clearEvictedPages( stream );

    uint32_t lruThreshold = 0;
    {
        std::lock_guard evictionLock( eviction_.mutex );
        lruThreshold = eviction_.lruThreshold;
    }

    launchNum_++;
    void* arguments[] = { &deviceContext_, &launchNum_, &lruThreshold };
    memcpyDtoH( eviction_.candidates, deviceContext.evictionCandidates, stream );
    HIP_CHECK( hipModuleLaunchKernel( kernels_.collectRequests, gridSize, 1, 1, blockSize, 1, 1, 0, stream, arguments, nullptr ) );
    memcpyDtoH( requestedResources_, deviceContext.requestedResources, stream );
    memcpyDtoH( counters_, deviceContext.counters, stream );
    ProcessRequestCallback::enqueue( stream, new ProcessRequestCallback( *this, ticket ) );
    return ticket;
}

void DemandTextureLoaderImpl::processRequestsCallback( Ticket ticket )
{
    // The device counters may exceed the list sizes; clamping here avoids an atomicMin on the device.
    const uint32_t requestCount  = std::min( static_cast<uint32_t>( requestedResources_.len ),
                                             counters_.ptr[static_cast<uint32_t>( CounterIndex::RequestedResources )] );
    const uint32_t evictionCount = std::min( static_cast<uint32_t>( eviction_.candidates.len ),
                                             counters_.ptr[static_cast<uint32_t>( CounterIndex::EvictionCandidates )] );

    if( options_.enableEviction )
    {
        std::lock_guard lock( eviction_.mutex );

        // Residence bits of the pending candidates were cleared on the device before this pass, so no kernel launched
        // after it reads their pages and they can be unmapped.
        if( eviction_.pendingUnmapCount > 0 )
            releaseEvictedPages( requestCount );

        updateLruThreshold();
        eviction_.pendingUnmapCount = 0;
        eviction_.clearCount        = evictionCount;
    }

    requestProcessor_.submit( requestedResources_.ptr, requestCount, std::move( ticket ) );
}

void DemandTextureLoaderImpl::addTransferredBytes( size_t bytes )
{
    std::lock_guard lock( stats_.mutex );
    stats_.bytesTransferredToDevice += bytes;
}

void DemandTextureLoaderImpl::releaseEvictedPages( uint32_t reusablePageCount )
{
    const HostSpan<EvictionCandidate> candidates = eviction_.candidates.subspan( 0, eviction_.pendingUnmapCount );

    std::lock_guard lock( bits_.mutex );
    pageSystem_.enqueueEvictedPages( candidates, reusablePageCount );
    for( size_t i = 0; i < candidates.len; ++i )
        bits_.residence.set( resourceTable_.textureTiles.getResourceId( candidates.ptr[i].pageId ), false );
}

void DemandTextureLoaderImpl::updateLruThreshold()
{
    uint32_t&      threshold = eviction_.lruThreshold;
    const uint32_t count     = eviction_.pendingUnmapCount;
    if( count < options_.maxEvictedPages / 2 )
        threshold -= std::min( threshold - lruThresholdMin, 4u );
    else if( count < options_.maxEvictedPages )
        threshold -= std::min( threshold - lruThresholdMin, 2u );
    else  // The candidate list is full: raise the threshold if most candidates are older.
    {
        const auto byLru = []( EvictionCandidate a, EvictionCandidate b ) { return a.lru < b.lru; };
        std::sort( eviction_.candidates.ptr, eviction_.candidates.ptr + count, byLru );

        if( eviction_.candidates.ptr[count / 2].lru > threshold )
            ++threshold;
    }
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
        const Resource resource = decodeResource( resourceId );
        switch( resource.type )
        {
            case Resource::Type::TextureId:
                processTextureInfo( stream, resource.textureId );
                break;
            case Resource::Type::Tile:
                processTile( stream, resource.tile );
                break;
            case Resource::Type::MipTail:
                processMipTail( stream, resource.mipTail );
                break;
            case Resource::Type::Cascade:
                processCascade( stream, resource.cascade );
                break;
            default:
                throw std::logic_error( "Unhandled resource type: " + std::to_string( static_cast<uint32_t>( resource.type ) ) );
        }

        {
            std::lock_guard lock( processed_.mutex );
            assert( processed_.count < processed_.resources.len );
            processed_.resources.ptr[processed_.count++] = resourceId;
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

DemandTextureImpl& DemandTextureLoaderImpl::getTexture( uint32_t textureId )
{
    std::lock_guard lock( mutex_ );
    return *textures_.at( textureId );
}

DeviceTextureInfo DemandTextureLoaderImpl::textureInfo( const DemandTextureImpl& texture ) const
{
    std::lock_guard lock( metadataMutex_ );
    return texture.info;
}

void DemandTextureLoaderImpl::processTextureInfo( hipStream_t stream, uint32_t textureId )
{
    DemandTextureImpl& texture = getTexture( textureId );
    auto               upload  = uploadPool_.acquire<DeviceTextureInfo>( 1, stream );

    // An empty UDIM cell has no image and no pages: its descriptor only marks it loaded and empty, with the grid.
    if( texture.isEmpty() )
    {
        DeviceTextureInfo info{};
        info.setLoaded( true );
        info.setEmpty( true );
        setUdimFields( info, texture.udim );

        std::lock_guard lock( metadataMutex_ );
        texture.info = info;
        *upload      = info;
        memcpyHtoD( deviceContext_.textureInfos.subspan( textureId, 1 ), upload.span(), stream );
        return;
    }

    TextureInfo image{};
    texture.image->open( &image );
    validateImage( image );
    image.numMipLevels = std::min( image.numMipLevels, maxMipLevels );

    const uint32_t bytesPerTexel = getBytesPerChannel( image.format ) * image.numChannels;
    const uint2    tileShape     = tileShapeForPageBytes( pageSystem_.pageBytes(), bytesPerTexel );
    uint32_t       mipTailFirstLevel = computeMipTailFirstLevel( image, bytesPerTexel, pageSystem_.pageBytes() );

    // Only the tiled levels have a word in DeviceTextureInfo::mips; the tail holds any number of levels. A texture
    // with more tiled levels than words keeps the finest ones and loses its tail.
    if( mipTailFirstLevel > MAX_TILED_MIP_LEVELS )
        image.numMipLevels = mipTailFirstLevel = MAX_TILED_MIP_LEVELS;

    DeviceTextureInfo info{};
    info.setAddressMode( 0, texture.descriptor.addressMode[0] );
    info.setAddressMode( 1, texture.descriptor.addressMode[1] );
    info.setFilterMode( texture.descriptor.filterMode );
    info.setMipmapFilterMode( texture.descriptor.mipmapFilterMode );
    info.setNormalizedCoords( texture.descriptor.normalizedCoords );
    info.setMaxAnisotropy( std::clamp( texture.descriptor.maxAnisotropy, 1u, 16u ) );
    info.setTexel( image.format, image.numChannels, bytesPerTexel );
    info.setLayout( powerOfTwoExponent( tileShape.x ), powerOfTwoExponent( tileShape.y ), image.numMipLevels, mipTailFirstLevel );
    info.width  = image.width;
    info.height = image.height;
    info.setLoaded( true );

    // The grid fields are set at creation, before any request for the texture can arrive.
    setUdimFields( info, texture.udim );

    std::lock_guard lock( metadataMutex_ );
    texture.info = info;

    const uint32_t firstLevel = options_.enableCascades ? std::min( mipTailFirstLevel, info.mipCount() - 1 ) : 0;
    assignLevelPages( texture, firstLevel, info.mipCount() );

    *upload = texture.info;
    memcpyHtoD( deviceContext_.textureInfos.subspan( textureId, 1 ), upload.span(), stream );
}

void DemandTextureLoaderImpl::assignLevelPages( DemandTextureImpl& texture, uint32_t firstLevel, uint32_t endLevel )
{
    DeviceTextureInfo& info = texture.info;
    assert( firstLevel < endLevel && endLevel <= info.mipCount() );

    const uint32_t tailFirstLevel = info.mipTailFirstLevel();
    const uint32_t tiledEndLevel  = std::min( endLevel, tailFirstLevel );
    const bool     withTail       = endLevel > tailFirstLevel;

    uint32_t pageCount = withTail ? 1 : 0;
    for( uint32_t level = firstLevel; level < tiledEndLevel; ++level )
        pageCount += info.getMipLevel( level ).pageCount();

    // TODO_BS: how to handle it?
    PageTable::Range& pages = pageTable_.textureTiles;
    if( pageCount > pages.startPage + pages.pageCount - pages.nextAvailablePage )
        throw std::runtime_error( "Maximum demand virtual page count exceeded" );

    uint32_t page = pages.nextAvailablePage;
    for( uint32_t level = firstLevel; level < tiledEndLevel; ++level )
    {
        info.mips[level] = page;
        pageRanges_.push_back( { page, level, &texture } );
        page += info.getMipLevel( level ).pageCount();
    }
    if( withTail )
    {
        info.mipTailPage = page;
        pageRanges_.push_back( { page, tailFirstLevel, &texture } );
        page++;
    }
    pages.nextAvailablePage = page;
    info.cascadeLevel       = firstLevel;

    // The sampler addresses texels with 32-bit offsets from a level's first page unless a level is larger than that.
    // The finest level has the most pages.
    if( firstLevel < tiledEndLevel
        && static_cast<uint64_t>( info.getMipLevel( firstLevel ).pageCount() ) * pageSystem_.pageBytes() > ( uint64_t( 1 ) << 32 ) )
        info.setWideOffsets( true );
}

void DemandTextureLoaderImpl::processCascade( hipStream_t stream, const Resource::Cascade& cascade )
{
    DemandTextureImpl& texture = getTexture( cascade.textureId );
    assert( texture.info.isLoaded() );

    auto upload = uploadPool_.acquire<DeviceTextureInfo>( 1, stream );

    std::lock_guard lock( metadataMutex_ );

    // The requests of one pass are processed in any order: a coarser one may find the texture extended already.
    if( cascade.level >= texture.info.cascadeLevel )
        return;

    assignLevelPages( texture, cascade.level, texture.info.cascadeLevel );
    *upload = texture.info;
    memcpyHtoD( deviceContext_.textureInfos.subspan( cascade.textureId, 1 ), upload.span(), stream );
}

void DemandTextureLoaderImpl::processTile( hipStream_t stream, const Resource::Tile& t )
{
    const DemandTextureImpl& texture = getTexture( t.textureId );
    const DeviceTextureInfo  info    = textureInfo( texture );

    Tile tile{};
    tile.x      = t.tileX;
    tile.y      = t.tileY;
    tile.width  = info.tileWidth();
    tile.height = info.tileHeight();
    // A tile of a texel that is not a power of two in size (rgb8, rgb16, rgb32f) leaves the end of its page unused.
    const size_t tileBytes = static_cast<size_t>( tile.width ) * tile.height * info.bytesPerTexel();
    assert( tileBytes <= pageSystem_.pageBytes() );

    auto                   pageBuffer = uploadPool_.acquire<uint8_t>( pageSystem_.pageBytes(), stream );
    const HostSpan<uint8_t> tileSpan   = pageBuffer.span().subspan( 0, tileBytes );
    if( !texture.image->readTile( reinterpret_cast<char*>( tileSpan.ptr ), t.mipLevel, tile, stream ) )
    {
        throw std::runtime_error( "Failed to read texture tile for texture " + std::to_string( t.textureId )
                                  + ", mip " + std::to_string( t.mipLevel ) + ", tile ("
                                  + std::to_string( t.tileX ) + ", " + std::to_string( t.tileY ) + ")" );
    }

    // A tile on the right or bottom edge of its level is cut by it.
    const uint32_t visibleTileWidth  = std::min( tile.width, info.levelWidth( t.mipLevel ) - t.tileX * tile.width );
    const uint32_t visibleTileHeight = std::min( tile.height, info.levelHeight( t.mipLevel ) - t.tileY * tile.height );
    const bool     insideLevel   = visibleTileWidth == tile.width && visibleTileHeight == tile.height;

    std::string key;
    if( options_.coalesceWhiteBlackTiles
        && whiteBlackTileKey( tileSpan.ptr, tile.width, visibleTileWidth, visibleTileHeight, info.format(), info.numChannels(), key ) )
    {
        std::lock_guard lock( whiteBlack_.mutex );
        const auto      shared = whiteBlack_.pages.find( key );
        if( shared != whiteBlack_.pages.end() )
        {
            pageSystem_.mapAlias( t.pageId, shared->second );
            return;
        }

        // Only a tile inside its level starts a shared page: the page is read whole by the tiles that join it later.
        // An edge tile that finds none gets a page of its own.
        if( insideLevel )
        {
            const uint32_t physicalPageId = pageSystem_.map( t.pageId );
            pageSystem_.retain( physicalPageId );
            memcpyHtoD( pageSystem_.page( t.pageId ).subspan( 0, tileBytes ), tileSpan, stream );
            addTransferredBytes( tileBytes );
            whiteBlack_.pages.emplace( std::move( key ), physicalPageId );
            return;
        }
    }

    pageSystem_.map( t.pageId );
    memcpyHtoD( pageSystem_.page( t.pageId ).subspan( 0, tileBytes ), tileSpan, stream );
    addTransferredBytes( tileBytes );
}

void DemandTextureLoaderImpl::processMipTail( hipStream_t stream, const Resource::MipTail& mipTail )
{
    const DemandTextureImpl& texture = getTexture( mipTail.textureId );
    const DeviceTextureInfo  info    = textureInfo( texture );
    assert( info.mipTailFirstLevel() < info.mipCount() );

    assert( info.mipTailLevelOffset( info.mipCount() ) <= pageSystem_.pageBytes() );
    assert( info.mipCount() <= maxMipLevels );
    std::array<uint2, maxMipLevels> levelDims{};
    for( uint32_t level = info.mipTailFirstLevel(); level < info.mipCount(); ++level )
        levelDims[level] = make_uint2( info.levelWidth( level ), info.levelHeight( level ) );

    auto pageBuffer = uploadPool_.acquire<uint8_t>( pageSystem_.pageBytes(), stream );
    std::memset( pageBuffer.span().ptr, 0, pageBuffer.span().sizeInBytes() );
    if( !texture.image->readMipTail( reinterpret_cast<char*>( pageBuffer.span().ptr ), info.mipTailFirstLevel(),
                                     info.mipCount(), levelDims.data(), stream ) )
    {
        throw std::runtime_error( "Failed to read mip tail for texture " + std::to_string( mipTail.textureId ) );
    }

    pageSystem_.map( mipTail.pageId );
    memcpyHtoD( pageSystem_.page( mipTail.pageId ), pageBuffer.span(), stream );
    addTransferredBytes( pageBuffer.span().sizeInBytes() );
}

Statistics DemandTextureLoaderImpl::getStatistics() const
{
    std::lock_guard lock( mutex_ );

    Statistics stats{};
    {
        std::lock_guard statsLock( stats_.mutex );
        stats.bytesTransferredToDevice = stats_.bytesTransferredToDevice;
    }
    stats.numEvictions     = pageSystem_.unmappedPageCount();
    stats.deviceMemoryUsed = static_cast<size_t>( pageSystem_.physicalPageCount() ) * pageSystem_.pageBytes()
                             + deviceContext_.residenceBits.sizeInBytes() + deviceContext_.textureInfos.sizeInBytes()
                             + deviceContext_.referenceBits.sizeInBytes() + deviceContext_.requestedResources.sizeInBytes()
                             + deviceContext_.evictionCandidates.sizeInBytes() + deviceContext_.processedResources.sizeInBytes()
                             + deviceContext_.lru.sizeInBytes() + deviceContext_.counters.sizeInBytes();

    std::set<const ImageSource*> images;
    for( const auto& texture : textures_ )
    {
        if( texture->isEmpty() )
            continue;
        stats.numTextures++;

        const ImageSource* image = texture->image.get();
        if( !images.insert( image ).second )
            continue;
        stats.numTilesRead += image->getNumTilesRead();
        stats.numBytesRead += image->getNumBytesRead();
        stats.readTime += image->getTotalReadTime();
        stats.virtualTextureBytes += getTextureSizeInBytes( image->getInfo() );
    }
    return stats;
}

uint32_t DemandTextureLoaderImpl::whiteBlackTileCount() const
{
    std::lock_guard lock( whiteBlack_.mutex );
    return static_cast<uint32_t>( whiteBlack_.pages.size() );
}

Resource DemandTextureLoaderImpl::decodeResource( uint32_t resourceId ) const
{
    std::lock_guard lock( metadataMutex_ );

    if( resourceTable_.textureInfos.contains( resourceId ) )
        return Resource( resourceTable_.textureInfos.getLocalId( resourceId ) );

    if( resourceTable_.cascades.contains( resourceId ) )
    {
        const uint32_t cascadeId = resourceTable_.cascades.getLocalId( resourceId );
        return Resource( Resource::Cascade{ cascadeId / MAX_TILED_MIP_LEVELS, cascadeId % MAX_TILED_MIP_LEVELS } );
    }

    // Find the level owning the page: the last range starting at or before it.
    const uint32_t pageId = resourceTable_.textureTiles.getLocalId( resourceId );
    const auto     next   = std::upper_bound( pageRanges_.cbegin(), pageRanges_.cend(), pageId,
                                              []( uint32_t page, const PageRange& range ) { return page < range.startPage; } );
    if( next == pageRanges_.cbegin() )
        throw std::out_of_range( "Cannot decode resourceId " + std::to_string( resourceId )
                                 + ": it does not belong to any registered texture" );

    const PageRange&         range   = *std::prev( next );
    const DemandTextureImpl& texture = *range.texture;
    const DeviceTextureInfo& info    = texture.info;
    if( range.level >= info.mipTailFirstLevel() )
    {
        if( pageId == info.mipTailPage )
            return Resource( Resource::MipTail{ texture.id, pageId } );
    }
    else
    {
        const DeviceMipLevel mip         = info.getMipLevel( range.level );
        const uint32_t       pageInLevel = pageId - mip.startPage;
        if( pageInLevel < mip.pageCount() )
            return Resource( Resource::Tile{ texture.id, range.level, pageInLevel % mip.tilesX, pageInLevel / mip.tilesX, pageId } );
    }

    // we should never be here!
    throw std::out_of_range( "Cannot decode resourceId " + std::to_string( resourceId )
                             + ": it is outside the mip ranges of texture " + std::to_string( texture.id ) );
}

void DemandTextureLoaderImpl::updateProcessedResources( hipStream_t stream )
{
    std::lock_guard lock( processed_.mutex );
    if( processed_.count == 0 )
        return;

    const uint32_t resourcesPerThread = 2;
    const uint32_t blockSize          = 256;
    const uint32_t gridSize           = ceilDiv( processed_.count, resourcesPerThread * blockSize );
    void*          args[]             = { &deviceContext_, &processed_.count };

    memcpyHtoD( deviceContext_.processedResources, processed_.resources, processed_.count, stream );
    HIP_CHECK( hipModuleLaunchKernel( kernels_.updateProcessedResources, gridSize, 1, 1, blockSize, 1, 1, 0, stream, args, nullptr ) );
    processed_.count = 0;
}

std::unique_ptr<DemandTextureLoader> createDemandTextureLoader( const Options& options )
{
    auto loader = std::make_unique<DemandTextureLoaderImpl>( options );
    loader->init();
    return loader;
}

}  // namespace hip_demand::vmm
