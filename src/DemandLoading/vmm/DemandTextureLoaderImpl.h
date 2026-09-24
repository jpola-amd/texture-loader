#pragma once

#include "../Internal/HipCheck.h"
#include "../Internal/Utils.h"
#include "Allocator.h"
#include "Lru.h"
#include "PageSystem.h"
#include "PinnedUploadPool.h"
#include "RequestProcessor.h"
#include <DemandLoading/VmmDemandTextureLoader.h>
#include <cassert>
#include <mutex>
#include <utility>

namespace hip_demand::vmm {

struct Resource
{
    enum class Type
    {
        TextureId,
        Tile,
        MipTail
    };

    struct Tile
    {
        uint32_t textureId = 0;
        uint32_t mipLevel  = 0;
        uint32_t tileX     = 0;
        uint32_t tileY     = 0;
        uint32_t pageId    = 0;
    };

    struct MipTail
    {
        uint32_t textureId = 0;
        uint32_t pageId    = 0;
    };

    Type type;
    union
    {
        uint32_t textureId;
        Tile     tile;
        MipTail  mipTail;
    };

    explicit Resource( uint32_t id )
        : type( Type::TextureId )
        , textureId( id )
    {
    }

    explicit Resource( Tile value )
        : type( Type::Tile )
        , tile( value )
    {
    }

    explicit Resource( MipTail value )
        : type( Type::MipTail )
        , mipTail( value )
    {
    }
};

struct ResourceBits
{
    mutable std::mutex mutex;
    Bitset             residence;
    Bitset             loading;

    explicit ResourceBits( uint32_t resourceCount )
        : residence( resourceCount )
        , loading( resourceCount )
    {
    }

    uint32_t wordCount() const { return residence.wordCount(); }
    uint32_t bitCount() const { return residence.bitCount(); }
};

class DemandTextureImpl : public DemandTexture, NonCopyble
{
  public:
    DemandTextureImpl( uint32_t textureId, std::shared_ptr<ImageSource> imageSource, const TextureDescriptor& textureDescriptor )
        : id( textureId )
        , image( imageSource )
        , descriptor( textureDescriptor )
    {
    }
    uint32_t getId() const override { return id; }

    uint32_t                     id                  = INVALID_TEXTURE;
    uint32_t                     loadedTextureInfoId = INVALID_TEXTURE;
    std::shared_ptr<ImageSource> image{};
    TextureDescriptor            descriptor;
};

class DemandTextureLoaderImpl : public DemandTextureLoader, NonCopyble
{
  public:
    explicit DemandTextureLoaderImpl( const Options& options );
    ~DemandTextureLoaderImpl() override;

    const DemandTexture& createTexture( std::shared_ptr<ImageSource> imageSource, const TextureDescriptor& textureDesc ) override;
    void   launchPrepare( hipStream_t stream, DeviceContext& deviceContext ) override;
    Ticket processRequests( hipStream_t stream, const DeviceContext& deviceContext ) override;
    void   processRequestsCallback( Ticket ticket );
    void   processRequest( hipStream_t stream, uint32_t resourceId );
    void   updateProccedResources( hipStream_t stream );
    int    device() const { return pageSystem_.device(); }

  private:
    Resource                     decode( uint32_t resourceId );
    void                         clearEvictedPages( hipStream_t stream );
    DevicePtr<DeviceTextureInfo> processTextureInfo( hipStream_t stream, const uint32_t textureId );
    void                         processTile( hipStream_t stream, const Resource::Tile& tile );
    void                         processMipTail( hipStream_t stream, const Resource::MipTail& mipTail );

    mutable std::mutex mutex_;
    Options            options_{};
    PageSystem         pageSystem_;

    std::vector<std::unique_ptr<DemandTextureImpl>> textures_{};

    // metadata
    mutable std::mutex           metadataMutex_;
    Allocator<DeviceTextureInfo> textureInfoAllocator_;
    ResourceTable                resourceTable_{};
    PageTable                    pageTable_{};
    // note that this list should be accessed by texture.loadedTextureInfoId
    // and it's ordered by startPage in order to use std::upper_bound
    std::vector<DeviceTextureInfo> loadedTextureInfos_{};

    RequestProcessor requestProcessor_;
    PinnedUploadPool uploadPool_;
    struct
    {
        hipModule_t   module;
        hipFunction_t collectRequests;
        hipFunction_t updateProcessedResources;
        hipFunction_t updateEvictedPages;
    } kernels_{};
    uint32_t launchNum_    = 0;
    uint32_t lruThreshold_ = lruThresholdMin;

    DeviceContext      deviceContext_{};
    HostSpan<uint32_t> requestedResources_{};
    HostSpan<uint32_t> counters_{};

    ResourceBits bits_;

    struct
    {
        mutable std::mutex          mutex;
        HostSpan<EvictionCandidate> candidates;
        uint32_t                    clearCount        = 0;  // Device candidates awaiting residence-bit clearing.
        uint32_t                    pendingUnmapCount = 0;  // Cleared candidates awaiting the next host copy.

    } eviction_{};

    struct
    {
        mutable std::mutex          mutex;
        HostSpan<ProcessedResource> resources;
        uint32_t                    count;
    } processed_{};
};

}  // namespace hip_demand::vmm
