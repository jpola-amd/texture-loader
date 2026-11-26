#pragma once

#include "../Internal/HipCheck.h"
#include "../Internal/Utils.h"
#include "Lru.h"
#include "PageSystem.h"
#include "PinnedUploadPool.h"
#include "RequestProcessor.h"
#include <DemandLoading/VmmDemandTextureLoader.h>
#include <cassert>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace hip_demand::vmm {

struct Resource
{
    enum class Type
    {
        TextureId,
        Tile,
        MipTail,
        Cascade
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

    // A request for the texture to get pages down to the given mip level; see DeviceTextureInfo::cascadeLevel.
    struct Cascade
    {
        uint32_t textureId = 0;
        uint32_t level     = 0;
    };

    Type type;
    union
    {
        uint32_t textureId;
        Tile     tile;
        MipTail  mipTail;
        Cascade  cascade;
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

    explicit Resource( Cascade value )
        : type( Type::Cascade )
        , cascade( value )
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
    // The UDIM grid a texture is an entry point to; see DeviceTextureInfo::udimStartTexture.
    struct UdimGrid
    {
        uint32_t startTexture = INVALID_TEXTURE;
        uint32_t udim         = 0;
        uint32_t vdim         = 0;
        uint32_t channels     = 1;  // Textures per cell.

        bool isUdim() const { return startTexture != INVALID_TEXTURE; }
    };

    DemandTextureImpl( uint32_t textureId, std::shared_ptr<ImageSource> imageSource, const TextureDescriptor& textureDescriptor )
        : id( textureId )
        , image( std::move( imageSource ) )
        , descriptor( textureDescriptor )
    {
    }
    uint32_t getId() const override { return id; }
    bool     isEmpty() const { return image == nullptr; }

    uint32_t                     id = INVALID_TEXTURE;
    std::shared_ptr<ImageSource> image{};
    TextureDescriptor            descriptor{};
    DeviceTextureInfo            info{};
    UdimGrid                     udim{};
};

class DemandTextureLoaderImpl : public DemandTextureLoader, NonCopyble
{
  public:
    explicit DemandTextureLoaderImpl( const Options& options );
    ~DemandTextureLoaderImpl() override;

    void init();

    const DemandTexture& createTexture( std::shared_ptr<ImageSource> imageSource, const TextureDescriptor& textureDesc ) override;
    const DemandTexture& createUdimTexture( const std::vector<std::shared_ptr<ImageSource>>& imageSources,
                                            const std::vector<TextureDescriptor>&            textureDescs,
                                            uint32_t                                         udim,
                                            uint32_t                                         vdim,
                                            uint32_t                                         numChannels ) override;
    void   launchPrepare( hipStream_t stream, DeviceContext& deviceContext ) override;
    Ticket processRequests( hipStream_t stream, const DeviceContext& deviceContext ) override;
    Statistics getStatistics() const override;

    void processRequestsCallback( Ticket ticket );
    void processRequest( hipStream_t stream, uint32_t resourceId );
    void updateProcessedResources( hipStream_t stream );
    int  device() const { return pageSystem_.device(); }

    uint32_t physicalPageCount() const { return pageSystem_.physicalPageCount(); }
    uint32_t whiteBlackTileCount() const;

  private:
    struct PageRange
    {
        uint32_t           startPage = INVALID_PAGE;
        uint32_t           level     = 0;
        DemandTextureImpl* texture   = nullptr;
    };

    Resource           decodeResource( uint32_t resourceId ) const;
    DemandTextureImpl& getTexture( uint32_t textureId );
    DeviceTextureInfo  textureInfo( const DemandTextureImpl& texture ) const;
    void          setUdimFields( DeviceTextureInfo& info, const DemandTextureImpl::UdimGrid& udim ) const;
    void          clearEvictedPages( hipStream_t stream );
    void          releaseEvictedPages( uint32_t reusablePageCount );
    void          updateLruThreshold();
    void          processTextureInfo( hipStream_t stream, uint32_t textureId );
    void          processTile( hipStream_t stream, const Resource::Tile& tile );
    void          processMipTail( hipStream_t stream, const Resource::MipTail& mipTail );
    void          processCascade( hipStream_t stream, const Resource::Cascade& cascade );
    void          assignLevelPages( DemandTextureImpl& texture, uint32_t firstLevel, uint32_t endLevel );
    void          addTransferredBytes( size_t bytes );

    mutable std::mutex mutex_;
    Options            options_{};
    PageSystem         pageSystem_;

    std::vector<std::unique_ptr<DemandTextureImpl>> textures_{};

    // Guards the layout of textures over the virtual pages: pageTable_.textureTiles.nextAvailablePage (the page
    // allocator), pageRanges_, and DemandTextureImpl::info of every texture, which encodes that layout for the device.
    // resourceTable_ is fixed at construction and needs no lock.
    mutable std::mutex metadataMutex_;
    ResourceTable      resourceTable_{};
    PageTable          pageTable_{};
    // Levels get their pages from one growing range, when their texture's metadata is loaded or when the texture is
    // extended to finer levels (processCascade), so the list is ordered by startPage, which decodeResource() relies on.
    std::vector<PageRange> pageRanges_{};

    RequestProcessor requestProcessor_;
    PinnedUploadPool uploadPool_;
    struct
    {
        hipModule_t   module;
        hipFunction_t collectRequests;
        hipFunction_t updateProcessedResources;
        hipFunction_t updateEvictedPages;
    } kernels_{};
    uint32_t launchNum_ = 0;

    DeviceContext      deviceContext_{};
    HostSpan<uint32_t> requestedResources_{};
    HostSpan<uint32_t> counters_{};

    ResourceBits bits_;

    // Eviction is pipelined over passes: candidates collected by one pass are cleared on the device at the start of the
    // next one (clearEvictedPages) and unmapped on the host after that pass's requests are read back.
    struct
    {
        mutable std::mutex          mutex;
        HostSpan<EvictionCandidate> candidates;
        uint32_t                    clearCount        = 0;  // Device candidates awaiting residence-bit clearing.
        uint32_t                    pendingUnmapCount = 0;  // Cleared candidates awaiting the next host copy.
        uint32_t                    lruThreshold      = lruThresholdMin;
    } eviction_{};

    // Resources loaded by the workers since the last device update. The loader expects one stream and a completed ticket
    // before the next processRequests(), so a pass never adds more than maxRequests resources, and the copy from this
    // buffer runs on the stream before the next pass's workers can overwrite it.
    struct
    {
        mutable std::mutex mutex;
        HostSpan<uint32_t> resources;
        uint32_t           count = 0;
    } processed_{};

    // The physical pages shared by the white and black tiles (Options::coalesceWhiteBlackTiles), by texel. They are
    // retained in the page system, so they outlive the eviction of any page mapped to them.
    struct
    {
        mutable std::mutex                        mutex;
        std::unordered_map<std::string, uint32_t> pages;
    } whiteBlack_{};

    struct
    {
        mutable std::mutex mutex;
        size_t             bytesTransferredToDevice = 0;
    } stats_{};
};

}  // namespace hip_demand::vmm
