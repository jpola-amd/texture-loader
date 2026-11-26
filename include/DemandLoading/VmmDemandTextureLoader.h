#pragma once

#include <DemandLoading/VmmDeviceContext.h>
#include <ImageSource/ImageSource.h>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <utility>
#include <vector>

namespace hip_demand::vmm {

struct TextureDescriptor
{
    hipTextureAddressMode addressMode[2]   = { hipAddressModeWrap, hipAddressModeWrap };
    hipTextureFilterMode  filterMode       = hipFilterModeLinear;
    hipTextureFilterMode  mipmapFilterMode = hipFilterModeLinear;
    bool                  normalizedCoords = true;
    /// Maximum number of samples tex2DGrad takes along the major axis of a pixel's footprint, 1 to 16, like
    /// hipTextureDesc::maxAnisotropy. 1 selects isotropic filtering.
    unsigned int maxAnisotropy = 1;
};

// Declared here, so that the names in Ticket mean this class also where hip_demand::TicketImpl of Ticket.h is visible.
class TicketImpl;

class Ticket
{
  public:
    Ticket() {}
    int  numTasksTotal() const;
    int  numTasksRemaining() const;
    void wait( hipEvent_t* event = nullptr );

  private:
    std::shared_ptr<TicketImpl> impl_;
    friend class TicketImpl;
    Ticket( std::shared_ptr<TicketImpl>&& impl )
        : impl_( std::move( impl ) )
    {
    }
};

struct Options
{
    /// Maximum number of VMM pages in the reserved virtual address space, including texture metadata and image tiles.
    uint32_t maxVirtualPages = 64 * 1024 * 1024;

    /// Memory budget of the image tiles, in 64 KiB pages. ~0u applies no limit. When the tiles a pass asks for do
    /// not fit, tiles that the last launch did not read are evicted, least recently used first, and the requests
    /// they make room for are served by the next pass. Nothing is evicted while the budget has room.
    uint32_t maxPhysicalPages = ~0u;

    /// What happens when the tiles that the last launch read do not fit in maxPhysicalPages by themselves, so that
    /// nothing can be evicted for a request. False: the loader maps beyond the budget, so that the launch can
    /// complete; Statistics::peakResidentTileBytes shows by how much. True: such requests are refused and counted in
    /// Statistics::numDeniedRequests, and Statistics::budgetExhausted is set; a renderer that relaunches until every
    /// tile is resident has to stop on it.
    bool strictPhysicalLimit = false;

    /// Maximum number of demand textures that can be registered with the loader.
    uint32_t maxTextures = 256 * 1024;

    /// Maximum number of missing-resource requests collected and queued in one processing pass.
    uint32_t maxRequests = 8192;

    /// Enables LRU tracking and the eviction of resident image-tile pages to stay within maxPhysicalPages. Texture
    /// metadata is not evicted.
    bool enableEviction = true;

    /// Gives textures virtual pages for their mip levels on demand, starting from the mip tail, instead of for all
    /// levels at once. Saves virtual pages on textures seen only from afar; costs one extra pass the first time a
    /// finer level is sampled.
    bool enableCascades = false;

    /// Maximum number of tiles one pass can evict. A value of zero disables eviction.
    uint32_t maxEvictedPages = 8192;

    /// Number of CPU request-processing threads. Zero selects hardware concurrency.
    uint32_t maxThreads = 0;

    /// Backs all white and all black tiles with one shared physical page per texel value.
    bool coalesceWhiteBlackTiles = false;
};

struct Statistics
{
    size_t       numTilesRead             = 0;
    size_t       numBytesRead             = 0;
    double       readTime                 = 0.0;
    size_t       numTextures              = 0;
    size_t       virtualTextureBytes      = 0;
    size_t       deviceMemoryUsed         = 0;
    size_t       bytesTransferredToDevice = 0;
    unsigned int numEvictions             = 0;
    /// Memory of the image tiles that are mapped now, and the most it has been.
    size_t residentTileBytes     = 0;
    size_t peakResidentTileBytes = 0;
    /// Tile requests a pass did not serve for lack of room, over all passes. With room made by eviction they are
    /// served by a later pass; see Options::strictPhysicalLimit for those that are not.
    size_t numDeniedRequests = 0;
    /// The last pass refused requests and had no tile to evict for them (Options::strictPhysicalLimit).
    bool budgetExhausted = false;
    /// Tiles and mip tails that the image source failed to read. They are resident and read as zero.
    size_t numFailedReads = 0;
};

class DemandTexture
{
  public:
    virtual ~DemandTexture()       = default;
    virtual uint32_t getId() const = 0;
};

class DemandTextureLoader
{
  public:
    virtual ~DemandTextureLoader() = default;

    virtual const DemandTexture& createTexture( std::shared_ptr<ImageSource> imageSource, const TextureDescriptor& textureDesc ) = 0;

    virtual const DemandTexture& createUdimTexture( const std::vector<std::shared_ptr<ImageSource>>& imageSources,
                                                    const std::vector<TextureDescriptor>&            textureDescs,
                                                    uint32_t                                         udim,
                                                    uint32_t                                         vdim,
                                                    uint32_t numChannels = 1 ) = 0;
    virtual void   launchPrepare( hipStream_t stream, DeviceContext& deviceContext )         = 0;
    virtual Ticket processRequests( hipStream_t stream, const DeviceContext& deviceContext ) = 0;
    virtual Statistics getStatistics() const                                               = 0;
};

std::unique_ptr<DemandTextureLoader> createDemandTextureLoader( const Options& options );

}  // namespace hip_demand::vmm
