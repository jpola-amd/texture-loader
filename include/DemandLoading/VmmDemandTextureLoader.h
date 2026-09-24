#pragma once

#include <DemandLoading/DeviceContext.h>
#include <ImageSource/ImageSource.h>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <utility>

namespace hip_demand::vmm {

struct TextureDescriptor
{
    hipTextureAddressMode addressMode[2]   = { hipAddressModeWrap, hipAddressModeWrap };
    hipTextureFilterMode  filterMode       = hipFilterModeLinear;
    hipTextureFilterMode  mipmapFilterMode = hipFilterModeLinear;
    bool                  normalizedCoords = true;
};

class Ticket
{
  public:
    Ticket() {}
    int  numTasksTotal() const;
    int  numTasksRemaining() const;
    void wait( hipEvent_t* event = nullptr );

  private:
    std::shared_ptr<class TicketImpl> impl_;
    friend class TicketImpl;
    Ticket( std::shared_ptr<TicketImpl>&& impl )
        : impl_( std::move( impl ) )
    {
    }
};

struct Options
{
    /// Maximum number of VMM pages in the reserved virtual address space, including texture metadata and image tiles.
    uint32_t maxVirtualPages  = 64 * 1024 * 1024;

    /// Maximum number of physical VMM pages that may be allocated. ~0u applies no loader-imposed limit.
    uint32_t maxPhysicalPages = ~0u;

    /// Maximum number of demand textures that can be registered with the loader.
    uint32_t maxTextures = 256 * 1024;

    /// Maximum number of missing-resource requests collected and queued in one processing pass.
    uint32_t maxRequests = 8192;

    /// Enables LRU tracking and eviction of resident image-tile pages. Texture metadata is not evicted.
    bool enableEviction = true;

    /// Maximum number of eviction candidates collected per pass. A value of zero disables eviction.
    uint32_t maxEvictedPages = 8192;

    /// Number of CPU request-processing threads. Zero selects hardware concurrency.
    uint32_t maxThreads = 0;
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
    virtual void   launchPrepare( hipStream_t stream, DeviceContext& deviceContext )         = 0;
    virtual Ticket processRequests( hipStream_t stream, const DeviceContext& deviceContext ) = 0;
};

std::unique_ptr<DemandTextureLoader> createDemandTextureLoader( const Options& options );

}  // namespace hip_demand::vmm
