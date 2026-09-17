#pragma once

/// @file DemandTextureLoader.h
/// @brief Demand texture loading API for HIP applications.
/// @note This header requires HIP types. Include <hip/hip_runtime.h> before this header.

#include "DemandLoading/DeviceContext.h"
#include "DemandLoading/Ticket.h"
#include "DemandLoading/Contracts.h"
#include "DemandLoading/CubicContext.h"
#include <string>
#include <memory>
#include <vector>
#include <cstdint>

namespace hip_demand {

class ImageSource;  // Forward declaration

// Error codes
enum class LoaderError {
    Success = 0,
    InvalidTextureId,
    MaxTexturesExceeded,
    FileNotFound,
    ImageLoadFailed,
    OutOfMemory,
    InvalidParameter,
    HipError,
    Unsupported
};

const char* getErrorString(LoaderError error);

// Forward declarations
class TextureRegistry;
class RequestBuffer;
class ImageReader;

// Eviction priority for textures
enum class EvictionPriority {
    Normal = 0,    // Default - standard LRU eviction
    Low = 1,       // Evict first (temporary/preview textures)
    High = 2,      // Evict last (important textures)
    KeepResident = 3  // Never evict (UI, hero textures)
};

// Configuration options
struct LoaderOptions {
    size_t maxTextureMemory = 2ULL * 1024 * 1024 * 1024;  // 2 GB default
    size_t maxTextures = 4096;  // 1..UINT32_MAX; UINT32_MAX is never a texture ID
    size_t maxRequestsPerLaunch = 1024;  // 1..UINT32_MAX
    bool enableEviction = true;
    unsigned int maxThreads = 0;  // Legacy pool size (0 = auto); decode/upload currently serialized
    unsigned int minResidentFrames = 3;  // Thrashing prevention: don't evict textures younger than this
};

// Texture descriptor
struct TextureDesc {
    hipTextureAddressMode addressMode[2] = {hipAddressModeWrap, hipAddressModeWrap};
    hipTextureFilterMode filterMode = hipFilterModeLinear;
    hipTextureFilterMode mipmapFilterMode = hipFilterModeLinear;
    bool normalizedCoords = true;
    bool sRGB = false;
    bool generateMipmaps = true;  // Generate mipmaps for better quality
    unsigned int maxMipLevel = 0;  // 0 = auto-generate all levels
    EvictionPriority evictionPriority = EvictionPriority::Normal;  // Eviction priority hint
};

inline bool operator==(const TextureDesc& a, const TextureDesc& b) {
    return (a.addressMode[0] == b.addressMode[0] &&
            a.addressMode[1] == b.addressMode[1] &&
            a.filterMode == b.filterMode &&
            a.mipmapFilterMode == b.mipmapFilterMode &&
            a.normalizedCoords == b.normalizedCoords &&
            a.sRGB == b.sRGB &&
            a.generateMipmaps == b.generateMipmaps &&
            a.maxMipLevel == b.maxMipLevel &&
            a.evictionPriority == b.evictionPriority);
}

// Texture information returned after creation
inline constexpr uint32_t InvalidTextureId = UINT32_MAX;

struct TextureHandle {
    uint32_t id = InvalidTextureId;
    bool valid = false;
    int width = 0;
    int height = 0;
    int channels = 0;
    LoaderError error = LoaderError::InvalidTextureId;
};

namespace capability_v1 {

constexpr uint32_t Version = 1;
enum class MipPolicy : uint32_t { LegacyCompatibility, Disabled, Required, AllowBaseLevelFallback };
enum class State : uint32_t { Registered, Pending, Resident, Degraded, Failed, Unloaded, Cancelled };
enum class Support : uint32_t { Unknown, OperationSupported, BehaviorQualified, Unsupported };
enum class Resource : uint32_t { None, Array, MipmappedArray };
enum class Reason : uint32_t { None, Disabled, Singleton, LevelLimit, LegacyBaseOnly, CapabilityFallback };
enum class Operation : uint32_t {
    None, SelectDevice, SourceRead, ProbeAllocate, ProbeGetLevel, ProbeUpload,
    ProbeCreateSampler, ProbeDestroySampler, ProbeFree, AllocateMipmapped,
    AllocateArray, GetLevel, Upload, CreateSampler, ReadSampler, Publish,
    DestroySampler, FreeMipmapped, FreeArray, QualifySampler, AllocateHost, FreeHost, FreeDevice
};

struct Policy {
    contract_v1::AbiHeader abi{Version, sizeof(Policy)};
    MipPolicy mipPolicy = MipPolicy::Required;
};

struct Failure {
    contract_v1::Outcome outcome = contract_v1::Outcome::Success;
    Operation operation = Operation::None;
    int32_t rawHipError = 0;
};

// Host-only C++/HIP ABI: use matching headers, SDK and compiler. No legacy
// descriptor, handle, Ticket or DeviceContext layout is changed.
struct Status {
    contract_v1::AbiHeader abi{Version, sizeof(Status)};
    uint32_t textureId = InvalidTextureId;
    MipPolicy policy = MipPolicy::LegacyCompatibility;
    State state = State::Registered;
    Support capability = Support::Unknown;
    Resource resource = Resource::None;
    Reason reason = Reason::None;
    uint32_t originalWidth = 0, originalHeight = 0, originalLevels = 0;
    uint32_t firstResidentMip = UINT32_MAX, lastResidentMip = UINT32_MAX;
    uint32_t resourceWidth = 0, resourceHeight = 0, resourceLevels = 0;
    uint64_t payloadBytes = 0;
    uint64_t attempts = 0;
    int32_t device = -1, runtimeVersion = 0, driverVersion = 0;
    uint64_t ownerContext = 0;
    char deviceName[256]{};
    char architecture[256]{};
    TextureDesc requested{};
    hipTextureDesc submittedSampler{};
    hipTextureDesc returnedSampler{};
    uint32_t submitted = 0, returned = 0, published = 0;
    Failure primary{}, cleanup{}, fallback{};
};

static_assert(sizeof(Policy) == 12 && std::is_standard_layout<Policy>::value,
              "Capability policy ABI changed");
static_assert(std::is_standard_layout<Status>::value && std::is_trivially_copyable<Status>::value,
              "Capability status must remain a plain host snapshot");

} // namespace capability_v1

namespace anisotropy_v1 {

constexpr uint32_t Version = 1;
enum class Profile : uint32_t { LegacyCompatibility, Explicit, Parity16 };
enum class Requirement : uint32_t { AllowUnqualified, RequireQualified };
enum class Qualification : uint32_t { Unqualified, Qualified };
enum Limitation : uint32_t {
    None = 0, LegacySetting = 1, UnqualifiedBehavior = 2,
    SingleLevel = 4, DescriptorMismatch = 8, BaseLevelFallback = 16
};

// Additive host API; TextureDesc and capability_v1 layouts remain unchanged.
// The 1..16 range is a loader policy, not a measured hardware capability.
struct Request {
    contract_v1::AbiHeader abi{Version, sizeof(Request)};
    Profile profile = Profile::Explicit;
    uint32_t maxAnisotropy = 1;
    Requirement requirement = Requirement::AllowUnqualified;

    static Request legacy() {
        return {{Version, sizeof(Request)}, Profile::LegacyCompatibility, 0, Requirement::AllowUnqualified};
    }
    static Request parity() {
        return {{Version, sizeof(Request)}, Profile::Parity16, 16, Requirement::RequireQualified};
    }
};

inline bool operator==(const Request& a, const Request& b) {
    return a.abi.version == b.abi.version && a.abi.byteSize == b.abi.byteSize &&
           a.profile == b.profile && a.maxAnisotropy == b.maxAnisotropy && a.requirement == b.requirement;
}

struct Status {
    contract_v1::AbiHeader abi{Version, sizeof(Status)};
    Request requested{};
    capability_v1::Status texture{};
    Qualification qualification = Qualification::Unqualified;
    capability_v1::Support samplerSupport = capability_v1::Support::Unknown;
    uint32_t limitations = None;
    uint32_t requirementRejected = 0;
};

static_assert(sizeof(Request) == 20 && offsetof(Request, maxAnisotropy) == 12 &&
              std::is_standard_layout<Request>::value && std::is_trivially_copyable<Request>::value,
              "Anisotropy request ABI changed");
static_assert(std::is_standard_layout<Status>::value && std::is_trivially_copyable<Status>::value,
              "Anisotropy status must remain a plain host snapshot");

} // namespace anisotropy_v1

class DemandTextureLoader {
public:
    explicit DemandTextureLoader(const LoaderOptions& options = LoaderOptions());
    ~DemandTextureLoader();

    // Disable copy
    DemandTextureLoader(const DemandTextureLoader&) = delete;
    DemandTextureLoader& operator=(const DemandTextureLoader&) = delete;

    // Register a descriptor-specific sampler (pixels remain demand-loaded).
    // Equal complete descriptors for the same filename/source identity reuse
    // an ID, including at capacity. Every distinct descriptor consumes one ID.
    // Compatible variants share image backing; sRGB and mip-generation/limit
    // policies conservatively split storage. All descriptor fields participate
    // in sampler identity, including eviction priority.
    // These legacy handles are not owning leases: unload does not release
    // registration capacity, and IDs are never recycled.
    TextureHandle createTexture(const std::string& filename, 
                                const TextureDesc& desc = TextureDesc());

    // Create a texture from an ImageSource (pixels not loaded until requested).
    // Invalid metadata or an exception while opening/hashing fails registration.
    // Content hashes use the ImageSource identity contract plus validated native
    // dimensions/format/channels/mip metadata. Filename and content identities
    // occupy separate namespaces. Storage retains its selected source through
    // registration and active reads; an equivalent incoming alias need not be
    // retained. Source content/metadata must remain stable while registered.
    TextureHandle createTexture(std::shared_ptr<ImageSource> imageSource,
                                const TextureDesc& desc = TextureDesc());
    
    // Copy byte pixels into a fresh registration/storage identity on every call.
    TextureHandle createTextureFromMemory(const void* data, 
                                         int width, int height, int channels,
                                         const TextureDesc& desc = TextureDesc());

    // Explicit mip semantics. Required/allowed policies retain authored levels
    // even with generation disabled; missing required levels then fail. Disabled
    // publishes only the original base. maxMipLevel remains a level count.
    // Policy participates in sampler identity, not compatible backing identity.
    TextureHandle createTextureV1(const std::string& filename, const TextureDesc& desc,
                                  const capability_v1::Policy& policy = {});
    TextureHandle createTextureV1(std::shared_ptr<ImageSource> imageSource, const TextureDesc& desc,
                                  const capability_v1::Policy& policy = {});
    TextureHandle createTextureFromMemoryV1(const void* data, int width, int height, int channels,
                                            const TextureDesc& desc, const capability_v1::Policy& policy = {});

    // Pollable per-registration snapshot, including while a source read blocks.
    // Validates version/size and ID before writing output. Completion is not
    // success; degraded base fallback is never mip-filter qualification.
    // Support reports operations only; no runtime probe certifies pixel behavior.
    contract_v1::Outcome getTextureStatusV1(uint32_t textureId, capability_v1::Status& status) const;

    // Immutable anisotropy request, independent of compatible image backing.
    // Legacy APIs still request zero. Explicit 1 requests isotropic filtering;
    // 2..16 request native, unqualified anisotropy. By default the shared native
    // override submits zero for every request. Only the exact environment
    // string HDT_DISABLE_TEXTURE_ANISO_OVERRIDE=1 disables it. See CUBIC.md.
    // Strict requests (including parity()) fail residency until the exact
    // configuration has pixel qualification. No configurations are certified
    // by this implementation; successful HIP calls/readback are not proof.
    TextureHandle createTextureAnisotropyV1(const std::string& filename, const TextureDesc& desc,
        const anisotropy_v1::Request& request, const capability_v1::Policy& policy = {});
    TextureHandle createTextureAnisotropyV1(std::shared_ptr<ImageSource> source, const TextureDesc& desc,
        const anisotropy_v1::Request& request, const capability_v1::Policy& policy = {});
    TextureHandle createTextureFromMemoryAnisotropyV1(const void* data, int width, int height, int channels,
        const TextureDesc& desc, const anisotropy_v1::Request& request, const capability_v1::Policy& policy = {});
    // Requested, submitted and returned fields are separate from qualification.
    // The nested snapshot retains resource, owner and primary/cleanup evidence.
    contract_v1::Outcome getTextureAnisotropyStatusV1(uint32_t textureId, anisotropy_v1::Status& status) const;

    // Opt in before residency. Retains sampler identity and returns a tagged key
    // (revision 1). Requires normalized coordinates and native linear spatial
    // filtering. launchPrepare publishes the separate cubic snapshot.
    contract_v1::RegistrationResult enableCubicV1(uint32_t textureId);
    cubic_v1::DeviceContext getCubicContextV1() const;

    // Prepare for launch (updates device context). This implementation uses a
    // serialized, device-quiescent publication/retirement baseline; table copies
    // complete before return. Callers must not concurrently submit consumers
    // while preparing or unloading, or overlap reuse of the request context.
    void launchPrepare(hipStream_t stream = 0);

    // Get device context to pass to kernel
    DeviceContext getDeviceContext() const;

    // Process texture requests after kernel launch using the provided device context
    // Returns the number of sampler IDs newly made resident, not backing uploads.
    size_t processRequests(hipStream_t stream, const DeviceContext& deviceContext);

    // Completes request readback before handing decode/upload to a background
    // worker. Runtime operations are serialized; a Ticket reports completion,
    // not per-sampler success.
    Ticket processRequestsAsync(hipStream_t stream, const DeviceContext& deviceContext);

    // Statistics
    size_t getResidentTextureCount() const;
    // Array texel payload, counted once per shared allocation, including failed
    // cleanup until freed. Excludes opaque HIP/sampler overhead and host caches;
    // this is neither a physical-VRAM measurement nor a hard budget guarantee.
    size_t getTotalTextureMemory() const;
    size_t getRequestCount() const;
    bool hadRequestOverflow() const;
    LoaderError getLastError() const;

    // Eviction control
    void enableEviction(bool enable);
    void setMaxTextureMemory(size_t bytes);
    size_t getMaxTextureMemory() const;
    
    /// Update the eviction priority for a texture dynamically.
    /// Use this to adjust priorities based on camera distance, LOD importance, etc.
    /// Existing IDs never merge. Lookup of equal keys chooses the oldest ID.
    /// Storage uses the highest sibling priority; any registered KeepResident
    /// variant protects its backing even before that sampler is requested.
    void updateEvictionPriority(uint32_t textureId, EvictionPriority priority);

    // Unload only this sampler; live or failed-to-destroy sibling objects keep
    // their backing alive. Cancels older requests, waits for active operations
    // and GPU consumers, and invalidates device mappings before resource free.
    // Whole-storage eviction invalidates every sibling. Cleanup errors retain
    // resources and charges for a later retry. Unload is not registration release.
    // Invalid/unregistered IDs set InvalidTextureId without mutation.
    void unloadTexture(uint32_t textureId);
    void unloadAll();

    /// Abort all pending operations and halt the loader gracefully.
    /// After calling abort(), no new requests will be processed.
    /// Safe to call from any thread. Blocks until all in-flight operations complete.
    void abort();

    /// Check if the loader has been aborted.
    bool isAborted() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace hip_demand
