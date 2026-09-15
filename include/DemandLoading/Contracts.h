// SPDX-License-Identifier: MIT
#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#if defined(__HIPCC__)
#define HIP_DEMAND_CONTRACT_HD __host__ __device__
#else
#define HIP_DEMAND_CONTRACT_HD
#endif

namespace hip_demand {
namespace contract_v1 {

// This is an additive contract, not a replacement for the legacy loader ABI.
constexpr uint32_t Version = 1;
constexpr uint32_t InvalidSlot = UINT32_MAX;

enum class Outcome : uint32_t {
    Success, Pending, Deferred, InvalidKey, InvalidInput, SourceFailure,
    Unsupported, DeviceOutOfMemory, CapacityExhausted, DemandTooLarge,
    ProtectedBudgetExhausted, Cancelled, NoProgress, RequestOverflow,
    IdentityExhausted, AbiMismatch, InvalidTransition, RuntimeFailure
};

HIP_DEMAND_CONTRACT_HD constexpr bool retryable(Outcome value) {
    return value == Outcome::Pending || value == Outcome::Deferred ||
           value == Outcome::RequestOverflow;
}

enum class Operation : uint32_t {
    None, SourceRead, Allocate, Upload, CreateSampler, Publish, Destroy
};

struct Failure {
    Outcome outcome = Outcome::Success;
    Operation operation = Operation::None;
    int32_t rawError = 0;
};

struct OperationStatus {
    Failure primary{};
    Failure cleanup{};

    void recordPrimary(Failure failure) {
        if (primary.outcome == Outcome::Success)
            primary = failure;
    }
    void recordCleanup(Failure failure) {
        if (cleanup.outcome == Outcome::Success)
            cleanup = failure;
    }
};

enum class Completion : uint32_t { NotStarted, Running, Complete };

struct BatchResult {
    Completion completion = Completion::NotStarted;
    Outcome outcome = Outcome::Pending;

    HIP_DEMAND_CONTRACT_HD bool succeeded() const {
        return completion == Completion::Complete && outcome == Outcome::Success;
    }
};

struct alignas(8) GpuKey {
    uint32_t slot = InvalidSlot;
    uint32_t generation = 0;
    uint64_t incarnation = 0;
};

HIP_DEMAND_CONTRACT_HD constexpr bool valid(GpuKey key) {
    return key.slot != InvalidSlot && key.generation != 0 && key.incarnation != 0;
}
HIP_DEMAND_CONTRACT_HD constexpr bool operator==(GpuKey a, GpuKey b) {
    return a.slot == b.slot && a.generation == b.generation && a.incarnation == b.incarnation;
}

struct RegistrationResult {
    GpuKey key{};
    Outcome outcome = Outcome::InvalidKey;

    HIP_DEMAND_CONTRACT_HD bool succeeded() const {
        return outcome == Outcome::Success && valid(key);
    }
};

enum class ImageNamespace : uint32_t { Filename, Content, SourceObject, Memory };

// token is a collision-resolved identity, not an unchecked filename hash or pointer.
struct alignas(8) ImageIdentity {
    uint64_t token = 0;
    uint64_t revision = 0;
    ImageNamespace nameSpace = ImageNamespace::Memory;
    uint32_t reserved = 0;
};

inline bool operator==(const ImageIdentity& a, const ImageIdentity& b) {
    return a.token == b.token && a.revision == b.revision && a.nameSpace == b.nameSpace;
}

struct AbiHeader {
    uint32_t version;
    uint32_t byteSize;
};

HIP_DEMAND_CONTRACT_HD constexpr bool compatible(AbiHeader header, uint32_t bytes) {
    return header.version == Version && header.byteSize == bytes;
}

enum class AddressMode : uint32_t { Wrap, Clamp, Mirror, Border };
enum class FilterMode : uint32_t { Point, Linear };
enum class MipPolicy : uint32_t { Disabled, Required, AllowBaseLevelFallback };
enum class SamplingPolicy : uint32_t { Strict, AllowCoarsePreview };
enum class Priority : uint32_t { Normal, Low, High, KeepResident };

struct SamplerDesc {
    AbiHeader abi{Version, sizeof(SamplerDesc)};
    AddressMode addressMode[2]{AddressMode::Wrap, AddressMode::Wrap};
    FilterMode spatialFilter = FilterMode::Linear;
    FilterMode mipFilter = FilterMode::Linear;
    uint32_t normalizedCoords = 1;
    uint32_t sRGB = 0;
    uint32_t generateMipmaps = 1;
    uint32_t maxMipLevels = 0; // Count, not an original mip index; zero means the full chain.
    Priority priority = Priority::Normal;
    uint32_t maxAnisotropy = 1;
    MipPolicy mipPolicy = MipPolicy::Required;
    SamplingPolicy samplingPolicy = SamplingPolicy::Strict;
};

HIP_DEMAND_CONTRACT_HD inline Outcome validate(const SamplerDesc& desc) {
    if (!compatible(desc.abi, sizeof(SamplerDesc)))
        return Outcome::AbiMismatch;
    if (desc.addressMode[0] > AddressMode::Border || desc.addressMode[1] > AddressMode::Border ||
        desc.spatialFilter > FilterMode::Linear || desc.mipFilter > FilterMode::Linear ||
        desc.normalizedCoords > 1 || desc.sRGB > 1 || desc.generateMipmaps > 1 || desc.maxMipLevels > 32 ||
        desc.priority > Priority::KeepResident || desc.maxAnisotropy < 1 ||
        desc.maxAnisotropy > 16 || desc.mipPolicy > MipPolicy::AllowBaseLevelFallback ||
        desc.samplingPolicy > SamplingPolicy::AllowCoarsePreview)
        return Outcome::InvalidInput;
    return Outcome::Success;
}

inline bool operator==(const SamplerDesc& a, const SamplerDesc& b) {
    return a.abi.version == b.abi.version && a.abi.byteSize == b.abi.byteSize &&
           a.addressMode[0] == b.addressMode[0] && a.addressMode[1] == b.addressMode[1] &&
           a.spatialFilter == b.spatialFilter && a.mipFilter == b.mipFilter &&
           a.normalizedCoords == b.normalizedCoords && a.sRGB == b.sRGB &&
           a.generateMipmaps == b.generateMipmaps && a.maxMipLevels == b.maxMipLevels &&
           a.priority == b.priority && a.maxAnisotropy == b.maxAnisotropy &&
           a.mipPolicy == b.mipPolicy && a.samplingPolicy == b.samplingPolicy;
}

struct StorageKey {
    ImageIdentity image{};
    uint32_t uploadRepresentation = 0; // Assigned by the source/upload adapter.
    uint32_t sRGB = 0;
    uint32_t generateMipmaps = 1;
    uint32_t maxMipLevels = 0;
};

inline bool operator==(const StorageKey& a, const StorageKey& b) {
    return a.image == b.image && a.uploadRepresentation == b.uploadRepresentation &&
           a.sRGB == b.sRGB && a.generateMipmaps == b.generateMipmaps &&
           a.maxMipLevels == b.maxMipLevels;
}

inline StorageKey storageKey(ImageIdentity image, uint32_t representation, const SamplerDesc& desc) {
    return {image, representation, desc.sRGB, desc.generateMipmaps, desc.maxMipLevels};
}

struct alignas(8) RequestKey {
    GpuKey texture{};
    uint64_t revision = 0;
    uint32_t originalMip = 0;
    uint32_t reserved = 0;
};

HIP_DEMAND_CONTRACT_HD constexpr bool operator==(const RequestKey& a, const RequestKey& b) {
    return a.texture == b.texture && a.revision == b.revision && a.originalMip == b.originalMip;
}

struct MipRange {
    uint32_t first = InvalidSlot;
    uint32_t last = InvalidSlot;
};

struct MipLayout {
    uint32_t originalWidth = 0;
    uint32_t originalHeight = 0;
    uint32_t originalLevels = 0; // Policy-limited count, including level zero.
    uint32_t firstResidentMip = InvalidSlot;
    uint32_t resourceWidth = 0;
    uint32_t resourceHeight = 0;
    uint32_t resourceLevels = 0;
};

HIP_DEMAND_CONTRACT_HD inline uint32_t mipDimension(uint32_t value, uint32_t level) {
    return level >= 32 || (value >> level) == 0 ? 1 : value >> level;
}

HIP_DEMAND_CONTRACT_HD inline uint32_t fullMipCount(uint32_t width, uint32_t height) {
    if (width == 0 || height == 0)
        return 0;
    uint32_t levels = 1;
    for (uint32_t dimension = width > height ? width : height; dimension > 1; dimension >>= 1)
        ++levels;
    return levels;
}

HIP_DEMAND_CONTRACT_HD inline Outcome validate(const MipLayout& layout) {
    if (layout.originalLevels == 0 ||
        layout.originalLevels > fullMipCount(layout.originalWidth, layout.originalHeight))
        return Outcome::InvalidInput;
    if (layout.firstResidentMip == InvalidSlot)
        return layout.resourceWidth == 0 && layout.resourceHeight == 0 && layout.resourceLevels == 0
            ? Outcome::Success : Outcome::InvalidInput;
    if (layout.firstResidentMip >= layout.originalLevels ||
        layout.resourceLevels != layout.originalLevels - layout.firstResidentMip ||
        layout.resourceWidth != mipDimension(layout.originalWidth, layout.firstResidentMip) ||
        layout.resourceHeight != mipDimension(layout.originalHeight, layout.firstResidentMip))
        return Outcome::InvalidInput;
    return Outcome::Success;
}

struct LevelRequirement {
    Outcome outcome = Outcome::InvalidInput;
    MipRange levels{};
    float clampedOriginalLod = 0;
};

HIP_DEMAND_CONTRACT_HD inline LevelRequirement requiredLevels(
    float originalLod, uint32_t originalLevels, FilterMode filter, uint32_t maxAnisotropy = 1) {
    // Reject NaN/infinity before conversion to an integer.
    if (!(originalLod >= -3.402823466e+38F && originalLod <= 3.402823466e+38F) ||
        originalLevels == 0 || originalLevels > 32 || filter > FilterMode::Linear ||
        maxAnisotropy == 0 || maxAnisotropy > 16)
        return {};
    if (maxAnisotropy != 1)
        return {Outcome::Unsupported, {}, 0};
    const float last = static_cast<float>(originalLevels - 1);
    const float lod = originalLod < 0 ? 0 : (originalLod > last ? last : originalLod);
    const uint32_t low = static_cast<uint32_t>(lod);
    const float fraction = lod - static_cast<float>(low);
    if (fraction == 0)
        return {Outcome::Success, {low, low}, lod};
    if (filter == FilterMode::Linear || fraction == 0.5f)
        // Exact point ties conservatively require either possible hardware selection.
        return {Outcome::Success, {low, low + 1}, lod};
    const uint32_t selected = fraction < 0.5f ? low : low + 1;
    return {Outcome::Success, {selected, selected}, lod};
}

enum class SampleValidity : uint32_t { Invalid, Missing, Complete, CoarsePreview };

struct SampleDecision {
    Outcome outcome = Outcome::InvalidKey;
    SampleValidity validity = SampleValidity::Invalid;
    MipRange required{};
    float resourceLod = 0;

    HIP_DEMAND_CONTRACT_HD bool needsRequest() const {
        return validity == SampleValidity::Missing || validity == SampleValidity::CoarsePreview;
    }
    HIP_DEMAND_CONTRACT_HD bool contributesStrictSample() const {
        return validity == SampleValidity::Complete;
    }
};

HIP_DEMAND_CONTRACT_HD inline SampleDecision evaluate(
    const MipLayout& layout, const LevelRequirement& requirement, SamplingPolicy policy) {
    if (validate(layout) != Outcome::Success || policy > SamplingPolicy::AllowCoarsePreview)
        return {Outcome::InvalidInput};
    if (requirement.outcome != Outcome::Success)
        return {requirement.outcome};
    if (requirement.levels.first > requirement.levels.last ||
        requirement.levels.last >= layout.originalLevels ||
        !(requirement.clampedOriginalLod >= 0 &&
          requirement.clampedOriginalLod <= static_cast<float>(layout.originalLevels - 1)))
        return {Outcome::InvalidInput};
    const bool resident = layout.firstResidentMip != InvalidSlot;
    if (resident && requirement.levels.first >= layout.firstResidentMip)
        return {Outcome::Success, SampleValidity::Complete, requirement.levels,
                requirement.clampedOriginalLod - static_cast<float>(layout.firstResidentMip)};
    if (resident && policy == SamplingPolicy::AllowCoarsePreview)
        return {Outcome::Pending, SampleValidity::CoarsePreview, requirement.levels, 0};
    return {Outcome::Pending, SampleValidity::Missing, requirement.levels, 0};
}

enum class RegistrationState : uint32_t { Live, Retiring, Reclaimable };

struct alignas(8) PublishedTexture {
    GpuKey key{};
    uint64_t revision = 0;
    uint64_t textureObject = 0;
    MipLayout mips{};
    RegistrationState state = RegistrationState::Retiring;
    Outcome residency = Outcome::Pending;
    uint32_t reserved = 0;
};

// Immutable for the duration of every consumer. A tagged key cannot make a freed table safe.
struct alignas(8) DeviceContext {
    AbiHeader abi{Version, sizeof(DeviceContext)};
    uint64_t incarnation = 0;
    const PublishedTexture* textures = nullptr;
    uint32_t maxTextures = 0;
    uint32_t reserved = 0;
};

HIP_DEMAND_CONTRACT_HD inline SampleDecision resolve(
    const DeviceContext& context, GpuKey key, uint64_t revision,
    float originalLod, const SamplerDesc& desc) {
    if (!compatible(context.abi, sizeof(DeviceContext)))
        return {Outcome::AbiMismatch};
    const Outcome descriptorOutcome = validate(desc);
    if (descriptorOutcome != Outcome::Success)
        return {descriptorOutcome};
    if (context.reserved != 0)
        return {Outcome::InvalidInput};
    if (!valid(key) || key.incarnation != context.incarnation ||
        key.slot >= context.maxTextures || !context.textures)
        return {Outcome::InvalidKey};
    const PublishedTexture& texture = context.textures[key.slot];
    if (!(key == texture.key) || revision == 0 || revision != texture.revision ||
        texture.state != RegistrationState::Live)
        return {Outcome::InvalidKey};
    if (texture.reserved != 0)
        return {Outcome::InvalidInput};
    if (texture.residency != Outcome::Success && texture.residency != Outcome::Pending &&
        texture.residency != Outcome::Deferred)
        return {texture.residency};
    if ((desc.mipPolicy == MipPolicy::Disabled && texture.mips.originalLevels != 1) ||
        (desc.maxMipLevels != 0 && texture.mips.originalLevels > desc.maxMipLevels))
        return {Outcome::InvalidInput};
    uint32_t requiredCount = desc.mipPolicy == MipPolicy::Disabled
        ? 1 : fullMipCount(texture.mips.originalWidth, texture.mips.originalHeight);
    if (desc.maxMipLevels != 0 && desc.maxMipLevels < requiredCount)
        requiredCount = desc.maxMipLevels;
    // A reduced original count must not disguise lost mip capability as a legal LOD clamp.
    // Base-only capability fallback needs a distinct degraded representation (item 05).
    if (texture.mips.originalLevels != requiredCount)
        return {Outcome::Unsupported};
    const bool hasBacking = texture.mips.firstResidentMip != InvalidSlot;
    if ((texture.residency == Outcome::Success && (!hasBacking || texture.textureObject == 0)) ||
        (texture.residency != Outcome::Success && (hasBacking || texture.textureObject != 0)))
        return {Outcome::InvalidTransition};
    return evaluate(texture.mips, requiredLevels(originalLod, texture.mips.originalLevels,
                    desc.mipFilter, desc.maxAnisotropy), desc.samplingPolicy);
}

struct AbiInfo {
    AbiHeader abi{Version, sizeof(AbiInfo)};
    uint32_t keyBytes = sizeof(GpuKey);
    uint32_t requestBytes = sizeof(RequestKey);
    uint32_t descriptorBytes = sizeof(SamplerDesc);
    uint32_t contextBytes = sizeof(DeviceContext);
    uint32_t publicationBytes = sizeof(PublishedTexture);
    uint32_t pointerBytes = sizeof(void*);
    uint32_t contractPrimitives = 1;
    uint32_t productionIntegration = 0;
};

static_assert(sizeof(void*) == 8, "Contract v1 requires a 64-bit host/device ABI");
static_assert(sizeof(GpuKey) == 16 && alignof(GpuKey) == 8 && offsetof(GpuKey, incarnation) == 8);
static_assert(sizeof(RequestKey) == 32 && offsetof(RequestKey, revision) == 16 &&
              offsetof(RequestKey, originalMip) == 24);
static_assert(sizeof(SamplerDesc) == 56 && offsetof(SamplerDesc, maxAnisotropy) == 44);
static_assert(sizeof(MipLayout) == 28 && sizeof(PublishedTexture) == 72 &&
              offsetof(PublishedTexture, mips) == 32 && offsetof(PublishedTexture, residency) == 64);
static_assert(sizeof(DeviceContext) == 32 && offsetof(DeviceContext, textures) == 16 &&
              offsetof(DeviceContext, maxTextures) == 24);
static_assert(sizeof(AbiInfo) == 40 && sizeof(SampleDecision) == 20);
static_assert(std::is_standard_layout<DeviceContext>::value &&
              std::is_trivially_copyable<DeviceContext>::value &&
              std::is_trivially_copyable<PublishedTexture>::value &&
              std::is_trivially_copyable<SamplerDesc>::value);

} // namespace contract_v1
} // namespace hip_demand

// Identical unmangled symbol on Windows/Linux. No HIP initialization or C++ ownership crosses it.
extern "C" uint32_t hipDemandGetContractAbiV1(
    uint32_t requestedVersion, uint32_t outputBytes, hip_demand::contract_v1::AbiInfo* output) noexcept;

#undef HIP_DEMAND_CONTRACT_HD
