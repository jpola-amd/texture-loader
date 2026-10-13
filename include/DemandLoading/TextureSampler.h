#pragma once

/// @file TextureSampler.h
/// @brief Device-side texture sampler record, shaped after OptiX Toolkit demandLoading::TextureSampler.
/// @note No HIP headers required. This header uses platform-agnostic types.
///
/// The loader stores whole textures with whole mip chains, so the tiling fields of the
/// OptiX Toolkit record are fixed: a texture is one load unit, isSparseTexture is 0, the tile
/// sizes are 0, and mipTailFirstLevel is 0 (every level becomes resident with the sampler).

#include <cstddef>
#include <cstdint>

namespace hip_demand {

/// Device-side texture info. Published for a texture ID only while that texture is resident.
struct TextureSampler
{
    unsigned long long texture;  // Binary-compatible with hipTextureObject_t

    // Bit layout matches demandLoading::TextureSampler::Description.
    struct Description
    {
        unsigned int isInitialized : 1;
        unsigned int reserved1 : 2;
        unsigned int numMipLevels : 5;
        unsigned int logTileWidth : 4;
        unsigned int logTileHeight : 4;
        unsigned int reserved2 : 4;
        unsigned int isSparseTexture : 1;
        unsigned int isUdimBaseTexture : 1;
        unsigned int wrapMode0 : 2;
        unsigned int wrapMode1 : 2;
        unsigned int mipmapFilterMode : 1;
        unsigned int maxAnisotropy : 5;
    } desc;

    // Texture dimensions
    unsigned int width;
    unsigned int height;
    unsigned int mipTailFirstLevel;

    unsigned int filterMode : 2;  // hipTextureFilterMode of the hardware sampler
    unsigned int pad : 30;
};

static_assert(sizeof(TextureSampler::Description) == 4, "TextureSampler::Description must stay 32 bits");
static_assert(sizeof(TextureSampler) == 32 && offsetof(TextureSampler, mipTailFirstLevel) == 20,
              "TextureSampler is a device ABI; update consumers together with its layout");

}  // namespace hip_demand
