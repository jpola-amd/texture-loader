// SPDX-License-Identifier: MIT
#pragma once
#include "DemandLoading/DeviceContext.h"
#include "DemandLoading/WholeMipContext.h"

namespace hip_demand {
namespace cubic_v1 {
constexpr uint32_t Version = 1;
enum class SpatialMode : uint32_t
{
    Bicubic,
    SmartBicubic
};
struct Config
{
    contract_v1::AbiHeader abi{ Version, sizeof( Config ) };
    SpatialMode            spatial       = SpatialMode::Bicubic;
    float                  maxAnisotropy = 1;
    uint32_t               conservative  = 0;
    uint32_t               reserved      = 0;
};
struct Entry
{
    contract_v1::PublishedTexture texture{};
    contract_v1::SamplerDesc      descriptor{};
    // Ordinary point-filtered views, indexed by physical suffix level.
    TextureObject points[32]{};
    uint32_t      pointSRGB = 0;
    uint32_t      reserved  = 0;
};
enum class Backing : uint32_t
{
    Legacy,
    WholeMip
};
struct DeviceContext
{
    contract_v1::AbiHeader      abi{ Version, sizeof( DeviceContext ) };
    uint64_t                    incarnation = 0;
    const Entry*                entries     = nullptr;
    uint32_t                    count       = 0;
    Backing                     backing     = Backing::Legacy;
    hip_demand::DeviceContext   legacy{};
    whole_mip_v1::DeviceContext whole{};
};
static_assert( sizeof( Config ) == 24 && offsetof( Config, maxAnisotropy ) == 12 );
static_assert( sizeof( Entry ) == 392 && offsetof( Entry, points ) == 128 );
static_assert( sizeof( DeviceContext ) == 136 && offsetof( DeviceContext, whole ) == 80 );
static_assert( std::is_trivially_copyable<DeviceContext>::value );
}  // namespace cubic_v1
}  // namespace hip_demand
