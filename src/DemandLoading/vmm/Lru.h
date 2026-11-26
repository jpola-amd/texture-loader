#pragma once
#include <DemandLoading/VmmDeviceContext.h>

namespace hip_demand::vmm {

inline constexpr uint32_t lruMax          = 14u;
inline constexpr uint32_t lruNonEvictable = 15u;
// A tile becomes an eviction candidate with the first launch that does not read it.
inline constexpr uint32_t lruEvictable = 1u;

}  // namespace hip_demand::vmm