#pragma once
#include <DemandLoading/DeviceContext.h>

namespace hip_demand::vmm {

inline constexpr uint32_t lruMax          = 14u;
inline constexpr uint32_t lruNonEvictable = 15u;
inline constexpr uint32_t lruThresholdMin = 2u;

}  // namespace hip_demand::vmm