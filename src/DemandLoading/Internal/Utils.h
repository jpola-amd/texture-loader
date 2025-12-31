// SPDX-License-Identifier: MIT
// Internal utility functions

#pragma once

#include <hip/hip_runtime.h>

namespace hip_demand {
namespace internal {

/// Calculate total memory needed for mipmaps
inline size_t calculateMipmapMemory(int width, int height, int bytesPerPixel) {
    size_t total = 0;
    while (width > 0 && height > 0) {
        total += width * height * bytesPerPixel;
        width /= 2;
        height /= 2;
    }
    return total;
}

/// Calculate number of mip levels
inline int calculateMipLevels(int width, int height) {
    int levels = 1;
    while (width > 1 || height > 1) {
        width = std::max(1, width / 2);
        height = std::max(1, height / 2);
        levels++;
    }
    return levels;
}

} // namespace internal
} // namespace hip_demand
