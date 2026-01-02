#pragma once

/// @file DeviceContext.h
/// @brief Device context for GPU texture sampling.
/// @note No HIP headers required. This header uses platform-agnostic types.

#include <cstdint>

namespace hip_demand {

/// Platform-agnostic texture object handle.
/// This is binary-compatible with hipTextureObject_t (unsigned long long).
using TextureObject = unsigned long long;

// Device context passed to kernels
// This structure contains GPU-accessible data for texture sampling
struct DeviceContext {
    uint32_t* residentFlags;      // Bit flags for texture residency
    TextureObject* textures;      // Array of texture objects
    uint32_t* requests;           // Request buffer
    uint32_t* requestCount;       // Atomic counter for requests
    uint32_t* requestOverflow;    // Flag set when request buffer overflows
    uint32_t maxTextures;
    uint32_t maxRequests;
};

} // namespace hip_demand
