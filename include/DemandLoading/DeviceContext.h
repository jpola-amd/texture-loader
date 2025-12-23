#pragma once

#include <hip/hip_runtime.h>
#include <cstdint>

namespace hip_demand {

// Device context passed to kernels
// This structure contains GPU-accessible data for texture sampling
struct DeviceContext {
    uint32_t* residentFlags;      // Bit flags for texture residency
    hipTextureObject_t* textures; // Array of texture objects
    uint32_t* requests;           // Request buffer
    uint32_t* requestCount;       // Atomic counter for requests
    uint32_t* requestOverflow;    // Flag set when request buffer overflows
    uint32_t maxTextures;
    uint32_t maxRequests;
};

} // namespace hip_demand
