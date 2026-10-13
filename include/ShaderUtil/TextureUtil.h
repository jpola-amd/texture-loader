#pragma once

/// @file TextureUtil.h
/// @brief Texture helpers that mirror OptiXToolkit/ShaderUtil/TextureUtil.h.
/// @note getMipLevel is in the global namespace, as in the OptiX Toolkit, so shared renderer
///       device code can call it without a backend-specific qualifier. It requires the HIP compiler.

#if defined(__HIPCC__)
#include <hip/hip_runtime.h>

/// Compute the mip level from the texture gradients (same EWA estimate as the OptiX Toolkit).
/// invAnisotropy is 1 / maxAnisotropy; use 1 for isotropic filtering.
__host__ __device__ inline float getMipLevel(float2 ddx, float2 ddy, int texWidth, int texHeight,
                                             float invAnisotropy)
{
    ddx = make_float2(ddx.x * texWidth, ddx.y * texHeight);
    ddy = make_float2(ddy.x * texWidth, ddy.y * texHeight);

    const float A    = ddy.x * ddy.x + ddy.y * ddy.y;
    const float B    = -2.0f * (ddx.x * ddy.x + ddx.y * ddy.y);
    const float C    = ddx.x * ddx.x + ddx.y * ddx.y;
    const float root = sqrtf(fmaxf(A * A - 2.0f * A * C + C * C + B * B, 0.0f));

    // Squared major and minor ellipse radii; half the log converts them to a mip level.
    const float minorRadius2 = (A + C - root) * 0.5f;
    const float majorRadius2 = (A + C + root) * 0.5f;
    const float filterWidth2 = fmaxf(minorRadius2, majorRadius2 * invAnisotropy * invAnisotropy);
    return 0.5f * log2f(filterWidth2);
}
#endif
