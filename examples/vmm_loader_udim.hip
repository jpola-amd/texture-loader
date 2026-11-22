#include <DemandLoading/VmmTextureSampling.h>
#include <hip/hip_runtime.h>

namespace {

__device__ __forceinline__ uint8_t toUnorm8( float value )
{
    value = fminf( 1.0f, fmaxf( 0.0f, value ) );
    return static_cast<uint8_t>( value * 255.0f + 0.5f );
}

}  // namespace

// Renders the whole UDIM grid onto the output: grid coordinates run from (0, 0) at the top-left pixel to (udim, vdim)
// at the bottom-right one. footprintScale enlarges the pixel footprint to emulate a distant view. The alpha channel
// records residency, 255 for a final sample and 0 for a provisional one, so the host can tell when rendering is done.
extern "C" __global__ void renderUdimTexture( hip_demand::vmm::DeviceContext context,
                                              uint8_t*                       output,
                                              uint32_t                       outputWidth,
                                              uint32_t                       outputHeight,
                                              uint32_t                       textureId,
                                              uint32_t                       channel,
                                              uint32_t                       udim,
                                              uint32_t                       vdim,
                                              float                          footprintScale )
{
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if( x >= outputWidth || y >= outputHeight )
        return;

    const float  cellsPerPixelX = static_cast<float>( udim ) / static_cast<float>( outputWidth );
    const float  cellsPerPixelY = static_cast<float>( vdim ) / static_cast<float>( outputHeight );
    const float  u              = ( static_cast<float>( x ) + 0.5f ) * cellsPerPixelX;
    const float  v              = ( static_cast<float>( y ) + 0.5f ) * cellsPerPixelY;
    const float2 ddx            = make_float2( cellsPerPixelX * footprintScale, 0.0f );
    const float2 ddy            = make_float2( 0.0f, cellsPerPixelY * footprintScale );

    bool         resident = false;
    const float3 color    = hip_demand::vmm::tex2DGradUdim<float3>( context, textureId, channel, u, v, ddx, ddy, resident );

    const size_t offset = ( static_cast<size_t>( y ) * outputWidth + x ) * 4;
    output[offset + 0]  = toUnorm8( color.x );
    output[offset + 1]  = toUnorm8( color.y );
    output[offset + 2]  = toUnorm8( color.z );
    output[offset + 3]  = resident ? 255 : 0;
}
