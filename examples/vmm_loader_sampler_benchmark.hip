#include <DemandLoading/VmmTextureSampling.h>
#include <hip/hip_runtime.h>

namespace {

__device__ __forceinline__ uint8_t toUnorm8( float value )
{
    value = fminf( 1.0f, fmaxf( 0.0f, value ) );
    return static_cast<uint8_t>( value * 255.0f + 0.5f );
}

__device__ __forceinline__ void writePixel( uint8_t* output, size_t offset, uint8_t red, uint8_t green, uint8_t blue )
{
    output[offset + 0] = red;
    output[offset + 1] = green;
    output[offset + 2] = blue;
    output[offset + 3] = 255;
}

}  // namespace

extern "C" __global__ void renderVmmTex2DLodBenchmark( hip_demand::vmm::DeviceContext context,
                                                       uint8_t*                       output,
                                                       uint32_t                       outputWidth,
                                                       uint32_t                       outputHeight,
                                                       float                          mipLevel )
{
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if( x >= outputWidth || y >= outputHeight )
        return;

    const float  u            = ( static_cast<float>( x ) + 0.5f ) / static_cast<float>( outputWidth );
    const float  v            = ( static_cast<float>( y ) + 0.5f ) / static_cast<float>( outputHeight );
    bool         resident     = false;
    const float3 color        = hip_demand::vmm::tex2DLod<float3>( context, 0, u, v, mipLevel, resident );
    const size_t outputOffset = ( static_cast<size_t>( y ) * outputWidth + x ) * 4;
    if( resident )
        writePixel( output, outputOffset, toUnorm8( color.x ), toUnorm8( color.y ), toUnorm8( color.z ) );
    else
        writePixel( output, outputOffset, 255, 0, 255 );
}

extern "C" __global__ void renderHipTex2DLodBenchmark( hipTextureObject_t texture, uint8_t* output, uint32_t outputWidth, uint32_t outputHeight, float mipLevel )
{
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if( x >= outputWidth || y >= outputHeight )
        return;

    const float  u            = ( static_cast<float>( x ) + 0.5f ) / static_cast<float>( outputWidth );
    const float  v            = ( static_cast<float>( y ) + 0.5f ) / static_cast<float>( outputHeight );
    const float4 color        = tex2DLod<float4>( texture, u, v, mipLevel );
    const size_t outputOffset = ( static_cast<size_t>( y ) * outputWidth + x ) * 4;
    writePixel( output, outputOffset, toUnorm8( color.x ), toUnorm8( color.y ), toUnorm8( color.z ) );
}
