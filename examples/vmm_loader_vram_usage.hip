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

struct MipRegion
{
    uint32_t level;
    uint32_t x;
    uint32_t y;
    uint32_t width;
    uint32_t height;
};

__device__ MipRegion findMipRegion( uint32_t localX, uint32_t localY, uint32_t cellWidth, uint32_t cellHeight,
                                    uint32_t mipCount )
{
    const uint32_t baseWidth = max( 1u, cellWidth * 2u / 3u );
    if( localX < baseWidth )
        return { 0, 0, 0, baseWidth, cellHeight };

    uint32_t y = 0;
    for( uint32_t level = 1; level < mipCount; ++level )
    {
        const uint32_t width  = max( 1u, baseWidth >> level );
        const uint32_t height = max( 1u, cellHeight >> level );
        if( localX < baseWidth + width && localY >= y && localY < y + height )
            return { level, baseWidth, y, width, height };
        y += height;
    }
    return { 0, 0, 0, 0, 0 };
}

__device__ void renderGridPixel( const hip_demand::vmm::DeviceContext& context, uint8_t* output,
                                 uint32_t outputWidth, uint32_t outputHeight, uint32_t textureCount,
                                 uint32_t columnCount, uint32_t rowCount, uint32_t x, uint32_t y )
{
    const size_t   outputOffset = ( static_cast<size_t>( y ) * outputWidth + x ) * 4;
    const uint32_t column       = ( ( x + 1 ) * columnCount - 1 ) / outputWidth;
    const uint32_t row          = ( ( y + 1 ) * rowCount - 1 ) / outputHeight;
    const uint32_t textureId    = row * columnCount + column;
    if( textureId >= textureCount )
    {
        writePixel( output, outputOffset, 0, 0, 0 );
        return;
    }

    const uint32_t cellX0     = column * outputWidth / columnCount;
    const uint32_t cellY0     = row * outputHeight / rowCount;
    const uint32_t cellWidth  = ( column + 1 ) * outputWidth / columnCount - cellX0;
    const uint32_t cellHeight = ( row + 1 ) * outputHeight / rowCount - cellY0;
    const uint32_t localX     = x - cellX0;
    const uint32_t localY     = y - cellY0;

    const uint32_t textureResourceId = context.resourceTable.textureInfos.getResourceId( textureId );
    if( !hip_demand::vmm::isResourceResident( context, textureResourceId ) )
    {
        bool resident = false;
        hip_demand::vmm::tex2DLod<float3>( context, textureId, 0.5f, 0.5f, 0.0f, resident );
        writePixel( output, outputOffset, 255, 0, 255 );
        return;
    }

    const hip_demand::vmm::DeviceTextureInfo& texture = *context.textureInfos.ptr[textureId];
    const MipRegion region = findMipRegion( localX, localY, cellWidth, cellHeight, texture.mipCount );
    if( region.width == 0 )
    {
        writePixel( output, outputOffset, 0, 0, 0 );
        return;
    }

    const float u = ( static_cast<float>( localX - region.x ) + 0.5f ) / static_cast<float>( region.width );
    const float v = ( static_cast<float>( localY - region.y ) + 0.5f ) / static_cast<float>( region.height );
    bool        resident = false;
    const auto  color = hip_demand::vmm::tex2DLod<float3>( context, textureId, u, v,
                                                           static_cast<float>( region.level ), resident );

    if( resident )
        writePixel( output, outputOffset, toUnorm8( color.x ), toUnorm8( color.y ), toUnorm8( color.z ) );
    else
        writePixel( output, outputOffset, 255, 0, 255 );
}

}  // namespace

extern "C" __global__ void renderVmmTextureRegion( hip_demand::vmm::DeviceContext context, uint8_t* output,
                                                     uint32_t outputWidth, uint32_t outputHeight,
                                                     uint32_t textureCount, uint32_t columnCount, uint32_t rowCount,
                                                     uint32_t regionX, uint32_t regionY, uint32_t regionWidth,
                                                     uint32_t regionHeight )
{
    const uint32_t localX = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t localY = blockIdx.y * blockDim.y + threadIdx.y;
    if( localX >= regionWidth || localY >= regionHeight )
        return;

    const uint32_t x = regionX + localX;
    const uint32_t y = regionY + localY;
    if( x < outputWidth && y < outputHeight )
        renderGridPixel( context, output, outputWidth, outputHeight, textureCount, columnCount, rowCount, x, y );
}
