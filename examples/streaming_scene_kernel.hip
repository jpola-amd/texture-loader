// SPDX-License-Identifier: MIT
// Streaming scene kernel - samples visible tile textures

#include "DemandLoading/TextureSampling.h"
#include <hip/hip_runtime.h>

using namespace hip_demand;

// Tile info passed to kernel - use int for visible to ensure consistent struct layout
struct TileInfo {
    uint32_t textureId;
    float worldX;      // Tile center X in world space
    float worldZ;      // Tile center Z in world space
    int visible;       // Whether tile is visible this frame (int for cross-compiler compatibility)
};

// Camera info
struct CameraInfo {
    float posX, posY, posZ;     // Camera position
    float targetX, targetZ;     // Look-at point (for direction)
    float fov;                  // Field of view in radians
    float nearPlane, farPlane;  // View distances
};

// Sample visible tiles and render a top-down debug view
extern "C" __global__ void streamingSceneKernel(
    DeviceContext ctx,
    float4* output,
    int width, int height,
    const TileInfo* tiles,
    int numTilesX, int numTilesY,
    float tileSize,
    CameraInfo camera)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    
    // Map pixel to world position (top-down view of the tile grid)
    float worldExtent = numTilesX * tileSize;
    float worldX = (float(x) / width) * worldExtent;
    float worldZ = (float(y) / height) * worldExtent;
    
    // Find which tile this pixel is in
    int tileX = int(worldX / tileSize);
    int tileZ = int(worldZ / tileSize);
    
    // Clamp to valid range
    tileX = max(0, min(tileX, numTilesX - 1));
    tileZ = max(0, min(tileZ, numTilesY - 1));
    
    int tileIdx = tileZ * numTilesX + tileX;
    const TileInfo& tile = tiles[tileIdx];
    
    // UV within this tile
    float localX = fmodf(worldX, tileSize) / tileSize;
    float localZ = fmodf(worldZ, tileSize) / tileSize;
    
    float4 color;
    
    if (tile.visible) {
        // Sample the texture - this generates a request if not resident
        bool resident = tex2D(ctx, tile.textureId, localX, localZ, color);
        
        // Tint based on residency status
        if (!resident) {
            // Pink tint for loading
            color = make_float4(0.8f, 0.3f, 0.5f, 1.0f);
        }
    } else {
        // Gray for non-visible tiles (not requested)
        color = make_float4(0.2f, 0.2f, 0.2f, 1.0f);
    }
    
    // Draw tile grid lines
    float edgeDist = min(min(localX, 1.0f - localX), min(localZ, 1.0f - localZ));
    if (edgeDist < 0.02f) {
        color = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
    }
    
    // Draw camera position as a bright dot
    float camGridX = (camera.posX / worldExtent) * width;
    float camGridZ = (camera.posZ / worldExtent) * height;
    float distToCam = sqrtf((x - camGridX) * (x - camGridX) + (y - camGridZ) * (y - camGridZ));
    if (distToCam < 8.0f) {
        color = make_float4(1.0f, 1.0f, 0.0f, 1.0f);  // Yellow camera marker
    }
    
    // Draw view frustum approximation (simple circle for now)
    float viewRadius = camera.farPlane / worldExtent * width * 0.5f;
    if (fabsf(distToCam - viewRadius) < 2.0f) {
        color = make_float4(0.0f, 1.0f, 0.0f, 1.0f);  // Green view radius
    }
    
    output[idx] = color;
}
