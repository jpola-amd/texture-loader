// Path tracing kernel for HIP demand texture loading demonstration
// Demonstrates non-coherent texture access patterns typical of ray tracing

#include <hip/hip_runtime.h>
#include "DemandLoading/TextureSampling.h"
#include "DemandLoading/DeviceContext.h"

// Simple vector math
struct Vec3 {
    float x, y, z;
    
    __device__ Vec3() : x(0), y(0), z(0) {}
    __device__ Vec3(float v) : x(v), y(v), z(v) {}
    __device__ Vec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
    
    __device__ Vec3 operator+(const Vec3& b) const { return Vec3(x + b.x, y + b.y, z + b.z); }
    __device__ Vec3 operator-(const Vec3& b) const { return Vec3(x - b.x, y - b.y, z - b.z); }
    __device__ Vec3 operator*(float s) const { return Vec3(x * s, y * s, z * s); }
    __device__ Vec3 operator*(const Vec3& b) const { return Vec3(x * b.x, y * b.y, z * b.z); }
    __device__ Vec3& operator+=(const Vec3& b) { x += b.x; y += b.y; z += b.z; return *this; }
    __device__ Vec3 operator-() const { return Vec3(-x, -y, -z); }
};

__device__ float dot(const Vec3& a, const Vec3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__device__ Vec3 normalize(const Vec3& v) { float inv = rsqrtf(dot(v, v) + 1e-8f); return v * inv; }
__device__ Vec3 cross(const Vec3& a, const Vec3& b) {
    return Vec3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
__device__ float length(const Vec3& v) { return sqrtf(dot(v, v)); }
__device__ Vec3 reflect(const Vec3& v, const Vec3& n) { return v - n * (2.0f * dot(v, n)); }

// Simple LCG random number generator
__device__ uint32_t pcg_hash(uint32_t input) {
    uint32_t state = input * 747796405u + 2891336453u;
    uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

__device__ float randomFloat(uint32_t& seed) {
    seed = pcg_hash(seed);
    return static_cast<float>(seed) / static_cast<float>(0xFFFFFFFFu);
}

__device__ Vec3 randomInHemisphere(const Vec3& normal, uint32_t& seed) {
    float u1 = randomFloat(seed);
    float u2 = randomFloat(seed);
    
    float r = sqrtf(u1);
    float theta = 2.0f * 3.14159265f * u2;
    
    float x = r * cosf(theta);
    float y = r * sinf(theta);
    float z = sqrtf(1.0f - u1);
    
    // Create orthonormal basis
    Vec3 up = fabsf(normal.y) < 0.999f ? Vec3(0, 1, 0) : Vec3(1, 0, 0);
    Vec3 tangent = normalize(cross(up, normal));
    Vec3 bitangent = cross(normal, tangent);
    
    return normalize(tangent * x + bitangent * y + normal * z);
}

// Ray structure
struct Ray {
    Vec3 origin;
    Vec3 direction;
    
    __device__ Ray() {}
    __device__ Ray(const Vec3& o, const Vec3& d) : origin(o), direction(d) {}
    __device__ Vec3 at(float t) const { return origin + direction * t; }
};

// Hit information
struct HitInfo {
    float t;
    Vec3 point;
    Vec3 normal;
    float u, v;        // Texture coordinates
    int materialId;    // Which texture to use
    bool hit;
};

// Scene geometry: spheres and planes
struct Sphere {
    Vec3 center;
    float radius;
    int materialId;
};

struct Plane {
    Vec3 point;
    Vec3 normal;
    Vec3 uAxis, vAxis;  // For texture mapping
    float uvScale;
    int materialId;
};

// Intersect ray with sphere
__device__ bool intersectSphere(const Ray& ray, const Sphere& sphere, float tMin, float tMax, HitInfo& hit) {
    Vec3 oc = ray.origin - sphere.center;
    float a = dot(ray.direction, ray.direction);
    float half_b = dot(oc, ray.direction);
    float c = dot(oc, oc) - sphere.radius * sphere.radius;
    float discriminant = half_b * half_b - a * c;
    
    if (discriminant < 0) return false;
    
    float sqrtd = sqrtf(discriminant);
    float root = (-half_b - sqrtd) / a;
    if (root < tMin || root > tMax) {
        root = (-half_b + sqrtd) / a;
        if (root < tMin || root > tMax) return false;
    }
    
    hit.t = root;
    hit.point = ray.at(root);
    hit.normal = normalize(hit.point - sphere.center);
    hit.materialId = sphere.materialId;
    hit.hit = true;
    
    // Spherical UV mapping
    float theta = acosf(-hit.normal.y);
    float phi = atan2f(-hit.normal.z, hit.normal.x) + 3.14159265f;
    hit.u = phi / (2.0f * 3.14159265f);
    hit.v = theta / 3.14159265f;
    
    return true;
}

// Intersect ray with plane
__device__ bool intersectPlane(const Ray& ray, const Plane& plane, float tMin, float tMax, HitInfo& hit) {
    float denom = dot(plane.normal, ray.direction);
    if (fabsf(denom) < 1e-6f) return false;
    
    float t = dot(plane.point - ray.origin, plane.normal) / denom;
    if (t < tMin || t > tMax) return false;
    
    hit.t = t;
    hit.point = ray.at(t);
    hit.normal = plane.normal;
    if (denom > 0) hit.normal = -hit.normal;  // Face the ray
    hit.materialId = plane.materialId;
    hit.hit = true;
    
    // Planar UV mapping
    Vec3 local = hit.point - plane.point;
    hit.u = dot(local, plane.uAxis) * plane.uvScale;
    hit.v = dot(local, plane.vAxis) * plane.uvScale;
    
    return true;
}

// Scene constants
constexpr int NUM_SPHERES = 4;
constexpr int NUM_PLANES = 5;  // Floor + 4 walls

// Scene geometry in constant memory (no dynamic initialization)
__device__ __constant__ float g_sphereData[NUM_SPHERES * 5] = {
    // center.x, center.y, center.z, radius, materialId
    -1.5f, 0.8f, 0.0f, 0.8f, 0.0f,   // Left sphere
     1.5f, 0.8f, 0.0f, 0.8f, 1.0f,   // Right sphere  
     0.0f, 0.5f, 1.5f, 0.5f, 2.0f,   // Front sphere
     0.0f, 1.2f, -1.0f, 0.4f, 3.0f,  // Back sphere (emissive)
};

__device__ __constant__ float g_planeData[NUM_PLANES * 13] = {
    // point.x, point.y, point.z, normal.x, normal.y, normal.z, uAxis.x, uAxis.y, uAxis.z, vAxis.x, vAxis.y, vAxis.z, uvScale, materialId(as float)
    // Floor
    0.0f, 0.0f, 0.0f,  0.0f, 1.0f, 0.0f,  1.0f, 0.0f, 0.0f,  0.0f, 0.0f, 1.0f,  0.25f,
    // Back wall
    0.0f, 0.0f, -3.0f, 0.0f, 0.0f, 1.0f,  1.0f, 0.0f, 0.0f,  0.0f, 1.0f, 0.0f,  0.5f,
    // Left wall (red)
    -4.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,  0.0f, 0.0f, 1.0f,  0.0f, 1.0f, 0.0f,  0.5f,
    // Right wall (green)
    4.0f, 0.0f, 0.0f,  -1.0f, 0.0f, 0.0f, 0.0f, 0.0f, -1.0f, 0.0f, 1.0f, 0.0f,  0.5f,
    // Ceiling (light source area near center)
    0.0f, 4.0f, 0.0f,  0.0f, -1.0f, 0.0f, 1.0f, 0.0f, 0.0f,  0.0f, 0.0f, 1.0f,  0.25f,
};

// Material IDs for planes (stored separately to avoid float conversion issues)
__device__ __constant__ int g_planeMaterialIds[NUM_PLANES] = { 4, 5, 6, 7, 8 };

// Helper to get sphere from constant data
__device__ Sphere getSphere(int i) {
    Sphere s;
    int base = i * 5;
    s.center = Vec3(g_sphereData[base], g_sphereData[base + 1], g_sphereData[base + 2]);
    s.radius = g_sphereData[base + 3];
    s.materialId = static_cast<int>(g_sphereData[base + 4]);
    return s;
}

// Helper to get plane from constant data
__device__ Plane getPlane(int i) {
    Plane p;
    int base = i * 13;
    p.point = Vec3(g_planeData[base], g_planeData[base + 1], g_planeData[base + 2]);
    p.normal = Vec3(g_planeData[base + 3], g_planeData[base + 4], g_planeData[base + 5]);
    p.uAxis = Vec3(g_planeData[base + 6], g_planeData[base + 7], g_planeData[base + 8]);
    p.vAxis = Vec3(g_planeData[base + 9], g_planeData[base + 10], g_planeData[base + 11]);
    p.uvScale = g_planeData[base + 12];
    p.materialId = g_planeMaterialIds[i];
    return p;
}

// Trace ray through scene
__device__ HitInfo traceScene(const Ray& ray) {
    HitInfo closest;
    closest.hit = false;
    closest.t = 1e20f;
    
    HitInfo temp;
    
    // Test spheres
    for (int i = 0; i < NUM_SPHERES; ++i) {
        Sphere sphere = getSphere(i);
        if (intersectSphere(ray, sphere, 0.001f, closest.t, temp)) {
            closest = temp;
        }
    }
    
    // Test planes
    for (int i = 0; i < NUM_PLANES; ++i) {
        Plane plane = getPlane(i);
        if (intersectPlane(ray, plane, 0.001f, closest.t, temp)) {
            closest = temp;
        }
    }
    
    return closest;
}

// Sample texture with demand loading
__device__ Vec3 sampleMaterial(int materialId, float u, float v,
                                const uint32_t* textureIds, int numTextures,
                                hip_demand::DeviceContext& ctx, bool& requested) {
    // Map material to texture (cycle through available textures)
    int texIdx = materialId % numTextures;
    uint32_t texId = textureIds[texIdx];
    
    // Wrap UVs
    u = u - floorf(u);
    v = v - floorf(v);
    
    float4 sample;
    bool isResident = hip_demand::tex2DLod(ctx, texId, u, v, 0.0f, sample);
    requested = !isResident;
    
    // If texture not loaded, return placeholder color based on material
    if (requested) {
        // Return a distinguishable color while waiting for texture
        float hue = static_cast<float>(materialId) * 0.37f;
        return Vec3(0.3f + 0.2f * sinf(hue * 6.28f),
                    0.3f + 0.2f * sinf(hue * 6.28f + 2.09f),
                    0.3f + 0.2f * sinf(hue * 6.28f + 4.19f));
    }
    
    return Vec3(sample.x, sample.y, sample.z);
}

// Get emission for emissive materials
__device__ Vec3 getEmission(int materialId, const Vec3& point) {
    // Ceiling light (material 8) - area light
    if (materialId == 8) {
        // Only emit in center region
        if (fabsf(point.x) < 1.0f && fabsf(point.z) < 1.0f) {
            return Vec3(8.0f, 7.5f, 7.0f);  // Warm white light
        }
    }
    // Small emissive sphere (material 3)
    if (materialId == 3) {
        return Vec3(3.0f, 2.5f, 2.0f);
    }
    return Vec3(0.0f);
}

// Path trace a single ray
__device__ Vec3 pathTrace(Ray ray, const uint32_t* textureIds, int numTextures,
                          hip_demand::DeviceContext& ctx, uint32_t& seed,
                          int maxBounces, bool& anyRequested) {
    Vec3 throughput(1.0f);
    Vec3 radiance(0.0f);
    anyRequested = false;
    
    for (int bounce = 0; bounce < maxBounces; ++bounce) {
        HitInfo hit = traceScene(ray);
        
        if (!hit.hit) {
            // Sky/environment - subtle gradient
            float t = 0.5f * (ray.direction.y + 1.0f);
            Vec3 sky = Vec3(0.1f, 0.12f, 0.15f) * (1.0f - t) + Vec3(0.05f, 0.07f, 0.1f) * t;
            radiance += throughput * sky;
            break;
        }
        
        // Add emission
        Vec3 emission = getEmission(hit.materialId, hit.point);
        radiance += throughput * emission;
        
        // Sample material texture (this is where demand loading kicks in)
        bool requested = false;
        Vec3 albedo = sampleMaterial(hit.materialId, hit.u, hit.v, 
                                      textureIds, numTextures, ctx, requested);
        if (requested) anyRequested = true;
        
        // Russian roulette termination after a few bounces
        if (bounce > 2) {
            float p = fmaxf(albedo.x, fmaxf(albedo.y, albedo.z));
            if (randomFloat(seed) > p) break;
            throughput = throughput * (1.0f / (p + 0.001f));
        }
        
        // Update throughput
        throughput = throughput * albedo;
        
        // Generate new direction (diffuse BRDF - cosine weighted)
        Vec3 newDir = randomInHemisphere(hit.normal, seed);
        ray = Ray(hit.point + hit.normal * 0.001f, newDir);
    }
    
    return radiance;
}

// Camera
struct Camera {
    Vec3 origin;
    Vec3 lowerLeft;
    Vec3 horizontal;
    Vec3 vertical;
};

__device__ Ray getCameraRay(const Camera& cam, float u, float v) {
    Vec3 dir = normalize(cam.lowerLeft + cam.horizontal * u + cam.vertical * v - cam.origin);
    return Ray(cam.origin, dir);
}

// Main path tracing kernel
extern "C" __device__ void pathTraceKernel(
    float4* output,
    float4* accumulator,
    int width, int height,
    int sampleIndex,
    int samplesPerPixel,
    const uint32_t* textureIds,
    int numTextures,
    int maxBounces,
    float cameraAngle,
    hip_demand::DeviceContext ctx
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    
    // Setup camera - small orbit inside the Cornell box room
    // Room bounds: x=[-4,4], y=[0,4], z=[-3, front open]
    // Use a gentle sway instead of full orbit to stay well inside
    float swayX = 1.5f * sinf(cameraAngle);        // Sway left-right ±1.5
    float swayZ = 0.5f * cosf(cameraAngle * 0.5f); // Slight forward-back
    float camHeight = 1.8f;
    Vec3 camPos(swayX, camHeight, 3.5f + swayZ);   // Stay at front of room (z=3 to 4)
    Vec3 target(0.0f, 1.2f, -1.0f);                // Look at center/back of room
    Vec3 up(0.0f, 1.0f, 0.0f);
    
    Vec3 w = normalize(camPos - target);
    Vec3 u_cam = normalize(cross(up, w));
    Vec3 v_cam = cross(w, u_cam);
    
    float fov = 0.8f;  // ~45 degrees
    float aspect = static_cast<float>(width) / static_cast<float>(height);
    
    Camera cam;
    cam.origin = camPos;
    cam.horizontal = u_cam * (2.0f * fov * aspect);
    cam.vertical = v_cam * (2.0f * fov);
    cam.lowerLeft = camPos - w - cam.horizontal * 0.5f - cam.vertical * 0.5f;
    
    // Per-pixel random seed
    uint32_t seed = (idx + sampleIndex * width * height) * 1973 + sampleIndex * 9277;
    
    // Accumulate samples
    Vec3 color(0.0f);
    bool anyRequested = false;
    
    for (int s = 0; s < samplesPerPixel; ++s) {
        // Jittered sampling
        float jx = randomFloat(seed);
        float jy = randomFloat(seed);
        float px = (static_cast<float>(x) + jx) / static_cast<float>(width);
        float py = (static_cast<float>(y) + jy) / static_cast<float>(height);
        
        Ray ray = getCameraRay(cam, px, py);
        
        bool requested = false;
        Vec3 sample = pathTrace(ray, textureIds, numTextures, ctx, seed, maxBounces, requested);
        if (requested) anyRequested = true;
        
        color += sample;
    }
    
    color = color * (1.0f / static_cast<float>(samplesPerPixel));
    
    // For moving camera, don't accumulate across frames - just output current frame's result
    // (Progressive accumulation only makes sense for static camera)
    float4 result;
    result.x = color.x;
    result.y = color.y;
    result.z = color.z;
    result.w = 1.0f;
    
    accumulator[idx] = result;
    
    // Tone mapping (simple Reinhard) for display
    float4 display;
    display.x = result.x / (1.0f + result.x);
    display.y = result.y / (1.0f + result.y);
    display.z = result.z / (1.0f + result.z);
    display.w = anyRequested ? 0.5f : 1.0f;  // Mark if textures were requested
    
    output[idx] = display;
}

// Kernel wrapper for hipModuleLaunchKernel
extern "C" __global__ void pathTraceKernelWrapper(
    float4* output,
    float4* accumulator,
    int width, int height,
    int sampleIndex,
    int samplesPerPixel,
    const uint32_t* textureIds,
    int numTextures,
    int maxBounces,
    float cameraAngle,
    hip_demand::DeviceContext ctx
) {
    pathTraceKernel(output, accumulator, width, height, sampleIndex, samplesPerPixel,
                    textureIds, numTextures, maxBounces, cameraAngle, ctx);
}
