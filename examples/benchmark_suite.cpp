#include <hip/hip_runtime.h>
#include "DemandLoading/DemandTextureLoader.h"
#include "DemandLoading/Logging.h"
#include <iostream>
#include <chrono>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <filesystem>

using namespace hip_demand;
using namespace std::chrono;

static std::string resolveKernelPath(const char* argv0) {
    namespace fs = std::filesystem;

    // Best effort: locate the code object next to the executable.
    fs::path exePath(argv0 ? argv0 : "");
    fs::path exeDir = exePath.has_parent_path() ? exePath.parent_path() : fs::current_path();
    fs::path coPath = exeDir / "benchmark_kernel.co";
    if (fs::exists(coPath)) {
        return coPath.string();
    }

    // Fallback to CWD-relative behavior.
    return "benchmark_kernel.co";
}

// Simple kernel that just requests textures
extern "C" __global__ void benchmarkKernelWrapper(
    DeviceContext ctx,
    const uint32_t* textureIds,
    int numTextureIds,
    int numRequesters,
    int iterations);

struct BenchmarkResult {
    std::string name;
    double mean_us;
    double median_us;
    double min_us;
    double max_us;
    double stddev_us;
    
    void print() const {
        std::cout << "  " << name << ":\n";
        std::cout << "    Mean:   " << mean_us << " us\n";
        std::cout << "    Median: " << median_us << " us\n";
        std::cout << "    Min:    " << min_us << " us\n";
        std::cout << "    Max:    " << max_us << " us\n";
        std::cout << "    StdDev: " << stddev_us << " us\n";
    }
};

BenchmarkResult analyze(const std::string& name, const std::vector<double>& samples_us) {
    BenchmarkResult result;
    result.name = name;
    
    std::vector<double> sorted = samples_us;
    std::sort(sorted.begin(), sorted.end());
    
    result.min_us = sorted.front();
    result.max_us = sorted.back();
    result.median_us = sorted[sorted.size() / 2];
    result.mean_us = std::accumulate(sorted.begin(), sorted.end(), 0.0) / sorted.size();
    
    double variance = 0.0;
    for (double s : sorted) {
        variance += (s - result.mean_us) * (s - result.mean_us);
    }
    result.stddev_us = std::sqrt(variance / sorted.size());
    
    return result;
}

// Benchmark 1: launchPrepare() overhead
void benchmark_launch_prepare(DemandTextureLoader& loader, hipStream_t stream, int iterations) {
    std::cout << "\n=== Benchmark 1: launchPrepare() Overhead ===\n";
    std::cout << "Measuring incremental update cost after textures are resident...\n";
    
    std::vector<double> samples;
    samples.reserve(iterations);
    
    // Warmup
    for (int i = 0; i < 10; ++i) {
        loader.launchPrepare(stream);
        hipStreamSynchronize(stream);
    }
    
    // Measure
    for (int i = 0; i < iterations; ++i) {
        auto start = high_resolution_clock::now();
        loader.launchPrepare(stream);
        hipStreamSynchronize(stream);
        auto end = high_resolution_clock::now();
        
        samples.push_back(duration_cast<nanoseconds>(end - start).count() / 1000.0);
    }
    
    auto result = analyze("launchPrepare (incremental)", samples);
    result.print();
    
    // Baseline HIP overhead is ~60-80 us (hipMemsetAsync + driver + stream coordination)
    // When dirty-range tracking works, we copy 0 bytes but still pay this API cost.
    if (result.mean_us < 100.0) {
        std::cout << "  [PASS] Good! Baseline HIP overhead only (0 bytes copied when clean).\n";
    } else if (result.mean_us < 200.0) {
        std::cout << "  [WARN] Moderate overhead. Some data may be copied unnecessarily.\n";
    } else {
        std::cout << "  [FAIL] High overhead detected. Full array copy happening?\n";
    }
    std::cout << "  Note: Run with --debug to see actual bytes copied per frame.\n";
}

// Benchmark 2: Async overlap
void benchmark_async_overlap(DemandTextureLoader& loader, hipStream_t stream,
                              hipModule_t module, hipFunction_t kernel,
                              uint32_t* d_textureIds, int numTextures, int numRequesters) {
    std::cout << "\n=== Benchmark 2: Async Processing Overlap ===\n";
    std::cout << "Measuring whether background loading overlaps with CPU work...\n";
    
    const int cpuWorkMicros = 5000; // 5ms of artificial CPU work
    const int passes = 20;
    
    // Scenario A: Call processRequestsAsync and wait immediately (no overlap)
    auto start_no_overlap = high_resolution_clock::now();
    for (int pass = 0; pass < passes; ++pass) {
        loader.launchPrepare(stream);
        auto ctx = loader.getDeviceContext();
        
        int k_numTextureIds = numTextures;
        int k_numRequesters = numRequesters;
        int k_iterations = 100;
        void* args[] = { &ctx, &d_textureIds, &k_numTextureIds, &k_numRequesters, &k_iterations };
        
        dim3 blockSize(256);
        dim3 gridSize(64);
        
        hipModuleLaunchKernel(kernel, gridSize.x, gridSize.y, gridSize.z,
                              blockSize.x, blockSize.y, blockSize.z,
                              0, stream, args, nullptr);

        // No explicit stream sync needed here: processRequestsAsync records an event on
        // the render stream and ensures the copy/processing waits on it.
        auto ticket = loader.processRequestsAsync(stream, ctx);
        ticket.wait(); // Wait immediately - no CPU work
    }
    auto end_no_overlap = high_resolution_clock::now();
    double time_no_overlap_ms = duration_cast<microseconds>(end_no_overlap - start_no_overlap).count() / 1000.0;
    
    // Scenario B: Do CPU work between async call and wait (potential overlap)
    auto start_with_overlap = high_resolution_clock::now();
    for (int pass = 0; pass < passes; ++pass) {
        loader.launchPrepare(stream);
        auto ctx = loader.getDeviceContext();
        
        int k_numTextureIds = numTextures;
        int k_numRequesters = numRequesters;
        int k_iterations = 100;
        void* args[] = { &ctx, &d_textureIds, &k_numTextureIds, &k_numRequesters, &k_iterations };
        
        dim3 blockSize(256);
        dim3 gridSize(64);
        
        hipModuleLaunchKernel(kernel, gridSize.x, gridSize.y, gridSize.z,
                              blockSize.x, blockSize.y, blockSize.z,
                              0, stream, args, nullptr);

        // Let the async request pipeline wait on the stream dependency.
        auto ticket = loader.processRequestsAsync(stream, ctx);
        
        // Simulate CPU work (e.g., preparing next frame, UI updates, etc.)
        auto cpu_start = high_resolution_clock::now();
        volatile double sink = 0.0;
        while (duration_cast<microseconds>(high_resolution_clock::now() - cpu_start).count() < cpuWorkMicros) {
            sink += std::sin(sink + 1.0);
        }
        
        ticket.wait();
    }
    auto end_with_overlap = high_resolution_clock::now();
    double time_with_overlap_ms = duration_cast<microseconds>(end_with_overlap - start_with_overlap).count() / 1000.0;
    
    std::cout << "  Scenario A (wait immediately): " << time_no_overlap_ms << " ms\n";
    std::cout << "  Scenario B (CPU work first):   " << time_with_overlap_ms << " ms\n";
    std::cout << "  Expected CPU work overhead:    " << (passes * cpuWorkMicros / 1000.0) << " ms\n";
    
    double overlap_benefit = (time_no_overlap_ms + passes * cpuWorkMicros / 1000.0) - time_with_overlap_ms;
    const double cpu_work_total_ms = (passes * cpuWorkMicros) / 1000.0;
    const double max_possible_overlap_ms = std::min(cpu_work_total_ms, time_no_overlap_ms);
    const double overlap_ratio = (max_possible_overlap_ms > 0.0) ? (overlap_benefit / max_possible_overlap_ms) : 0.0;

    if (overlap_ratio > 0.85) {
        std::cout << "  [PASS] Strong overlap (~" << (overlap_ratio * 100.0) << "% of max possible, " << overlap_benefit << " ms).\n";
    } else if (overlap_ratio > 0.30) {
        std::cout << "  [WARN] Partial overlap (~" << (overlap_ratio * 100.0) << "% of max possible, " << overlap_benefit << " ms).\n";
    } else {
        std::cout << "  [WARN] Little overlap (~" << (overlap_ratio * 100.0) << "%). Check stream/event wiring.\n";
    }
}

// Benchmark 3: Texture load throughput
void benchmark_load_throughput(const std::string& kernelPath, int textureCount) {
    std::cout << "\n=== Benchmark 3: Texture Load Throughput ===\n";
    std::cout << "Loading " << textureCount << " textures and measuring time...\n";
    
    LoaderOptions opts;
    opts.maxTextures = textureCount + 100;
    opts.maxTextureMemory = 0; // Unlimited
    opts.enableEviction = false;
    
    DemandTextureLoader loader(opts);
    
    // Create procedural textures in memory
    const int texSize = 256;
    std::vector<uint8_t> pixels(texSize * texSize * 4);
    
    auto create_start = high_resolution_clock::now();
    std::vector<uint32_t> textureIds;
    for (int i = 0; i < textureCount; ++i) {
        // Generate unique pattern
        for (int y = 0; y < texSize; ++y) {
            for (int x = 0; x < texSize; ++x) {
                int idx = (y * texSize + x) * 4;
                pixels[idx + 0] = (i * 17 + x) % 256;
                pixels[idx + 1] = (i * 31 + y) % 256;
                pixels[idx + 2] = (i * 47 + x + y) % 256;
                pixels[idx + 3] = 255;
            }
        }
        
        TextureDesc desc;
        desc.generateMipmaps = true;
        auto handle = loader.createTextureFromMemory(pixels.data(), texSize, texSize, 4, desc);
        if (handle.valid) {
            textureIds.push_back(handle.id);
        }
    }
    auto create_end = high_resolution_clock::now();
    double create_time_ms = duration_cast<milliseconds>(create_end - create_start).count();
    
    std::cout << "  Texture creation: " << create_time_ms << " ms\n";
    
    // Now trigger loading by requesting all textures
    hipStream_t stream;
    hipStreamCreate(&stream);
    
    uint32_t* d_textureIds;
    hipMalloc(&d_textureIds, textureIds.size() * sizeof(uint32_t));
    hipMemcpy(d_textureIds, textureIds.data(), textureIds.size() * sizeof(uint32_t), hipMemcpyHostToDevice);
    
    // Load kernel module
    hipModule_t module;
    hipFunction_t kernel;
    if (hipModuleLoad(&module, kernelPath.c_str()) != hipSuccess) {
        std::cerr << "Failed to load kernel module\n";
        hipFree(d_textureIds);
        hipStreamDestroy(stream);
        return;
    }
    if (hipModuleGetFunction(&kernel, module, "benchmarkKernelWrapper") != hipSuccess) {
        std::cerr << "Failed to get kernel function\n";
        hipModuleUnload(module);
        hipFree(d_textureIds);
        hipStreamDestroy(stream);
        return;
    }
    
    auto load_start = high_resolution_clock::now();
    
    // First pass: trigger all texture loads
    loader.launchPrepare(stream);
    auto ctx = loader.getDeviceContext();
    
    int k_numTextureIds = static_cast<int>(textureIds.size());
    int k_numRequesters = static_cast<int>(textureIds.size());
    int k_iterations = 1;
    void* args[] = { &ctx, &d_textureIds, &k_numTextureIds, &k_numRequesters, &k_iterations };
    
    dim3 blockSize(256);
    dim3 gridSize((textureCount + blockSize.x - 1) / blockSize.x);
    
    hipModuleLaunchKernel(kernel, gridSize.x, gridSize.y, gridSize.z,
                          blockSize.x, blockSize.y, blockSize.z,
                          0, stream, args, nullptr);

    // processRequests() performs the required stream synchronization internally.
    size_t loaded = loader.processRequests(stream, ctx);
    
    auto load_end = high_resolution_clock::now();
    double load_time_ms = duration_cast<milliseconds>(load_end - load_start).count();
    
    std::cout << "  Textures loaded: " << loaded << " / " << textureCount << "\n";
    std::cout << "  Load time: " << load_time_ms << " ms\n";
    if (load_time_ms > 0.0) {
        std::cout << "  Throughput: " << (loaded / (load_time_ms / 1000.0)) << " textures/sec\n";
    } else {
        std::cout << "  Throughput: (n/a)\n";
    }
    if (loaded > 0) {
        std::cout << "  Average per texture: " << (load_time_ms / loaded) << " ms\n";
    } else {
        std::cout << "  Average per texture: (n/a)\n";
    }
    std::cout << "  Total GPU memory: " << (loader.getTotalTextureMemory() / (1024.0 * 1024.0)) << " MB\n";
    
    if (loaded / (load_time_ms / 1000.0) > 100) {
        std::cout << "  [PASS] Good throughput!\n";
    } else if (loaded / (load_time_ms / 1000.0) > 50) {
        std::cout << "  [WARN] Moderate throughput. Consider parallel loading.\n";
    } else {
        std::cout << "  [FAIL] Low throughput. Parallel loading would help significantly.\n";
    }
    
    hipFree(d_textureIds);
    hipModuleUnload(module);
    hipStreamDestroy(stream);
}

// Benchmark 4: Request processing overhead
void benchmark_request_processing(DemandTextureLoader& loader, hipStream_t stream,
                                   hipModule_t module, hipFunction_t kernel,
                                   uint32_t* d_textureIds, int numTextures, int numRequesters) {
    std::cout << "\n=== Benchmark 4: Request Processing Overhead ===\n";
    std::cout << "Measuring processRequests() vs processRequestsAsync() performance...\n";
    
    const int iterations = 100;
    
    // Sync version
    std::vector<double> sync_samples;
    for (int i = 0; i < iterations; ++i) {
        loader.launchPrepare(stream);
        auto ctx = loader.getDeviceContext();
        
        int k_numTextureIds = numTextures;
        int k_numRequesters = numRequesters;
        int k_iterations = 50;
        void* args[] = { &ctx, &d_textureIds, &k_numTextureIds, &k_numRequesters, &k_iterations };
        
        dim3 blockSize(256);
        dim3 gridSize(64);
        
        hipModuleLaunchKernel(kernel, gridSize.x, gridSize.y, gridSize.z,
                              blockSize.x, blockSize.y, blockSize.z,
                              0, stream, args, nullptr);

        // processRequests() performs a stream sync internally.
        auto start = high_resolution_clock::now();
        loader.processRequests(stream, ctx);
        auto end = high_resolution_clock::now();
        
        sync_samples.push_back(duration_cast<nanoseconds>(end - start).count() / 1000.0);
    }
    
    // Async version
    std::vector<double> async_samples;
    for (int i = 0; i < iterations; ++i) {
        loader.launchPrepare(stream);
        auto ctx = loader.getDeviceContext();
        
        int k_numTextureIds = numTextures;
        int k_numRequesters = numRequesters;
        int k_iterations = 50;
        void* args[] = { &ctx, &d_textureIds, &k_numTextureIds, &k_numRequesters, &k_iterations };
        
        dim3 blockSize(256);
        dim3 gridSize(64);
        
        hipModuleLaunchKernel(kernel, gridSize.x, gridSize.y, gridSize.z,
                              blockSize.x, blockSize.y, blockSize.z,
                              0, stream, args, nullptr);

        // processRequestsAsync() establishes stream dependencies via events.
        auto start = high_resolution_clock::now();
        auto ticket = loader.processRequestsAsync(stream, ctx);
        ticket.wait();
        auto end = high_resolution_clock::now();
        
        async_samples.push_back(duration_cast<nanoseconds>(end - start).count() / 1000.0);
    }
    
    auto sync_result = analyze("processRequests (sync)", sync_samples);
    auto async_result = analyze("processRequestsAsync (async)", async_samples);
    
    sync_result.print();
    async_result.print();
    
    double overhead = async_result.mean_us - sync_result.mean_us;
    std::cout << "  Async overhead: " << overhead << " us\n";
    
    if (overhead < 10.0) {
        std::cout << "  [PASS] Excellent! Minimal async overhead with worker thread.\n";
    } else if (overhead < 50.0) {
        std::cout << "  [WARN] Moderate overhead. Worker thread is helping.\n";
    } else {
        std::cout << "  [WARN] High overhead. Check worker thread implementation.\n";
    }
}

int main(int argc, char** argv) {
    // Parse command line: --debug enables debug logging
    bool enableDebug = false;
    int requesters = 256; // default request pressure for Bench 2/4
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--debug") {
            enableDebug = true;
        } else if (std::string(argv[i]) == "--requesters" && (i + 1) < argc) {
            requesters = std::max(1, std::atoi(argv[i + 1]));
            ++i;
        }
    }
    
    if (enableDebug) {
        setLogLevel(LogLevel::Debug);
        std::cout << "Debug logging enabled\n";
    } else {
        setLogLevel(LogLevel::Warn);
    }
    
    std::cout << "=================================================\n";
    std::cout << "  HIP Demand Texture Loader - Benchmark Suite\n";
    std::cout << "=================================================\n";
    
    // Initialize HIP
    int deviceCount = 0;
    hipGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        std::cerr << "No HIP devices found\n";
        return 1;
    }
    
    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, 0);
    std::cout << "\nDevice: " << prop.name << "\n";
    std::cout << "Compute units: " << prop.multiProcessorCount << "\n";
    std::cout << "Memory: " << (prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0)) << " GB\n";
    
    // Setup
    LoaderOptions opts;
    opts.maxTextures = 512;
    opts.maxTextureMemory = 512 * 1024 * 1024; // 512 MB
    opts.enableEviction = true;
    
    DemandTextureLoader loader(opts);

    const std::string kernelPath = resolveKernelPath((argc > 0) ? argv[0] : nullptr);
    // Keep requesters within the default maxRequestsPerLaunch (1024) unless the user has
    // explicitly built the loader with a higher cap.
    requesters = std::min(requesters, 1024);

    std::cout << "\nRun config: kernel=\"" << kernelPath << "\" requesters=" << requesters << "\n";
    
    // Create some textures
    std::vector<uint32_t> textureIds;
    const int numTextures = 64;
    const int texSize = 256;
    std::vector<uint8_t> pixels(texSize * texSize * 4);
    
    for (int i = 0; i < numTextures; ++i) {
        for (int y = 0; y < texSize; ++y) {
            for (int x = 0; x < texSize; ++x) {
                int idx = (y * texSize + x) * 4;
                pixels[idx + 0] = (i * 17 + x) % 256;
                pixels[idx + 1] = (i * 31 + y) % 256;
                pixels[idx + 2] = (i * 47) % 256;
                pixels[idx + 3] = 255;
            }
        }
        
        auto handle = loader.createTextureFromMemory(pixels.data(), texSize, texSize, 4);
        if (handle.valid) {
            textureIds.push_back(handle.id);
        }
    }
    
    hipStream_t stream;
    hipStreamCreate(&stream);
    
    uint32_t* d_textureIds;
    hipMalloc(&d_textureIds, textureIds.size() * sizeof(uint32_t));
    hipMemcpy(d_textureIds, textureIds.data(), textureIds.size() * sizeof(uint32_t), hipMemcpyHostToDevice);
    
    // Load kernel
    hipModule_t module;
    hipFunction_t kernel;
    if (hipModuleLoad(&module, kernelPath.c_str()) != hipSuccess) {
        std::cerr << "Failed to load " << kernelPath << "\n";
        hipFree(d_textureIds);
        hipStreamDestroy(stream);
        return 1;
    }
    if (hipModuleGetFunction(&kernel, module, "benchmarkKernelWrapper") != hipSuccess) {
        std::cerr << "Failed to get kernel function\n";
        hipModuleUnload(module);
        hipFree(d_textureIds);
        hipStreamDestroy(stream);
        return 1;
    }
    
    // Initial pass to load textures
    loader.launchPrepare(stream);
    auto ctx = loader.getDeviceContext();
    int k_numTextureIds = numTextures;
    int k_numRequesters = numTextures;
    int k_iterations = 1;
    void* args[] = { &ctx, &d_textureIds, &k_numTextureIds, &k_numRequesters, &k_iterations };
    dim3 blockSize(256);
    dim3 gridSize(64);
    hipModuleLaunchKernel(kernel, gridSize.x, gridSize.y, gridSize.z,
                          blockSize.x, blockSize.y, blockSize.z,
                          0, stream, args, nullptr);
    loader.processRequests(stream, ctx);
    
    // Run benchmarks
    benchmark_launch_prepare(loader, stream, 1000);
    benchmark_async_overlap(loader, stream, module, kernel, d_textureIds, numTextures, requesters);
    benchmark_request_processing(loader, stream, module, kernel, d_textureIds, numTextures, requesters);
    benchmark_load_throughput(kernelPath, 500);
    
    // Cleanup
    hipFree(d_textureIds);
    hipModuleUnload(module);
    hipStreamDestroy(stream);
    
    std::cout << "\n=================================================\n";
    std::cout << "  Benchmark Suite Complete\n";
    std::cout << "=================================================\n";
    
    return 0;
}
