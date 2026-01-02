# Improvement Notes

## Completed Improvements

### Thread Safety (High Priority)
- ✅ **Fixed race condition in loadTexture()**: Changed `resident` and `loading` flags from `bool` to `std::atomic<bool>`. Uses atomic compare-exchange for `loading` flag to prevent multiple threads from loading the same texture simultaneously.
- ✅ **Fixed async task lifetime issue**: Refactored `processRequestsAsync()` to use RAII guards for `inFlightAsync_` counter. Moved counter increment before `destroying_` check to prevent race with destructor. Tasks now check `destroying_` flag before processing.
- ✅ **Improved destructor synchronization**: Uses `seq_cst` memory ordering for `destroying_` flag to establish total ordering with async tasks.

### AMD GPU Optimizations (Medium Priority)
- ✅ **Added AMD wave size detection**: Added compile-time detection for wave32 (RDNA) vs wave64 (GCN) architectures via `__gfx10__`, `__gfx11__`, `__gfx12__` defines.
- ✅ **Native AMD intrinsics**: Added `getWaveSize()`, `getLaneId()`, and `getActiveMask()` helpers using `__builtin_amdgcn_*` intrinsics.
- ✅ **Improved overflow check**: Uses `__atomic_load_n` with `__ATOMIC_RELAXED` for overflow flag check.
- ✅ **Reduced redundant bounds checks**: Combined bounds check with residency check in `tex2D()` family of functions.

### Performance (Medium Priority)
- ✅ **Lock-free logging**: Replaced `std::mutex` with a spinlock using `std::atomic_flag`. Uses exponential backoff with `_mm_pause()` (MSVC/x86) or `__builtin_ia32_pause()` (GCC/Clang) to reduce cache-line contention.
- ✅ **Thread-local log buffer**: Formats messages into thread-local buffer before acquiring lock to minimize lock hold time.
- ✅ **Parallel texture loading**: Added `Internal/ThreadPool.h` with configurable worker count for concurrent I/O-bound image loading. Textures are loaded in parallel when multiple requests arrive in the same frame.
- ✅ **Pinned memory pooling**: Added `Internal/PinnedMemoryPool.h` to reuse `hipHostMalloc` buffers. Avoids expensive pinned memory allocation/deallocation per texture load.
- ✅ **HIP event pooling**: Added `Internal/HipEventPool.h` to reuse HIP events for async operations. Reduces ~5µs overhead per event create/destroy cycle.

### Eviction Policy (Medium Priority)
- ✅ **Priority hints**: Added `EvictionPriority` enum (`Low`, `Normal`, `High`, `KeepResident`) to `TextureDesc`. Low-priority textures are evicted first, `KeepResident` textures are never evicted.
- ✅ **Thrashing prevention**: Added `minResidentFrames` option (default: 3). Textures must be resident for this many frames before becoming eviction candidates, preventing rapid load-evict-reload cycles.
- ✅ **Improved eviction scoring**: Eviction now considers both priority and age (LRU within same priority tier).

### Code Quality (Low Priority)
- ✅ **Separated STB_IMAGE_IMPLEMENTATION**: Moved `#define STB_IMAGE_IMPLEMENTATION` to dedicated `stb_image_impl.cpp` file to prevent multiple definition issues when library is linked multiple times.
- ✅ **File organization**: Split `DemandTextureLoaderImpl.cpp` into modular components: `Internal/TextureMetadata.h`, `Internal/Utils.h`, `DemandTextureLoaderImpl.h`.

## Remaining Improvements

### Performance
- Consider block-level request aggregation using shared memory to further reduce global atomics.

### AMD-Specific
- Add support for BC7/DXT compressed textures which are highly efficient on AMD GPUs.
- Load pre-computed mipmaps from DDS/KTX formats instead of runtime generation.
- Consider using `hipMallocAsync`/`hipFreeAsync` with memory pools (HIP 5.3+) for texture allocations.
- Investigate Infinity Cache (L3) optimization for texture access patterns on RDNA2/3.

### Architecture
- Add per-stream DeviceContext for multi-stream rendering scenarios.
- Consider sparse/tiled texture support for very large textures (similar to OptiX reference).

### Testing
- Add unit tests for concurrent texture creation, eviction, and async/sync equivalence.
- Add stress tests for loader lifetime in multi-threaded scenarios.
