# Build Instructions

Complete guide for building the HIP Demand Texture Loader on Windows and Linux.

The root [CMakeLists.txt](CMakeLists.txt) configures dependencies and the loader
library. Optional targets are defined in
[examples/CMakeLists.txt](examples/CMakeLists.txt) and
[tests/CMakeLists.txt](tests/CMakeLists.txt). Continue configuring from the
repository root; build commands, executable locations and top-level CTest
discovery are unchanged.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Windows Build](#windows-build-visual-studio-2022)
- [Linux Build](#linux-build)
- [HIP Module API](#hip-module-api)
- [GPU Architecture](#gpu-architecture-configuration)
- [Troubleshooting](#troubleshooting)

## Prerequisites

### All Platforms

- **ROCm/HIP**: Version 6.0+ (tested with 6.4)
- **CMake**: Version 3.21+
- **Git** and internet access for the first dependency restore
- **C++17 Compiler**:
  - Windows: Visual Studio 2022
  - Linux: GCC 9+ or Clang 10+

### Automatic dependencies

CMake uses **vcpkg manifest mode by default** for standalone builds on Windows
and Linux. No separate clone, bootstrap, `vcpkg install`, or global
`vcpkg integrate install` step is needed.

| Dependency | Downloaded when |
|------------|-----------------|
| stb (image loading and example image writing) | Always |
| OpenImageIO and its transitive dependencies | `USE_OIIO=ON` |
| GoogleTest | `BUILD_TESTS=ON` |

`USE_OIIO` remains **OFF** by default. Enabling it adds EXR, TIFF and other
OpenImageIO formats; it does not remove stb. Unneeded OIIO tools, viewers,
Python bindings and optional codecs are not enabled by the manifest.

The `builtin-baseline` in [vcpkg.json](vcpkg.json) pins dependency versions
and the automatically downloaded vcpkg checkout. Checkouts are cached under
`external/vcpkg/<baseline>`; installed packages belong to each build directory
under `vcpkg_installed`. vcpkg's normal binary/download caches can be reused.
The first OIIO restore builds a substantial dependency tree and may take a while.

An existing vcpkg can be selected with the `VCPKG_ROOT` environment variable or
`CMAKE_TOOLCHAIN_FILE`. Its checkout is not modified by this project. To use
another compiler toolchain with vcpkg, pass `VCPKG_CHAINLOAD_TOOLCHAIN_FILE`.
Start a new build directory when changing toolchains, triplets, or dependency
providers. For offline builds, prepopulate vcpkg's download/binary caches and
provide an existing vcpkg checkout.

When updating `builtin-baseline`, configure a fresh build directory to select
the new pinned checkout as well as the new dependency versions.

ROCm/HIP, the GPU driver, and the host compiler are **not** installed by vcpkg.
Linux also needs the standard native build utilities (for example, on Debian/
Ubuntu: `build-essential git curl zip unzip tar pkg-config`).

Use `-DUSE_VCPKG=OFF` for system/custom packages or an embedding parent project.
When included via `add_subdirectory`, dependency management defaults to OFF;
the parent must select its toolchain before its own `project()` call.

### Supported GPUs

- RDNA2 (gfx1030): RX 6000 series
- RDNA3 (gfx1100): RX 7000 series
- CDNA2 (gfx90a): MI200 series
- Vega (gfx900/906): RX Vega, Radeon VII

## Windows Build (Visual Studio 2022)

### Method 1: Using vcpkg (Recommended)

Configure once and build either configuration with matching dependencies:

```powershell
$env:HIP_PATH = "C:\Program Files\AMD\ROCm\6.4" # Or your supported SDK
$env:PATH = "$env:HIP_PATH\bin;$env:PATH"
cmake -B build\oiio -S . `
      -G "Visual Studio 17 2022" -A x64 `
      -DBUILD_EXAMPLES=ON `
      -DBUILD_TESTS=ON `
      -DUSE_OIIO=ON

cmake --build build\oiio --config Release --parallel
cmake --build build\oiio --config Debug --parallel
ctest --test-dir build\oiio -C Release --output-on-failure
ctest --test-dir build\oiio -C Debug --output-on-failure

.\build\oiio\Release\texture_loader_example.exe
```

The standard `x64-windows` triplet provides both Release and Debug libraries.
CMake selects the correct imported libraries and `/MD` (Release) or `/MDd`
(Debug) CRT automatically. Do not point Debug at Release-only OIIO libraries or
use HART's release-only triplet: STL/CRT ABIs must match. vcpkg copies dependency
DLLs next to build-tree binaries.
The build also copies the HIP runtime (`amdhip64_<major>.dll` or `amdhip64.dll`)
and matching COMGR DLL from the configured `HIP_PATH/bin` next to the loader,
examples and tests. These two DLLs are also installed into `bin`. Application-local
copies take precedence over a different HIP runtime supplied by the graphics
driver in Windows `System32`; adding the SDK to `PATH` alone does not guarantee
that the intended runtime is loaded. Other SDK support files and third-party
dependencies still need to be available when deploying the installation.

### Method 2: Basic Build (stb_image only)

For basic image format support (PNG, JPEG, BMP, TGA, HDR):

```powershell
# 1. Set HIP_PATH
$env:HIP_PATH = "C:\Program Files\AMD\ROCm\6.4"

# 2. Configure (stb is restored automatically; OIIO is not downloaded)
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -DBUILD_EXAMPLES=ON
cmake --build build --config Release
cmake --build build --config Debug

# 3. Run
.\build\Release\texture_loader_example.exe
```

### Custom HIP Path

If using a custom ROCm installation:

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -DHIP_PATH="E:\Custom\Path\To\hip\win64"
```

### Building with Examples

Examples are disabled by default. To enable:

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -DBUILD_EXAMPLES=ON
```

### Method 3: Custom OpenImageIO Build

Set `USE_VCPKG=OFF` for existing OpenImageIO installations. The sample script
sets this option and shows explicit dependency paths. Alternatively, pass
`CMAKE_PREFIX_PATH` containing your package installation prefixes. All libraries
must provide compatible Debug and Release configurations.

If you have OpenImageIO built from source or installed elsewhere:

1. Copy the example configuration script:
   ```powershell
   Copy-Item cmake_configure_vs17_oiio.cmd.example cmake_configure_vs17.cmd
   ```

2. Edit `cmake_configure_vs17.cmd` and update all `<PATH_TO_*>` placeholders with your actual installation paths

3. Run the configuration script:
   ```powershell
   .\cmake_configure_vs17.cmd
   cmake --build build --config Release
   ```

The example script includes:
- Detailed path configuration for OpenImageIO and all dependencies
- Multiple installation examples (vcpkg, custom builds, Conan)
- Complete troubleshooting guide
- Dependency overview

**Required Dependencies** (when building OIIO from source):
- Imath 3.1+
- OpenEXR 3.1+
- libtiff 4.0+
- libpng 1.6+
- libjpeg or libjpeg-turbo
- zlib 1.2+

## Linux Build

### Automatic vcpkg Build (Recommended)

```bash
# Install a supported ROCm SDK and host build utilities first.
export HIP_PATH=/opt/rocm
cmake -S . -B build/release -DCMAKE_BUILD_TYPE=Release \
      -DUSE_OIIO=ON -DBUILD_TESTS=ON -DBUILD_EXAMPLES=ON
cmake --build build/release --parallel
ctest --test-dir build/release --output-on-failure

cmake -S . -B build/debug -DCMAKE_BUILD_TYPE=Debug \
      -DUSE_OIIO=ON -DBUILD_TESTS=ON -DBUILD_EXAMPLES=ON
cmake --build build/debug --parallel
ctest --test-dir build/debug --output-on-failure
```

For stb-only builds, omit `-DUSE_OIIO=ON`; no OIIO dependency is restored.
The standard `x64-linux` triplet provides both dependency configurations and
position-independent static libraries suitable for the shared loader. Other
supported target triplets can be selected with `VCPKG_TARGET_TRIPLET`.

### Using System Packages Instead

```bash
sudo apt install libopenimageio-dev libstb-dev libgtest-dev
cmake -S . -B build/system -DCMAKE_BUILD_TYPE=Release \
      -DUSE_VCPKG=OFF -DUSE_OIIO=ON -DBUILD_TESTS=ON \
      -DSTB_INCLUDE_DIR=/usr/include/stb
cmake --build build/system --parallel
```

With vcpkg disabled, stb defaults to the bundled `external/stb` headers unless
overridden. Tests use an installed GTest package if available, otherwise the
existing FetchContent fallback downloads GoogleTest.

## HIP Module API

For a HIP-only cross-platform mip-filtering comparison, use the
[standalone LOD reproducer](examples/lod_repro/README.md). Configure its
directory directly; it does not build the loader or download its dependencies.
It prints expected versus measured mip blends and distinguishes numerical
differences from unsupported/setup failures.

### Why Module API?

Visual Studio 2022 doesn't support HIP language in CMake. We use Module API to:
- Compile device code separately with `hipcc`
- Load compiled kernels at runtime
- Support any CMake generator

### Build Process

```
.hip source → hipcc → .co (code object) → Runtime loading
```

### Kernel Requirements

Kernels must use `extern "C"`:

```cpp
// Correct
extern "C" __global__ void myKernel(int* data) { }

// Wrong - name mangling
__global__ void myKernel(int* data) { }
```

### Runtime Loading

```cpp
hipModule_t module;
hipModuleLoad(&module, "kernel.co");
hipFunction_t kernel;
hipModuleGetFunction(&kernel, module, "myKernel");
hipModuleLaunchKernel(kernel, ...);
```

## GPU Architecture Configuration

### Find Your Architecture

```bash
# Linux
rocminfo | grep gfx

# Windows
& "$env:HIP_PATH\bin\rocminfo.exe" | Select-String "gfx"
```

### Architecture Table

| GPU | Architecture | CMake Value |
|-----|-------------|-------------|
| RX 6000 series | RDNA2 | gfx1030 |
| RX 7000 series | RDNA3 | gfx1100 |
| RX Vega | Vega | gfx900 |
| Radeon VII | Vega 20 | gfx906 |
| MI100 | CDNA | gfx908 |
| MI200 | CDNA2 | gfx90a |

### Set Architecture

In `CMakeLists.txt`:

```cmake
hip_add_executable(
    TARGET render_kernel
    SOURCES examples/simple_render_kernel.hip
    ARCHITECTURES gfx1030 gfx1100  # Your GPU(s)
    OPTIONS -O3 --std=c++17
    INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/include
)
```

## Unit Tests

The release-specific [handoff guide](HANDOFF.md) describes manifest verification,
installed-header consumers, runtime dependency staging and relocation checks.
Installed fixtures accept `HDT_TEST_FILE_ROOT` (absolute, test-owned); the default
is `test-files` beside the executable. For packaged modules set
`HIP_DEMAND_TEST_KERNEL_DIR` to package `bin`, and use `HDT_TEST_LOADER_LIBRARY`
for an explicit ABI-test library path. None requires the old build directory.

### Cubic and smart-bicubic v1

See [CUBIC.md](CUBIC.md) for the additive host/device API, normalized-coordinate
derivatives, frozen numerical bounds and strict demand contract. Enable cubic
before residency. `image_data_tests` contains the GPU-independent double oracle
and ABI tests; `texture_loader_tests` stages `cubic_kernel.co` using the same HIP
module mechanism as the existing sampling tests.

```powershell
cmake --build build --config Release --target texture_loader_tests image_data_tests shared_contract_tests
ctest --test-dir build -C Release -N -R "Cubic"
ctest --test-dir build -C Release -R "Cubic" --output-on-failure
```

Do not exclude `CubicNativeOptOut` or `NativeDependentSmart` from the qualification
run. They retain native anisotropy creation and native mip-filtering failures as
failed tests, not successful skips. Pure cubic/override success and native
qualification are separate results. Existing tests retaining known-native skips
have not had their numerical thresholds changed.

Native anisotropy defaults to an observable zero-submission override.
`HDT_DISABLE_TEXTURE_ANISO_OVERRIDE=1` alone disables it; unset, `0`, empty or
any other string retains the override. Requested sampler identity and effective
mathematical cubic A are unchanged. Set the environment before loading and do
not mutate it concurrently with loader operations. `RequireQualified` continues
to reject unqualified native behavior. On Windows, set
`HDT_TEST_MODULE_IDENTITY=1` to log actual loader/HIP/COMGR module paths; hash
those files together with the printed cubic code-object path for runtime proof.

The project includes a comprehensive test suite using Google Test.

### Building Tests

```powershell
# Windows
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -DBUILD_TESTS=ON
cmake --build build --config Release --target texture_loader_tests image_data_tests
cmake --build build --config Debug --target texture_loader_tests image_data_tests
```

```bash
# Linux
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/opt/rocm -DBUILD_TESTS=ON
cmake --build build -j$(nproc)
```

**Note**: Configuring with tests enabled restores GoogleTest through vcpkg.

### CTest Discovery

Individual GoogleTest cases, including parameterized cases, are registered with
CTest automatically after the test executables are built. The
`vcpkg_bootstrap_config` CMake regression test is registered during configuration.

```powershell
ctest --test-dir build -C Release -N
ctest --test-dir build -C Debug -N
```

With Visual Studio, specify `-C Release` or `-C Debug` and build that configuration
first. A `texture_loader_tests_NOT_BUILT` or `image_data_tests_NOT_BUILT` entry
means the corresponding test target still needs to be built; configuring alone
does not enumerate its GoogleTest cases. Use the build directory, not the source
directory, when invoking CTest.

### Running Tests

```powershell
# Windows - Run all tests
ctest --test-dir build -C Release --output-on-failure

# Run with verbose output
ctest --test-dir build -C Release -V

# Run specific test suite
ctest --test-dir build -C Release -R TextureInfo

# List available tests
ctest --test-dir build -C Release -N
```

```bash
# Linux - Run all tests
ctest --test-dir build --output-on-failure

# Or run the test executable directly
./build/texture_loader_tests
```

### Test Coverage

For registration and real sampling validation, use a separate test-enabled
build with `USE_OIIO=ON` and explicit `HIP_ARCHITECTURES` for the target device.
`image_data_tests` remains GPU-independent. Loader/sampling fixtures accept
`HIP_DEMAND_TEST_DEVICE` (default zero); an unavailable device is a test failure,
not a successful skip.

The sampling code object is built and staged beside `texture_loader_tests`.
It is also installed with the test executables through the `tests` install
component. `HIP_DEMAND_TEST_KERNEL_DIR` can override code-object discovery;
an invalid override fails rather than loading a stale build-tree module.
Installed tests still require their GoogleTest/OIIO dependencies in the
platform's supported library search path.

```powershell
ctest --test-dir build -C Release -N -R "^Registration|/Registration|SamplingTestUtils|LegacyTextureSamplingTest|SamplingHarnessDeviceTest"
ctest --test-dir build -C Release -R "^Registration|/Registration|SamplingTestUtils|LegacyTextureSamplingTest|SamplingHarnessDeviceTest" --output-on-failure
```

The [item 01 evidence record](docs/implementation-plan/01-tests-and-registration-errors.md)
documents the current Windows gfx1201 results. The two authored-mip tests run
all checks, then report **skipped: expected incorrect behavior** only if their
numerical differences match the known upstream mip-blend defect on that target.
Different errors still fail; a corrected runtime passes normally. Expected
pixels and tolerances are unchanged, and neither these skips nor descriptor
readback count as filtering qualification. The standalone reproducer remains
strict and still returns a numerical-difference exit code. Private fault seams
are compiled only with `BUILD_TESTS=ON`; production calls are real by default.

For generated byte-mip rounding and preservation, use the same test-enabled
build and explicit device architecture:

```text
ctest --test-dir build -C Release -N -R "ImageData|SamplingTestUtils|MipRoundingUploadTest|GeneratedByteRoundingKeeps"
ctest --test-dir build -C Release -R "ImageData|SamplingTestUtils|MipRoundingUploadTest|GeneratedByteRoundingKeeps" --output-on-failure
```

Expected discovery is 82 tests (62 GPU-independent host tests and 20 device
tests). See the [item 03 evidence record](docs/implementation-plan/03-byte-mip-rounding.md)
for exact half-up semantics, sRGB/float preservation, runtime identity, and
independent-platform qualification requirements. Base-only capability fallback
does not pass the new multilevel preservation tests.

The test suite covers:

| Test Suite | Description |
|------------|-------------|
| `TextureInfoTests` | `TextureInfo` struct, `getBytesPerChannel()`, `getTextureSizeInBytes()` |
| `DemandTextureLoaderTests` | Loader creation, texture creation, device context, statistics, abort |
| `ThreadPoolTests` | Thread pool construction, task execution, concurrency, shutdown |
| `MemoryPoolTests` | `PinnedMemoryPool` and `HipEventPool` allocation and reuse |
| `TicketTests` | Async ticket construction and wait behavior |
| `RegistrationErrorTests` | Invalid identities, all-overload capacity boundaries, transaction rollback, failure injection, and device non-aliasing |
| `SamplingTestUtilsTests` | GPU-independent authored mip/boundary fixtures and device/module selection |
| `TextureSamplingTests` | Real HIP implicit/LOD/gradient sampling, native comparison, and fixture cleanup |

### Test Requirements

- **GPU Required**: Most tests require a HIP-compatible AMD GPU
- **ROCm/HIP**: Must be properly installed and configured
- **Internet**: First configure restores enabled dependencies, unless cached

### Combining Build Options

```powershell
# Build everything: library, examples, and tests
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 `
      -DBUILD_EXAMPLES=ON `
      -DBUILD_TESTS=ON `
      -DUSE_OIIO=ON
cmake --build build --config Release

# Run tests
ctest --test-dir build -C Release --output-on-failure
```

## Troubleshooting

### GPU tests hang during loader construction

Rebuild the test targets so the selected SDK's HIP runtime and COMGR DLLs are
copied next to the executables. A mismatched driver/runtime/compiler combination
can hang in `hipStreamCreateWithFlags`; HIP tracing may report missing ROCm
device libraries or failure to create internal blit kernels.

```powershell
cmake --build build --config Release --target texture_loader_tests image_data_tests
ctest --test-dir build -C Release -R '^HipTestFixture\.DefaultConstruction$' `
      --timeout 30 --output-on-failure
```

For diagnostics, set `$env:AMD_LOG_LEVEL = "4"` and `$env:AMD_LOG_MASK = "1"`
before running CTest with `-V` to see HIP API calls. These settings are not
required for normal test execution.

### "HIP not found"

**Solution**:
```powershell
# Windows - set HIP_PATH
$env:HIP_PATH = "C:\Program Files\AMD\ROCm\6.4"

# Linux - check installation
ls /opt/rocm
```

### "amd_comgr not found"

**Solution**: FindHIP.cmake automatically tries multiple library names (amd_comgr_2, amd_comgr0604, amd_comgr). If still failing:

```powershell
# Check what's available
ls "$env:HIP_PATH\lib" | Select-String comgr
```

### "stb_image.h not found"

**Solution**: Check the vcpkg restore log in the build directory and configure
again. If using `USE_VCPKG=OFF`, set `STB_INCLUDE_DIR` to a directory containing
`stb_image.h` (and `stb_image_write.h` when building examples). A stale
`STB_INCLUDE_DIR` cache entry may need to be removed or updated when switching
providers; use a fresh build directory.

### "Failed to load HIP module"

**Solution**: Check .co file exists:
```powershell
ls build\Release\render_kernel.co
```

### "Failed to get kernel function"

**Solution**:
1. Verify kernel has `extern "C"`
2. Check name matches exactly (case-sensitive)
3. List symbols: `llvm-nm render_kernel.co`

### "Request buffer overflow"

**Solution**: Increase buffer size:
```cpp
options.maxRequestsPerLaunch = renderWidth * renderHeight;
```

### Slow Loading

**Optimize**:
```cpp
desc.generateMipmaps = false;  // Disable if not needed
desc.maxMipLevel = 4;          // Limit mip levels
```

## Build Options

### CMake Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HIP_PATH` | Auto | Path to HIP installation |
| `USE_VCPKG` | ON (standalone) | Automatically provision vcpkg and restore dependencies |
| `STB_INCLUDE_DIR` | Auto with vcpkg; otherwise `external/stb` | Path to stb headers |
| `BUILD_EXAMPLES` | OFF | Build example applications |
| `BUILD_TESTS` | OFF | Build unit tests (restores GoogleTest) |
| `USE_OIIO` | OFF | Enable OpenImageIO support |
| `CMAKE_BUILD_TYPE` | Generator default | Set Release or Debug for single-config generators |
| `VCPKG_TARGET_TRIPLET` | vcpkg platform default | Target architecture/linkage; normally `x64-windows` or `x64-linux` |

### Compiler Flags

```cmake
# Optimize for speed
OPTIONS -O3 --std=c++17 -ffast-math

# Debug with symbols
OPTIONS -O0 -g --std=c++17
```

### Clean Build

```powershell
# Windows
Remove-Item -Recurse -Force build
cmake -S . -B build -G "Visual Studio 17 2022" -A x64

# Linux
rm -rf build && cmake -S . -B build
```

## Advanced

### Multiple Architectures

```cmake
ARCHITECTURES gfx900 gfx906 gfx1030 gfx1100 gfx90a
```

Creates "fat binary" supporting multiple GPUs.

### Verbose Build

```powershell
cmake -S . -B build --trace-expand
cmake --build build --config Release --verbose
```

### Verify Installation

```powershell
& "$env:HIP_PATH\bin\hipcc.bat" --version
& "$env:HIP_PATH\bin\rocminfo.exe"
```

## Getting Help

Include in bug reports:
- ROCm version
- GPU model (from `rocminfo`)
- CMake version
- Full error message

See [README.md](README.md) for API reference and usage examples.
