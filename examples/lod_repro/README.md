# Native HIP fractional-LOD reproducer

This small program compares the GPU's mip blending with the requested blend.
It uses **HIP only**: no texture-loader library, GoogleTest, OIIO, or image files.
It does not change driver settings or modify private GPU descriptors.

There are two authored FLOAT RGBA mip levels:

- Level 0: an 8x8 black image.
- Level 1: a 4x4 white image.

At LOD **0.25**, the expected RGB value is therefore **0.25**. On the affected
Windows stack, it is **0.1875** instead. Alpha should stay one.
The program first checks both uploaded levels by exact readback, then prints
explicit-LOD and isotropic-gradient samples from LOD 0 through 1.

## Requirements

- An AMD GPU and a HIP SDK/runtime supporting that GPU.
- CMake 3.21+ and a C++17 host compiler.
- Windows: Visual Studio 2022 C++ build tools.
- Linux: a supported GCC/Clang host compiler and ROCm/HIP installation.

Use **the same GPU architecture on both systems** if possible, to isolate the
OS/runtime difference. Otherwise record the GPU difference too.

The source is compiled twice: native host code, and a HIP device code object.
This follows the repository's module-loading convention and works with Visual
Studio without requiring CMake's native HIP language support.

## Windows (PowerShell)

Open PowerShell in this directory (`examples\lod_repro` in the repository,
or the `hip-lod-reproducer` directory extracted from the source archive):

```powershell
$env:HIP_PATH = "C:\path\to\your\HIP-SDK"
$env:PATH = "$env:HIP_PATH\bin;$env:PATH"

cmake -S . -B build `
  -G "Visual Studio 17 2022" -A x64 `
  "-DHIP_PATH=$env:HIP_PATH" -DHIP_ARCHITECTURES=gfx1201
cmake --build build --config Release

.\build\Release\lod_repro.exe `
  .\build\Release\lod_repro_kernel.co 0 |
  Tee-Object windows-lod.txt
$LASTEXITCODE
```

CMake copies the HIP runtime and matching COMGR DLL from the chosen SDK beside
the executable. This avoids silently loading a different HIP runtime from
Windows System32. Preserve those DLLs when moving the executable.

## Linux (bash)

Open a shell in this directory (`examples/lod_repro` in the repository,
or `hip-lod-reproducer` extracted from the source archive):

```bash
export HIP_PATH=/opt/rocm
export PATH="$HIP_PATH/bin:$PATH"

cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DHIP_PATH="$HIP_PATH" -DHIP_ARCHITECTURES=gfx1201
cmake --build build

set -o pipefail
./build/lod_repro ./build/lod_repro_kernel.co 0 |
  tee linux-lod.txt
echo "exit: $?"
```

Replace `gfx1201` with the actual target, for example `gfx1100` or `gfx1151`.
Use the same replacement on Windows. The final argument selects the device
ordinal; zero is the default if omitted.

## Reading the output

```text
LOD,expected,explicit_LOD,gradient,explicit_error,gradient_error
...
0.250000,0.250000,0.187500,0.187500,0.062500,0.062500
...
```

- If both systems produce `0.187500`, both reproduce the discrepancy.
- If one produces `0.250000` and the other `0.187500`, their filtering differs.
- Compare the complete table, GPU architecture, and runtime versions, not only
  one row. The maximum error checks all RGBA components, not only displayed red.

Exit codes:

| Code | Meaning |
|---:|---|
| 0 | Samples match analytical blending within the stated tolerances |
| 1 | Sampling succeeded but the numerical result differs; this is the expected reproducer result on the affected stack |
| 2 | Setup, upload, module, unsupported operation, or cleanup failed; **not** a sampling comparison |

Explicit-LOD tolerance is `1e-6`. Gradient tolerance is `1e-6 + 1/128`,
allowing the existing harness's gradient/weight precision allowance while still
detecting the observed optimization. These thresholds are fixed, not tuned to
the observed result.

The HIP driver API version printed by the program is not the Windows display
driver package version. Also record the display-driver version (Windows) or
kernel/driver version (Linux), plus `hipconfig --full`, when sharing results.

## Using the standalone source archive

The provided source archive contains this directory plus the existing
repository `FindHIP.cmake` helper in `cmake/`, and the repository license.
After extracting it, enter the `hip-lod-reproducer` directory and run the
commands above. It does not need a clone of texture-loader and does not
download any dependencies.

If assembling an archive yourself, include:

```text
CMakeLists.txt
lod_repro.cpp
README.md
LICENSE
cmake/FindHIP.cmake
```

Compile on each machine with its own supported HIP SDK. A Windows executable
is not a Linux executable, and compiling for a GPU does not prove runtime
support on a different GPU.

## Local verification

On 15 September 2026, this independent reproducer was built and run on Windows,
AMD Radeon RX 9070 XT (`gfx1201`), HIP headers `7.14.60850`, runtime/driver API
version `71460850`, and display driver `32.0.31035.1003`.
Both uploads read back exactly. LOD 0.25 produced 0.1875 and LOD 0.75 produced
0.8125 through both sampling paths. Maximum error was 0.09375, so exit code 1
correctly reported the discrepancy. The source-archive build and invalid-input,
missing-module, and unavailable-device error paths were also checked.

Linux and other GPUs have not been executed here; those are the independent
comparisons this reproducer is intended to enable.
