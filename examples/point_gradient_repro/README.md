# Native HIP NPOT point-gradient reproducer

HIP only: no texture-loader DLL, GoogleTest, OIIO, input images, or private
descriptor modifications. This isolates a **spatial-point/mip-point**
`tex2DGrad` discrepancy, separate from the known linear-mip interpolation issue.

The program uploads a nonconstant FLOAT RGBA 19x19 pyramid and a second dense
suffix starting at original mip 1 (9x9). Every level's dimensions and pixels
are checked by readback. Both native samplers use normalized wrap coordinates,
point spatial/mip filtering, no sRGB/bias, and legacy anisotropy zero.

At UV `(1.125,-0.125)`, it compares native explicit LOD and isotropic gradients
at original LODs `0.515625`, `0.75`, `1`, and `1.25`. Point mip selection is
unambiguous (no half-way ties). Per-level affine RGB values make a wrong texel
visible. Explicit suffix LOD is rebased once; suffix gradients correct only
NPOT rounding, by `19/(9*2)`, preserving the original texel-space footprint.
The full-chain native gradient path has **no suffix correction at all**.

## Build and run

Select a supported HIP SDK, C++17 compiler, CMake 3.21+, and the architecture of
the actual device. No dependencies are downloaded. In this directory:

```powershell
$env:HIP_PATH = "C:\path\to\HIP-SDK"
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 `
    "-DHIP_PATH=$env:HIP_PATH" -DHIP_ARCHITECTURES=gfx1201
cmake --build build --config Release
.\build\Release\point_gradient_repro.exe `
    .\build\Release\point_gradient_kernel.co 0 | Tee-Object results.txt
$LASTEXITCODE
```

On Linux, use the same CMake source with your native generator:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
    -DHIP_PATH=/opt/rocm -DHIP_ARCHITECTURES=gfx1201
cmake --build build
./build/point_gradient_repro ./build/point_gradient_kernel.co 0
```

Replace `gfx1201` for independent `gfx1151`/`gfx1100` qualification. The last
argument is an optional device ordinal. Windows packaging copies the selected
SDK's HIP and COMGR DLLs beside the executable; retain them to avoid loading a
different System32 runtime.

Exit **0** means analytical pixels pass at `1e-6` absolute per channel.
Exit **1** means operations/readback succeeded but sampling differs.
Exit **2** means setup/unsupported-operation/cleanup failure, not a pixel test.
The CSV reports all four channels and full/suffix gradient differences.

For a standalone source archive include these three files, the repository
license, and the existing `cmake/FindHIP.cmake` helper under `cmake/`. No checkout
is required after extraction. Send the full output plus SDK/compiler, OS and
display-driver versions; the printed HIP driver API number is not the display
driver package version.

## Observed Windows result (16 September 2026)

AMD Radeon RX 9070 XT (`gfx1201`), TheRock `7.14.0rc3`, HIP
headers/runtime/driver API `71460850`, Windows build 26200, display driver
`32.0.31035.1003`: all uploaded dimensions and pixels read back exactly.
All four tested LODs produced:

| Native path | R | G | Analytical max error |
|---|---:|---:|---:|
| Full-chain explicit LOD | 1.5625 | 1.0625 | 0 |
| Full-chain gradient | 1.4375 | 1.125 | 0.125 |
| Suffix explicit LOD | 1.5625 | 1.0625 | 0 |
| Suffix gradient | 1.5625 | 1.0625 | 0 |

Blue and alpha stayed one. The full/suffix gradient difference was **0.125**;
exit code **1** correctly reported numerical failure. This is a native behavior
reproducer, not a driver root-cause diagnosis or a production workaround.
Other operating systems/devices remain unexecuted.
