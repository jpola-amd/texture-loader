@echo off
REM ==============================================================================
REM Test Image Generation Script using oiiotool
REM ==============================================================================
REM
REM This script creates test images in various formats for testing OpenImageIO
REM support in the HIP Demand Texture Loader.
REM
REM USAGE:
REM   create_test_images.cmd <OIIO_BIN_DIR>
REM
REM EXAMPLE:
REM   create_test_images.cmd "E:\Arnold\arnold-core\build\windows\dependencies\oiio\autodesk-arnold-2.6.3.1-1\bin"
REM
REM ==============================================================================

setlocal enabledelayedexpansion

REM Check if OIIO bin directory was provided
if "%~1"=="" (
    echo ERROR: OIIO bin directory not specified
    echo.
    echo Usage: %~nx0 ^<OIIO_BIN_DIR^>
    echo Example: %~nx0 "E:\path\to\oiio\bin"
    exit /b 1
)

set OIIO_BIN_DIR=%~1
set OIIOTOOL=%OIIO_BIN_DIR%\oiiotool.exe

REM Verify oiiotool exists
if not exist "%OIIOTOOL%" (
    echo ERROR: oiiotool.exe not found at: %OIIOTOOL%
    echo.
    echo Please verify the OIIO bin directory path.
    exit /b 1
)

echo ==============================================================================
echo Test Image Generation
echo ==============================================================================
echo Using oiiotool: %OIIOTOOL%
echo.

REM Create output directory
set OUTPUT_DIR=test_images
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"
echo Output directory: %OUTPUT_DIR%
echo.

REM Add OIIO bin to PATH temporarily for any DLL dependencies
set PATH=%OIIO_BIN_DIR%;%PATH%

echo ------------------------------------------------------------------------------
echo Generating test images...
echo ------------------------------------------------------------------------------
echo.

REM 1. Generate gradient pattern (512x512)
echo [1/8] Creating gradient pattern (PNG, UINT8)...
"%OIIOTOOL%" --pattern fill:topleft=0,0,0:topright=1,0,0:bottomleft=0,1,0:bottomright=0,0,1 512x512 4 -o "%OUTPUT_DIR%\gradient.png"
if errorlevel 1 (
    echo   ERROR: Failed to create gradient.png
) else (
    echo   SUCCESS: gradient.png
)

REM 2. Convert gradient to EXR (32-bit float)
echo [2/8] Converting to EXR (FLOAT32)...
"%OIIOTOOL%" "%OUTPUT_DIR%\gradient.png" -d float -o "%OUTPUT_DIR%\gradient.exr"
if errorlevel 1 (
    echo   ERROR: Failed to create gradient.exr
) else (
    echo   SUCCESS: gradient.exr
)

REM 3. Convert gradient to HDR (Radiance format)
echo [3/8] Converting to HDR (Radiance format)...
"%OIIOTOOL%" "%OUTPUT_DIR%\gradient.png" -d float -o "%OUTPUT_DIR%\gradient.hdr"
if errorlevel 1 (
    echo   ERROR: Failed to create gradient.hdr
) else (
    echo   SUCCESS: gradient.hdr
)

REM 4. Convert gradient to 16-bit TIFF
echo [4/8] Converting to TIFF (UINT16)...
"%OIIOTOOL%" "%OUTPUT_DIR%\gradient.png" -d uint16 -o "%OUTPUT_DIR%\gradient_16bit.tif"
if errorlevel 1 (
    echo   ERROR: Failed to create gradient_16bit.tif
) else (
    echo   SUCCESS: gradient_16bit.tif
)

REM 5. Generate checker pattern
echo [5/8] Creating checker pattern (PNG, UINT8)...
"%OIIOTOOL%" --pattern checker:width=64:height=64:color1=0.2,0.2,0.2:color2=0.8,0.8,0.8 512x512 4 -o "%OUTPUT_DIR%\checker.png"
if errorlevel 1 (
    echo   ERROR: Failed to create checker.png
) else (
    echo   SUCCESS: checker.png
)

REM 6. Convert checker to EXR
echo [6/8] Converting checker to EXR (FLOAT32)...
"%OIIOTOOL%" "%OUTPUT_DIR%\checker.png" -d float -o "%OUTPUT_DIR%\checker.exr"
if errorlevel 1 (
    echo   ERROR: Failed to create checker.exr
) else (
    echo   SUCCESS: checker.exr
)

REM 7. Generate noise pattern
echo [7/8] Creating noise pattern (PNG, UINT8)...
"%OIIOTOOL%" --pattern noise:min=0:max=1:seed=42 512x512 4 -o "%OUTPUT_DIR%\noise.png"
if errorlevel 1 (
    echo   ERROR: Failed to create noise.png
) else (
    echo   SUCCESS: noise.png
)

REM 8. Generate HDR test pattern (bright values for HDR testing)
echo [8/8] Creating HDR test pattern with bright values...
"%OIIOTOOL%" --pattern fill:topleft=1,0,0:topright=5,5,0:bottomleft=0,1,0:bottomright=0,0,10 512x512 4 -d float -o "%OUTPUT_DIR%\hdr_bright.exr"
if errorlevel 1 (
    echo   ERROR: Failed to create hdr_bright.exr
) else (
    echo   SUCCESS: hdr_bright.exr
)

echo.
echo ==============================================================================
echo Generation Complete
echo ==============================================================================
echo.
echo Generated images in: %OUTPUT_DIR%\
echo.
dir /B "%OUTPUT_DIR%"
echo.
echo ------------------------------------------------------------------------------
echo Test the images with:
echo   .\build\Release\oiio_formats_example.exe ^
echo     %OUTPUT_DIR%\gradient.exr ^
echo     %OUTPUT_DIR%\gradient.hdr ^
echo     %OUTPUT_DIR%\gradient_16bit.tif ^
echo     %OUTPUT_DIR%\checker.png ^
echo     %OUTPUT_DIR%\noise.png ^
echo     %OUTPUT_DIR%\hdr_bright.exr
echo ------------------------------------------------------------------------------
echo.
echo Image Details:
echo   gradient.png       - 8-bit RGBA gradient (red->blue, green->bottom)
echo   gradient.exr       - 32-bit float EXR version
echo   gradient.hdr       - Radiance HDR version
echo   gradient_16bit.tif - 16-bit TIFF version
echo   checker.png        - 8-bit checkerboard pattern (64x64 checks)
echo   checker.exr        - 32-bit float EXR version
echo   noise.png          - 8-bit random noise
echo   hdr_bright.exr     - HDR with values >1.0 for HDR testing
echo.
echo ==============================================================================

endlocal
