@echo off
REM ==============================================================================
REM Run OIIO Formats Example with Test Images
REM ==============================================================================
REM
REM This script runs the oiio_formats_example with all generated test images.
REM
REM USAGE:
REM   run_oiio_example.cmd [build_dir] [config]
REM
REM PARAMETERS:
REM   build_dir  - Build directory (default: .\build)
REM   config     - Build configuration: Debug, Release, etc. (default: Release)
REM
REM EXAMPLES:
REM   run_oiio_example.cmd
REM   run_oiio_example.cmd .\build Release
REM   run_oiio_example.cmd .\build Debug
REM
REM ==============================================================================

setlocal enabledelayedexpansion

REM Parse arguments
set BUILD_DIR=.\build
set CONFIG=Release
set TEST_IMAGE_DIR=test_images

if not "%~1"=="" set BUILD_DIR=%~1
if not "%~2"=="" set CONFIG=%~2

set EXECUTABLE=%BUILD_DIR%\%CONFIG%\oiio_formats_example.exe

echo ==============================================================================
echo OIIO Formats Example Test Runner
echo ==============================================================================
echo Build directory: %BUILD_DIR%
echo Configuration:   %CONFIG%
echo Executable:      %EXECUTABLE%
echo Test images:     %TEST_IMAGE_DIR%
echo.

REM Check if executable exists
if not exist "%EXECUTABLE%" (
    echo ERROR: Executable not found: %EXECUTABLE%
    echo.
    echo Please build the project first with:
    echo   cmake --build %BUILD_DIR% --config %CONFIG%
    echo.
    echo Make sure to configure with -DBUILD_EXAMPLES=ON -DUSE_OIIO=ON
    exit /b 1
)

REM Check if test images directory exists
if not exist "%TEST_IMAGE_DIR%" (
    echo ERROR: Test images directory not found: %TEST_IMAGE_DIR%
    echo.
    echo Please generate test images first with:
    echo   .\create_test_images.cmd "path\to\oiio\bin"
    exit /b 1
)

REM Count available test images
set IMAGE_COUNT=0
set IMAGE_LIST=

for %%f in (
    "%TEST_IMAGE_DIR%\gradient.png"
    "%TEST_IMAGE_DIR%\gradient.exr"
    "%TEST_IMAGE_DIR%\gradient.hdr"
    "%TEST_IMAGE_DIR%\gradient_16bit.tif"
    "%TEST_IMAGE_DIR%\checker.png"
    "%TEST_IMAGE_DIR%\checker.exr"
    "%TEST_IMAGE_DIR%\noise.png"
    "%TEST_IMAGE_DIR%\hdr_bright.exr"
) do (
    if exist "%%~f" (
        set /a IMAGE_COUNT+=1
        set IMAGE_LIST=!IMAGE_LIST! "%%~f"
    ) else (
        echo WARNING: Image not found: %%~f
    )
)

if %IMAGE_COUNT%==0 (
    echo ERROR: No test images found in %TEST_IMAGE_DIR%
    echo.
    echo Please generate test images first with:
    echo   .\create_test_images.cmd "path\to\oiio\bin"
    exit /b 1
)

echo Found %IMAGE_COUNT% test images
echo.

echo ==============================================================================
echo Running OIIO Formats Example
echo ==============================================================================
echo.

REM Run the executable with all test images
"%EXECUTABLE%" %IMAGE_LIST%

set EXIT_CODE=%ERRORLEVEL%

echo.
echo ==============================================================================
echo Test Complete
echo ==============================================================================
echo Exit code: %EXIT_CODE%
echo.

if %EXIT_CODE% neq 0 (
    echo ERROR: Example execution failed
    echo.
    echo Possible issues:
    echo   - OIIO not enabled: Rebuild with -DUSE_OIIO=ON
    echo   - Missing DLLs: Check that OIIO DLLs are in PATH or same directory
    echo   - Image format issues: Verify test images are valid
    exit /b %EXIT_CODE%
)

echo SUCCESS: All images processed successfully
echo.

endlocal
