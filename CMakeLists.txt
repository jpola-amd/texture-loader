cmake_minimum_required(VERSION 3.21)

# Manifest features and the toolchain must be selected before project().
option(BUILD_EXAMPLES "Build example applications" OFF)
option(BUILD_TESTS "Build unit tests" OFF)
option(USE_OIIO "Use OpenImageIO for image loading" OFF)
set(_use_vcpkg_default OFF)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(_use_vcpkg_default ON)
endif()
option(USE_VCPKG "Download and build dependencies with vcpkg" ${_use_vcpkg_default})
if(USE_VCPKG)
    include(cmake/VcpkgBootstrap.cmake)
endif()

project(HIPDemandTextureLoader)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Enable exception handling for MSVC
if(MSVC)
    add_compile_options(/EHsc)
endif()

if (UNIX)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)

    # Let installed executables locate the shared library relative to their
    # own location (bin/ -> ../lib) without requiring LD_LIBRARY_PATH.
    if(APPLE)
        set(CMAKE_INSTALL_RPATH "@loader_path/../lib")
    else()
        set(CMAKE_INSTALL_RPATH "$ORIGIN/../lib")
    endif()
    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON)
endif()

# Add cmake module path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Find HIP using our custom module
find_package(HIP REQUIRED)

# Prefer the managed headers; retain the bundled/manual path when opted out.
if(NOT USE_VCPKG)
    set(STB_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/stb" CACHE PATH "Path to stb headers")
endif()
find_path(STB_INCLUDE_DIR NAMES stb_image.h PATH_SUFFIXES stb REQUIRED)
if(NOT EXISTS "${STB_INCLUDE_DIR}/stb_image.h")
    message(FATAL_ERROR "STB_INCLUDE_DIR must contain stb_image.h")
endif()

# GPU architectures to compile for
set(HIP_ARCHITECTURES "" CACHE STRING "GPU architectures to compile for (semicolon-separated)")
if(NOT HIP_ARCHITECTURES)
    set(HIP_ARCHITECTURES "gfx1030;gfx1100;gfx1101;gfx1102;gfx1150;gfx1151;gfx1200;gfx1201")
    message(STATUS "Using default HIP architectures: ${HIP_ARCHITECTURES}")
else()
    message(STATUS "Using custom HIP architectures: ${HIP_ARCHITECTURES}")
endif()

# Find OpenImageIO if requested
if(USE_OIIO)
    find_package(Imath REQUIRED)
    find_package(OpenEXR REQUIRED)
    find_package(OpenImageIO REQUIRED)
    message(STATUS "Using OpenImageIO for image loading")
endif()

# Library
set(TEXTURE_LOADER_SOURCES
    src/DemandLoading/vmm/DemandTextureLoaderImpl.cpp
    src/DemandLoading/vmm/PageSystem.cpp
    src/DemandLoading/vmm/RequestProcessor.cpp
    src/DemandLoading/vmm/Ticket.cpp
    src/DemandLoading/DemandTextureLoader.cpp
    src/DemandLoading/DemandTextureLoaderImpl.cpp
    src/DemandLoading/Ticket.cpp
    src/DemandLoading/Logging.cpp
    src/DemandLoading/stb_image_impl.cpp
)

set(TEXTURE_LOADER_HEADERS
    src/DemandLoading/Internal/HipCheck.h
    src/DemandLoading/Internal/HipEventPool.h
    src/DemandLoading/Internal/PinnedMemoryPool.h
    src/DemandLoading/Internal/TextureMetadata.h
    src/DemandLoading/Internal/ThreadPool.h
    src/DemandLoading/Internal/Utils.h
    src/DemandLoading/vmm/DemandTextureLoaderImpl.h
    src/DemandLoading/vmm/HipEventPool.h
    src/DemandLoading/vmm/Lru.h
    src/DemandLoading/vmm/Allocator.h
    src/DemandLoading/vmm/PageSystem.h
    src/DemandLoading/vmm/PinnedUploadPool.h
    src/DemandLoading/vmm/RequestProcessor.h
    src/DemandLoading/vmm/TicketImpl.h
    src/DemandLoading/DemandTextureLoaderImpl.h
)

hip_add_executable(
    TARGET hip_demand_texture_kernels
    SOURCES src/DemandLoading/vmm/VmmTextureLoaderKernels.hip
    ARCHITECTURES ${HIP_ARCHITECTURES}
    OPTIONS -O3 --std=c++17
    INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/include
    DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/include/DemandLoading/DeviceContext.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/DemandLoading/VmmDeviceUtils.h
        src/DemandLoading/vmm/Lru.h
)

if(USE_OIIO)
    list(APPEND TEXTURE_LOADER_SOURCES
        src/ImageSource/OIIOReader.cpp
    )
endif()

source_group(
    TREE "${CMAKE_CURRENT_SOURCE_DIR}/src"
    PREFIX "Source Files"
    FILES
        ${TEXTURE_LOADER_SOURCES}
        ${TEXTURE_LOADER_HEADERS}
)

add_library(hip_demand_texture SHARED
    ${TEXTURE_LOADER_SOURCES}
    ${TEXTURE_LOADER_HEADERS}
)

# Export all symbols on Windows so the shared library is usable without
# explicit __declspec(dllexport) annotations.
set_target_properties(hip_demand_texture PROPERTIES
    WINDOWS_EXPORT_ALL_SYMBOLS ON
)

target_include_directories(hip_demand_texture
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${STB_INCLUDE_DIR}
)

target_link_libraries(hip_demand_texture
    PUBLIC
        hip::host
)

add_dependencies(hip_demand_texture hip_demand_texture_kernels)

add_custom_command(TARGET hip_demand_texture POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_PROPERTY:hip_demand_texture_kernels,HIP_CODE_OBJECT>
        $<TARGET_FILE_DIR:hip_demand_texture>/hip_demand_texture_kernels.co
    COMMENT "Copying request compaction HIP kernel code object"
    VERBATIM
)

target_compile_definitions(hip_demand_texture PRIVATE __HIP_PLATFORM_AMD__)

if(WIN32)
    # Stage once per build: Windows searches System32 before PATH, which can
    # otherwise mix the driver's HIP runtime with the configured SDK's COMGR.
    add_custom_target(hip_runtime_dlls
        COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:hip_demand_texture>"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${HIP_RUNTIME_DLLS} "$<TARGET_FILE_DIR:hip_demand_texture>"
        COMMENT "Deploying HIP runtime and COMGR from the configured SDK"
        VERBATIM
    )
    add_dependencies(hip_demand_texture hip_runtime_dlls)
endif()

# Add OpenImageIO support if enabled
if(USE_OIIO)
    target_compile_definitions(hip_demand_texture PRIVATE USE_OIIO)
    target_link_libraries(hip_demand_texture PRIVATE OpenImageIO::OpenImageIO)
endif()

if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Installation
install(TARGETS hip_demand_texture
    EXPORT HIPDemandTextureTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

if(WIN32)
    install(FILES
        $<TARGET_PROPERTY:hip_demand_texture_kernels,HIP_CODE_OBJECT>
        DESTINATION bin
        RENAME hip_demand_texture_kernels.co
    )
else()
    install(FILES
        $<TARGET_PROPERTY:hip_demand_texture_kernels,HIP_CODE_OBJECT>
        DESTINATION lib
        RENAME hip_demand_texture_kernels.co
    )
endif()

install(DIRECTORY include/
    DESTINATION include
)

if(WIN32)
    install(FILES ${HIP_RUNTIME_DLLS} DESTINATION bin)
endif()
