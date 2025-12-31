cmake_minimum_required(VERSION 3.21)
project(HIPDemandTextureLoader)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Enable exception handling for MSVC
if(MSVC)
    add_compile_options(/EHsc)
endif()

# Add cmake module path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Find HIP using our custom module
find_package(HIP REQUIRED)

# Find stb_image for image loading (header-only)
# You can download from: https://github.com/nothings/stb
set(STB_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/stb" CACHE PATH "Path to stb headers")

# Build options
option(BUILD_EXAMPLES "Build example applications" OFF)
option(USE_OIIO "Use OpenImageIO for image loading" OFF)

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
    src/DemandLoading/DemandTextureLoader.cpp
    src/DemandLoading/DemandTextureLoaderImpl.cpp
    src/DemandLoading/Ticket.cpp
    src/DemandLoading/Logging.cpp
    src/DemandLoading/stb_image_impl.cpp
)

if(USE_OIIO)
    list(APPEND TEXTURE_LOADER_SOURCES
        src/ImageSource/OIIOReader.cpp
    )
endif()

add_library(hip_demand_texture ${TEXTURE_LOADER_SOURCES})

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

target_compile_definitions(hip_demand_texture PRIVATE __HIP_PLATFORM_AMD__)

# Add OpenImageIO support if enabled
if(USE_OIIO)
    target_compile_definitions(hip_demand_texture PRIVATE USE_OIIO)
    target_link_libraries(hip_demand_texture PRIVATE OpenImageIO::OpenImageIO)
endif()

# Examples (optional)
if(BUILD_EXAMPLES)
    # Compile HIP kernel to code object
    hip_add_executable(
        TARGET render_kernel
        SOURCES examples/simple_render_kernel.hip
        ARCHITECTURES ${HIP_ARCHITECTURES}
            OPTIONS -O3 --std=c++17 -DHIP_ENABLE_WARP_SYNC_BUILTINS
        INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/include
    )

    hip_add_executable(
        TARGET tiled_render_kernel
        SOURCES examples/tiled_render_kernel.hip
        ARCHITECTURES ${HIP_ARCHITECTURES}
            OPTIONS -O3 --std=c++17 -DHIP_ENABLE_WARP_SYNC_BUILTINS
        INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/include
    )

    hip_add_executable(
        TARGET mega_texture_kernel
        SOURCES examples/mega_texture_kernel.hip
        ARCHITECTURES ${HIP_ARCHITECTURES}
            OPTIONS -O3 --std=c++17 -DHIP_ENABLE_WARP_SYNC_BUILTINS
        INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/include
    )

    hip_add_executable(
        TARGET atlas_render_kernel
        SOURCES examples/atlas_render_kernel.hip
        ARCHITECTURES ${HIP_ARCHITECTURES}
            OPTIONS -O3 --std=c++17 -DHIP_ENABLE_WARP_SYNC_BUILTINS
        INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/include
    )

    # Example application (host code only)
    add_executable(texture_loader_example
        examples/simple_render.cpp
    )

    add_executable(tiled_render_example
        examples/tiled_render.cpp
    )

    add_executable(mega_texture_example
        examples/mega_texture_render.cpp
    )

    add_executable(atlas_render_example
        examples/atlas_render.cpp
    )

    add_executable(async_vs_sync_example
        examples/async_vs_sync.cpp
    )

    target_include_directories(texture_loader_example
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${STB_INCLUDE_DIR}
    )

    target_include_directories(tiled_render_example
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${STB_INCLUDE_DIR}
    )

    target_include_directories(mega_texture_example
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${STB_INCLUDE_DIR}
    )

    target_include_directories(atlas_render_example
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${STB_INCLUDE_DIR}
    )

    target_include_directories(async_vs_sync_example
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${STB_INCLUDE_DIR}
    )

    target_link_libraries(texture_loader_example
        PRIVATE
            hip_demand_texture
    )
    target_compile_definitions(texture_loader_example PRIVATE __HIP_PLATFORM_AMD__)

    target_link_libraries(tiled_render_example
        PRIVATE
            hip_demand_texture
    )
    target_compile_definitions(tiled_render_example PRIVATE __HIP_PLATFORM_AMD__)

    target_link_libraries(mega_texture_example
        PRIVATE
            hip_demand_texture
    )
    target_compile_definitions(mega_texture_example PRIVATE __HIP_PLATFORM_AMD__)

    target_link_libraries(atlas_render_example
        PRIVATE
            hip_demand_texture
    )
    target_compile_definitions(atlas_render_example PRIVATE __HIP_PLATFORM_AMD__)

    target_link_libraries(async_vs_sync_example
        PRIVATE
            hip_demand_texture
    )
    target_compile_definitions(async_vs_sync_example PRIVATE __HIP_PLATFORM_AMD__)

    # Copy kernel code object to output directory
    add_custom_command(TARGET texture_loader_example POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_PROPERTY:render_kernel,HIP_CODE_OBJECT>
            $<TARGET_FILE_DIR:texture_loader_example>/render_kernel.co
        COMMENT "Copying HIP kernel code object"
    )

    add_custom_command(TARGET tiled_render_example POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_PROPERTY:tiled_render_kernel,HIP_CODE_OBJECT>
            $<TARGET_FILE_DIR:tiled_render_example>/tiled_render_kernel.co
        COMMENT "Copying HIP kernel code object"
    )

    add_custom_command(TARGET mega_texture_example POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_PROPERTY:mega_texture_kernel,HIP_CODE_OBJECT>
            $<TARGET_FILE_DIR:mega_texture_example>/mega_texture_kernel.co
        COMMENT "Copying HIP kernel code object"
    )

    add_custom_command(TARGET atlas_render_example POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_PROPERTY:atlas_render_kernel,HIP_CODE_OBJECT>
            $<TARGET_FILE_DIR:atlas_render_example>/atlas_render_kernel.co
        COMMENT "Copying HIP kernel code object"
    )

    add_custom_command(TARGET async_vs_sync_example POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_PROPERTY:render_kernel,HIP_CODE_OBJECT>
            $<TARGET_FILE_DIR:async_vs_sync_example>/render_kernel.co
        COMMENT "Copying HIP kernel code object"
    )

    # Add dependency to ensure kernel is built first
    add_dependencies(texture_loader_example render_kernel)
    add_dependencies(tiled_render_example tiled_render_kernel)
    add_dependencies(mega_texture_example mega_texture_kernel)
    add_dependencies(atlas_render_example atlas_render_kernel)
    add_dependencies(async_vs_sync_example render_kernel)
    
    # Benchmark suite
    hip_add_executable(
        TARGET benchmark_kernel
        SOURCES examples/benchmark_kernel.hip
        ARCHITECTURES ${HIP_ARCHITECTURES}
            OPTIONS -O3 --std=c++17 -DHIP_ENABLE_WARP_SYNC_BUILTINS
        INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    
    add_executable(benchmark_suite
        examples/benchmark_suite.cpp
    )
    
    target_include_directories(benchmark_suite
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    
    target_link_libraries(benchmark_suite
        PRIVATE
            hip_demand_texture
            hip::host
    )
    
    target_compile_definitions(benchmark_suite PRIVATE __HIP_PLATFORM_AMD__)
    
    add_custom_command(TARGET benchmark_suite POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_PROPERTY:benchmark_kernel,HIP_CODE_OBJECT>
            $<TARGET_FILE_DIR:benchmark_suite>/benchmark_kernel.co
        COMMENT "Copying HIP kernel code object"
    )
    
    add_dependencies(benchmark_suite benchmark_kernel)
    
    # OpenImageIO format example (only if OIIO is enabled)
    if(USE_OIIO)
        add_executable(oiio_formats_example
            examples/oiio_formats.cpp
        )
        
        target_include_directories(oiio_formats_example
            PRIVATE
                ${CMAKE_CURRENT_SOURCE_DIR}/include
        )
        
        target_link_libraries(oiio_formats_example
            PRIVATE
                hip_demand_texture
        )
        
        target_compile_definitions(oiio_formats_example PRIVATE __HIP_PLATFORM_AMD__ USE_OIIO)
        
        message(STATUS "Building OpenImageIO format example")
    endif()
endif()

# Installation
install(TARGETS hip_demand_texture
    EXPORT HIPDemandTextureTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/
    DESTINATION include
)
