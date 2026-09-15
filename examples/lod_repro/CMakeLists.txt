cmake_minimum_required(VERSION 3.21)
project(HipLodReproducer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
set(HIP_ARCHITECTURES "" CACHE STRING "Target GPU architecture(s), e.g. gfx1201 or gfx1100")
if(NOT HIP_ARCHITECTURES)
    message(FATAL_ERROR "Set HIP_ARCHITECTURES to the GPU being tested, e.g. -DHIP_ARCHITECTURES=gfx1201")
endif()

# The source archive includes the existing repository helper in cmake/.
find_file(LOD_FIND_HIP FindHIP.cmake
    PATHS "${CMAKE_CURRENT_SOURCE_DIR}/cmake" "${CMAKE_CURRENT_SOURCE_DIR}/../../cmake"
    NO_DEFAULT_PATH REQUIRED)
get_filename_component(LOD_CMAKE_DIR "${LOD_FIND_HIP}" DIRECTORY)
list(PREPEND CMAKE_MODULE_PATH "${LOD_CMAKE_DIR}")
find_package(HIP REQUIRED)

hip_add_executable(
    TARGET lod_repro_kernel
    SOURCES lod_repro.cpp
    ARCHITECTURES ${HIP_ARCHITECTURES}
    OPTIONS -O3 --std=c++17 -DLOD_REPRO_KERNEL
)

add_executable(lod_repro lod_repro.cpp)
target_compile_definitions(lod_repro PRIVATE __HIP_PLATFORM_AMD__)
target_link_libraries(lod_repro PRIVATE hip::host)
if(MSVC)
    target_compile_options(lod_repro PRIVATE /EHsc)
endif()
add_dependencies(lod_repro lod_repro_kernel)
add_custom_command(TARGET lod_repro POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${lod_repro_kernel_CODE_OBJECT}" "$<TARGET_FILE_DIR:lod_repro>/lod_repro_kernel.co"
    VERBATIM)
if(WIN32)
    add_custom_command(TARGET lod_repro POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${HIP_RUNTIME_DLLS} "$<TARGET_FILE_DIR:lod_repro>"
        VERBATIM)
endif()
