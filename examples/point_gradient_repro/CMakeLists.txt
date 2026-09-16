cmake_minimum_required(VERSION 3.21)
project(HipPointGradientReproducer LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
set(HIP_ARCHITECTURES "" CACHE STRING "Actual GPU target, e.g. gfx1201")
if(NOT HIP_ARCHITECTURES)
    message(FATAL_ERROR "Set HIP_ARCHITECTURES to the GPU being tested")
endif()
find_file(POINT_FIND_HIP FindHIP.cmake
    PATHS "${CMAKE_CURRENT_SOURCE_DIR}/cmake" "${CMAKE_CURRENT_SOURCE_DIR}/../../cmake"
    NO_DEFAULT_PATH REQUIRED)
get_filename_component(POINT_CMAKE_DIR "${POINT_FIND_HIP}" DIRECTORY)
list(PREPEND CMAKE_MODULE_PATH "${POINT_CMAKE_DIR}")
find_package(HIP REQUIRED)
hip_add_executable(
    TARGET point_gradient_kernel
    SOURCES point_gradient_repro.cpp
    ARCHITECTURES ${HIP_ARCHITECTURES}
    OPTIONS -O3 --std=c++17 -DPOINT_GRADIENT_KERNEL)
add_executable(point_gradient_repro point_gradient_repro.cpp)
target_compile_definitions(point_gradient_repro PRIVATE __HIP_PLATFORM_AMD__)
target_link_libraries(point_gradient_repro PRIVATE hip::host)
if(MSVC)
    target_compile_options(point_gradient_repro PRIVATE /EHsc)
endif()
add_dependencies(point_gradient_repro point_gradient_kernel)
add_custom_command(TARGET point_gradient_repro POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${point_gradient_kernel_CODE_OBJECT}" "$<TARGET_FILE_DIR:point_gradient_repro>/point_gradient_kernel.co"
    VERBATIM)
if(WIN32)
    add_custom_command(TARGET point_gradient_repro POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${HIP_RUNTIME_DLLS}
            "$<TARGET_FILE_DIR:point_gradient_repro>"
        VERBATIM)
endif()
