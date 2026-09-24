# FindHIP.cmake - HIP module for Visual Studio compatibility
# Provides functions to compile HIP device code separately and load via module API

# Find HIP installation
if(NOT DEFINED HIP_PATH)
    if(DEFINED ENV{HIP_PATH})
        set(HIP_PATH $ENV{HIP_PATH} CACHE PATH "Path to HIP installation")
    elseif(WIN32)
        set(HIP_PATH "C:/Program Files/AMD/ROCm/6.4" CACHE PATH "Path to HIP installation")
    else()
        set(HIP_PATH "/opt/rocm" CACHE PATH "Path to HIP installation")
    endif()
endif()

# Verify HIP installation
if(NOT EXISTS "${HIP_PATH}")
    message(FATAL_ERROR "HIP not found at ${HIP_PATH}. Please set HIP_PATH.")
endif()

message(STATUS "Using HIP_PATH: ${HIP_PATH}")
set(HIP_INCLUDE_DIR "${HIP_PATH}/include")
set(HIP_LIB_DIR "${HIP_PATH}/lib")
set(HIP_BIN_DIR "${HIP_PATH}/bin")

# Read HIP version for versioned library names
set(HIP_VERSION_FILE "${HIP_INCLUDE_DIR}/hip/hip_version.h")
if(EXISTS "${HIP_VERSION_FILE}")
    file(READ "${HIP_VERSION_FILE}" HIP_VERSION_CONTENT)
    string(REGEX MATCH "#define HIP_VERSION_MAJOR ([0-9]+)" _ "${HIP_VERSION_CONTENT}")
    set(HIP_VERSION_MAJOR ${CMAKE_MATCH_1})
    string(REGEX MATCH "#define HIP_VERSION_MINOR ([0-9]+)" _ "${HIP_VERSION_CONTENT}")
    set(HIP_VERSION_MINOR ${CMAKE_MATCH_1})
    
    # Format version as 0604 for HIP 6.4
    string(LENGTH "${HIP_VERSION_MAJOR}" MAJOR_LEN)
    string(LENGTH "${HIP_VERSION_MINOR}" MINOR_LEN)
    if(MAJOR_LEN EQUAL 1)
        set(HIP_VERSION_MAJOR "${HIP_VERSION_MAJOR}")
    endif()
    if(MINOR_LEN EQUAL 1)
        set(HIP_VERSION_MINOR "${HIP_VERSION_MINOR}")
    endif()
    set(HIP_VERSION_STRING "0${HIP_VERSION_MAJOR}0${HIP_VERSION_MINOR}")
    message(STATUS "Detected HIP version: ${HIP_VERSION_MAJOR}.${HIP_VERSION_MINOR} (${HIP_VERSION_STRING})")
endif()

# Find required libraries
find_library(HIP_LIBRARY
    names amdhip64
    PATHS ${HIP_PATH}
    PATH_SUFFIXES lib
    NO_DEFAULT_PATH
)

# Find amd_comgr with multiple possible names
# Try: amd_comgr_2 (standard), amd_comgr0604 (versioned), amd_comgr (fallback)
find_library(HIP_COMGR_LIBRARY
    NAMES amd_comgr_2 amd_comgr${HIP_VERSION_STRING} amd_comgr
    PATHS ${HIP_PATH}
    PATH_SUFFIXES lib
    NO_DEFAULT_PATH
)

# Find the Windows runtime libraries that must be next to HIP executables.
if(WIN32)
    find_file(HIP_COMGR_RUNTIME_LIBRARY
        NAMES amd_comgr.dll
        PATHS ${HIP_BIN_DIR}
        NO_DEFAULT_PATH
    )
    find_file(HIP_RUNTIME_LIBRARY
        NAMES amdhip64_7.dll
        PATHS ${HIP_BIN_DIR}
        NO_DEFAULT_PATH
    )
    find_file(HIP_KPACK_RUNTIME_LIBRARY
        NAMES rocm_kpack.dll
        PATHS ${HIP_BIN_DIR}
        NO_DEFAULT_PATH
    )

    set(HIP_RUNTIME_LIBRARIES
        ${HIP_COMGR_RUNTIME_LIBRARY}
        ${HIP_RUNTIME_LIBRARY}
    )
    if(HIP_KPACK_RUNTIME_LIBRARY AND EXISTS "${HIP_KPACK_RUNTIME_LIBRARY}")
        list(APPEND HIP_RUNTIME_LIBRARIES ${HIP_KPACK_RUNTIME_LIBRARY})
    else()
        message(STATUS "Optional HIP runtime library rocm_kpack.dll not found in ${HIP_BIN_DIR}")
    endif()
endif()

# Find compilers
find_program(HIP_HIPCC_EXECUTABLE
    NAMES hipcc hipcc.bat
    PATHS ${HIP_PATH}
    PATH_SUFFIXES bin
    NO_DEFAULT_PATH
)

find_program(HIP_CLANG_EXECUTABLE
    NAMES clang++ amdclang
    PATHS ${HIP_PATH}/lib
    PATH_SUFFIXES bin llvm/bin
    NO_DEFAULT_PATH
)

find_path(HIP_DEVICE_LIB_PATH
    NAME bitcode
    PATHS ${HIP_PATH}/lib/llvm/
    PATH_SUFFIXES llvm amdgcn llvm/amdgcn
    NO_DEFAULT_PATH
)


# Verify we found everything
if(NOT HIP_LIBRARY)
    message(FATAL_ERROR "HIP runtime library (amdhip64) not found in ${HIP_LIB_DIR}")
endif()

if(NOT HIP_COMGR_LIBRARY)
    message(FATAL_ERROR "HIP code object manager library (amd_comgr) not found in ${HIP_LIB_DIR}. Searched for: amd_comgr_2, amd_comgr${HIP_VERSION_STRING}, amd_comgr")
endif()

if(WIN32)
    if(NOT HIP_COMGR_RUNTIME_LIBRARY)
        message(FATAL_ERROR "HIP runtime library amd_comgr.dll not found in ${HIP_BIN_DIR}")
    endif()
    if(NOT HIP_RUNTIME_LIBRARY)
        message(FATAL_ERROR "HIP runtime library amdhip64_7.dll not found in ${HIP_BIN_DIR}")
    endif()
endif()

if(NOT HIP_HIPCC_EXECUTABLE)
    message(FATAL_ERROR "hipcc compiler not found in ${HIP_BIN_DIR}")
endif()

if (HIP_DEVICE_LIB_PATH)
    set(HIP_DEVICE_LIB_PATH ${HIP_DEVICE_LIB_PATH}/bitcode)
else()
    message(FATAL_ERROR "HIP device lib path not found (search for amdgcn/bitcode)")
endif()


message(STATUS "Found HIP: ${HIP_PATH}")
message(STATUS "Include: ${HIP_INCLUDE_DIR}")
message(STATUS "Libraries: ${HIP_LIBRARY}, ${HIP_COMGR_LIBRARY}")
if(WIN32)
    message(STATUS "Runtime libraries: ${HIP_RUNTIME_LIBRARIES}")
endif()
message(STATUS "Compiler:\n  -- hipcc: ${HIP_HIPCC_EXECUTABLE}\n  -- clang: ${HIP_CLANG_EXECUTABLE}")
message(STATUS "Device libs: ${HIP_DEVICE_LIB_PATH}")

if(WIN32)
    find_file(HIP_RUNTIME_DLL
        NAMES amdhip64_${HIP_VERSION_MAJOR}.dll amdhip64.dll
        PATHS "${HIP_BIN_DIR}"
        NO_DEFAULT_PATH
        REQUIRED
    )
    get_filename_component(_hip_comgr_name "${HIP_COMGR_LIBRARY}" NAME_WE)
    find_file(HIP_COMGR_RUNTIME_DLL
        NAMES "${_hip_comgr_name}.dll"
        PATHS "${HIP_BIN_DIR}"
        NO_DEFAULT_PATH
        REQUIRED
    )
    set(HIP_RUNTIME_DLLS "${HIP_RUNTIME_DLL}" "${HIP_COMGR_RUNTIME_DLL}")
endif()

# Create interface targets
if(NOT TARGET hip::include)
    add_library(hip::include INTERFACE IMPORTED)
    set_target_properties(hip::include PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${HIP_INCLUDE_DIR}"
    )
endif()

if(NOT TARGET hip::host)
    add_library(hip::host INTERFACE IMPORTED)
    set_target_properties(hip::host PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${HIP_INCLUDE_DIR}"
        INTERFACE_LINK_LIBRARIES "${HIP_LIBRARY};${HIP_COMGR_LIBRARY}"
    )
endif()

# Alias for convenience
if(NOT TARGET hip::hip)
    add_library(hip::hip INTERFACE IMPORTED)
    set_target_properties(hip::hip PROPERTIES
        INTERFACE_LINK_LIBRARIES hip::host
    )
endif()

# Function to compile HIP device code to code object
# Usage:
#   hip_add_executable(
#       TARGET my_kernels
#       SOURCES kernel1.hip kernel2.hip
#       ARCHITECTURES gfx1030 gfx1100
#       OPTIONS -O3 --std=c++17
#       INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/include
#   )
function(hip_add_executable)
    set(options "")
    set(oneValueArgs TARGET)
    set(multiValueArgs SOURCES ARCHITECTURES OPTIONS INCLUDES DEPENDS)
    cmake_parse_arguments(HIP "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    
    if(NOT HIP_TARGET)
        message(FATAL_ERROR "hip_add_executable: TARGET is required")
    endif()
    
    if(NOT HIP_SOURCES)
        message(FATAL_ERROR "hip_add_executable: SOURCES is required")
    endif()
    
    # Default architectures if not specified
    if(NOT HIP_ARCHITECTURES)
        set(HIP_ARCHITECTURES "gfx1030" "gfx1100")
    endif()
    
    # Build offload-arch arguments
    set(OFFLOAD_ARCH_FLAGS "")
    foreach(arch ${HIP_ARCHITECTURES})
        list(APPEND OFFLOAD_ARCH_FLAGS "--offload-arch=${arch}")
    endforeach()
    
    # Build include flags
    set(INCLUDE_FLAGS "")
    foreach(inc ${HIP_INCLUDES})
        list(APPEND INCLUDE_FLAGS "-I${inc}")
    endforeach()
    
    # Output file
    set(OUTPUT_FILE "${CMAKE_CURRENT_BINARY_DIR}/${HIP_TARGET}.co")

    # for theRock project we must pass this explicitly
    if(UNIX)
        set(ROCM_DEVICE_LIB_PATH_OPTION "--rocm-device-lib-path=${HIP_DEVICE_LIB_PATH}")
    endif()
    
    # Build compile command
    add_custom_command(
        OUTPUT ${OUTPUT_FILE}
        COMMAND ${HIP_HIPCC_EXECUTABLE}
            --genco
            ${OFFLOAD_ARCH_FLAGS}
            ${HIP_OPTIONS}
            ${INCLUDE_FLAGS}
            ${HIP_SOURCES}
            ${ROCM_DEVICE_LIB_PATH_OPTION}
            -o ${OUTPUT_FILE}
        DEPENDS ${HIP_SOURCES} ${HIP_DEPENDS}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Compiling HIP device code: ${HIP_TARGET}"
        VERBATIM
    )
    
    # Create a custom target
    add_custom_target(${HIP_TARGET} ALL DEPENDS ${OUTPUT_FILE})
    
    # Set properties so other targets can find the output
    set_target_properties(${HIP_TARGET} PROPERTIES
        HIP_CODE_OBJECT "${OUTPUT_FILE}"
    )
    
    # Export the output file path for easy access
    set(${HIP_TARGET}_CODE_OBJECT "${OUTPUT_FILE}" PARENT_SCOPE)
endfunction()

# Mark as found
set(HIP_FOUND TRUE)
