# Mempulse.cmake - builds MemPulse (https://github.com/jpola-amd/mempulse) from source
#
# Provides:
#   Mempulse::Mempulse                 imported shared library target
#   mempulse_copy_runtime(<target>)    copies the MemPulse DLL beside <target> (Windows only)
#
# Also installs the MemPulse runtime library with the project.
# Requires HIP_PATH (set by FindHIP.cmake).

include_guard(GLOBAL)
include(ExternalProject)

set(MEMPULSE_INSTALL_DIR "${CMAKE_BINARY_DIR}/_deps/mempulse")

# The examples use the DRM backend on Linux, which needs libdrm_amdgpu (libdrm-dev).
set(MEMPULSE_BACKEND_DRM OFF)
if(UNIX AND NOT APPLE)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(LIBDRM_AMDGPU REQUIRED libdrm_amdgpu)
    set(MEMPULSE_BACKEND_DRM ON)
endif()

set(MEMPULSE_CMAKE_ARGS
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_INSTALL_PREFIX=${MEMPULSE_INSTALL_DIR}
    -DCMAKE_INSTALL_LIBDIR=lib
    -DHIP_DIR=${HIP_PATH}
    -DBUILD_BACKEND_DRM=${MEMPULSE_BACKEND_DRM}
    -DBUILD_TESTS=OFF
    -DBUILD_APPLICATION=OFF
)
if(NOT CMAKE_GENERATOR MATCHES "Visual Studio")
    list(APPEND MEMPULSE_CMAKE_ARGS -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER})
endif()

if(WIN32)
    set(MEMPULSE_RUNTIME "${MEMPULSE_INSTALL_DIR}/bin/mempulse.dll")
    set(MEMPULSE_IMPLIB "${MEMPULSE_INSTALL_DIR}/lib/mempulse.lib")
else()
    set(MEMPULSE_RUNTIME "${MEMPULSE_INSTALL_DIR}/lib/${CMAKE_SHARED_LIBRARY_PREFIX}mempulse${CMAKE_SHARED_LIBRARY_SUFFIX}")
    set(MEMPULSE_IMPLIB "")
endif()

# Version 1.0.4
ExternalProject_Add(mempulse_external
    URL https://github.com/jpola-amd/mempulse/archive/24506d0944380825fea13cea83e141ada2a9a125.tar.gz
    URL_HASH SHA256=d030154157ed101bc9a035e369a84fa7fa724d6486874525b6a5e51b64d69905
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    CMAKE_ARGS ${MEMPULSE_CMAKE_ARGS}
    BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --config Release
    INSTALL_COMMAND ${CMAKE_COMMAND} --install <BINARY_DIR> --config Release
    BUILD_BYPRODUCTS ${MEMPULSE_RUNTIME} ${MEMPULSE_IMPLIB}
)

# The include directory must exist before the imported target is used.
file(MAKE_DIRECTORY "${MEMPULSE_INSTALL_DIR}/include")
add_library(Mempulse::Mempulse SHARED IMPORTED GLOBAL)
set_target_properties(Mempulse::Mempulse PROPERTIES
    IMPORTED_LOCATION "${MEMPULSE_RUNTIME}"
    INTERFACE_INCLUDE_DIRECTORIES "${MEMPULSE_INSTALL_DIR}/include"
)
if(WIN32)
    set_target_properties(Mempulse::Mempulse PROPERTIES IMPORTED_IMPLIB "${MEMPULSE_IMPLIB}")
endif()
add_dependencies(Mempulse::Mempulse mempulse_external)

if(WIN32)
    install(FILES ${MEMPULSE_RUNTIME} DESTINATION bin)
else()
    install(DIRECTORY "${MEMPULSE_INSTALL_DIR}/lib/"
        DESTINATION lib
        FILES_MATCHING PATTERN "${CMAKE_SHARED_LIBRARY_PREFIX}mempulse${CMAKE_SHARED_LIBRARY_SUFFIX}*"
        PATTERN "cmake" EXCLUDE
        PATTERN "pkgconfig" EXCLUDE
    )
endif()

# On Linux the build-tree RPATH already points at the MemPulse library.
function(mempulse_copy_runtime target)
    if(WIN32)
        add_custom_command(TARGET ${target} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${MEMPULSE_RUNTIME}
                $<TARGET_FILE_DIR:${target}>
            COMMENT "Copying MemPulse runtime library"
            VERBATIM
        )
    endif()
endfunction()
