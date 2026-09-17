// SPDX-License-Identifier: MIT
#pragma once
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <vector>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace hip_demand { namespace test {
inline std::filesystem::path testExecutablePath() {
#ifdef _WIN32
    std::vector<wchar_t> buffer(512);
    for(;;) {
        const DWORD size=GetModuleFileNameW(nullptr,buffer.data(),DWORD(buffer.size()));
        if(!size) throw std::runtime_error("cannot locate test executable");
        if(size<buffer.size()) return std::filesystem::path(std::wstring(buffer.data(),size));
        buffer.resize(buffer.size()*2);
    }
#else
    return std::filesystem::read_symlink("/proc/self/exe");
#endif
}
inline std::filesystem::path testFileRoot() {
    const char* overridePath=std::getenv("HDT_TEST_FILE_ROOT");
    auto root=overridePath ? std::filesystem::path(overridePath):testExecutablePath().parent_path()/"test-files";
    if(root.empty() || !root.is_absolute())
        throw std::invalid_argument("HDT_TEST_FILE_ROOT must be a nonempty absolute test-owned directory");
    std::filesystem::create_directories(root);
    return root.lexically_normal().make_preferred();
}
inline std::filesystem::path testLoaderLibraryPath() {
    if(const char* overridePath=std::getenv("HDT_TEST_LOADER_LIBRARY")) {
        const std::filesystem::path path(overridePath);
        if(!path.is_absolute() || !std::filesystem::is_regular_file(path))
            throw std::runtime_error("HDT_TEST_LOADER_LIBRARY does not name an existing absolute library");
        return path;
    }
    const auto directory=testExecutablePath().parent_path();
#ifdef _WIN32
    const auto path=directory/"hip_demand_texture.dll";
    if(std::filesystem::is_regular_file(path)) return path;
#else
    for(const auto& path:{directory/"libhip_demand_texture.so",directory.parent_path()/"lib"/"libhip_demand_texture.so"})
        if(std::filesystem::is_regular_file(path)) return path;
#endif
    throw std::runtime_error("loader library is missing beside installed tests");
}
} }
