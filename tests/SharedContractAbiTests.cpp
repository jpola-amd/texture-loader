// SPDX-License-Identifier: MIT
#include <DemandLoading/Contracts.h>
#include <gtest/gtest.h>
#include "TestPaths.h"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace {
class SharedAbiExport : public testing::Test {
protected:
    void SetUp() override {
#ifdef _WIN32
        library_ = LoadLibraryW(hip_demand::test::testLoaderLibraryPath().c_str());
#else
        library_ = dlopen(hip_demand::test::testLoaderLibraryPath().c_str(), RTLD_NOW | RTLD_LOCAL);
#endif
        ASSERT_NE(library_, nullptr);
    }
    void TearDown() override {
        if (library_) {
#ifdef _WIN32
            EXPECT_NE(FreeLibrary(library_), 0);
#else
            EXPECT_EQ(dlclose(library_), 0);
#endif
        }
    }
#ifdef _WIN32
    HMODULE library_ = nullptr;
#else
    void* library_ = nullptr;
#endif
};

TEST_F(SharedAbiExport, VersionedSymbolIsIdenticalOnBothPlatforms) {
    using namespace hip_demand::contract_v1;
    using Query = decltype(&hipDemandGetContractAbiV1);
#ifdef _WIN32
    const auto symbol = GetProcAddress(library_, "hipDemandGetContractAbiV1");
#else
    const auto symbol = dlsym(library_, "hipDemandGetContractAbiV1");
#endif
    ASSERT_NE(symbol, nullptr);
    const auto query = reinterpret_cast<Query>(symbol);
    AbiInfo info;
    EXPECT_EQ(query(Version, sizeof(info), &info), static_cast<uint32_t>(Outcome::Success));
    EXPECT_EQ(info.productionIntegration, 0);
    EXPECT_EQ(query(Version + 1, sizeof(info), &info), static_cast<uint32_t>(Outcome::AbiMismatch));
}
} // namespace
