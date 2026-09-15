// SPDX-License-Identifier: MIT
#include "TestUtils.h"
#include "ContractDeviceTestData.h"
#include <DemandLoading/ContractState.h>
#include <cstdlib>
#include <iostream>

namespace cv1 = hip_demand::contract_v1;

namespace {

class SharedContractDevice : public testing::Test {
protected:
    void SetUp() override {
        int device = 0;
        if (const char* selection = std::getenv("HIP_DEMAND_TEST_DEVICE")) {
            char* end = nullptr;
            const long parsed = std::strtol(selection, &end, 10);
            ASSERT_NE(end, selection);
            ASSERT_EQ(*end, '\0');
            ASSERT_GE(parsed, 0);
            ASSERT_LE(parsed, INT32_MAX);
            device = static_cast<int>(parsed);
        }
        ASSERT_EQ(hipSetDevice(device), hipSuccess);
        hipDeviceProp_t properties{};
        ASSERT_EQ(hipGetDeviceProperties(&properties, device), hipSuccess);
        int runtime = 0, driver = 0;
        ASSERT_EQ(hipRuntimeGetVersion(&runtime), hipSuccess);
        ASSERT_EQ(hipDriverGetVersion(&driver), hipSuccess);
        std::cout << "Contract device=" << device << " " << properties.name << " "
                  << properties.gcnArchName << " runtime=" << runtime << " driver=" << driver << '\n';
        ASSERT_EQ(hipModuleLoad(&module_, CONTRACT_KERNEL_PATH), hipSuccess);
        ASSERT_EQ(hipModuleGetFunction(&function_, module_, "evaluateContract"), hipSuccess);
        ASSERT_EQ(hipMalloc(&input_, sizeof(ContractDeviceCase)), hipSuccess);
        ASSERT_EQ(hipMalloc(&output_, sizeof(ContractDeviceResult)), hipSuccess);
    }
    void TearDown() override {
        if (output_)
            EXPECT_EQ(hipFree(output_), hipSuccess);
        if (input_)
            EXPECT_EQ(hipFree(input_), hipSuccess);
        if (module_)
            EXPECT_EQ(hipModuleUnload(module_), hipSuccess);
    }
    void run(const ContractDeviceCase& input, cv1::Outcome outcome, cv1::SampleValidity validity,
             uint32_t first = cv1::InvalidSlot, uint32_t last = cv1::InvalidSlot, float resourceLod = 0) {
        ASSERT_EQ(hipMemcpy(input_, &input, sizeof(input), hipMemcpyHostToDevice), hipSuccess);
        void* args[]{&input_, &output_};
        ASSERT_EQ(hipModuleLaunchKernel(function_, 1, 1, 1, 1, 1, 1, 0, nullptr, args, nullptr), hipSuccess);
        ContractDeviceResult result;
        ASSERT_EQ(hipMemcpy(&result, output_, sizeof(result), hipMemcpyDeviceToHost), hipSuccess);
        EXPECT_EQ(result.decision.outcome, outcome);
        EXPECT_EQ(result.decision.validity, validity);
        EXPECT_EQ(result.decision.required.first, first);
        EXPECT_EQ(result.decision.required.last, last);
        EXPECT_FLOAT_EQ(result.decision.resourceLod, resourceLod);
        EXPECT_EQ(result.needsRequest, validity == cv1::SampleValidity::Missing ||
                                      validity == cv1::SampleValidity::CoarsePreview);
        EXPECT_EQ(result.contributes, validity == cv1::SampleValidity::Complete);
        cv1::AbiInfo host;
        ASSERT_EQ(hipDemandGetContractAbiV1(cv1::Version, sizeof(host), &host), 0);
        EXPECT_EQ(result.abi.abi.version, host.abi.version);
        EXPECT_EQ(result.abi.abi.byteSize, host.abi.byteSize);
        EXPECT_EQ(result.abi.keyBytes, host.keyBytes);
        EXPECT_EQ(result.abi.requestBytes, host.requestBytes);
        EXPECT_EQ(result.abi.descriptorBytes, host.descriptorBytes);
        EXPECT_EQ(result.abi.contextBytes, host.contextBytes);
        EXPECT_EQ(result.abi.publicationBytes, host.publicationBytes);
        EXPECT_EQ(result.abi.pointerBytes, host.pointerBytes);
    }
    ContractDeviceCase base() const {
        ContractDeviceCase input;
        input.key = {0, 1, 1};
        input.texture.key = input.key;
        input.texture.revision = 1;
        input.texture.textureObject = 123;
        input.texture.mips = {32, 16, 6, 2, 8, 4, 4};
        input.texture.state = cv1::RegistrationState::Live;
        input.texture.residency = cv1::Outcome::Success;
        return input;
    }
    hipModule_t module_ = nullptr;
    hipFunction_t function_ = nullptr;
    ContractDeviceCase* input_ = nullptr;
    ContractDeviceResult* output_ = nullptr;
};

TEST_F(SharedContractDevice, InvalidAndCrossLoaderKeysNeverDemand) {
    for (cv1::GpuKey key : {cv1::GpuKey{}, cv1::GpuKey{1, 1, 1}, cv1::GpuKey{0, 2, 1},
                            cv1::GpuKey{0, 1, 2}, cv1::GpuKey{0, 0, 1}}) {
        auto input = base();
        input.key = key;
        run(input, cv1::Outcome::InvalidKey, cv1::SampleValidity::Invalid);
    }
    auto input = base();
    input.incarnation = 2;
    run(input, cv1::Outcome::InvalidKey, cv1::SampleValidity::Invalid);
    input = base();
    input.revision = 2;
    run(input, cv1::Outcome::InvalidKey, cv1::SampleValidity::Invalid);
    input = base();
    input.texture.state = cv1::RegistrationState::Retiring;
    run(input, cv1::Outcome::InvalidKey, cv1::SampleValidity::Invalid);
}

TEST_F(SharedContractDevice, TerminalFailuresAndSuccessAfterFailure) {
    for (cv1::Outcome outcome : {cv1::Outcome::SourceFailure, cv1::Outcome::DeviceOutOfMemory,
                                 cv1::Outcome::Unsupported, cv1::Outcome::Cancelled, cv1::Outcome::NoProgress}) {
        auto input = base();
        input.texture.residency = outcome;
        run(input, outcome, cv1::SampleValidity::Invalid);
    }
    auto input = base();
    input.lod = 3.25f;
    run(input, cv1::Outcome::Success, cv1::SampleValidity::Complete, 3, 4, 1.25f);
}

TEST_F(SharedContractDevice, PointLinearAndPreviewRequirements) {
    auto input = base();
    input.lod = 1.75f;
    run(input, cv1::Outcome::Pending, cv1::SampleValidity::Missing, 1, 2);
    input.descriptor.mipFilter = cv1::FilterMode::Point;
    run(input, cv1::Outcome::Success, cv1::SampleValidity::Complete, 2, 2, -0.25f);
    input.lod = 1.5f;
    run(input, cv1::Outcome::Pending, cv1::SampleValidity::Missing, 1, 2);
    input.descriptor.samplingPolicy = cv1::SamplingPolicy::AllowCoarsePreview;
    run(input, cv1::Outcome::Pending, cv1::SampleValidity::CoarsePreview, 1, 2);
    input.descriptor.maxAnisotropy = 16;
    run(input, cv1::Outcome::Unsupported, cv1::SampleValidity::Invalid);
}

TEST_F(SharedContractDevice, PhysicalSuffixAndLegalLodBoundaries) {
    auto input = base();
    input.texture.mips = {7, 3, 3, 1, 3, 1, 2};
    input.lod = 1.25f;
    run(input, cv1::Outcome::Success, cv1::SampleValidity::Complete, 1, 2, .25f);
    input.texture.mips = {1, 31, 5, 3, 1, 3, 2};
    input.lod = 100;
    run(input, cv1::Outcome::Success, cv1::SampleValidity::Complete, 4, 4, 1);
    input.lod = -100;
    run(input, cv1::Outcome::Pending, cv1::SampleValidity::Missing, 0, 0);
    input.texture.mips = {1, 1, 1, 0, 1, 1, 1};
    run(input, cv1::Outcome::Success, cv1::SampleValidity::Complete, 0, 0, 0);
    input.texture.mips = {7, 3, 3, 1, 7, 3, 3};
    run(input, cv1::Outcome::InvalidInput, cv1::SampleValidity::Invalid);
}

TEST_F(SharedContractDevice, MismatchedAbiIsRejectedOnDevice) {
    auto input = base();
    input.contextVersion = 0;
    run(input, cv1::Outcome::AbiMismatch, cv1::SampleValidity::Invalid);
    input = base();
    --input.contextBytes;
    run(input, cv1::Outcome::AbiMismatch, cv1::SampleValidity::Invalid);
    input = base();
    ++input.descriptor.abi.version;
    run(input, cv1::Outcome::AbiMismatch, cv1::SampleValidity::Invalid);
    input = base();
    ++input.descriptor.abi.byteSize;
    run(input, cv1::Outcome::AbiMismatch, cv1::SampleValidity::Invalid);
}

TEST_F(SharedContractDevice, RequiredMipPolicyRejectsUndeclaredBaseOnlyFallback) {
    auto input = base();
    input.texture.mips = {32, 16, 1, 0, 32, 16, 1};
    input.lod = 4;
    run(input, cv1::Outcome::Unsupported, cv1::SampleValidity::Invalid);
    input.descriptor.mipPolicy = cv1::MipPolicy::AllowBaseLevelFallback;
    run(input, cv1::Outcome::Unsupported, cv1::SampleValidity::Invalid);
    input.descriptor = {};
    input.descriptor.maxMipLevels = 1;
    run(input, cv1::Outcome::Success, cv1::SampleValidity::Complete, 0, 0, 0);
    input.descriptor = {};
    input.descriptor.mipPolicy = cv1::MipPolicy::Disabled;
    run(input, cv1::Outcome::Success, cv1::SampleValidity::Complete, 0, 0, 0);
}

TEST_F(SharedContractDevice, LegacyEntryPointAndTextureZeroAreUnchanged) {
    static_assert(sizeof(hip_demand::TextureDesc) == 28);
    static_assert(sizeof(hip_demand::TextureHandle) == 24);
    static_assert(sizeof(hip_demand::DeviceContext) == 48);
    static_assert(offsetof(hip_demand::DeviceContext, maxTextures) == 40);
    static_assert(offsetof(hip_demand::TextureDesc, evictionPriority) == 24);
    static_assert(offsetof(hip_demand::TextureHandle, error) == 20);
    hip_demand::LoaderOptions options;
    options.maxTextures = 2;
    options.maxThreads = 1;
    hip_demand::DemandTextureLoader loader(options);
    hip_demand::TextureDesc descriptor;
    descriptor.generateMipmaps = false;
    descriptor.filterMode = hipFilterModePoint;
    const uint8_t pixel[]{64, 128, 192, 255};
    auto handle = loader.createTextureFromMemory(pixel, 1, 1, 4, descriptor);
    ASSERT_TRUE(handle.valid);
    ASSERT_EQ(handle.id, 0);
    auto context = loader.getDeviceContext();
    const uint32_t count = 1;
    ASSERT_EQ(hipMemcpy(context.requests, &handle.id, sizeof(handle.id), hipMemcpyHostToDevice), hipSuccess);
    ASSERT_EQ(hipMemcpy(context.requestCount, &count, sizeof(count), hipMemcpyHostToDevice), hipSuccess);
    ASSERT_EQ(loader.processRequests(nullptr, context), 1);
    loader.launchPrepare(nullptr);
    hipFunction_t sampler = nullptr;
    ASSERT_EQ(hipModuleGetFunction(&sampler, module_, "sampleLegacyContract"), hipSuccess);
    void* args[]{&context, &output_};
    ASSERT_EQ(hipModuleLaunchKernel(sampler, 1, 1, 1, 1, 1, 1, 0, nullptr, args, nullptr), hipSuccess);
    float4 color{};
    ASSERT_EQ(hipMemcpy(&color, output_, sizeof(color), hipMemcpyDeviceToHost), hipSuccess);
    constexpr float tolerance = 1e-6f;
    EXPECT_NEAR(color.x, 64.0f / 255.0f, tolerance);
    EXPECT_NEAR(color.y, 128.0f / 255.0f, tolerance);
    EXPECT_NEAR(color.z, 192.0f / 255.0f, tolerance);
    EXPECT_NEAR(color.w, 1, tolerance);
    loader.unloadTexture(handle.id);
    EXPECT_EQ(loader.getResidentTextureCount(), 0);
}

} // namespace
