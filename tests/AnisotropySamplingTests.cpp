// SPDX-License-Identifier: MIT
#include "AnisotropyReference.h"
#include "ImageDataTestUtils.h"
#include "TestUtils.h"
#include "TextureSamplingHarness.h"
#include <cstring>
#include <iomanip>
#include <numeric>
#include <string>
#include <tuple>

namespace hip_demand { namespace test {
namespace {
namespace ref = anisotropy_reference;
namespace cap = capability_v1;
namespace aniso = anisotropy_v1;
using contract_v1::Outcome;

enum class ResourceCase { Authored, Generated, Ordinary };
using QualificationCase = std::tuple<unsigned int, hipTextureFilterMode, hipTextureFilterMode, ResourceCase>;

struct Geometry {
    unsigned int width, height;
    ref::Jacobian jacobian;
    const char* name;
};

std::array<Geometry, 4> geometries(unsigned int ratio) {
    const double major = 2.0 * ratio;
    auto swapped = ref::orientedJacobian(major, 2, 1.1, .37);
    std::swap(swapped.dx, swapped.dy);
    swapped.dx = swapped.dx * -1;
    return {{{128, 128, {{major, 0}, {0, 2}}, "AxisSquare"},
             {128, 64, ref::orientedJacobian(major, 2, .47), "RotatedRectangle"},
             {127, 61, ref::orientedJacobian(major, 2, .37, .61), "ShearedNpot"},
             {64, 128, swapped, "SwappedReflected"}}};
}

ref::Pixel pixel(float4 value) { return {value.x, value.y, value.z, value.w}; }

std::vector<unsigned char> floatBytes(const ref::Image& image) {
    std::vector<unsigned char> result(image.pixels.size() * 4 * sizeof(float));
    for (size_t i = 0; i < image.pixels.size(); ++i) {
        for (size_t c = 0; c < 4; ++c) {
            const float value = static_cast<float>(image.pixels[i][c]);
            std::memcpy(result.data() + (i * 4 + c) * sizeof(float), &value, sizeof(value));
        }
    }
    return result;
}

std::shared_ptr<TypedImageSource> sourceFor(const std::vector<ref::Image>& levels, ResourceCase resource) {
    auto source = std::make_shared<TypedImageSource>();
    source->info.width = levels.front().width;
    source->info.height = levels.front().height;
    source->info.numChannels = 4;
    source->info.numMipLevels = resource == ResourceCase::Authored
        ? static_cast<unsigned int>(levels.size()) : 1;
    source->info.format = HIP_AD_FORMAT_FLOAT;
    source->info.isValid = true;
    source->pixels = floatBytes(levels.front());
    if (resource == ResourceCase::Authored)
        for (const auto& level : levels)
            source->mipPixels.push_back(floatBytes(level));
    return source;
}

SamplingInput gradient(uint32_t id, const Geometry& geometry, ref::Vec2 position) {
    SamplingInput input;
    input.textureId = id;
    input.path = SamplingPath::Gradient;
    input.u = static_cast<float>(position.x / geometry.width);
    input.v = static_cast<float>(position.y / geometry.height);
    input.ddx = {static_cast<float>(geometry.jacobian.dx.x / geometry.width),
                 static_cast<float>(geometry.jacobian.dx.y / geometry.height)};
    input.ddy = {static_cast<float>(geometry.jacobian.dy.x / geometry.width),
                 static_cast<float>(geometry.jacobian.dy.y / geometry.height)};
    input.defaultColor = {-.25f, .125f, -.5f, .375f};
    return input;
}

ref::Vec2 actualPosition(const SamplingInput& input, const Geometry& geometry) {
    return {double(input.u) * geometry.width, double(input.v) * geometry.height};
}

ref::Jacobian actualJacobian(const SamplingInput& input, const Geometry& geometry) {
    return ref::texelJacobian({input.ddx.x, input.ddx.y}, {input.ddy.x, input.ddy.y},
                             geometry.width, geometry.height);
}

double rms(const std::vector<SamplingResult>& samples, const std::vector<ref::Pixel>& expected,
           size_t channel) {
    double squared = 0;
    for (size_t i = 0; i < samples.size(); ++i) {
        const double error = pixel(samples[i].value)[channel] - expected[i][channel];
        squared += error * error;
    }
    return std::sqrt(squared / samples.size());
}

double detailGain(const std::vector<SamplingResult>& samples, const std::vector<ref::Pixel>& expected) {
    double mean = 0, actualMean = 0;
    for (size_t i = 0; i < samples.size(); ++i) {
        mean += expected[i][1];
        actualMean += samples[i].value.y;
    }
    mean /= samples.size();
    actualMean /= samples.size();
    double energy = 0, correlation = 0;
    for (size_t i = 0; i < samples.size(); ++i) {
        const double delta = expected[i][1] - mean;
        energy += delta * delta;
        correlation += delta * (samples[i].value.y - actualMean);
    }
    return energy > 1e-8 ? correlation / energy : std::numeric_limits<double>::quiet_NaN();
}

class AnisotropySampling : public LoaderTestFixture, public testing::WithParamInterface<QualificationCase> {
protected:
    void SetUp() override {
        LoaderTestFixture::SetUp();
        if (HasFatalFailure())
            return;
        std::filesystem::path module;
        ASSERT_NO_THROW(module = samplingModulePath());
        ASSERT_EQ(harness_.open(module), hipSuccess);
        std::cout << "Anisotropy reference=principal-axis box, cell-clipped degree-2 quadrature"
                  << " majorRMS=" << ref::MajorRmsLimit << " minorRMS=" << ref::MinorRmsLimit
                  << " minorGain=[" << ref::MinimumMinorGain << ',' << ref::MaximumMinorGain << ']'
                  << " alpha=" << ref::AlphaTolerance << " reload=" << ref::ReloadTolerance << '\n';
    }
    void TearDown() override {
        EXPECT_EQ(harness_.close(), hipSuccess);
        EXPECT_TRUE(harness_.isClosed());
        LoaderTestFixture::TearDown();
    }
    unsigned int ratio() const { return std::get<0>(GetParam()); }
    ResourceCase resourceCase() const { return std::get<3>(GetParam()); }
    ref::Spatial spatial() const {
        return std::get<1>(GetParam()) == hipFilterModePoint ? ref::Spatial::Point : ref::Spatial::Linear;
    }
    TextureDesc descriptor() const {
        TextureDesc desc;
        desc.addressMode[0] = desc.addressMode[1] = hipAddressModeClamp;
        desc.filterMode = std::get<1>(GetParam());
        desc.mipmapFilterMode = std::get<2>(GetParam());
        desc.normalizedCoords = true;
        desc.sRGB = false;
        desc.generateMipmaps = resourceCase() == ResourceCase::Generated;
        return desc;
    }
    TextureHandle create(const std::shared_ptr<TypedImageSource>& source, unsigned int maximum) {
        aniso::Request request;
        request.maxAnisotropy = maximum;
        cap::Policy policy;
        if (resourceCase() == ResourceCase::Ordinary)
            policy.mipPolicy = cap::MipPolicy::Disabled;
        return loader_->createTextureAnisotropyV1(source, descriptor(), request, policy);
    }
    void loadAfterStrictMiss(TextureHandle handle, const Geometry& geometry) {
        ASSERT_TRUE(handle.valid) << "Registration failed: " << static_cast<unsigned int>(handle.error);
        loader_->launchPrepare();
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        const auto context = loader_->getDeviceContext();
        const auto input = gradient(handle.id, geometry, {geometry.width / 2.0, geometry.height / 2.0});
        std::vector<SamplingResult> miss;
        ASSERT_EQ(harness_.sample(context, {input}, miss), hipSuccess);
        ASSERT_EQ(miss.size(), 1u);
        EXPECT_EQ(miss.front().resident, 0u);
        EXPECT_EQ(pixel(miss.front().value), pixel(input.defaultColor));
        uint32_t count = 0, requested = InvalidTextureId, overflow = 1;
        ASSERT_EQ(hipMemcpy(&count, context.requestCount, sizeof(count), hipMemcpyDeviceToHost), hipSuccess);
        ASSERT_EQ(count, 1u);
        ASSERT_EQ(hipMemcpy(&requested, context.requests, sizeof(requested), hipMemcpyDeviceToHost), hipSuccess);
        ASSERT_EQ(hipMemcpy(&overflow, context.requestOverflow, sizeof(overflow), hipMemcpyDeviceToHost), hipSuccess);
        EXPECT_EQ(requested, handle.id);
        EXPECT_EQ(overflow, 0u);
        const auto loaded = loader_->processRequests(nullptr, context);
        cap::Status status;
        ASSERT_EQ(loader_->getTextureStatusV1(handle.id, status), Outcome::Success);
        ASSERT_EQ(loaded, 1u) << "Required native sampling proof blocker: operation="
            << static_cast<unsigned int>(status.primary.operation) << " HIP=" << status.primary.rawHipError
            << " outcome=" << static_cast<unsigned int>(status.primary.outcome)
            << ". Unsupported nonzero anisotropy is a failure, never a skip or legacy-zero substitution.";
        loader_->launchPrepare();
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        ASSERT_EQ(loader_->getTextureStatusV1(handle.id, status), Outcome::Success);
        ASSERT_EQ(status.primary.outcome, Outcome::Success)
            << "operation=" << static_cast<unsigned int>(status.primary.operation)
            << " HIP=" << status.primary.rawHipError;
        ASSERT_EQ(status.published, 1u) << "A completed load is not evidence of sampler publication";
        ASSERT_TRUE(status.state == cap::State::Resident || status.state == cap::State::Degraded);
    }
    void verifyResource(TextureHandle handle, unsigned int requestedRatio, const std::vector<ref::Image>& pyramid) {
        cap::Status status;
        ASSERT_EQ(loader_->getTextureStatusV1(handle.id, status), Outcome::Success);
        const bool mipmapped = resourceCase() != ResourceCase::Ordinary;
        const auto levels = mipmapped ? pyramid.size() : size_t{1};
        const bool degraded = requestedRatio > 1 ||
                              (status.returned && status.returnedSampler.maxAnisotropy != requestedRatio);
        ASSERT_EQ(status.state, degraded ? cap::State::Degraded : cap::State::Resident);
        ASSERT_EQ(status.primary.outcome, Outcome::Success);
        EXPECT_EQ(status.fallback.outcome, Outcome::Success);
        ASSERT_EQ(status.submitted, 1u);
        ASSERT_EQ(status.returned, 1u);
        ASSERT_EQ(status.published, 1u);
        EXPECT_NE(status.capability, cap::Support::BehaviorQualified);
        EXPECT_EQ(status.resource, mipmapped ? cap::Resource::MipmappedArray : cap::Resource::Array);
        EXPECT_EQ(status.resourceLevels, levels);
        EXPECT_EQ(status.firstResidentMip, 0u);
        EXPECT_EQ(status.lastResidentMip, levels - 1);
        EXPECT_EQ(status.resourceWidth, pyramid.front().width);
        EXPECT_EQ(status.resourceHeight, pyramid.front().height);
        EXPECT_NE(status.reason, cap::Reason::CapabilityFallback);
        EXPECT_EQ(status.submittedSampler.maxAnisotropy, requestedRatio);
        EXPECT_EQ(status.submittedSampler.filterMode, descriptor().filterMode);
        EXPECT_EQ(status.requested.mipmapFilterMode, descriptor().mipmapFilterMode);
        if (mipmapped)
            EXPECT_EQ(status.submittedSampler.mipmapFilterMode, descriptor().mipmapFilterMode);
        hipTextureObject_t object{};
        ASSERT_EQ(hipMemcpy(&object, loader_->getDeviceContext().textures + handle.id, sizeof(object),
                            hipMemcpyDeviceToHost), hipSuccess);
        ASSERT_NE(object, hipTextureObject_t{});
        hipTextureDesc returned{};
        hipResourceDesc resource{};
        ASSERT_EQ(hipGetTextureObjectTextureDesc(&returned, object), hipSuccess);
        ASSERT_EQ(hipGetTextureObjectResourceDesc(&resource, object), hipSuccess);
        // Readback may round an intermediate request to a hardware mode. The
        // immutable submission and the actual returned state remain separate;
        // pixels, not exact returned ratio equality, decide quality below.
        EXPECT_EQ(returned.maxAnisotropy, status.returnedSampler.maxAnisotropy);
        EXPECT_EQ(returned.filterMode, descriptor().filterMode);
        EXPECT_EQ(returned.normalizedCoords, 1);
        EXPECT_EQ(returned.sRGB, 0);
        ASSERT_EQ(resource.resType, mipmapped ? hipResourceTypeMipmappedArray : hipResourceTypeArray)
            << "Base fallback is not anisotropic mip qualification";
        size_t bytes = 0;
        for (size_t level = 0; level < levels; ++level) {
            SCOPED_TRACE(level);
            hipArray_t array{};
            if (mipmapped) {
                ASSERT_EQ(hipGetMipmappedArrayLevel(&array, resource.res.mipmap.mipmap,
                                                   static_cast<unsigned int>(level)), hipSuccess);
            } else {
                array = resource.res.array.array;
            }
            hipChannelFormatDesc channels{};
            hipExtent extent{};
            unsigned int flags = 0;
            ASSERT_EQ(hipArrayGetInfo(&channels, &extent, &flags, array), hipSuccess);
            EXPECT_EQ(extent.width, pyramid[level].width);
            EXPECT_EQ(extent.height, pyramid[level].height);
            EXPECT_EQ(channels.f, hipChannelFormatKindFloat);
            EXPECT_EQ(channels.x, 32);
            EXPECT_EQ(channels.y, 32);
            EXPECT_EQ(channels.z, 32);
            EXPECT_EQ(channels.w, 32);
            const size_t rowBytes = pyramid[level].width * 4 * sizeof(float);
            std::vector<float> actual(pyramid[level].pixels.size() * 4);
            ASSERT_EQ(hipMemcpy2DFromArray(actual.data(), rowBytes, array, 0, 0, rowBytes,
                                           pyramid[level].height, hipMemcpyDeviceToHost), hipSuccess);
            if (resourceCase() == ResourceCase::Authored) {
                const auto authored = floatBytes(pyramid[level]);
                EXPECT_EQ(std::memcmp(actual.data(), authored.data(), authored.size()), 0);
            } else {
                double maximumError = 0;
                for (size_t i = 0; i < pyramid[level].pixels.size(); ++i)
                    for (size_t c = 0; c < 4; ++c)
                        maximumError = std::max(maximumError,
                            std::abs(actual[i * 4 + c] - pyramid[level].pixels[i][c]));
                EXPECT_LE(maximumError, ref::UploadTolerance) << "Independent generated-mip oracle";
            }
            bytes += actual.size() * sizeof(float);
        }
        EXPECT_EQ(status.payloadBytes, bytes);
        EXPECT_EQ(loader_->getTotalTextureMemory(), bytes) << "Compatible anisotropy variants share backing";
        if (mipmapped) {
            EXPECT_EQ(returned.mipmapFilterMode, descriptor().mipmapFilterMode);
            hipArray_t extra{};
            EXPECT_NE(hipGetMipmappedArrayLevel(&extra, resource.res.mipmap.mipmap,
                                                static_cast<unsigned int>(levels)), hipSuccess);
        }
    }
    void checkResidentSamples(const std::vector<SamplingResult>& values) {
        for (size_t i = 0; i < values.size(); ++i) {
            SCOPED_TRACE(i);
            EXPECT_EQ(values[i].resident, 1u);
            for (double c : pixel(values[i].value))
                EXPECT_TRUE(std::isfinite(c));
            EXPECT_NEAR(values[i].value.w, ref::Alpha, ref::AlphaTolerance);
        }
        uint32_t requests = 1;
        ASSERT_EQ(hipMemcpy(&requests, loader_->getDeviceContext().requestCount, sizeof(requests),
                            hipMemcpyDeviceToHost), hipSuccess);
        EXPECT_EQ(requests, 0u);
    }
    void sample(TextureHandle handle, std::vector<SamplingInput> inputs, std::vector<SamplingResult>& values) {
        ASSERT_TRUE(handle.valid);
        cap::Status status;
        ASSERT_EQ(loader_->getTextureStatusV1(handle.id, status), Outcome::Success);
        ASSERT_EQ(status.primary.outcome, Outcome::Success)
            << "operation=" << static_cast<unsigned int>(status.primary.operation)
            << " HIP=" << status.primary.rawHipError;
        ASSERT_EQ(status.published, 1u);
        ASSERT_EQ(status.returned, 1u);
        ASSERT_TRUE(status.state == cap::State::Resident || status.state == cap::State::Degraded);
        ASSERT_FALSE(inputs.empty());
        ASSERT_LE(inputs.size(), TextureSamplingHarness::MaxSamples);
        for (auto& input : inputs)
            input.textureId = handle.id;
        ASSERT_EQ(harness_.sample(loader_->getDeviceContext(), inputs, values), hipSuccess);
        ASSERT_EQ(values.size(), inputs.size());
        ASSERT_NO_FATAL_FAILURE(checkResidentSamples(values));
    }
    void checkDisabledBaseline(const std::vector<ref::Image>& pyramid, const Geometry& geometry,
                               const std::vector<SamplingInput>& inputs, const std::vector<SamplingResult>& values) {
        // Disabled anisotropy uses the usual maximum derivative-length LOD,
        // independently of the SVD footprint quality reference. This is a
        // scalar disabled-control oracle, never an anisotropic demand rule.
        for (size_t i = 0; i < inputs.size(); ++i) {
            const auto j = actualJacobian(inputs[i], geometry);
            const double rho = std::max(std::hypot(j.dx.x, j.dx.y), std::hypot(j.dy.x, j.dy.y));
            const double lod = resourceCase() == ResourceCase::Ordinary ? 0 :
                std::clamp(std::log2(rho), 0.0, double(pyramid.size() - 1));
            const bool pointMip = descriptor().mipmapFilterMode == hipFilterModePoint;
            const auto lower = static_cast<size_t>(std::floor(lod + (pointMip ? .5 : 0)));
            const auto upper = pointMip ? lower : std::min(lower + 1, pyramid.size() - 1);
            const double weight = pointMip ? 0 : lod - lower;
            const auto a = ref::spatialSample(pyramid[lower],
                {double(inputs[i].u) * pyramid[lower].width, double(inputs[i].v) * pyramid[lower].height}, spatial());
            const auto b = ref::spatialSample(pyramid[upper],
                {double(inputs[i].u) * pyramid[upper].width, double(inputs[i].v) * pyramid[upper].height}, spatial());
            for (size_t c = 0; c < 3; ++c)
                EXPECT_NEAR(pixel(values[i].value)[c], a[c] + weight * (b[c] - a[c]), ref::DisabledPixelTolerance)
                    << "disabled baseline channel=" << c << " sample=" << i;
        }
    }
    void checkDirectional(const std::vector<SamplingResult>& baseline,
                          const std::vector<SamplingResult>& candidate,
                          const std::vector<ref::Pixel>& expected, bool compareBaseline) {
        const double major = rms(candidate, expected, 0), minor = rms(candidate, expected, 1);
        const double majorBaseline = rms(baseline, expected, 0), minorBaseline = rms(baseline, expected, 1);
        const double gain = detailGain(candidate, expected);
        std::cout << "ratio=" << ratio() << " majorRMS=" << major << " disabledMajorRMS=" << majorBaseline
                  << " minorRMS=" << minor << " disabledMinorRMS=" << minorBaseline << " minorGain=" << gain << '\n';
        EXPECT_LE(major, ref::MajorRmsLimit);
        EXPECT_LE(minor, ref::MinorRmsLimit);
        EXPECT_GE(gain, ref::MinimumMinorGain) << "A flat or isotropically blurred image cannot pass";
        EXPECT_LE(gain, ref::MaximumMinorGain);
        if (compareBaseline && ratio() >= 4) {
            if (resourceCase() == ResourceCase::Ordinary) {
                EXPECT_GT(majorBaseline, .05) << "Disabled control must expose major-axis aliasing";
                EXPECT_LE(major, ref::MajorImprovementFactor * majorBaseline);
            } else {
                // Disabled full-chain filtering already suppresses major
                // frequencies by destroying minor detail. Require recovery of
                // the latter WITHOUT introducing major-axis aliasing.
                EXPECT_GT(minorBaseline, .04) << "Disabled control must expose minor-axis blur";
                EXPECT_LE(minor, ref::MinorImprovementFactor * minorBaseline);
                EXPECT_LE(major, majorBaseline + ref::MajorNonRegression);
            }
        }
    }
    void unloadPair(TextureHandle baseline, TextureHandle candidate) {
        loader_->unloadTexture(candidate.id);
        if (baseline.id != candidate.id)
            loader_->unloadTexture(baseline.id);
        EXPECT_EQ(loader_->getResidentTextureCount(), 0u);
        EXPECT_EQ(loader_->getTotalTextureMemory(), 0u);
    }
    TextureSamplingHarness harness_;
};

TEST_P(AnisotropySampling, DirectionalFrequenciesAcrossGeometryAndGrazingSweep) {
    const auto cases = geometries(ratio());
    for (size_t scene = 0; scene < cases.size() + 1; ++scene) {
        const bool sweep = scene == cases.size();
        auto geometry = sweep ? Geometry{128, 64, {{2, 0}, {0, 2}}, "GrazingSweep"} : cases[scene];
        SCOPED_TRACE(geometry.name);
        const auto axes = ref::singularAxes(geometry.jacobian);
        const auto pyramid = ref::mipPyramid(
            ref::makeImage(geometry.width, geometry.height, ref::Pattern::Directional, axes.direction));
        auto source = sourceFor(pyramid, resourceCase());
        const auto baseline = create(source, 1), candidate = create(source, ratio());
        ASSERT_TRUE(baseline.valid && candidate.valid);
        std::vector<SamplingInput> inputs;
        std::vector<ref::Pixel> expected;
        const ref::Vec2 center{geometry.width / 2.0 + .5, geometry.height / 2.0 + .5};
        const std::array<double, 8> grazingMajor{{2, 4, 6, 8, 10, 14, 16, 32}};
        const unsigned int phases = sweep ? 4 : 8;
        for (unsigned int step = 0; step < (sweep ? grazingMajor.size() : 1); ++step) {
            auto current = geometry;
            if (sweep)
                current.jacobian = {{grazingMajor[step], 0}, {0, 2}};
            for (unsigned int y = 0; y < phases; ++y) {
                for (unsigned int x = 0; x < phases; ++x) {
                    const auto p = center + axes.direction * (4.0 * (x + .23) / phases - 2) +
                                   axes.perpendicular() * (8.0 * (y + .37) / phases - 4);
                    const auto input = gradient(candidate.id, current, p);
                    inputs.push_back(input);
                    expected.push_back(ref::integrate(pyramid.front(), actualPosition(input, current),
                        ref::footprint(actualJacobian(input, current), ratio()), spatial()));
                }
            }
        }
        ASSERT_LE(inputs.size(), TextureSamplingHarness::MaxSamples);
        ASSERT_NO_FATAL_FAILURE(loadAfterStrictMiss(baseline, geometry));
        if (candidate.id != baseline.id)
            ASSERT_NO_FATAL_FAILURE(loadAfterStrictMiss(candidate, geometry));
        ASSERT_NO_FATAL_FAILURE(verifyResource(baseline, 1, pyramid));
        ASSERT_NO_FATAL_FAILURE(verifyResource(candidate, ratio(), pyramid));
        std::vector<unsigned int> reads(resourceCase() == ResourceCase::Authored ? pyramid.size() : 1);
        std::iota(reads.begin(), reads.end(), 0u);
        EXPECT_EQ(source->readLevels, reads);
        std::vector<SamplingResult> disabled, anisotropic;
        ASSERT_NO_FATAL_FAILURE(sample(baseline, inputs, disabled));
        ASSERT_NO_FATAL_FAILURE(sample(candidate, inputs, anisotropic));
        if (ratio() == 1 && !sweep) {
            ASSERT_NO_FATAL_FAILURE(checkDisabledBaseline(pyramid, geometry, inputs, anisotropic));
        } else if (!sweep) {
            ASSERT_NO_FATAL_FAILURE(checkDirectional(disabled, anisotropic, expected, true));
        } else {
            // Evaluate each grazing step independently: averaging the sweep
            // could otherwise conceal loss of detail at the steepest angle.
            for (size_t step = 0; step < grazingMajor.size(); ++step) {
                SCOPED_TRACE(testing::Message() << "major=" << grazingMajor[step]);
                const size_t first = step * 16;
                const auto begin = static_cast<std::ptrdiff_t>(first);
                const auto end = static_cast<std::ptrdiff_t>(first + 16);
                const std::vector<SamplingResult> a(disabled.begin() + begin, disabled.begin() + end);
                const std::vector<SamplingResult> b(anisotropic.begin() + begin, anisotropic.begin() + end);
                const std::vector<ref::Pixel> e(expected.begin() + begin, expected.begin() + end);
                // Require minor-detail gain below the green signal's
                // coarse-mip Nyquist boundary. At/above that boundary, point
                // mip quantization can legitimately remove its phase detail;
                // major-axis suppression is still required.
                const double minorWidth = std::max(2.0, grazingMajor[step] / ratio());
                if (ratio() > 1 && grazingMajor[step] >= 4 && minorWidth < 4)
                    ASSERT_NO_FATAL_FAILURE(checkDirectional(a, b, e, grazingMajor[step] >= 8));
                else if (ratio() > 1 && grazingMajor[step] >= 4)
                    EXPECT_LE(rms(b, e, 0), ref::MajorRmsLimit);
                if (ratio() == 1 || step == 0)
                    ASSERT_NO_FATAL_FAILURE(checkDisabledBaseline(pyramid, geometry,
                        std::vector<SamplingInput>(inputs.begin() + begin, inputs.begin() + end), b));
            }
        }
        ASSERT_NO_FATAL_FAILURE(unloadPair(baseline, candidate));
    }
}

TEST_P(AnisotropySampling, ImpulseRampAlphaAndDeterministicReload) {
    for (ref::Pattern pattern : {ref::Pattern::Impulse, ref::Pattern::Ramp}) {
        for (unsigned int orientation : {0u, 1u}) {
            Geometry geometry{127, 61, ref::orientedJacobian(16, 2, orientation ? .47 : 0,
                                                           orientation ? .61 : 0),
                              orientation ? "ShearedRotated" : "AxisAligned"};
            SCOPED_TRACE(testing::Message() << geometry.name << " pattern=" << static_cast<unsigned int>(pattern));
            const auto axes = ref::footprint(geometry.jacobian, ratio());
            const auto pyramid = ref::mipPyramid(ref::makeImage(geometry.width, geometry.height, pattern));
            auto source = sourceFor(pyramid, resourceCase());
            const auto handle = create(source, ratio());
            ASSERT_TRUE(handle.valid);
            const ref::Vec2 center{double(geometry.width / 2) + .5, double(geometry.height / 2) + .5};
            std::vector<SamplingInput> inputs;
            std::vector<ref::Pixel> expected;
            for (unsigned int y = 0; y < 15; ++y) {
                for (unsigned int x = 0; x < 15; ++x) {
                    const auto p = center + axes.direction * ((2 * axes.major + 4) * (x / 14.0 - .5)) +
                                   axes.perpendicular() * ((2 * axes.minor + 4) * (y / 14.0 - .5));
                    const auto input = gradient(handle.id, geometry, p);
                    inputs.push_back(input);
                    expected.push_back(ref::integrate(pyramid.front(), actualPosition(input, geometry),
                        ref::footprint(actualJacobian(input, geometry), ratio()), spatial()));
                }
            }
            ASSERT_EQ(inputs.size(), 225u);
            ASSERT_NO_FATAL_FAILURE(loadAfterStrictMiss(handle, geometry));
            ASSERT_NO_FATAL_FAILURE(verifyResource(handle, ratio(), pyramid));
            std::vector<SamplingResult> initial;
            ASSERT_NO_FATAL_FAILURE(sample(handle, inputs, initial));
            if (ratio() == 1) {
                ASSERT_NO_FATAL_FAILURE(checkDisabledBaseline(pyramid, geometry, inputs, initial));
            } else if (pattern == ref::Pattern::Ramp) {
                // Spatial point filtering quantizes to a coarse mip texel.
                // Bound that known positional uncertainty by one capped
                // minor-axis width times the affine slope, in addition to
                // the fixed residual tolerance. Linear reconstruction has
                // no such affine quantization allowance.
                const double cellWidth = spatial() == ref::Spatial::Point &&
                    resourceCase() != ResourceCase::Ordinary ? axes.minor : 0;
                const ref::Pixel slope{.6 / geometry.width, .5 / geometry.height,
                                       .3 / geometry.width + .2 / geometry.height, 0};
                for (size_t i = 0; i < initial.size(); ++i)
                    for (size_t c = 0; c < 3; ++c)
                        EXPECT_NEAR(pixel(initial[i].value)[c], expected[i][c],
                                    ref::RampTolerance + cellWidth * slope[c])
                            << "sample=" << i << " channel=" << c;
            } else {
                EXPECT_LE(rms(initial, expected, 0), ref::ImpulseRmsLimit);
                double mass = 0, referenceMass = 0, maximumError = 0;
                ref::Vec2 moment{}, referenceMoment{};
                for (size_t i = 0; i < initial.size(); ++i) {
                    const double value = initial[i].value.x;
                    EXPECT_GE(value, -ref::AlphaTolerance);
                    EXPECT_LE(value, 1 + ref::AlphaTolerance);
                    EXPECT_NEAR(initial[i].value.y, value, ref::AlphaTolerance);
                    EXPECT_NEAR(initial[i].value.z, value, ref::AlphaTolerance);
                    const auto offset = actualPosition(inputs[i], geometry) - center;
                    mass += value;
                    referenceMass += expected[i][0];
                    moment = moment + offset * value;
                    referenceMoment = referenceMoment + offset * expected[i][0];
                    maximumError = std::max(maximumError, std::abs(value - expected[i][0]));
                }
                ASSERT_GT(referenceMass, 0);
                EXPECT_LE(maximumError, ref::ImpulseMaximumError);
                EXPECT_GE(mass / referenceMass, ref::ImpulseMinimumMass);
                EXPECT_LE(mass / referenceMass, ref::ImpulseMaximumMass);
                ASSERT_GT(mass, 0) << "An absent impulse cannot pass an absolute-error-only test";
                const auto centroid = moment * (1 / mass) - referenceMoment * (1 / referenceMass);
                // Coarse mip cells localize an authored impulse only to their
                // scale; exact subtexel centroids would dictate vendor taps.
                const double localization = resourceCase() == ResourceCase::Ordinary ? 1 : axes.minor;
                EXPECT_LE(std::hypot(centroid.x, centroid.y), ref::ImpulseCentroidTolerance * localization);
            }
            loader_->unloadTexture(handle.id);
            ASSERT_EQ(loader_->getTotalTextureMemory(), 0u);
            EXPECT_EQ(create(source, ratio()).id, handle.id);
            ASSERT_NO_FATAL_FAILURE(loadAfterStrictMiss(handle, geometry));
            ASSERT_NO_FATAL_FAILURE(verifyResource(handle, ratio(), pyramid));
            std::vector<SamplingResult> reloaded;
            ASSERT_NO_FATAL_FAILURE(sample(handle, inputs, reloaded));
            ASSERT_EQ(reloaded.size(), initial.size());
            for (size_t i = 0; i < initial.size(); ++i)
                for (size_t c = 0; c < 4; ++c)
                    EXPECT_NEAR(pixel(reloaded[i].value)[c], pixel(initial[i].value)[c], ref::ReloadTolerance);
            loader_->unloadTexture(handle.id);
            EXPECT_EQ(loader_->getResidentTextureCount(), 0u);
            EXPECT_EQ(loader_->getTotalTextureMemory(), 0u);
        }
    }
}

using NativeCase = std::tuple<unsigned int, bool>;

// Independent of DemandTextureLoader, its probes, fault seams, source reads,
// descriptor translation and publication. The harness receives one already
// resident native object; its Gradient branch forwards to ::tex2DGrad.
class AnisotropyNative : public HipTestFixture, public testing::WithParamInterface<NativeCase> {
protected:
    void TearDown() override {
        EXPECT_EQ(harness_.close(), hipSuccess);
        if (table_)
            EXPECT_EQ(hipFree(table_), hipSuccess);
        if (flags_)
            EXPECT_EQ(hipFree(flags_), hipSuccess);
        const auto destroyed = object_ ? hipDestroyTextureObject(object_) : hipSuccess;
        EXPECT_EQ(destroyed, hipSuccess);
        if (destroyed == hipSuccess) {
            if (mipmap_)
                EXPECT_EQ(hipFreeMipmappedArray(mipmap_), hipSuccess);
            if (array_)
                EXPECT_EQ(hipFreeArray(array_), hipSuccess);
        }
        HipTestFixture::TearDown();
    }
    TextureSamplingHarness harness_;
    hipTextureObject_t object_{};
    hipArray_t array_ = nullptr;
    hipMipmappedArray_t mipmap_ = nullptr;
    TextureObject* table_ = nullptr;
    uint32_t* flags_ = nullptr;
};

TEST_P(AnisotropyNative, DirectDescriptorCreateReadbackAndGradient) {
    const auto [ratio, mipmapped] = GetParam();
    const auto pyramid = ref::mipPyramid(ref::makeImage(64, 32, ref::Pattern::Directional));
    const auto channels = hipCreateChannelDesc<float4>();
    const auto levels = mipmapped ? static_cast<unsigned int>(pyramid.size()) : 1u;
    hipResourceDesc resource{};
    if (mipmapped) {
        ASSERT_EQ(hipMallocMipmappedArray(&mipmap_, &channels, make_hipExtent(64, 32, 0), levels, 0), hipSuccess);
        resource.resType = hipResourceTypeMipmappedArray;
        resource.res.mipmap.mipmap = mipmap_;
    } else {
        ASSERT_EQ(hipMallocArray(&array_, &channels, 64, 32), hipSuccess);
        resource.resType = hipResourceTypeArray;
        resource.res.array.array = array_;
    }
    for (unsigned int level = 0; level < levels; ++level) {
        hipArray_t array = array_;
        if (mipmapped)
            ASSERT_EQ(hipGetMipmappedArrayLevel(&array, mipmap_, level), hipSuccess);
        const auto bytes = floatBytes(pyramid[level]);
        const size_t rowBytes = pyramid[level].width * 4 * sizeof(float);
        ASSERT_EQ(hipMemcpy2DToArray(array, 0, 0, bytes.data(), rowBytes, rowBytes,
                                     pyramid[level].height, hipMemcpyHostToDevice), hipSuccess);
    }
    hipTextureDesc submitted{};
    submitted.addressMode[0] = submitted.addressMode[1] = hipAddressModeClamp;
    submitted.filterMode = hipFilterModeLinear;
    submitted.mipmapFilterMode = hipFilterModePoint;
    submitted.readMode = hipReadModeElementType;
    submitted.normalizedCoords = 1;
    submitted.maxAnisotropy = ratio;
    submitted.minMipmapLevelClamp = 0;
    submitted.maxMipmapLevelClamp = static_cast<float>(levels - 1);
    // Zero is a separately named native legacy control, never a replacement
    // for a failed explicit request. Only this field varies within a path.
    const auto created = hipCreateTextureObject(&object_, &resource, &submitted, nullptr);
    std::cout << "Direct HIP resource=" << (mipmapped ? "mipmapped" : "array")
              << " FLOAT_RGBA=64x32 levels=" << levels << " spatial=linear mip=point"
              << " submittedMaxAnisotropy=" << ratio << " createRawHIP=" << static_cast<int>(created)
              << " (" << hipGetErrorString(created) << ')' << '\n';
    ASSERT_EQ(created, hipSuccess) << "Native descriptor rejected; no loader/probe/fallback was involved";
    ASSERT_NE(object_, hipTextureObject_t{});
    hipTextureDesc returned{};
    const auto read = hipGetTextureObjectTextureDesc(&returned, object_);
    std::cout << "Direct HIP readbackRawHIP=" << static_cast<int>(read)
              << " (" << hipGetErrorString(read) << ')';
    if (read == hipSuccess)
        std::cout << " returnedMaxAnisotropy=" << returned.maxAnisotropy;
    std::cout << '\n';
    ASSERT_EQ(read, hipSuccess) << "Native creation succeeded but native descriptor readback failed";
    EXPECT_EQ(returned.filterMode, submitted.filterMode);
    EXPECT_EQ(returned.normalizedCoords, submitted.normalizedCoords);
    EXPECT_EQ(returned.readMode, submitted.readMode);
    hipResourceDesc actualResource{};
    ASSERT_EQ(hipGetTextureObjectResourceDesc(&actualResource, object_), hipSuccess);
    ASSERT_EQ(actualResource.resType, resource.resType);
    if (mipmapped)
        EXPECT_EQ(actualResource.res.mipmap.mipmap, mipmap_);
    else
        EXPECT_EQ(actualResource.res.array.array, array_);
    std::filesystem::path module;
    ASSERT_NO_THROW(module = samplingModulePath());
    ASSERT_EQ(harness_.open(module), hipSuccess);
    ASSERT_EQ(hipMalloc(&flags_, 4 * sizeof(uint32_t)), hipSuccess);
    ASSERT_EQ(hipMalloc(&table_, sizeof(TextureObject)), hipSuccess);
    const std::array<uint32_t, 4> initial{{1, 0, 0, InvalidTextureId}};
    ASSERT_EQ(hipMemcpy(flags_, initial.data(), sizeof(initial), hipMemcpyHostToDevice), hipSuccess);
    ASSERT_EQ(hipMemcpy(table_, &object_, sizeof(object_), hipMemcpyHostToDevice), hipSuccess);
    DeviceContext context{};
    context.residentFlags = flags_;
    context.textures = table_;
    context.requestCount = flags_ + 1;
    context.requestOverflow = flags_ + 2;
    context.requests = flags_ + 3;
    context.maxTextures = context.maxRequests = 1;
    const Geometry geometry{64, 32, {{16, 0}, {0, 2}}, "NativeDirect"};
    std::vector<SamplingInput> inputs;
    for (unsigned int y = 0; y < 4; ++y)
        for (unsigned int x = 0; x < 4; ++x)
            inputs.push_back(gradient(0, geometry, {30.25 + x, 12.75 + 2 * y}));
    std::vector<SamplingResult> samples;
    const auto sampled = harness_.sample(context, inputs, samples);
    std::cout << "Direct HIP gradientRawHIP=" << static_cast<int>(sampled)
              << " (" << hipGetErrorString(sampled) << ')' << '\n';
    ASSERT_EQ(sampled, hipSuccess);
    ASSERT_EQ(samples.size(), inputs.size());
    for (size_t i = 0; i < samples.size(); ++i) {
        const auto& sample = samples[i];
        ASSERT_EQ(sample.resident, 1u);
        for (double value : pixel(sample.value))
            EXPECT_TRUE(std::isfinite(value));
        EXPECT_NEAR(sample.value.w, ref::Alpha, ref::AlphaTolerance);
        if (ratio == 0) {
            // Separate legacy correctness control: the fixed 16x2 gradient
            // selects LOD 4 with disabled anisotropy, or level zero on an
            // ordinary array. It is not a passing nonzero-anisotropy result.
            const auto& level = pyramid[mipmapped ? 4 : 0];
            const auto expected = ref::spatialSample(level,
                {double(inputs[i].u) * level.width, double(inputs[i].v) * level.height}, ref::Spatial::Linear);
            for (size_t c = 0; c < 3; ++c)
                EXPECT_NEAR(pixel(sample.value)[c], expected[c], ref::DisabledPixelTolerance)
                    << "legacy-zero baseline sample=" << i << " channel=" << c;
        }
    }
    std::array<uint32_t, 4> after{};
    ASSERT_EQ(hipMemcpy(after.data(), flags_, sizeof(after), hipMemcpyDeviceToHost), hipSuccess);
    EXPECT_EQ(after, initial);
}

std::string nativeCaseName(const testing::TestParamInfo<NativeCase>& info) {
    const auto [ratio, mipmapped] = info.param;
    return (ratio ? "Ratio" + std::to_string(ratio) : std::string{"LegacyZeroControl"}) +
           (mipmapped ? "Mipmapped" : "Ordinary");
}

std::string qualificationCaseName(const testing::TestParamInfo<QualificationCase>& info) {
    const auto [ratio, spatial, mip, resource] = info.param;
    return "Ratio" + std::to_string(ratio) + (spatial == hipFilterModePoint ? "SpatialPoint" : "SpatialLinear") +
        (mip == hipFilterModePoint ? "MipPoint" : "MipLinear") +
        (resource == ResourceCase::Authored ? "Authored" :
         resource == ResourceCase::Generated ? "Generated" : "Ordinary");
}

INSTANTIATE_TEST_SUITE_P(RatiosAndResources, AnisotropyNative,
    testing::Combine(testing::Values(0u, 1u, 2u, 3u, 4u, 5u, 7u, 8u, 16u), testing::Bool()),
    nativeCaseName);

INSTANTIATE_TEST_SUITE_P(RatiosFiltersAndResources, AnisotropySampling,
    testing::Combine(testing::ValuesIn(ref::Ratios),
                     testing::Values(hipFilterModePoint, hipFilterModeLinear),
                     testing::Values(hipFilterModePoint, hipFilterModeLinear),
                     testing::Values(ResourceCase::Authored, ResourceCase::Generated, ResourceCase::Ordinary)),
    qualificationCaseName);

} // namespace
} }
