// SPDX-License-Identifier: MIT
#include "AnisotropyReference.h"
#include <gtest/gtest.h>
#include <tuple>

namespace hip_demand { namespace test {
namespace {
namespace ref = anisotropy_reference;

TEST(AnisotropyReference, SingularValuesUseTheWholeTexelJacobian) {
    const auto diagonal = ref::singularAxes({{16, 0}, {0, 2}});
    EXPECT_DOUBLE_EQ(diagonal.major, 16);
    EXPECT_DOUBLE_EQ(diagonal.minor, 2);
    // Both derivative vectors have length five, but are not orthogonal.
    const auto shear = ref::singularAxes({{5, 0}, {3, 4}});
    EXPECT_NEAR(shear.major, std::sqrt(40.0), 1e-13);
    EXPECT_NEAR(shear.minor, std::sqrt(10.0), 1e-13);
    EXPECT_NEAR(shear.major / shear.minor, 2, 1e-13);
    EXPECT_GT(shear.major / shear.minor, std::hypot(5, 0) / std::hypot(3, 4));
    const auto rankOne = ref::singularAxes({{3, 4}, {6, 8}});
    EXPECT_NEAR(rankOne.major, std::sqrt(125.0), 1e-13);
    EXPECT_DOUBLE_EQ(rankOne.minor, 0);
    const auto zero = ref::singularAxes({});
    EXPECT_DOUBLE_EQ(zero.major, 0);
    EXPECT_DOUBLE_EQ(zero.minor, 0);
}

TEST(AnisotropyReference, RotationsSwapsAndReflectionsPreserveSingularValues) {
    for (unsigned int ratio : ref::Ratios) {
        for (double angle : {0.0, .37, 1.2}) {
            for (double screen : {0.0, .63}) {
                auto j = ref::orientedJacobian(2.0 * ratio, 2, angle, screen);
                for (const auto transformed : {j, ref::Jacobian{j.dy, j.dx},
                                               ref::Jacobian{j.dx * -1, j.dy}}) {
                    const auto axes = ref::singularAxes(transformed);
                    EXPECT_NEAR(axes.major, 2.0 * ratio, 1e-12);
                    EXPECT_NEAR(axes.minor, 2, 1e-12);
                    if (ratio > 1)
                        EXPECT_NEAR(std::abs(ref::dot(axes.direction, {std::cos(angle), std::sin(angle)})),
                                    1, 1e-12);
                }
            }
        }
    }
}

TEST(AnisotropyReference, RectangularScalingAndNearlyCollinearAxesRemainAccurate) {
    const auto j = ref::texelJacobian({16.0 / 127, 0}, {0, 2.0 / 61}, 127, 61);
    const auto axes = ref::singularAxes(j);
    EXPECT_DOUBLE_EQ(axes.major, 16);
    EXPECT_DOUBLE_EQ(axes.minor, 2);
    const auto thin = ref::singularAxes({{1, 0}, {1, 1e-10}});
    EXPECT_NEAR(thin.major, std::sqrt(2.0), 1e-14);
    EXPECT_NEAR(thin.minor, 1e-10 / std::sqrt(2.0), 1e-24);
    const auto large = ref::singularAxes({{1e150, 0}, {0, 2e149}});
    EXPECT_NEAR(large.major / 1e150, 1, 1e-14);
    EXPECT_NEAR(large.minor / 1e150, .2, 1e-14);
}

TEST(AnisotropyReference, RatioCapsBroadenMinorAxisWithoutShrinkingMajorAxis) {
    for (unsigned int ratio : ref::Ratios) {
        const auto axes = ref::footprint({{16, 0}, {0, 1}}, ratio);
        EXPECT_DOUBLE_EQ(axes.major, 16);
        EXPECT_DOUBLE_EQ(axes.minor, 16.0 / ratio);
        const auto isotropic = ref::footprint({{2, 0}, {0, 2}}, ratio);
        EXPECT_DOUBLE_EQ(isotropic.major, 2);
        EXPECT_DOUBLE_EQ(isotropic.minor, 2);
    }
    const auto magnification = ref::footprint({{.1, 0}, {0, .2}}, 16);
    EXPECT_DOUBLE_EQ(magnification.major, 1);
    EXPECT_DOUBLE_EQ(magnification.minor, 1);
}

TEST(AnisotropyReference, InvalidAndUnboundedInputsAreRejected) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    EXPECT_THROW(ref::singularAxes({{nan, 0}, {0, 1}}), std::invalid_argument);
    EXPECT_THROW(ref::singularAxes({{1, 0}, {inf, 1}}), std::invalid_argument);
    EXPECT_THROW(ref::texelJacobian({}, {}, 0, 2), std::invalid_argument);
    EXPECT_THROW(ref::makeImage(2, 0, ref::Pattern::Ramp), std::invalid_argument);
    for (unsigned int ratio : {0u, 17u, UINT32_MAX})
        EXPECT_THROW(ref::footprint({}, ratio), std::invalid_argument);
    const auto image = ref::makeImage(8, 8, ref::Pattern::Ramp);
    EXPECT_THROW(ref::integrate(image, {}, {65, 1, {1, 0}}, ref::Spatial::Linear), std::invalid_argument);
    EXPECT_THROW(ref::integrate(image, {}, {1, 1, {1, 0}}, ref::Spatial::Linear, 3), std::invalid_argument);
    EXPECT_THROW(ref::midpointIntegral(image, {}, {1, 1, {1, 0}}, ref::Spatial::Linear, 0),
                 std::invalid_argument);
}

TEST(AnisotropyReference, DirectionalFixtureSeparatesFrequencyAxesAndKeepsAlpha) {
    const auto image = ref::makeImage(64, 32, ref::Pattern::Directional);
    const auto center = image.at(32, 16);
    EXPECT_NEAR(center[0], .9, 3e-8);
    EXPECT_NEAR(center[1], .9, 3e-8);
    EXPECT_NEAR(image.at(34, 16)[0], .1, 3e-8);
    EXPECT_DOUBLE_EQ(image.at(34, 16)[1], center[1]);
    EXPECT_NEAR(image.at(32, 20)[1], .1, 3e-8);
    EXPECT_DOUBLE_EQ(image.at(32, 20)[0], center[0]);
    for (const auto& pixel : image.pixels)
        EXPECT_DOUBLE_EQ(pixel[3], ref::Alpha);
}

TEST(AnisotropyReference, IndependentPyramidPreservesAreaRampsAndOddDimensions) {
    ref::Image odd{3, 1, {{{0, 0, 0, ref::Alpha}}, {{.5, .5, .5, ref::Alpha}},
                          {{1, 1, 1, ref::Alpha}}}};
    const auto levels = ref::mipPyramid(odd);
    ASSERT_EQ(levels.size(), 2u);
    EXPECT_EQ(levels[1].width, 1u);
    EXPECT_EQ(levels[1].height, 1u);
    EXPECT_EQ(levels[1].pixels.front(), (ref::Pixel{.5, .5, .5, ref::Alpha}));
    const auto pyramid = ref::mipPyramid(ref::makeImage(127, 61, ref::Pattern::Ramp));
    ASSERT_EQ(pyramid.size(), 7u);
    for (size_t level = 0; level < pyramid.size(); ++level) {
        EXPECT_EQ(pyramid[level].width, std::max(1u, 127u >> level));
        EXPECT_EQ(pyramid[level].height, std::max(1u, 61u >> level));
        for (const auto& pixel : pyramid[level].pixels)
            EXPECT_DOUBLE_EQ(pixel[3], ref::Alpha);
    }
}

TEST(AnisotropyReference, CellQuadratureIsExactForConstantAffineAndImpulseFields) {
    auto constant = ref::makeImage(16, 16, ref::Pattern::Ramp);
    for (auto& pixel : constant.pixels)
        pixel = {.25, .5, .75, ref::Alpha};
    const auto axes = ref::footprint(ref::orientedJacobian(9, 2, .63, .37), 8);
    for (auto mode : {ref::Spatial::Point, ref::Spatial::Linear}) {
        const auto value = ref::integrate(constant, {-.2, 15.7}, axes, mode);
        for (size_t c = 0; c < 4; ++c)
            EXPECT_NEAR(value[c], constant.pixels.front()[c], ref::QuadratureTolerance);
    }
    auto affine = constant;
    for (unsigned int y = 0; y < affine.height; ++y)
        for (unsigned int x = 0; x < affine.width; ++x)
            affine.pixels[y * affine.width + x] = {double(x) + .5, double(y) + .5,
                                                  (x + .5) * (y + .5), ref::Alpha};
    const auto value = ref::integrate(affine, {8.25, 7.125}, {4, 2, {1, 0}}, ref::Spatial::Linear);
    EXPECT_NEAR(value[0], 8.25, ref::QuadratureTolerance);
    EXPECT_NEAR(value[1], 7.125, ref::QuadratureTolerance);
    EXPECT_NEAR(value[2], 8.25 * 7.125, ref::QuadratureTolerance);
    const auto impulse = ref::makeImage(32, 32, ref::Pattern::Impulse);
    for (auto mode : {ref::Spatial::Point, ref::Spatial::Linear}) {
        const auto value = ref::integrate(impulse, {16.5, 16.5}, {8, 2, {1, 0}}, mode);
        EXPECT_NEAR(value[0], 1.0 / 16, ref::QuadratureTolerance);
        EXPECT_NEAR(value[3], ref::Alpha, ref::QuadratureTolerance);
    }
}

TEST(AnisotropyReference, QualityFixtureRejectsAliasingAndIsotropicOverblur) {
    const auto image = ref::makeImage(64, 64, ref::Pattern::Directional);
    const auto ideal = ref::footprint({{16, 0}, {0, 2}}, 8);
    const auto isotropic = ref::footprint({{16, 0}, {0, 2}}, 1);
    for (auto mode : {ref::Spatial::Point, ref::Spatial::Linear}) {
        double unfilteredMajor = 0, isotropicMinor = 0, flatMinor = 0;
        for (unsigned int y = 0; y < 8; ++y) {
            for (unsigned int x = 0; x < 8; ++x) {
                const ref::Vec2 p{30.5 + .5 * (x + .23), 28.5 + y + .37};
                const auto expected = ref::integrate(image, p, ideal, mode);
                const auto unfiltered = ref::spatialSample(image, p, mode);
                const auto blurred = ref::integrate(image, p, isotropic, mode);
                unfilteredMajor += std::pow(unfiltered[0] - expected[0], 2);
                isotropicMinor += std::pow(blurred[1] - expected[1], 2);
                flatMinor += std::pow(.5 - expected[1], 2);
            }
        }
        EXPECT_GT(std::sqrt(unfilteredMajor / 64), ref::MajorRmsLimit);
        EXPECT_GT(std::sqrt(isotropicMinor / 64), ref::MinorRmsLimit);
        EXPECT_GT(std::sqrt(flatMinor / 64), ref::MinorRmsLimit);
    }
}

TEST(AnisotropyReference, IntegratedImpulseHasUnitMassForRotatedAndAxisAlignedBoxes) {
    const auto image = ref::makeImage(32, 32, ref::Pattern::Impulse);
    for (auto mode : {ref::Spatial::Point, ref::Spatial::Linear}) {
        for (double angle : {0.0, .47}) {
            const auto axes = ref::footprint(ref::orientedJacobian(8, 2, angle), 4);
            double mass = 0;
            ref::Vec2 moment{};
            for (int y = -7; y <= 7; ++y) {
                for (int x = -7; x <= 7; ++x) {
                    const auto sample = ref::integrate(image, {16.5 + x, 16.5 + y}, axes, mode);
                    mass += sample[0];
                    moment = moment + ref::Vec2{double(x), double(y)} * sample[0];
                }
            }
            EXPECT_NEAR(mass, 1, ref::QuadratureTolerance);
            EXPECT_NEAR(moment.x, 0, ref::QuadratureTolerance);
            EXPECT_NEAR(moment.y, 0, ref::QuadratureTolerance);
        }
    }
}

using QuadratureCase = std::tuple<ref::Pattern, ref::Spatial, unsigned int>;
class AnisotropyQuadrature : public testing::TestWithParam<QuadratureCase> {};

TEST_P(AnisotropyQuadrature, SubdivisionAndIndependentMidpointsConverge) {
    const auto [pattern, mode, geometry] = GetParam();
    const std::array<ref::Jacobian, 3> jacobians{{
        {{1.5, 0}, {0, 1.5}},
        ref::orientedJacobian(16, 2, .47),
        {{12, 1}, {9, 3}}
    }};
    const auto axes = ref::footprint(jacobians[geometry], 16);
    const auto image = ref::makeImage(64, 48, pattern, axes.direction);
    for (ref::Vec2 center : {ref::Vec2{32.5, 24.5}, ref::Vec2{31.23, 25.37}}) {
        const auto exact = ref::integrate(image, center, axes, mode);
        for (unsigned int refinement : {1u, 2u}) {
            const auto refined = ref::integrate(image, center, axes, mode, refinement);
            for (size_t c = 0; c < 4; ++c)
                EXPECT_NEAR(refined[c], exact[c], ref::QuadratureTolerance);
        }
        for (unsigned int nodes : {128u, 256u}) {
            const auto midpoint = ref::midpointIntegral(image, center, axes, mode, nodes);
            // Point discontinuities converge more slowly than continuous
            // bilinear cells. Both bounds were set before device sampling.
            const double tolerance = mode == ref::Spatial::Point
                ? (nodes == 128 ? .02 : .01) : (nodes == 128 ? .0016 : .0004);
            for (size_t c = 0; c < 4; ++c)
                EXPECT_NEAR(midpoint[c], exact[c], tolerance) << "nodes=" << nodes << " channel=" << c;
        }
        EXPECT_NEAR(exact[3], ref::Alpha, ref::QuadratureTolerance);
    }
}

INSTANTIATE_TEST_SUITE_P(PatternsModesAndGeometry, AnisotropyQuadrature,
    testing::Combine(testing::Values(ref::Pattern::Directional, ref::Pattern::Impulse, ref::Pattern::Ramp),
                     testing::Values(ref::Spatial::Point, ref::Spatial::Linear),
                     testing::Values(0u, 1u, 2u)));

} // namespace
} }
