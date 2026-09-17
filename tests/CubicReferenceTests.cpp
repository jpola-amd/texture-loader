// SPDX-License-Identifier: MIT
#include "CubicReference.h"
#include "CubicExpected.h"
#include <gtest/gtest.h>
#include <limits>
#include <numeric>

using namespace cubic_reference;
TEST(CubicReference, FrozenWeightsAndSymmetry) {
    for (size_t n = 0; n < cubic_expected::fractions.size(); ++n) {
        const double a = cubic_expected::fractions[n];
        const auto w = weights(a), d = derivatives(a), reverse = weights(1-a);
        EXPECT_NEAR(std::accumulate(w.begin(),w.end(),0.),1,hostTolerance);
        EXPECT_NEAR(std::accumulate(d.begin(),d.end(),0.),0,hostTolerance);
        for (int k = 0; k < 4; ++k) {
            EXPECT_NEAR(w[k],cubic_expected::weights[n][k],hostTolerance);
            EXPECT_NEAR(w[k],reverse[3-k],hostTolerance);
        }
    }
}
TEST(CubicReference, ImpulseSensitivityAndRampDerivatives) {
    Level image{8,8,std::vector<Pixel>(64)};
    image.pixels[3*8+3] = {1,1,1,1};
    const auto impulse = reconstruct(image,3.5/8,3.5/8,Clamp,Clamp);
    EXPECT_NEAR(impulse.value[0],cubic_expected::impulseCenter,hostTolerance);
    EXPECT_GT(cubic_expected::bilinearImpulseCenter-impulse.value[0],
              100*valueTolerance(impulse.value[0]));
    for (unsigned y=0;y<8;++y) for (unsigned x=0;x<8;++x)
        image.pixels[y*8+x] = {double(x),double(y),1,0};
    for (double a : cubic_expected::fractions) for (double b : cubic_expected::fractions) {
        const auto r = reconstruct(image,(3.5+a)/8,(3.5+b)/8,Clamp,Clamp);
        EXPECT_NEAR(r.value[0],3+a,hostTolerance);
        EXPECT_NEAR(r.value[1],3+b,hostTolerance);
        EXPECT_NEAR(r.ds[0],8,hostTolerance);
        EXPECT_NEAR(r.dt[1],8,hostTolerance);
    }
}
TEST(CubicReference, SmartThresholdsAndPreparation) {
    for (size_t i=0;i<cubic_expected::rho.size();++i) {
        const auto f = footprint({cubic_expected::rho[i]/8,0},{0,0},8,8,1,false);
        EXPECT_NEAR(f.weight,cubic_expected::smartWeight[i],hostTolerance);
    }
    for (double a : cubic_expected::anisotropy) for (bool conservative : {false,true}) {
        auto f = footprint({100,0},{0,0},19,9,a,conservative);
        EXPECT_TRUE(std::isfinite(f.lod));
        EXPECT_NEAR(std::hypot(f.dy.x,f.dy.y),.99/19,hostTolerance);
        EXPECT_NEAR(std::hypot(f.dx.x,f.dx.y),conservative ? 100 : 16*.99/19,hostTolerance);
    }
}
TEST(CubicReference, AnalyticFiniteDifference) {
    Level image{8,8,std::vector<Pixel>(64)};
    for (unsigned i=0;i<64;++i) image.pixels[i] = {double(i%7)/7,0,0,0};
    constexpr double s=.421,t=.537,e=1./128;
    const auto r = reconstruct(image,s,t,Wrap,Mirror);
    // All five points stay within one polynomial span. This stencil is exact
    // for a cubic, so there is no truncation allowance on the host check.
    const double ds = (-reconstruct(image,s+2*e,t,Wrap,Mirror).value[0]+
                       8*reconstruct(image,s+e,t,Wrap,Mirror).value[0]-
                       8*reconstruct(image,s-e,t,Wrap,Mirror).value[0]+
                       reconstruct(image,s-2*e,t,Wrap,Mirror).value[0])/(12*e);
    EXPECT_NEAR(ds,r.ds[0],hostTolerance);
}
