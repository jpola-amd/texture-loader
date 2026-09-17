// SPDX-License-Identifier: MIT
#pragma once
#include <array>

namespace cubic_expected {
// Frozen data-only uniform B-spline fixtures, before candidate implementation.
constexpr std::array<std::array<double, 4>, 4> weights{{
    {1./6, 4./6, 1./6, 0},
    {27./384, 235./384, 121./384, 1./384},
    {1./48, 23./48, 23./48, 1./48},
    {1./384, 121./384, 235./384, 27./384}
}};
constexpr std::array<double, 4> fractions{0,.25,.5,.75};
constexpr double impulseCenter = 4./9;
constexpr double bilinearImpulseCenter = 1;
constexpr std::array<double, 6> rho{0,.5,1,1.5,2,4};
constexpr std::array<double, 6> smartWeight{1,1,1,.5,0,0};
constexpr std::array<unsigned, 5> anisotropy{1,2,4,8,16};
constexpr std::array<std::array<unsigned, 2>, 6> dimensions{{
    {8,8},{19,19},{17,9},{1,9},{9,1},{1,1}
}};
// Legacy native zero maps to effective mathematical A=1, never zero.
constexpr unsigned legacyEffectiveAnisotropy = 1;
// Native branch conservatively requires every policy-allowed original level.
constexpr unsigned nativeFirstRequired = 0;
}
