// SPDX-License-Identifier: MIT
#pragma once
#include <array>
namespace cubic_smart_expected {
// Authored scalar mip pixels, expanded identically into each RGBA channel.
// 8x8 + 4x4 + 2x2 + 1x1, no source conversion or generation.
constexpr std::array<float, 85>   pixels{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3 };
constexpr unsigned                width = 8, height = 8, levels = 4;
constexpr float                   s = .4375f, t = .4375f, dxS = .1875f, dxT = 0, dyS = 0, dyT = 0;
constexpr float                   a = 1, weight = .5f;
constexpr std::array<unsigned, 8> cubicLevel{ 0, 0, 1, 1, 1, 1, 1, 1 };
constexpr std::array<float, 8>    value{ 0, .5f, 0, 1, 0, 1, 0, 1 };
constexpr float                   ds = 0, dt = 0, jitterS = 0, jitterT = 0;
constexpr unsigned                firstRequired = 0, lastRequired = 3, firstResident = 0;
// Successful completion is contract_v1::Outcome::Success (0),
// SampleValidity::Complete (2); derivative-request mask is bits 1 and 2.
constexpr unsigned outcome = 0, validity = 2;
}  // namespace cubic_smart_expected
