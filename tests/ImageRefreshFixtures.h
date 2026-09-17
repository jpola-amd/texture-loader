// SPDX-License-Identifier: MIT
#pragma once
#include <array>
#include <cstdint>

namespace image_refresh_data {
constexpr double                  pointTolerance = 1e-6;
constexpr std::array<uint8_t, 4>  red{ 255, 0, 0, 255 }, green{ 0, 255, 0, 255 }, blue{ 0, 0, 255, 255 };
constexpr std::array<uint8_t, 64> redPattern{ 32, 0, 0, 16,  64,  0, 0, 32,  96,  0, 0, 48,  128, 0, 0, 64,
                                              48, 0, 0, 80,  80,  0, 0, 96,  112, 0, 0, 112, 144, 0, 0, 128,
                                              64, 0, 0, 144, 96,  0, 0, 160, 128, 0, 0, 176, 160, 0, 0, 192,
                                              80, 0, 0, 208, 112, 0, 0, 224, 144, 0, 0, 240, 176, 0, 0, 255 };
constexpr std::array<uint8_t, 64> greenPattern{ 0, 176, 0, 255, 0, 144, 0, 240, 0, 112, 0, 224, 0, 80, 0, 208,
                                                0, 160, 0, 192, 0, 128, 0, 176, 0, 96,  0, 160, 0, 64, 0, 144,
                                                0, 144, 0, 128, 0, 112, 0, 112, 0, 80,  0, 96,  0, 48, 0, 80,
                                                0, 128, 0, 64,  0, 96,  0, 48,  0, 64,  0, 32,  0, 32, 0, 16 };
constexpr std::array<std::array<uint8_t, 4>, 4> redAuthored{
    { { 255, 0, 0, 255 }, { 128, 0, 0, 192 }, { 64, 0, 0, 128 }, { 32, 0, 0, 64 } } };
constexpr std::array<std::array<uint8_t, 4>, 4> greenAuthored{
    { { 0, 255, 0, 255 }, { 0, 128, 0, 192 }, { 0, 64, 0, 128 }, { 0, 32, 0, 64 } } };
constexpr std::array<float, 4> redFloat{ 2, -.5f, .25f, .375f }, greenFloat{ -.25f, 3, .5f, .875f };
constexpr unsigned             alternations = 10;
}  // namespace image_refresh_data
