// SPDX-License-Identifier: MIT
#pragma once

#include <cerrno>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace hip_demand { namespace test {

inline bool hasKnownMipBlendIssue(bool windows, std::string_view architecture) {
    return windows && architecture.substr(0, architecture.find(':')) == "gfx1201";
}

enum class MipBlendComparison { Correct, KnownIncorrect, Unexpected };

inline MipBlendComparison compareMipBlend(
    const std::array<float, 4>& actual, const std::array<float, 4>& first,
    const std::array<float, 4>& second, float fraction,
    const std::array<float, 4>& tolerance, bool allowKnownIssue) {
    if (!std::isfinite(fraction) || fraction < 0 || fraction > 1)
        return MipBlendComparison::Unexpected;
    const float knownWeight = std::clamp(1.25f * fraction - .125f, 0.f, 1.f);
    bool correct = true, known = true;
    for (size_t channel = 0; channel < actual.size(); ++channel) {
        const float expected = first[channel] + fraction * (second[channel] - first[channel]);
        const float incorrect = first[channel] + knownWeight * (second[channel] - first[channel]);
        if (!std::isfinite(actual[channel]) || !std::isfinite(expected) || !std::isfinite(incorrect) ||
            !std::isfinite(tolerance[channel]) || tolerance[channel] < 0)
            return MipBlendComparison::Unexpected;
        correct &= std::abs(actual[channel] - expected) <= tolerance[channel];
        known &= std::abs(actual[channel] - incorrect) <= tolerance[channel];
    }
    if (correct)
        return MipBlendComparison::Correct;
    return allowKnownIssue && known ? MipBlendComparison::KnownIncorrect : MipBlendComparison::Unexpected;
}

inline int parseTestDevice(const char* selection) {
    if (!selection)
        return 0;
    if (*selection < '0' || *selection > '9')
        throw std::invalid_argument("HIP_DEMAND_TEST_DEVICE must be a nonnegative device ordinal");
    char* end = nullptr;
    errno = 0;
    const long value = std::strtol(selection, &end, 10);
    if (errno == ERANGE || *end != '\0' || value > std::numeric_limits<int>::max())
        throw std::invalid_argument("HIP_DEMAND_TEST_DEVICE must be a nonnegative device ordinal");
    return static_cast<int>(value);
}

inline std::vector<std::filesystem::path> samplingModuleCandidates(
    const std::filesystem::path& executable, const std::filesystem::path& buildModule,
    const char* overrideDirectory = nullptr) {
    const auto filename = buildModule.filename();
    // An explicit override is authoritative: a typo must not load a stale build artifact.
    if (overrideDirectory)
        return {std::filesystem::path(overrideDirectory) / filename};
    return {executable.parent_path() / filename,
            executable.parent_path().parent_path() / "tests" / filename,
            buildModule};
}

inline std::filesystem::path findSamplingModule(
    const std::vector<std::filesystem::path>& candidates) {
    std::string attempted;
    for (const auto& candidate : candidates) {
        std::error_code error;
        if (std::filesystem::is_regular_file(candidate, error))
            return std::filesystem::absolute(candidate);
        attempted += "\n  " + candidate.string();
    }
    throw std::runtime_error("Sampling HIP code object not found; attempted:" + attempted);
}

} }
