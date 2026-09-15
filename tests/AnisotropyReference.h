// SPDX-License-Identifier: MIT
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace hip_demand { namespace test { namespace anisotropy_reference {

using Pixel = std::array<double, 4>;
inline constexpr double Pi = 3.14159265358979323846;
inline constexpr double Alpha = .625;
inline constexpr std::array<unsigned int, 8> Ratios{{1, 2, 3, 4, 5, 7, 8, 16}};

// Predetermined quality gates, not a specification of vendor tap positions.
inline constexpr double AlphaTolerance = 1e-6;
inline constexpr double ReloadTolerance = 1e-6;
inline constexpr double DisabledPixelTolerance = .004;
inline constexpr double UploadTolerance = 3e-6;
inline constexpr double QuadratureTolerance = 2e-11;
inline constexpr double MajorRmsLimit = .12;
inline constexpr double MinorRmsLimit = .12;
inline constexpr double MinimumMinorGain = .55;
inline constexpr double MaximumMinorGain = 1.45;
inline constexpr double MajorImprovementFactor = .75;
inline constexpr double MinorImprovementFactor = .75;
inline constexpr double MajorNonRegression = .04;
inline constexpr double RampTolerance = .025;
inline constexpr double ImpulseRmsLimit = .055;
inline constexpr double ImpulseMaximumError = .18;
inline constexpr double ImpulseMinimumMass = .55;
inline constexpr double ImpulseMaximumMass = 1.45;
inline constexpr double ImpulseCentroidTolerance = .75;

struct Vec2 {
    double x = 0, y = 0;
};
inline Vec2 operator+(Vec2 a, Vec2 b) { return {a.x + b.x, a.y + b.y}; }
inline Vec2 operator-(Vec2 a, Vec2 b) { return {a.x - b.x, a.y - b.y}; }
inline Vec2 operator*(Vec2 a, double s) { return {a.x * s, a.y * s}; }
inline double dot(Vec2 a, Vec2 b) { return a.x * b.x + a.y * b.y; }
inline double cross(Vec2 a, Vec2 b) { return a.x * b.y - a.y * b.x; }

struct Jacobian {
    Vec2 dx, dy;
};

inline Jacobian texelJacobian(Vec2 normalizedDx, Vec2 normalizedDy,
                              unsigned int width, unsigned int height) {
    if (!width || !height)
        throw std::invalid_argument("Empty texture Jacobian");
    return {{normalizedDx.x * width, normalizedDx.y * height},
            {normalizedDy.x * width, normalizedDy.y * height}};
}

struct Axes {
    double major = 0, minor = 0;
    Vec2 direction{1, 0};
    Vec2 perpendicular() const { return {-direction.y, direction.x}; }
};

// Left singular vectors and singular values of the texel-space J=[dx dy].
// Scaling avoids overflow; |det(J)|/sigmaMax avoids cancellation in sigmaMin.
inline Axes singularAxes(Jacobian j) {
    const double scale = std::max({std::abs(j.dx.x), std::abs(j.dx.y),
                                   std::abs(j.dy.x), std::abs(j.dy.y)});
    if (!std::isfinite(j.dx.x) || !std::isfinite(j.dx.y) ||
        !std::isfinite(j.dy.x) || !std::isfinite(j.dy.y))
        throw std::invalid_argument("Nonfinite Jacobian");
    if (scale == 0)
        return {};
    j.dx = {j.dx.x / scale, j.dx.y / scale};
    j.dy = {j.dy.x / scale, j.dy.y / scale};
    const double a = j.dx.x * j.dx.x + j.dy.x * j.dy.x;
    const double b = j.dx.x * j.dx.y + j.dy.x * j.dy.y;
    const double d = j.dx.y * j.dx.y + j.dy.y * j.dy.y;
    const double largest = std::sqrt((a + d + std::hypot(a - d, 2 * b)) / 2);
    const double angle = .5 * std::atan2(2 * b, a - d);
    return {largest * scale, std::abs(cross(j.dx, j.dy)) / largest * scale,
            {std::cos(angle), std::sin(angle)}};
}

inline Jacobian orientedJacobian(double major, double minor, double angle, double screenRotation = 0) {
    const Vec2 u{std::cos(angle), std::sin(angle)}, v{-u.y, u.x};
    const double c = std::cos(screenRotation), s = std::sin(screenRotation);
    return {u * (major * c) + v * (minor * s),
            u * (-major * s) + v * (minor * c)};
}

// Test-only quality convention: average a unit-area, principal-axis box of
// width max(1,sigmaMax) and height max(1,sigmaMin,sigmaMax/requestedRatio).
// A ratio cap broadens the minor axis; it never shortens the major footprint.
// This box is NOT the screen-pixel parallelogram, an EWA ellipse, native tap
// emulation, or a production required-mip/demand calculation.
inline Axes footprint(Jacobian j, unsigned int requestedRatio) {
    if (!requestedRatio || requestedRatio > 16)
        throw std::invalid_argument("Reference anisotropy outside [1,16]");
    auto axes = singularAxes(j);
    axes.major = std::max(1.0, axes.major);
    axes.minor = std::max({1.0, axes.minor, axes.major / requestedRatio});
    return axes;
}

enum class Spatial { Point, Linear };
enum class Pattern { Directional, Impulse, Ramp };

struct Image {
    unsigned int width = 0, height = 0;
    std::vector<Pixel> pixels;
    const Pixel& at(int x, int y) const {
        x = std::clamp(x, 0, static_cast<int>(width) - 1);
        y = std::clamp(y, 0, static_cast<int>(height) - 1);
        return pixels.at(static_cast<size_t>(y) * width + x);
    }
};

inline Image makeImage(unsigned int width, unsigned int height, Pattern pattern, Vec2 major = {1, 0}) {
    if (!width || !height)
        throw std::invalid_argument("Empty reference image");
    Image image{width, height, std::vector<Pixel>(static_cast<size_t>(width) * height)};
    const Vec2 minor{-major.y, major.x};
    const Vec2 center{width / 2.0 + .5, height / 2.0 + .5};
    for (unsigned int y = 0; y < height; ++y) {
        for (unsigned int x = 0; x < width; ++x) {
            const Vec2 p{x + .5, y + .5}, offset = p - center;
            Pixel value{};
            if (pattern == Pattern::Directional) {
                value = {.5 + .4 * std::cos(2 * Pi * .25 * dot(offset, major)),
                         .5 + .4 * std::cos(2 * Pi * .125 * dot(offset, minor)),
                         .2 + .3 * p.x / width + .2 * p.y / height, Alpha};
            } else if (pattern == Pattern::Impulse) {
                const double pulse = x == width / 2 && y == height / 2 ? 1.0 : 0.0;
                value = {pulse, pulse, pulse, Alpha};
            } else {
                value = {.1 + .6 * p.x / width, .15 + .5 * p.y / height,
                         .2 + .3 * p.x / width + .2 * p.y / height, Alpha};
            }
            for (auto& channel : value)
                channel = static_cast<float>(channel);
            image.pixels[static_cast<size_t>(y) * width + x] = value;
        }
    }
    return image;
}

// Independent area-overlap reduction, recursively rounded to source FLOAT.
// No loader mip generator is used as its own upload oracle.
inline std::vector<Image> mipPyramid(const Image& base) {
    std::vector<Image> levels{base};
    while (levels.back().width > 1 || levels.back().height > 1) {
        const auto& src = levels.back();
        Image dst{std::max(1u, src.width / 2), std::max(1u, src.height / 2), {}};
        dst.pixels.resize(static_cast<size_t>(dst.width) * dst.height);
        for (unsigned int y = 0; y < dst.height; ++y) {
            for (unsigned int x = 0; x < dst.width; ++x) {
                const double x0 = double(x) * src.width / dst.width;
                const double x1 = double(x + 1) * src.width / dst.width;
                const double y0 = double(y) * src.height / dst.height;
                const double y1 = double(y + 1) * src.height / dst.height;
                Pixel sum{};
                for (int sy = static_cast<int>(std::floor(y0)); sy < std::ceil(y1); ++sy) {
                    for (int sx = static_cast<int>(std::floor(x0)); sx < std::ceil(x1); ++sx) {
                        const double weight = (std::min(x1, sx + 1.0) - std::max(x0, double(sx))) *
                                              (std::min(y1, sy + 1.0) - std::max(y0, double(sy)));
                        const auto& value = src.at(sx, sy);
                        for (size_t c = 0; c < 4; ++c)
                            sum[c] += weight * value[c];
                    }
                }
                for (auto& c : sum)
                    c = static_cast<float>(c / ((x1 - x0) * (y1 - y0)));
                dst.pixels[static_cast<size_t>(y) * dst.width + x] = sum;
            }
        }
        levels.push_back(std::move(dst));
    }
    return levels;
}

inline Pixel spatialSample(const Image& image, Vec2 p, Spatial mode) {
    if (mode == Spatial::Point)
        return image.at(static_cast<int>(std::floor(p.x)), static_cast<int>(std::floor(p.y)));
    const double x = p.x - .5, y = p.y - .5;
    const int ix = static_cast<int>(std::floor(x)), iy = static_cast<int>(std::floor(y));
    const double fx = x - ix, fy = y - iy;
    Pixel result{};
    for (size_t c = 0; c < 4; ++c)
        result[c] = (1 - fy) * ((1 - fx) * image.at(ix, iy)[c] + fx * image.at(ix + 1, iy)[c]) +
                    fy * ((1 - fx) * image.at(ix, iy + 1)[c] + fx * image.at(ix + 1, iy + 1)[c]);
    return result;
}

using Polygon = std::vector<Vec2>;
inline Polygon clip(const Polygon& input, unsigned int axis, double boundary, bool lower) {
    Polygon output;
    if (input.empty())
        return output;
    const auto coordinate = [axis](Vec2 p) { return axis == 0 ? p.x : p.y; };
    auto previous = input.back();
    double dp = (coordinate(previous) - boundary) * (lower ? 1 : -1);
    for (Vec2 current : input) {
        const double dc = (coordinate(current) - boundary) * (lower ? 1 : -1);
        if ((dp >= 0) != (dc >= 0))
            output.push_back(previous + (current - previous) * (dp / (dp - dc)));
        if (dc >= 0)
            output.push_back(current);
        previous = current;
        dp = dc;
    }
    return output;
}

inline void integrateTriangle(const Image& image, Spatial mode, Vec2 a, Vec2 b, Vec2 c,
                              unsigned int refinement, Pixel& sum) {
    if (refinement) {
        const auto ab = (a + b) * .5, bc = (b + c) * .5, ca = (c + a) * .5;
        integrateTriangle(image, mode, a, ab, ca, refinement - 1, sum);
        integrateTriangle(image, mode, ab, b, bc, refinement - 1, sum);
        integrateTriangle(image, mode, ca, bc, c, refinement - 1, sum);
        integrateTriangle(image, mode, ab, bc, ca, refinement - 1, sum);
        return;
    }
    const double area = std::abs(cross(b - a, c - a)) / 2;
    if (area == 0)
        return;
    for (Vec2 p : {(a * 4 + b + c) * (1.0 / 6),
                   (a + b * 4 + c) * (1.0 / 6),
                   (a + b + c * 4) * (1.0 / 6)}) {
        const auto value = spatialSample(image, p, mode);
        for (size_t channel = 0; channel < 4; ++channel)
            sum[channel] += value[channel] * (area / 3);
    }
}

// The footprint is clipped at every point/bilinear reconstruction cell.
// A three-node degree-two triangle rule then integrates each polynomial
// exactly (up to double roundoff), including the bilinear x*y term. Thus this
// is converged quadrature, not a finite set of guessed hardware taps. Optional
// subdivision verifies convergence without changing the integration domain.
inline Pixel integrate(const Image& image, Vec2 center, Axes axes, Spatial mode,
                       unsigned int refinement = 0) {
    if (image.width == 0 || image.height == 0 || refinement > 2 ||
        !std::isfinite(center.x) || !std::isfinite(center.y) ||
        std::abs(center.x) > 1e6 || std::abs(center.y) > 1e6 ||
        !std::isfinite(axes.major) || !std::isfinite(axes.minor) ||
        !std::isfinite(axes.direction.x) || !std::isfinite(axes.direction.y) ||
        std::abs(dot(axes.direction, axes.direction) - 1) > 1e-10 ||
        axes.major < 1 || axes.minor < 1 || axes.major > 64 || axes.minor > 64)
        throw std::invalid_argument("Unbounded reference integration");
    const Vec2 u = axes.direction * (axes.major / 2);
    const Vec2 v = axes.perpendicular() * (axes.minor / 2);
    const Polygon box{center - u - v, center + u - v, center + u + v, center - u + v};
    const double offset = mode == Spatial::Point ? 0 : .5;
    const double radiusX = std::abs(u.x) + std::abs(v.x);
    const double radiusY = std::abs(u.y) + std::abs(v.y);
    const int x0 = static_cast<int>(std::floor(center.x - radiusX - offset));
    const int x1 = static_cast<int>(std::floor(center.x + radiusX - offset));
    const int y0 = static_cast<int>(std::floor(center.y - radiusY - offset));
    const int y1 = static_cast<int>(std::floor(center.y + radiusY - offset));
    Pixel sum{};
    for (int y = y0; y <= y1; ++y) {
        for (int x = x0; x <= x1; ++x) {
            auto cell = clip(box, 0, x + offset, true);
            cell = clip(cell, 0, x + offset + 1, false);
            cell = clip(cell, 1, y + offset, true);
            cell = clip(cell, 1, y + offset + 1, false);
            for (size_t i = 1; i + 1 < cell.size(); ++i)
                integrateTriangle(image, mode, cell[0], cell[i], cell[i + 1], refinement, sum);
        }
    }
    for (auto& c : sum)
        c /= axes.major * axes.minor;
    return sum;
}

inline Pixel midpointIntegral(const Image& image, Vec2 center, Axes axes, Spatial mode,
                              unsigned int count) {
    if (!count || count > 512)
        throw std::invalid_argument("Unbounded midpoint quadrature");
    Pixel sum{};
    for (unsigned int y = 0; y < count; ++y) {
        for (unsigned int x = 0; x < count; ++x) {
            const Vec2 p = center + axes.direction * (axes.major * ((x + .5) / count - .5)) +
                           axes.perpendicular() * (axes.minor * ((y + .5) / count - .5));
            const auto value = spatialSample(image, p, mode);
            for (size_t c = 0; c < 4; ++c)
                sum[c] += value[c];
        }
    }
    for (auto& c : sum)
        c /= double(count) * count;
    return sum;
}

} } }
