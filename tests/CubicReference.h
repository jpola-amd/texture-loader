// SPDX-License-Identifier: MIT
#pragma once

// Independent double-precision reference. Does not include device reconstruction.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

namespace cubic_reference {
using Pixel = std::array<double, 4>;
struct Level { unsigned width, height; std::vector<Pixel> pixels; };
struct Result { Pixel value{}, ds{}, dt{}; };
enum Address { Wrap, Clamp, Mirror, Border };
struct Gradient { double x, y; };
struct Footprint { Gradient dx, dy; double weight, lod; };

inline std::array<double, 4> weights(double a) {
    const double b = 1 - a;
    return {b*b*b/6, (4-6*a*a+3*a*a*a)/6,
            (1+3*a+3*a*a-3*a*a*a)/6, a*a*a/6};
}
inline std::array<double, 4> derivatives(double a) {
    return {-(1-a)*(1-a)/2, a*(1.5*a-2), .5+a-1.5*a*a, a*a/2};
}
inline int address(int i, int n, Address mode) {
    if (mode == Clamp) return std::clamp(i, 0, n-1);
    if (mode == Border) return i < 0 || i >= n ? -1 : i;
    const int period = mode == Mirror ? 2*n : n;
    i = (i % period + period) % period;
    return mode == Mirror && i >= n ? 2*n-1-i : i;
}
inline Result reconstruct(const Level& level, double s, double t, Address u, Address v) {
    Result r;
    const double x = s*level.width-.5, y = t*level.height-.5;
    const int i = int(std::floor(x)), j = int(std::floor(y));
    const auto wx = weights(x-i), wy = weights(y-j);
    const auto dx = derivatives(x-i), dy = derivatives(y-j);
    for (int b = 0; b < 4; ++b)
        for (int a = 0; a < 4; ++a) {
            const int xx = address(i+a-1, level.width, u), yy = address(j+b-1, level.height, v);
            if (xx < 0 || yy < 0) continue;
            const auto& p = level.pixels[yy*level.width+xx];
            for (int k = 0; k < 4; ++k) {
                r.value[k] += p[k]*wx[a]*wy[b];
                r.ds[k] += p[k]*dx[a]*wy[b]*level.width;
                r.dt[k] += p[k]*wx[a]*dy[b]*level.height;
            }
        }
    return r;
}
inline Result mix(Result a, const Result& b, double f) {
    for (int k = 0; k < 4; ++k) {
        a.value[k] += f*(b.value[k]-a.value[k]);
        a.ds[k] += f*(b.ds[k]-a.ds[k]);
        a.dt[k] += f*(b.dt[k]-a.dt[k]);
    }
    return a;
}
inline Result sample(const std::vector<Level>& levels, double s, double t, double lod,
                     bool linear, Address u = Clamp, Address v = Clamp) {
    lod = std::clamp(lod, 0., double(levels.size()-1));
    if (!linear) lod = std::ceil(lod-.5);
    const unsigned low = unsigned(std::max(lod, 0.));
    auto r = reconstruct(levels[low], s, t, u, v);
    return linear && lod > low ? mix(r, reconstruct(levels[low+1], s, t, u, v), lod-low) : r;
}
inline Footprint footprint(Gradient dx, Gradient dy, unsigned w, unsigned h, double a, bool conservative) {
    const double rho = std::max(std::hypot(dx.x*w, dx.y*h), std::hypot(dy.x*w, dy.y*h));
    const double q = .99/std::max(w,h);
    auto prepare = [q](Gradient g, bool y) {
        const double len = std::hypot(g.x,g.y);
        if (!len) return y ? Gradient{0,q} : Gradient{q,0};
        const double scale = std::max(1.,q/len);
        return Gradient{g.x*scale,g.y*scale};
    };
    dx = prepare(dx,false); dy = prepare(dy,true);
    if (!conservative) {
        const double x = std::hypot(dx.x,dx.y), y = std::hypot(dy.x,dy.y);
        const double sx = std::min(1.,16*y/x), sy = std::min(1.,16*x/y);
        dx = {dx.x*sx,dx.y*sx}; dy = {dy.x*sy,dy.y*sy};
    }
    const double px = dx.x*w, py = dx.y*h, qx = dy.x*w, qy = dy.y*h;
    const double trace = px*px+py*py+qx*qx+qy*qy;
    const double delta = std::hypot(qx*qx+qy*qy-px*px-py*py,2*(px*qx+py*qy));
    const double major = (trace+delta)/2;
    // determinant/major avoids cancellation in a highly eccentric ellipse.
    const double cross = px*qy-py*qx;
    const double minor = major ? cross*cross/major : 0;
    return {dx,dy,std::clamp(2-rho,0.,1.),.5*std::log2(std::max(minor,major/(a*a)))};
}
constexpr double hostTolerance = 1e-12;
inline double valueTolerance(double expected) { return 2e-5*(1+std::abs(expected)); }
inline double derivativeTolerance(double expectedPerOriginalTexel) {
    return 1e-4*(1+std::abs(expectedPerOriginalTexel));
}
}
