#!/bin/bash

set -e

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
root_dir="$(cd -- "$script_dir/../.." && pwd)"

cmake -S "$root_dir" -B "$root_dir/build" \
    -G "Visual Studio 18 2026" \
    -A x64 \
    -T "version=14.50" \
    -DBUILD_EXAMPLES=ON \
    -DUSE_VCPKG=ON \
    -DUSE_OIIO=ON \
    -DHIP_ARCHITECTURES=gfx1031
