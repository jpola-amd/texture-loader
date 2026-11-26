#!/bin/bash

set -e

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
root_dir="$(cd -- "$script_dir/../.." && pwd)"

hip_path="${HIP_PATH:-$HOME/my_space/code/therock-tarball/install}"
build_type="${1:-Release}"

cmake -S "$root_dir" -B "$root_dir/build" \
    -G Ninja \
    -DCMAKE_BUILD_TYPE="$build_type" \
    -DHIP_PATH="$hip_path" \
    -DBUILD_EXAMPLES=ON \
    -DUSE_VCPKG=ON \
    -DUSE_OIIO=ON \
    -DBUILD_TESTS=ON \
    -DHIP_ARCHITECTURES=gfx1031
