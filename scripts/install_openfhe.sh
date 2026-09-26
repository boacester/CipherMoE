#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPENFHE_ROOT="$WORKSPACE_ROOT/thirdparty/openfhe"
OPENFHE_SOURCE="$OPENFHE_ROOT/src"
OPENFHE_BUILD="$OPENFHE_ROOT/build"
OPENFHE_PREFIX="$OPENFHE_ROOT/install"
OPENFHE_CHECK_BUILD="$OPENFHE_ROOT/check-build"
OPENFHE_VERSION="v1.5.1"
OPENFHE_COMMIT="1306d14f8c26bb6150d3e6ad54f28dfe1007689e"
BUILD_JOBS="${BUILD_JOBS:-16}"

if [[ ! -e "$OPENFHE_SOURCE" ]]; then
    git clone --branch "$OPENFHE_VERSION" --depth 1 \
        --recurse-submodules --shallow-submodules \
        https://github.com/openfheorg/openfhe-development.git "$OPENFHE_SOURCE"
fi

if [[ "$(git -C "$OPENFHE_SOURCE" rev-parse HEAD)" != "$OPENFHE_COMMIT" ]]; then
    printf 'Expected OpenFHE %s at commit %s; inspect %s before building.\n' \
        "$OPENFHE_VERSION" "$OPENFHE_COMMIT" "$OPENFHE_SOURCE" >&2
    exit 1
fi
git -C "$OPENFHE_SOURCE" submodule update --init --recursive --depth 1

cmake -S "$OPENFHE_SOURCE" -B "$OPENFHE_BUILD" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$OPENFHE_PREFIX" \
    -DBUILD_SHARED=ON \
    -DBUILD_STATIC=OFF \
    -DBUILD_UNITTESTS=OFF \
    -DBUILD_BENCHMARKS=OFF \
    -DBUILD_EXAMPLES=ON \
    -DWITH_OPENMP=ON \
    -DNATIVE_SIZE=64 \
    -DGIT_SUBMOD_AUTO=OFF

cmake --build "$OPENFHE_BUILD" --parallel "$BUILD_JOBS" \
    --target simple-real-numbers boolean scheme-switching
cmake --install "$OPENFHE_BUILD"

cmake -S "$WORKSPACE_ROOT/scripts/openfhe_check" -B "$OPENFHE_CHECK_BUILD" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="$OPENFHE_PREFIX"
cmake --build "$OPENFHE_CHECK_BUILD" --parallel "$BUILD_JOBS"
OMP_NUM_THREADS=1 "$OPENFHE_CHECK_BUILD/openfhe-check"

printf 'OpenFHE %s installed and checked at %s\n' "$OPENFHE_VERSION" "$OPENFHE_PREFIX"
