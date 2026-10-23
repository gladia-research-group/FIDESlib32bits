#!/usr/bin/env bash
#
# Build the 32-bit stack into one prefix: OpenFHE at the openfhe-n32 base commit with the
# deps/openfhe-n32 series (NATIVE_SIZE=32), then FIDESlib32bits against it. This is what
# Dockerfile.fherma runs; it also runs on a host, which is how the image is checked
# without Docker.
#
#   build-fherma.sh <install-prefix>
#
# Env: FIDESLIB_ARCH (default "89-real;120-real": FHERMA's L40S and RTX PRO 6000 Blackwell),
#      JOBS (default nproc), OPENFHE_SRC (an OpenFHE clone to copy instead of fetching),
#      WORK_DIR (scratch, default a fresh temporary directory, removed on success),
#      FIDESLIB_REVISION (recorded in BUILD-INFO; defaults to this checkout's HEAD).
#      CUDACXX, CC, CXX, CMAKE_LIBRARY_PATH and CMAKE_INCLUDE_PATH are read by CMake itself.

set -euo pipefail

[ $# -eq 1 ] || { echo "usage: $0 <install-prefix>" >&2; exit 2; }
PREFIX="$(realpath -m "$1")"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
SERIES="$REPO/deps/openfhe-n32"
OPENFHE_URL=https://github.com/openfheorg/openfhe-development.git
OPENFHE_BASE=aa391988d354d4360f390f223a90e0d1b98839d7
ARCH="${FIDESLIB_ARCH:-89-real;120-real}"
JOBS="${JOBS:-$(nproc)}"
WORK="${WORK_DIR:-$(mktemp -d)}"
REVISION="${FIDESLIB_REVISION:-$(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo unknown)}"

echo "=== OpenFHE (NATIVE_SIZE=32) + FIDESlib32bits -> $PREFIX, arch $ARCH, $JOBS jobs ==="
mkdir -p "$PREFIX" "$WORK"

# A dropped connection should not cost a two-hour build.
retry() { local i; for i in 1 2 3; do "$@" && return 0; echo "retrying: $*" >&2; sleep $((10 * i)); done; return 1; }

# ---- OpenFHE
OPENFHE="$WORK/openfhe"
if [ -n "${OPENFHE_SRC:-}" ]; then
    cp -r "$OPENFHE_SRC" "$OPENFHE"
    git -C "$OPENFHE" checkout -q "$OPENFHE_BASE"
else
    git init -q "$OPENFHE"
    retry git -C "$OPENFHE" fetch -q --depth 1 "$OPENFHE_URL" "$OPENFHE_BASE"
    git -C "$OPENFHE" checkout -q FETCH_HEAD
fi
# Only cereal: google-test, google-benchmark and gperftools are behind options left off below.
retry git -C "$OPENFHE" submodule update -q --init --depth 1 third-party/cereal
git -C "$OPENFHE" -c user.name="FIDESlib32bits build" -c user.email="build@fideslib32bits" \
    am -q "$SERIES"/*.patch
# The key-switching-key seed expander exists on both sides and must be byte-identical.
cmp -s "$OPENFHE/src/core/include/utils/kskseedexpand.h" "$REPO/src/CKKS/KskSeedExpand.cuh" \
    || { echo "ERROR: kskseedexpand.h (patch 0007) differs from src/CKKS/KskSeedExpand.cuh" >&2; exit 1; }

cmake -S "$OPENFHE" -B "$WORK/openfhe-build" -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$PREFIX" -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DNATIVE_SIZE=32 -DBUILD_STATIC=ON -DWITH_BE2=OFF -DWITH_BE4=OFF -DGIT_SUBMOD_AUTO=OFF \
    -DBUILD_UNITTESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_BENCHMARKS=OFF
cmake --build "$WORK/openfhe-build" --parallel "$JOBS"
cmake --install "$WORK/openfhe-build"

# ---- FIDESlib32bits
cmake -S "$REPO" -B "$WORK/fideslib-build" -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$PREFIX" -DFIDESLIB_INSTALL_PREFIX="$PREFIX" \
    -DOPENFHE_INSTALL_PREFIX="$PREFIX" -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DFIDESLIB_ARCH="$ARCH" -DFIDESLIB_OPENFHE_NATIVE_SIZE=32 -DFIDESLIB_INSTALL_OPENFHE=OFF \
    -DFIDESLIB_COMPILE_TESTS=OFF -DFIDESLIB_COMPILE_BENCHMARKS=OFF
# NCCL is optional to CMake, but the multi-GPU sources are what Perseus measured with.
if grep -q '^nccl_FOUND:FILEPATH=.*NOTFOUND' "$WORK/fideslib-build/CMakeCache.txt"; then
    echo "ERROR: NCCL not found; set CMAKE_LIBRARY_PATH / CMAKE_INCLUDE_PATH to libnccl.so / nccl.h" >&2
    exit 1
fi
cmake --build "$WORK/fideslib-build" --parallel "$JOBS"
cmake --install "$WORK/fideslib-build"

# ---- licences and provenance
install -Dm644 "$REPO/LICENSE.txt" "$PREFIX/share/licenses/fideslib32bits/LICENSE.txt"
install -Dm644 "$OPENFHE/LICENSE" "$PREFIX/share/licenses/openfhe/LICENSE"
mkdir -p "$PREFIX/share/fideslib32bits"
cat > "$PREFIX/share/fideslib32bits/BUILD-INFO" <<EOF
fideslib32bits  $REVISION
openfhe         $OPENFHE_BASE + deps/openfhe-n32 ($(ls "$SERIES"/*.patch | wc -l) patches), NATIVE_SIZE=32
cuda            $("${CUDACXX:-nvcc}" --version | sed -n 's/.*release \([0-9.]*\).*/\1/p')
arch            $ARCH
EOF

[ -n "${WORK_DIR:-}" ] || rm -rf "$WORK"
echo "=== Installed into $PREFIX ==="
