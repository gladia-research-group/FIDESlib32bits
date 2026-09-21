#!/usr/bin/env bash
#
# Build and install the patched OpenFHE that FIDESlib links against.
#
#   build.sh <install-prefix> [native-size]
#
# native-size selects the OpenFHE NATIVEINT backend (64 default, 32 for the
# native32 port) and decides whether the NATIVEINT=32 patch is applied.

set -e
set -x

INSTALL_PREFIX="$1"
NATIVE_SIZE="${2:-${NATIVE_SIZE:-64}}"

#Remove previous installation.
rm -rf openfhe-install
rm -rf openfhe-src

# Target installation directory.
mkdir -p "$INSTALL_PREFIX"
git submodule update --init --recursive --remote

#Source submodule.
cd openfhe-src
git checkout v1.4.2
git config user.email "FIDESlib"
git config user.name "FIDESlib"
git am ../openfhe-1.4.2.patch
# FIDESlib HEAD requires SetScalingModSizePerLevel (api/CCParams.cpp), added by the
# mixedlimb patch; without it the FIDESlib build fails.
git am ../openfhe-1.4.2-mixedlimb.patch
# NATIVEINT=32 only: CKKSPackedEncoding::FitToNativeVector truncates the encoder's
# int64 coefficients to 32 bits, so negatives encode as 2^32+x (enc/dec(-0.4) ~ 31.57).
if [ "$NATIVE_SIZE" = "32" ]; then
	git am ../openfhe-1.4.2-native32.patch
fi

# Compilation and installation.
mkdir build
cd build
echo "Installing into $INSTALL_PREFIX"
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
	-DNATIVE_SIZE="$NATIVE_SIZE" ..
make -j12
make install -j12
