//
// FIDESLIB_GPU_ENCRYPT (default 1; 0 restores the host encryption): public-key encryption on the device. The same algorithm and distributions
// as OpenFHE's EncryptZeroCore (public-key form): v uniform ternary, e0/e1 discrete Gaussian of the context's
// sigma, (pk0 v + e0, pk1 v + e1), the plaintext then added to c0. The randomness is a ChaCha12 stream
// (KskSeedExpand.cuh's block function, its own domain word) under a per-context key from the OS entropy pool and
// a per-encryption nonce: fresh on every call, never reused.
//
#ifndef FIDESLIB_CKKS_GPUENCRYPT_CUH
#define FIDESLIB_CKKS_GPUENCRYPT_CUH

#include <cstdint>

namespace FIDESlib::CKKS {

class Ciphertext;

/** FIDESLIB_GPU_ENCRYPT (default 1), settable at run time for the gate. */
bool gpuEncryptOn();
void setGpuEncrypt(int on);

/** out.c0 = pk.c0 * v + e0 and out.c1 = pk.c1 * v + e1 over out's limbs (pk holds at least as many); out's
 *  metadata is left as it is. key: 8 words, nonce: unique per call. */
void encryptZeroDevice(Ciphertext& out, const Ciphertext& pk, const uint32_t key[8], uint64_t nonce, double sigma);

}  // namespace FIDESlib::CKKS

#endif  // FIDESLIB_CKKS_GPUENCRYPT_CUH
