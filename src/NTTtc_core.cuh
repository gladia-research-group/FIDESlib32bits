#pragma once
// Device side of the tensor-core NTT core (see NTTtc.cuh). Pulls in the tile macros (A/AS): include only
// from NTT.cu / NTTtc.cu, never from host-facing headers.
#include "NTTtc.cuh"
#include "AddSub.cuh"
#include "ModMult.cuh"
#include "NTT.cuh"
#include "NTThelper.cuh"

namespace FIDESlib {
namespace tc {
constexpr int BD = 128, ROWS = 8, KLEN = 256, PLANE = ROWS * KLEN;  // one u8 plane = 2 KB; 4 planes = the 8 KB tile

__device__ __forceinline__ uint32_t barrett64(uint64_t v, uint32_t p, uint64_t mu) {
    const uint64_t q = __umul64hi(v, mu);
    uint64_t r = v - q * (uint64_t)p;
    if (r >= p) r -= p;
    if (r >= p) r -= p;
    return (uint32_t)r;
}

__device__ __forceinline__ void mma_u8(int32_t (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// Replace the butterfly core on the tile held in `buffer` (rows A(i), swizzled as AS). The 8 KB tile is
// re-used in place as the four u8 chunk planes; the outputs wait in registers until every warp is done.
// Rplanes: this prime's 4 x 65536 B core planes (global, L2-resident, shared by every block of the prime).
template <typename T>
__device__ __forceinline__ void tc_core(char* buffer, const uint8_t* __restrict__ Rplanes, const int primeid,
                                        const uint64_t mu, const int tid) {
    static_assert(sizeof(T) == 4, "tensor-core core: u32 limbs only");
    // The two-pass kernels reach their core without a block barrier (each thread's first butterfly touches only
    // elements it stored itself); this core reads other threads' elements, so it must start with one.
    __syncthreads();
    // 1. tile -> registers (16 elements per thread: element idx = tid + 128*q -> row i = idx >> 8, e = idx & 255)
    T x[16];
#pragma unroll
    for (int q = 0; q < 16; ++q) {
        const int idx = tid + BD * q;
        x[q] = AS(idx >> 8, idx & 255);
    }
    __syncthreads();
    // 2. registers -> four u8 planes over the tile memory: plane c at buffer + c*PLANE, [row][k]
    uint8_t* planes = (uint8_t*)buffer;
#pragma unroll
    for (int q = 0; q < 16; ++q) {
        const int idx = tid + BD * q;
        const int i = idx >> 8, e = idx & 255;
#pragma unroll
        for (int c = 0; c < 4; ++c)
            planes[c * PLANE + i * KLEN + e] = (uint8_t)((x[q] >> (8 * c)) & 0xFFu);
    }
    __syncthreads();
    // 3. MMA: warp w computes output m-tiles w, w+4, w+8, w+12 (16 output elements each) for all 8 rows
    const int lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
    const uint32_t p = (uint32_t)C_.primes[primeid];
    uint32_t out[4][4];  // [m-tile slot][o]: o = {row g col 2t, row g col 2t+1, row g+8 col 2t, row g+8 col 2t+1}
#pragma unroll 1
    for (int slot = 0; slot < 4; ++slot) {
        const int mt = warp + 4 * slot;
        int32_t acc[7][4];
#pragma unroll
        for (int s = 0; s < 7; ++s)
#pragma unroll
            for (int o = 0; o < 4; ++o) acc[s][o] = 0;
#pragma unroll 1
        for (int ks = 0; ks < 8; ++ks) {
            uint32_t bfr[4][2], afr[4][4];
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                const uint8_t* pc = planes + c * PLANE + g * KLEN + 32 * ks + 4 * t;
                bfr[c][0] = *(const uint32_t*)(pc);
                bfr[c][1] = *(const uint32_t*)(pc + 16);
            }
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                const uint8_t* Rl = Rplanes + l * 65536 + (16 * mt + g) * KLEN + 32 * ks + 4 * t;
                afr[l][0] = __ldg((const uint32_t*)(Rl));
                afr[l][1] = __ldg((const uint32_t*)(Rl + 8 * KLEN));
                afr[l][2] = __ldg((const uint32_t*)(Rl + 16));
                afr[l][3] = __ldg((const uint32_t*)(Rl + 8 * KLEN + 16));
            }
#pragma unroll
            for (int l = 0; l < 4; ++l)
#pragma unroll
                for (int c = 0; c < 4; ++c) mma_u8(acc[l + c], afr[l], bfr[c]);
        }
        // 4. exact recombination (< 2^64) and reduction
#pragma unroll
        for (int o = 0; o < 4; ++o) {
            uint64_t v = 0;
#pragma unroll
            for (int s = 0; s < 7; ++s) v += (uint64_t)(uint32_t)acc[s][o] << (8 * s);
            out[slot][o] = barrett64(v, p, mu);
        }
    }
    __syncthreads();  // every warp is done reading the planes: the tile can be rewritten
#pragma unroll
    for (int slot = 0; slot < 4; ++slot) {
        const int mt = warp + 4 * slot;
#pragma unroll
        for (int o = 0; o < 4; ++o) {
            const int m = 16 * mt + g + ((o & 2) ? 8 : 0);  // output element index
            const int n = 2 * t + (o & 1);                 // tile row
            AS(n, m) = (T)out[slot][o];
        }
    }
    __syncthreads();
}

// ---- v2: two-stage radix-16 core. R[m][k] = c * g^(sigma(m) * tau(k)) (host-verified per prime, NTTtc.cu).
// With K = tau(k) = K1 + 16 K0 and M = sigma(m) = M0 + 16 M1:
//   stage 1 (over K0): Y[M0][K1] = sum_K0 g^(M0 (K1 + 16 K0)) x'[K]          A1[K1] : 16 x 16, one per K1
//   stage 2 (over K1): X'[M]      = sum_K1 c g^(16 M1 K1) Y[M0][K1]            A2     : 16 x 16
// Both stages are u8 x u8 -> s32 mma.m16n8k16 over 4 x 4 chunk planes (7 shifted accumulators, exact < 2^61),
// reduced once each with a 64-bit Barrett. Blob layout (bytes): A1 planes [l][K1][M0][K0] at 0 (4 x 4096),
// A2 planes [l][M1][K1] at 16384 (4 x 256), tau[256] at 17408, sigma^-1[256] at 17664; TC2_BLOB bytes total.
constexpr int TC2_A1 = 0, TC2_A2 = 16384, TC2_TAU = 17408, TC2_SIGINV = 17664, TC2_TAUINV = 17920, TC2_BLOB = 18176;

__device__ __forceinline__ void mma_u8_k16(int32_t (&d)[4], const uint32_t (&a)[2], const uint32_t b) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.s32.u8.u8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(b));
}

// u8 plane word (row, hi, w): hi = the n-tile index (K1 in stage 1, M0 in stage 2), w = k-quad (4 k values).
// XOR swizzle of hi by the row spreads the 8 rows of a B fragment over 8 distinct 16-B bank groups, and the
// word-level XOR by (row >> 1) & 3 spreads the 4 row pairs of an output fragment over the 4 words of a group:
// every B-fragment load and every packed plane store is bank-conflict-free. Writer and reader share this.
__device__ __forceinline__ int tc2_word(int row, int hi, int w) {
    return row * KLEN + ((hi ^ row) << 4) + ((w ^ ((row >> 1) & 3)) << 2);
}

template <typename T>
__device__ __forceinline__ void tc_core2(char* buffer, const uint8_t* __restrict__ blob, const int primeid,
                                         const uint64_t mu, const int tid) {
    static_assert(sizeof(T) == 4, "tensor-core core: u32 limbs only");
    __syncthreads();  // see tc_core
    const int lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
    const uint32_t p = (uint32_t)C_.primes[primeid];
    uint8_t* planes = (uint8_t*)buffer;
    // 1. tile -> registers, 2. permuted chunk planes [row][K1^row][K0]
    // planes alias the tile: gather every needed input into registers first, then (after a barrier) store
    uint32_t w_all[4][4];
#pragma unroll
    for (int it = 0; it < 4; ++it) {
        const int Q = tid + BD * it;  // cell: (row, K1, K0-quad)
        const int quad = Q & 3, K1 = (Q >> 2) & 15, row = Q >> 6;
#pragma unroll
        for (int c = 0; c < 4; ++c) w_all[it][c] = 0u;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int K = K1 + 16 * (4 * quad + i);
            const int e = __ldg(blob + TC2_TAUINV + K);
            const uint32_t v = (uint32_t)AS(row, e);
#pragma unroll
            for (int c = 0; c < 4; ++c) w_all[it][c] |= ((v >> (8 * c)) & 0xFFu) << (8 * i);
        }
    }
    __syncthreads();
#pragma unroll
    for (int it = 0; it < 4; ++it) {
        const int Q = tid + BD * it;
        const int quad = Q & 3, K1 = (Q >> 2) & 15, row = Q >> 6;
        const int pos = tc2_word(row, K1, quad);
#pragma unroll
        for (int c = 0; c < 4; ++c) *(uint32_t*)(planes + c * PLANE + pos) = w_all[it][c];
    }
    __syncthreads();
    // 3. stage 1: warp w owns n-tiles K1 = w + 4 slot; D rows = M0 (g, g+8), D cols = tile rows (2t, 2t+1)
    uint32_t y[4][4];
#pragma unroll
    for (int slot = 0; slot < 4; ++slot) {
        const int K1 = 4 * warp + slot;  // a thread's 4 slots = 4 consecutive K1 -> packed stage-1 output
        int32_t acc[7][4];
#pragma unroll
        for (int s_ = 0; s_ < 7; ++s_)
#pragma unroll
            for (int o = 0; o < 4; ++o) acc[s_][o] = 0;
        uint32_t a[4][2], b[4];
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint8_t* A = blob + TC2_A1 + l * 4096 + K1 * 256 + 4 * t;
            a[l][0] = __ldg((const uint32_t*)(A + g * 16));
            a[l][1] = __ldg((const uint32_t*)(A + (g + 8) * 16));
        }
#pragma unroll
        for (int c = 0; c < 4; ++c) b[c] = *(const uint32_t*)(planes + c * PLANE + tc2_word(g, K1, t));
#pragma unroll
        for (int l = 0; l < 4; ++l)
#pragma unroll
            for (int c = 0; c < 4; ++c) mma_u8_k16(acc[l + c], a[l], b[c]);
#pragma unroll
        for (int o = 0; o < 4; ++o) {
            uint64_t v = 0;
#pragma unroll
            for (int s_ = 0; s_ < 7; ++s_) v += (uint64_t)(uint32_t)acc[s_][o] << (8 * s_);
            y[slot][o] = barrett64(v, p, mu);
        }
    }
    __syncthreads();  // all B fragments of stage 1 consumed: the planes can be rewritten
    // 4. Y -> chunk planes [row][M0^row][K1]: the 4 slots are K1 = 4 warp + 0..3 -> one word per (o, plane)
#pragma unroll
    for (int o = 0; o < 4; ++o) {
        const int M0 = g + ((o & 2) ? 8 : 0), row = 2 * t + (o & 1);
        const int pos = tc2_word(row, M0, warp);
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            uint32_t w = 0u;
#pragma unroll
            for (int slot = 0; slot < 4; ++slot) w |= ((y[slot][o] >> (8 * c)) & 0xFFu) << (8 * slot);
            *(uint32_t*)(planes + c * PLANE + pos) = w;
        }
    }
    __syncthreads();
    // 5. stage 2: warp w owns n-tiles M0 = w + 4 slot; D rows = M1 (g, g+8), D cols = tile rows
    uint32_t a2[4][2];
#pragma unroll
    for (int l = 0; l < 4; ++l) {
        const uint8_t* A = blob + TC2_A2 + l * 256 + 4 * t;
        a2[l][0] = __ldg((const uint32_t*)(A + g * 16));
        a2[l][1] = __ldg((const uint32_t*)(A + (g + 8) * 16));
    }
    uint32_t out[4][4];
    int mo[4][4];
#pragma unroll
    for (int slot = 0; slot < 4; ++slot) {
        const int M0 = warp + 4 * slot;
        int32_t acc[7][4];
#pragma unroll
        for (int s_ = 0; s_ < 7; ++s_)
#pragma unroll
            for (int o = 0; o < 4; ++o) acc[s_][o] = 0;
        uint32_t b[4];
#pragma unroll
        for (int c = 0; c < 4; ++c) b[c] = *(const uint32_t*)(planes + c * PLANE + tc2_word(g, M0, t));
#pragma unroll
        for (int l = 0; l < 4; ++l)
#pragma unroll
            for (int c = 0; c < 4; ++c) mma_u8_k16(acc[l + c], a2[l], b[c]);
#pragma unroll
        for (int o = 0; o < 4; ++o) {
            uint64_t v = 0;
#pragma unroll
            for (int s_ = 0; s_ < 7; ++s_) v += (uint64_t)(uint32_t)acc[s_][o] << (8 * s_);
            out[slot][o] = barrett64(v, p, mu);
            const int M1 = g + ((o & 2) ? 8 : 0);
            mo[slot][o] = __ldg(blob + TC2_SIGINV + M0 + 16 * M1);
        }
    }
    __syncthreads();  // planes consumed: the tile is written back in place
#pragma unroll
    for (int slot = 0; slot < 4; ++slot)
#pragma unroll
        for (int o = 0; o < 4; ++o) {
            const int row = 2 * t + (o & 1);
            AS(row, mo[slot][o]) = (T)out[slot][o];
        }
    __syncthreads();
}
}  // namespace tc
}  // namespace FIDESlib
