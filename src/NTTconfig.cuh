// Rows (int2 pairs) per thread in the two-pass NTT for 32-bit limbs. 8 is the shipped value (byte parity with the u64
// kernels' M = 4); FIDES_NTT_M32=16 doubles the independent butterflies each thread interleaves (ILP experiment).
#ifndef FIDESLIB_NTTCONFIG_CUH
#define FIDESLIB_NTTCONFIG_CUH
#ifndef FIDES_NTT_M32
#define FIDES_NTT_M32 8
#endif
// Minimum resident blocks per SM for the two-pass NTT_ / INTT_ kernels (128-thread blocks at N = 2^16): a register cap
// (65536 / (128 * MINB)) that trades spills for occupancy. 0 = no bound (the compiler's choice, 72-76 registers).
#ifndef FIDES_NTT_MINB
#define FIDES_NTT_MINB 0
#endif
#if FIDES_NTT_MINB > 0
#define FIDES_NTT_LB __launch_bounds__(128, FIDES_NTT_MINB)
#else
#define FIDES_NTT_LB
#endif
#endif
