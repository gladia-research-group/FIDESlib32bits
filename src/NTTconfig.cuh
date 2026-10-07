// Rows (int2 pairs) per thread in the two-pass NTT for 32-bit limbs. 8 is the shipped value (byte parity with the u64
// kernels' M = 4); FIDES_NTT_M32=16 doubles the independent butterflies each thread interleaves (ILP experiment).
#ifndef FIDESLIB_NTTCONFIG_CUH
#define FIDESLIB_NTTCONFIG_CUH
#ifndef FIDES_NTT_M32
#define FIDES_NTT_M32 8
#endif
#endif
