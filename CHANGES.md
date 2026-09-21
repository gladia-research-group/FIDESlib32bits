# What this fork changes relative to upstream FIDESlib

Base: upstream [FIDESlib](https://github.com/CAPS-UMU/FIDESlib) `main` at *"Fix issue #31: Bug
on asymmetric chebychev intervals"* (OpenFHE 1.4.2 compatibility). Everything below was built
for encrypted transformer inference (Perseus) and is enabled by default; the environment
variables listed at the end are the only runtime knobs.

## 1. 32-bit composite backend

Upstream runs `NATIVEINT=64` chains (one ~54-bit prime per level). This fork adds
`NATIVEINT=32` composite-scaling chains: each level is two 27-bit primes (composite degree 2),
halving the word width the GPU moves. The elementwise and NTT kernel families gained 32-bit
paths (32-bit Barrett/Shoup multiplication, `u32` accumulators); composite mod-raise, centred
coefficient lift and narrowing kernels support the composite bootstrap. The CMake option is
`FIDESLIB_OPENFHE_NATIVE_SIZE` (32/64); the 32-bit OpenFHE is a fixed upstream commit plus the
patch series kept in the Perseus repository (`third_party/openfhe-n32/patches`).

## 2. Seed-expanded key-switching keys

The `a` half of every hybrid key-switching-key digit is uniform randomness. Upstream samples it
once and stores it. This fork derives it from a 256-bit per-key ChaCha12 seed
(`src/CKKS/KskSeedExpand.cuh`: ChaCha12 in counter mode over (block counter, digit index,
modulus as limb tag, domain separator); one 16-word block serves 16 slots; exact-uniform
rejection sampling; CPU and GPU bit-exact) and regenerates it inside the two kernels that
consume it (`hoistedRotateDotKSKRegen4_`, `fusedDotKSKRegen4_`, four lanes per ChaCha block),
so `a` is never resident in device memory. The `b` half is stored as a dense 28-bit bit-stream
unpacked with funnel shifts in the consuming kernel (16 slots = 14 words). Keys serialize as
seed + `b`. The OpenFHE keygen side is `deps/openfhe-1.4.2-kska-seed.patch` (a byte-identical
copy of the expander; the installer checks the two copies agree).

## 3. Kernels and bootstrap

Fused NTT/rescale passes, warp-cooperative key-switching dot products, a keyswitch that skips
the redundant per-digit source INTT (`FIDESLIB_KS_DIGIT_INTT=1` restores upstream's), an
EvalMod of degree 14 with 5 double-angle iterations, sparse bootstraps over `s` slots
(`EvalBootstrapSetup(levelBudget, dim1, slots, correctionFactor)`, an explicit correction
factor replaces upstream's defaulted signature), lazy metadata-only `Clone()` of GPU-resident
ciphertexts, and freed limbs kept in per-size free lists.

## 4. Host / API layer (`api/CryptoContext.*`)

Pinned ping-pong staging arenas for plaintext weights (`FHE_STAGE_ARENA_GB`,
`BeginStageBlock[Owned]` / `ReleaseStageBlock`, multi-consume and persistent staging),
coefficient-staged plaintexts expanded to their limbs on the GPU (`MarkCoeffStaged`), a
device-resident ciphertext store and a pinned KV-cache arena (`KV_ARENA_GB`, `PrewarmKvArena`,
`KvStoreStaged` / `KvLoadStaged` / `KvEvict`), a rotation-key band (`FIDESLIB_ROT_KEY_BAND`),
`DropToLevel`, and an asynchronous raw-ciphertext store for magnitude capture
(`StoreRaw` / `DecryptStoredRaw`, `FHE_MAG_RING_GB`).

## 5. Runtime knobs

| variable | default | meaning |
|---|---|---|
| `FIDESLIB_KSK_REGEN` | 2 | in-kernel regeneration of the `a` half (0 = stored keys) |
| `FIDESLIB_KSK_PACK` | 1 | bit-packed `b` half (0 = one 32-bit lane per limb) |
| `FIDESLIB_KS_DIGIT_INTT` | 0 | 1 = upstream's redundant per-digit INTT |
| `FIDESLIB_ROT_KEY_BAND` | -1 | keep only rotation keys of steps within ±2^band (−1 = all) |
| `FIDESLIB_SPARSE_ARCSINE` | unset | arcsine-corrected sparse bootstrap (set by the consumer) |
| `FHE_STAGE_ARENA_GB`, `KV_ARENA_GB`, `FHE_MAG_RING_GB` | 3 / 12 / 6 | pinned arena sizes |
| `FHE_STAGE_RELEASE_CPU` | unset | release host copies of staged plaintexts after upload |

## 6. Removed relative to upstream

The vendored `nvtx/` tree, the 1.2.3/1.5.1-era patches, `examples/serial/`, and the
micro-benchmark tests superseded by the width-parametrized `test/ParametrizedTest.cuh` suite.
