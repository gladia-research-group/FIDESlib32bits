# OpenFHE patches for the 32-bit chain

The OpenFHE series the `NATIVEINT=32` composite chain is built from, copied from Perseus
(`third_party/openfhe-n32/patches` at `83b3f45`) so that this repository builds the 32-bit
stack on its own. `docker/build-fherma.sh` applies it with `git am`; the 64-bit path keeps
using the `openfhe-1.4.2*.patch` files one directory up.

| field | value |
|---|---|
| upstream | `https://github.com/openfheorg/openfhe-development` |
| base commit | `aa391988d354d4360f390f223a90e0d1b98839d7` |
| series | `0001`–`0014` |

1. **0001** compatibility surface FIDESlib needs from the CryptoContext.
2. **0002–0004** mixed-size modulus chains: the bootstrap segment, the EvalMod output
   recovery (`scaleDec`), and composite-style bootstrap constants.
3. **0005–0006** 32-bit native integers: the encode path's coefficient truncation and the
   bootstrap DFT encoding under composite scaling.
4. **0007** seed expansion for the random half of a key-switching key, the OpenFHE side of
   the in-kernel regeneration. Its header must stay byte-equal to `src/CKKS/KskSeedExpand.cuh`;
   the build script checks this and refuses to build otherwise.
5. **0008–0009** exact arithmetic for moduli up to 2^31 and the matching `DoubleInteger`
   width cap.
6. **0010–0012** value and norm probes around q0 and the StC/CtS diagonals, off unless
   `OPENFHE_Q0_DIAG` is set. They are part of the series the measured binary was built
   from, so they stay in it.
7. **0013–0014** 128-bit keys: a bottom-only encapsulation key on composite chains and a
   uniform main key under `SPARSE_ENCAPSULATED`.

By hand:

```bash
git clone https://github.com/openfheorg/openfhe-development openfhe-src
cd openfhe-src && git checkout aa391988d354d4360f390f223a90e0d1b98839d7
git am /path/to/deps/openfhe-n32/*.patch
```
