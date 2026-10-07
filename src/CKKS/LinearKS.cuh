#pragma once
// Linear key switching (Jin et al., eprint 2024/1629) at our parameters, as a self-contained microbenchmark:
// single-limb digits lifted exactly into a 3-prime base T = {q0, q1, q2} (T > N * dnum * q_i * q_j), the
// (digit, output-limb) products accumulated in T, each output limb recovered by CRT and reduced mod q_j, then the
// usual ModDown. Gate: bit-exact against the paper's baseline (each single-limb digit base-converted to all of Q
// and NTT'd there, O(l^2) NTTs). Both are timed next to FIDESlib's hybrid key switch on the same operand.
#include <ostream>
namespace FIDESlib::CKKS {
class Ciphertext;
/// Returns the number of mismatching residues between the linear and the baseline path (0 = exact).
long linearKsBench(Ciphertext& any, int iters, std::ostream& os);
}  // namespace FIDESlib::CKKS
