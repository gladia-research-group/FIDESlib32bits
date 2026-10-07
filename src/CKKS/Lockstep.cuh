// FIDESLIB_EVALMOD_LOCKSTEP: run the dense EvalMod's Re and Im halves in lockstep on two host threads that never issue
// GPU work at the same time (a turn token), so each relinearization of the Re half is issued right after the same
// relinearization of the Im half and finds the relin key still in L2 (key-stream sharing without a paired kernel).
#ifndef FIDESLIB_CKKS_LOCKSTEP_CUH
#define FIDESLIB_CKKS_LOCKSTEP_CUH
namespace FIDESlib::CKKS {
/// Called around every relinearizing op (no-ops outside a lockstep region). The turn passes only once the op has
/// consumed the shared key-switch scratch buffer, never between the key switch and its use.
void lockstepBeforeKeySwitch();
void lockstepAfterKeySwitch();
struct LockstepGuard {
    LockstepGuard();
    ~LockstepGuard();
};
}  // namespace FIDESlib::CKKS
#endif
