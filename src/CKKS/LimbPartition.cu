//
// Created by carlosad on 27/04/24.
//
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <algorithm>
#include <array>
#include <variant>
#include <vector>

#include "CKKS/Context.cuh"
#include "CKKS/Conv.cuh"
#include "CKKS/ElemenwiseBatchKernels.cuh"
#include "CKKS/LimbPartition.cuh"
#include "LimbUtils.cuh"
#include "NTT.cuh"
#include "Rotation.cuh"
#include "VectorGPU.cuh"

#if defined(__clang__)
#include <experimental/source_location>
using sc = std::experimental::source_location;
constexpr int PREFIX_SIZE = 0;
#else
#include <source_location>
using sc = std::source_location;
constexpr int PREFIX_SIZE = 23;
#endif

namespace FIDESlib::CKKS {

LimbPartition::LimbPartition(LimbPartition&& l) noexcept
    : cc(l.cc),
      uid(l.uid),
      level(l.level),
      id(l.id),
      device((cudaSetDevice(l.device), l.device)),
      rank(l.rank),
      s(std::move(l.s)),
      meta(l.meta),
      SPECIALmeta(l.SPECIALmeta),
      digitid(l.digitid),
      DECOMPmeta(l.DECOMPmeta),
      DIGITmeta(l.DIGITmeta),
      GATHERmeta(l.GATHERmeta),
      limb(std::move(l.limb)),
      SPECIALlimb(std::move(l.SPECIALlimb)),
      DECOMPlimb(std::move(l.DECOMPlimb)),
      DIGITlimb(std::move(l.DIGITlimb)),
      GATHERlimb(std::move(l.GATHERlimb)),
      bufferAUXptrs(l.bufferAUXptrs),
      limbptr(std::move(l.limbptr)),
      auxptr(std::move(l.auxptr)),

      SPECIALlimbptr(std::move(l.SPECIALlimbptr)),
      SPECIALauxptr(std::move(l.SPECIALauxptr)),
      DECOMPlimbptr(std::move(l.DECOMPlimbptr)),
      //      DECOMPauxptr(std::move(l.DECOMPlimbptr)),
      DIGITlimbptr(std::move(l.DIGITlimbptr)),
      //      DIGITauxptr(std::move(l.DIGITlimbptr)),
      GATHERptr(std::move(l.GATHERptr)),
      DECOMPALLptr(std::move(l.DECOMPALLptr)),
      bufferDECOMPandDIGIT(l.bufferDECOMPandDIGIT),
      bufferSPECIAL(l.bufferSPECIAL),
      bufferLIMB(l.bufferLIMB),
      bufferGATHER(l.bufferGATHER),
      bufferDECOMPandDIGIT_handle(l.bufferDECOMPandDIGIT_handle),
      bufferGATHER_handle(l.bufferGATHER_handle) {
    key_q_band = l.key_q_band;
    key_pack_bits = l.key_pack_bits;
    bufferKSKPACK = l.bufferKSKPACK;
    bufferKSKPACKbytes = l.bufferKSKPACKbytes;
    bufferSPECIALbytes = l.bufferSPECIALbytes;
    bufferSPECIALcudaMalloc = l.bufferSPECIALcudaMalloc;
    bufferLIMBbytes = l.bufferLIMBbytes;
    l.bufferKSKPACK = nullptr;
    l.bufferSPECIAL = nullptr;
    l.bufferLIMB = nullptr;
    l.bufferDECOMPandDIGIT = nullptr;
    l.bufferAUXptrs = nullptr;
    l.bufferGATHER = nullptr;
    l.bufferDECOMPandDIGIT_handle = nullptr;
    l.bufferGATHER_handle = nullptr;
    for (int i = 0; i < SC_N; ++i) {
        scratch_[i] = l.scratch_[i];
        l.scratch_[i] = DevScratch{};   // else both dtors cudaFree the same pointer
    }
}

std::vector<VectorGPU<void*>> LimbPartition::generateDecompLimbptr(
    void** buffer, const std::vector<std::vector<LimbRecord>>& DECOMPmeta, const int device, int offset) {
    std::vector<VectorGPU<void*>> result;
    for (auto& d : DECOMPmeta) {
        result.emplace_back(buffer, std::max(1ul, d.size()), device, offset);
        offset += MAXP;
    }
    return result;
}


void* LimbPartition::scratchGet(const int slot, const size_t bytes) {
    if (bytes == 0)
        return nullptr;
    assert(slot >= 0 && slot < SC_N);
    DevScratch& sc = scratch_[slot];
    if (sc.p != nullptr && sc.bytes >= bytes)
        return sc.p;
    cudaSetDevice(device);
    void* p = nullptr;
    if (cudaMalloc(&p, bytes) != cudaSuccess || p == nullptr) {
        cudaGetLastError();   // never fail the op over a scratch buffer
        return nullptr;
    }
    if (sc.p != nullptr) {
        // The old buffer may still be read by work already enqueued on this partition's stream.
        cudaStreamSynchronize(s.ptr());
        cudaFree(sc.p);
    }
    sc.p = p;
    sc.bytes = bytes;
    return sc.p;
}

void LimbPartition::scratchFreeAll() {
    for (auto& sc : scratch_) {
        if (sc.p) {
            cudaFree(sc.p);
            sc.p = nullptr;
            sc.bytes = 0;
        }
    }
}

void** CudaMallocAuxBuffer(Stream& stream, unsigned long size, int device) {
    void** malloc;
    CudaCheckErrorModNoSync;
    malloc = (void**)GPUmalloc(device, MAXP * sizeof(void*) * (4ul + 4 * std::max(size, 1ul)), stream.ptr(), false);
    //cudaMallocAsync(&malloc, MAXP * sizeof(void*) * (4ul + 4 * std::max(size, 1ul)),
    //                stream.ptr());  // TODO: can reduce to 3
    CudaCheckErrorModNoSync;
    return malloc;
}

Stream initStream(bool default_) {
    Stream s;
    if (default_) {
        s.initDefault();
    } else {
        s.init(50);
    }
    return s;
}

LimbPartition::LimbPartition(ContextData& cc, const uint64_t& uid, int* level, const int id, const bool def_stream)
    : cc(cc),
      uid(uid),
      level(level),
      id(id),
      device((cudaSetDevice(cc.GPUid.at(id)), cc.GPUid.at(id))),
      rank(cc.GPUrank.at(id)),
      s(initStream(def_stream)),
      meta(cc.meta.at(id)),
      SPECIALmeta(cc.specialMeta.at(id)),
      digitid(cc.GPUdigits.at(id)),
      DECOMPmeta(cc.decompMeta.at(id)),
      DIGITmeta(cc.digitMeta.at(id)),
      GATHERmeta(cc.gatherMeta),
      DECOMPlimb(DECOMPmeta.size()),
      DIGITlimb(DIGITmeta.size()),
      bufferAUXptrs(CudaMallocAuxBuffer(s, cc.dnum, device)),
      /*
            limbptr(s, meta.size(), device),
            auxptr(s, meta.size(), device),
            SPECIALlimbptr(s, SPECIALmeta.size(), device),
            SPECIALauxptr(s, SPECIALmeta.size(), device),
            */

      limbptr(bufferAUXptrs, std::max(1ul, meta.size()), device, 0),
      auxptr(bufferAUXptrs, std::max(1ul, meta.size()), device, MAXP),
      SPECIALlimbptr(bufferAUXptrs, std::max(1ul, SPECIALmeta.size()), device, 2 * MAXP),
      SPECIALauxptr(bufferAUXptrs, std::max(1ul, SPECIALmeta.size()), device, 3 * MAXP),

      DECOMPlimbptr(generateDecompLimbptr(bufferAUXptrs, DECOMPmeta, device, 4 * MAXP)),
      //      DECOMPauxptr(generateDecompLimbptr(bufferAUXptrs, DECOMPmeta, device, (4 + DECOMPmeta.size()) * MAXP)),
      DIGITlimbptr(generateDecompLimbptr(bufferAUXptrs, DIGITmeta, device, (4 + DECOMPmeta.size()) * MAXP)),
      //      , DIGITauxptr(generateDecompLimbptr(bufferAUXptrs, DIGITmeta, device, (4 + 3 * DECOMPmeta.size()) * MAXP))
      GATHERptr(bufferAUXptrs, std::max(1ul, GATHERmeta.size()), device, (4 + 2 * DECOMPmeta.size()) * MAXP),
      // One spare MAXP-slot after GATHERptr — fits, the buffer holds (4 + 4*dnum)
      // slots and slots used so far are 4 + 2*dnum + 1 (see CudaMallocAuxBuffer).
      DECOMPALLptr(bufferAUXptrs, MAXP, device, (5 + 2 * DECOMPmeta.size()) * MAXP) {}

LimbPartition::~LimbPartition() {
    CudaNvtxRange r(std::string{sc::current().function_name()}.substr());
    cudaSetDevice(device);
    /*
        for (auto &i: limb) s.wait(STREAM(i));
        for (auto &i: SPECIALlimb) s.wait(STREAM(i));
        for (auto &i: DECOMPlimb) for (auto &j: i) s.wait(STREAM(j));
        for (auto &i: DIGITlimb) for (auto &j: i) s.wait(STREAM(j));
*/

    limbptr.free(s);
    auxptr.free(s);
    SPECIALlimbptr.free(s);
    SPECIALauxptr.free(s);
    GATHERptr.free(s);
    DECOMPALLptr.free(s);
    for (auto& d : DECOMPlimbptr)
        d.free(s);
    //    for (auto& d : DECOMPauxptr)
    //        d.free(s);
    for (auto& d : DIGITlimbptr)
        d.free(s);
    // for (auto& d : DIGITauxptr)
    //    d.free(s);

    //CudaCheckErrorMod;
    if (bufferDECOMPandDIGIT_handle) {
        cudaStreamSynchronize(s.ptr());
#ifdef NCCL
        if (bufferDECOMPandDIGIT_handle != (void*)-1)
            NCCLCHECK(ncclCommDeregister(rank, bufferDECOMPandDIGIT_handle));
        NCCLCHECK(ncclMemFree(bufferDECOMPandDIGIT));
#else
        assert(false);
#endif
    } else {
        if (bufferDECOMPandDIGIT)
            // bytes=0 would mis-file the block in GPUfree's 1 KB bucket; left as-is only because this
            // branch is unreachable (both allocation sites for bufferDECOMPandDIGIT are commented out).
            GPUfree(bufferDECOMPandDIGIT, id, 0, s.ptr());
        // cudaFreeAsync(bufferDECOMPandDIGIT, s.ptr());
    }
    freeSpecialBuffer();  // correct size AND route, not GPUfree(..., 0, ...)
    //cudaFreeAsync(bufferSPECIAL, s.ptr());
    if (bufferKSKPACK)
        GPUfree(bufferKSKPACK, id, (int)bufferKSKPACKbytes, s.ptr());
    if (bufferLIMB) {
        GPUfree(bufferLIMB, id, (int)bufferLIMBbytes, s.ptr());
        //cudaFreeAsync(bufferLIMB, s.ptr());
    }
    scratchFreeAll();   // plain cudaMalloc'd, so plain cudaFree
    if (bufferAUXptrs)
        GPUfree(bufferAUXptrs, id, MAXP * sizeof(void*) * (4ul + 4 * std::max(cc.dnum, 1)), s.ptr(), false);
    // cudaFreeAsync(bufferAUXptrs, s.ptr());
    if (bufferGATHER_handle) {
        cudaStreamSynchronize(s.ptr());

#ifdef NCCL
        if (bufferGATHER_handle != (void*)-1)
            NCCLCHECK(ncclCommDeregister(rank, bufferGATHER_handle));
        NCCLCHECK(ncclMemFree(bufferGATHER));
#else
        assert(false);
#endif
    } else {
        if (bufferGATHER)
            // Same bytes=0 defect, left as-is: bufferGATHER is allocated only by ncclMemAlloc, whose
            // companion handle sends it down the branch above.
            GPUfree(bufferGATHER, id, 0, s.ptr());
        //cudaFreeAsync(bufferGATHER, s.ptr());
    }
    limb.clear();
    SPECIALlimb.clear();
    DECOMPlimb.clear();
    DIGITlimb.clear();
}

Global::Globals* LimbPartition::getGlobals() {
    return cc.precom.globals->globals[id];
}

void LimbPartition::generate(std::vector<LimbRecord>& records, std::vector<LimbImpl>& limbs, VectorGPU<void*>& ptrs,
                             int pos, VectorGPU<void*>* auxptrs, uint64_t* buffer, size_t offset, uint64_t* buffer_aux,
                             size_t offset_aux, bool noptr) {
    CudaNvtxRange r(std::string{sc::current().function_name()}.substr());
    constexpr bool USE_PARTITION_STREAM = true;
    assert(pos < (int)records.size());
    cudaSetDevice(device);

    const int limbs_size = limbs.size();
    int size = std::max((int)(pos - limbs_size + 1), (int)0);
    std::vector<void*> cpu_ptr(size, nullptr);
    std::vector<void*> cpu_auxptr(size, nullptr);
    for (int i = limbs_size; i <= pos; ++i) {
        const LimbRecord& r = records.at(i);
        if (r.type == U32) {
            if (buffer && buffer_aux) {
                limbs.emplace_back(Limb<uint32_t>(cc, (uint32_t*)buffer, 2 * offset, id,
                                                  USE_PARTITION_STREAM ? s : records.at(i).stream, r.id,
                                                  (uint32_t*)buffer_aux, 2 * offset_aux));
                offset += cc.N;
                offset_aux += cc.N;
            } else if (buffer) {
                limbs.emplace_back(Limb<uint32_t>(cc, (uint32_t*)buffer, 2 * offset, id,
                                                  USE_PARTITION_STREAM ? s : records.at(i).stream, r.id, nullptr, 0));
                offset += cc.N;
            } else
                limbs.emplace_back(
                    Limb<uint32_t>(cc, id, USE_PARTITION_STREAM ? s : records.at(i).stream, r.id, !auxptrs));
            cpu_ptr[i - limbs_size] = {&(std::get<U32>(limbs.back()).v.data)[0]};
            cpu_auxptr[i - limbs_size] = {&(std::get<U32>(limbs.back()).aux.data)[0]};
        }
        if (r.type == U64) {
            if (buffer && buffer_aux) {
                limbs.emplace_back(Limb<uint64_t>(cc, buffer, offset, id,
                                                  USE_PARTITION_STREAM ? s : records.at(i).stream, r.id, buffer_aux,
                                                  offset_aux));
                offset += cc.N;
                offset_aux += cc.N;
            } else if (buffer) {
                limbs.emplace_back(Limb<uint64_t>(cc, buffer, offset, id,
                                                  USE_PARTITION_STREAM ? s : records.at(i).stream, r.id, nullptr, 0));
                offset += cc.N;
            } else {
                if (auxptrs) {
                    limbs.emplace_back(
                        Limb<uint64_t>(cc, id, USE_PARTITION_STREAM ? s : records.at(i).stream, r.id, false));
                } else {
                    limbs.emplace_back(
                        Limb<uint64_t>(cc, id, USE_PARTITION_STREAM ? s : records.at(i).stream, r.id, true));
                }
            }

            cpu_ptr[i - limbs_size] = {&(std::get<U64>(limbs.back()).v.data)[0]};
            cpu_auxptr[i - limbs_size] = {&(std::get<U64>(limbs.back()).aux.data)[0]};
            //cudaFreeHost(aux);
        }
        CudaCheckErrorModNoSync;
        s.wait(USE_PARTITION_STREAM ? s : records.at(i).stream);
        CudaCheckErrorModNoSync;
    }
    if (size > 0) {
        if (!noptr) {
        if (limbs_size + size > ptrs.size) { std::cerr << "PTRS BOUNDS ERROR: ptrs.size=" << ptrs.size << " limbs_size=" << limbs_size << " size=" << size << " total=" << (limbs_size + size) << " at " << __FILE__ << ":" << __LINE__ << std::endl; }
            cudaMemcpyAsync(ptrs.data + limbs_size, cpu_ptr.data(),
                            size * sizeof(void*), cudaMemcpyHostToDevice, s.ptr());
            CudaCheckErrorModNoSync;
            if (auxptrs) {
                cudaMemcpyAsync((*auxptrs).data + limbs_size, cpu_auxptr.data(),
                                size * sizeof(void*), cudaMemcpyHostToDevice, s.ptr());
            }
            CudaCheckErrorModNoSync;
        }
    }
    CudaCheckErrorModNoSync;
}
/*
void LimbPartition::generateLimb() {
    cudaSetDevice(device);
    generate(meta, limb, limbptr, (int)limb.size(), &auxptr);
}
*/

void LimbPartition::generateLimbToLevel(int new_level) {
    cudaSetDevice(device);
    int new_size = getLimbSize(new_level);
    if (new_size > limb.size()) {
        generate(meta, limb, limbptr, new_size - 1, &auxptr);
    }
}

/*
void LimbPartition::generateAllDecompLimb(uint64_t* pInt, size_t offset) {
    cudaSetDevice(device);
    DECOMPlimb.resize(DECOMPmeta.size());
    for (size_t i = 0; i < DECOMPmeta.size(); ++i) {
        generate(DECOMPmeta[i], DECOMPlimb[i], DECOMPlimbptr[i], (int)DECOMPmeta[i].size() - 1,
                 nullptr, pInt, offset, nullptr, 0);
        offset += cc.N * DECOMPmeta.at(i).size();
    }
}

*/

void LimbPartition::generateAllDigitLimb(uint64_t* pInt, size_t offset, int q_band) {
    cudaSetDevice(device);
    DIGITlimb.resize(DIGITmeta.size());
    const int specials   = (int)SPECIALmeta.size();
    int decomp_start     = 0;
    for (size_t i = 0; i < DIGITmeta.size(); ++i) {
        int n = (int)DIGITmeta[i].size() - 1;
        if (q_band >= 0) {
            // Digit-i key row layout = [specials...] ++ [Q-limbs in chain order
            // EXCLUDING digit i's own DECOMP window [s_i, e_i)] — the kernels
            // (dotKSK DIGIT phase) navigate the hole via start-decomp offsets.
            // A ciphertext at level <= q_band consumes digits with s_i <= q_band
            // and per-digit table entries head [0, min(band+1, s_i)) plus tail
            // [e_i, band+1) shifted left by the hole.
            const int s_i = decomp_start;
            const int e_i = decomp_start + (int)DECOMPmeta[i].size();
            if (s_i > q_band) {
                decomp_start += (int)DECOMPmeta[i].size();
                offset += cc.N * DIGITmeta.at(i).size();
                continue;
            }
            const int q_keep = std::min(q_band + 1, s_i) + std::max(0, q_band + 1 - e_i);
            n                = specials + q_keep - 1;
        }
        generate(DIGITmeta[i], DIGITlimb[i], DIGITlimbptr[i], n, nullptr /*&DIGITauxptr[i]*/,
                 pInt, offset, nullptr, 0);
        offset += cc.N * DIGITmeta.at(i).size();
        decomp_start += (int)DECOMPmeta[i].size();
    }
}

/* KSK bit-packing: repack this KEY partition's loaded DECOMP/DIGIT limbs into
 * dense `bits`-per-coefficient bitstreams, point the SAME device pointer tables (limbptr /
 * DIGITlimbptr[i]) at the packed streams, and free the dense u32 Limb storage back to the
 * pool. Consumers must launch the KSK_PACKED=true dot-kernel arms (selected via
 * key_pack_bits at the launch sites); any other reader of these tables is stale by
 * construction — the only such paths are dead (per-limb dotKSK, *ModupDotKSK, the batched
 * hoisted rotate). Lossless: residues are canonical < p < 2^bits.
 * All-u32 single-GPU chains only; composes with key_q_band (packs whatever limbs exist). */
void LimbPartition::packKeyLimbs(const int bits) {
    cudaSetDevice(device);
    assert(cc.GPUid.size() == 1);
    assert(bits > 0 && bits < 32);
    if (key_pack_bits)
        return;
    const size_t words = ((size_t)cc.N * bits + 31) / 32;
    const size_t slotBytes = ((words * 4 + 4) + 15) & ~15ull;  // +1 guard word, 16B-aligned slots
    size_t nlimbs = 0;
    for (auto& d : DECOMPlimb)
        nlimbs += d.size();
    for (auto& d : DIGITlimb)
        nlimbs += d.size();
    if (nlimbs == 0)
        return;
    bufferKSKPACKbytes = nlimbs * slotBytes;
    bufferKSKPACK = (uint64_t*)GPUmalloc(device, bufferKSKPACKbytes, s.ptr());

    size_t slot = 0;
    const uint32_t grid = (uint32_t)((words + 1 + 127) / 128);
    auto pack_one = [&](LimbImpl& l) -> void* {
        assert(l.index() == U32);
        void* dst = (char*)bufferKSKPACK + slot * slotBytes;
        ++slot;
        s.wait(STREAM(l));
        packKsk_<<<dim3{grid}, 128, 0, s.ptr()>>>((uint32_t*)dst, std::get<U32>(l).v.data, cc.N, bits);
        return dst;
    };

    // limbptr mirrors loadDecompDigit's mapping: entry k = the DECOMP limb with meta[k].id.
    // Key partitions never fill `limb`, so unmatched entries are nullptr by construction.
    std::vector<void*> h_limbptr(limbptr.size, nullptr);
    for (size_t i = 0; i < DECOMPlimb.size(); ++i) {
        for (auto& j : DECOMPlimb.at(i)) {
            void* p = pack_one(j);
            for (size_t k = 0; k < meta.size(); ++k)
                if (PRIMEID(j) == meta.at(k).id)
                    h_limbptr[k] = p;
        }
    }
    cudaMemcpyAsync(limbptr.data, h_limbptr.data(), limbptr.size * sizeof(void*), cudaMemcpyHostToDevice, s.ptr());
    for (size_t i = 0; i < DIGITlimb.size(); ++i) {
        if (DIGITlimb[i].empty())
            continue;
        std::vector<void*> h(DIGITlimb[i].size(), nullptr);
        for (size_t j = 0; j < DIGITlimb[i].size(); ++j)
            h[j] = pack_one(DIGITlimb[i][j]);
        cudaMemcpyAsync(DIGITlimbptr[i].data, h.data(), h.size() * sizeof(void*), cudaMemcpyHostToDevice, s.ptr());
    }
    CudaCheckErrorModNoSync;
    // The pack kernels must complete before the dense storage returns to the pool (a later
    // allocation could recycle and overwrite a block a pack kernel is still reading).
    cudaDeviceSynchronize();
    for (auto& d : DECOMPlimb)
        d.clear();
    for (auto& d : DIGITlimb)
        d.clear();
    key_pack_bits = bits;
}

/* FIDESLIB_KSK_REGEN level (default 2): 0 = stream the stored `a`; 1 = hoisted-rotation dot regenerates
 * `a` in-kernel from the key seed; 2 = both `a` readers regenerate, so `a` is never materialized on the
 * device; 3 = level 2 with the shared-memory (stage-A) fusedDotKSK arm. */
int kskRegenLevel() {
    static const int level = [] {
        const char* e = getenv("FIDESLIB_KSK_REGEN");
        return e != nullptr ? atoi(e) : 2;
    }();
    return level;
}

/* Record the seed and RELEASE this KEY partition's `a` rows
 * instead of expanding them. Legal only when every consumer regenerates (FIDESLIB_KSK_REGEN
 * >= 2 — see kskRegenLevel()); KeySwitchingKey::Initialize is the only caller and owns that
 * check. generateDecompAndDigit has already built the device pointer TABLES by the time we
 * get here — those stay (they are a few hundred pointers) and are nulled, so every host
 * staging path keeps working untouched while any missed reader gets a null deref rather than
 * stale key material. The dense storage returns to the pool exactly the way packKeyLimbs
 * releases it after packing. */
void LimbPartition::adoptKskASeed(const std::vector<uint32_t>& seed, int q_band) {
    cudaSetDevice(device);
    assert(cc.GPUid.size() == 1);
    assert(seed.size() == 8);
    for (int i = 0; i < 8; ++i)
        ksk_seed[i] = seed[i];
    ksk_seed_set = true;
    ksk_a_released = true;
    // generateAllDecompAndDigit would have set this; the caller now skips that call entirely,
    // so carry the band ourselves — dotKSK's banded-key guard reads it.
    if (q_band >= 0)
        key_q_band = q_band;

    // Works whether or not the rows were ever generated: KeySwitchingKey::Initialize skips
    // generateDecompAndDigit for a released `a` (never allocating), but the earlier form
    // allocated first and released here, and both must stay valid. Same ordering rule
    // packKeyLimbs documents: nothing may still be reading these blocks when they return to
    // the pool, or a later allocation recycles them under a live kernel.
    cudaDeviceSynchronize();
    for (auto& d : DECOMPlimb)
        d.clear();
    for (auto& d : DIGITlimb)
        d.clear();

    // Null EVERY pointer table this partition owns. They live in bufferAUXptrs and are built
    // by the constructor from the metas, so they exist (correctly sized) even when the storage
    // never was — which is exactly why skipping the generate call is safe for the host staging
    // paths. Uninitialized device memory would otherwise leave them holding garbage that reads
    // as a valid pointer; nulled, any reader this design missed faults instead.
    size_t maxsz = std::max<size_t>({(size_t)limbptr.size, (size_t)GATHERptr.size, (size_t)DECOMPALLptr.size, 1ul});
    for (auto& t : DIGITlimbptr)
        maxsz = std::max<size_t>(maxsz, (size_t)t.size);
    for (auto& t : DECOMPlimbptr)
        maxsz = std::max<size_t>(maxsz, (size_t)t.size);
    const std::vector<void*> nulls(maxsz, nullptr);
    auto null_table = [&](VectorGPU<void*>& t) {
        if (t.size > 0)
            cudaMemcpyAsync(t.data, nulls.data(), t.size * sizeof(void*), cudaMemcpyHostToDevice, s.ptr());
    };
    null_table(limbptr);
    null_table(GATHERptr);
    null_table(DECOMPALLptr);
    for (auto& t : DIGITlimbptr)
        null_table(t);
    for (auto& t : DECOMPlimbptr)
        null_table(t);
    CudaCheckErrorModNoSync;
    cudaStreamSynchronize(s.ptr());
}

void LimbPartition::expandKskADigits(const std::vector<uint32_t>& seed) {
    cudaSetDevice(device);
    assert(cc.GPUid.size() == 1);
    assert(seed.size() == 8);
    KskSeedWords sw;
    for (int i = 0; i < 8; ++i)
        sw.k[i] = seed[i];
    // Record for the 1b-ii in-kernel regen arms (dot kernels re-derive `a` from this).
    for (int i = 0; i < 8; ++i)
        ksk_seed[i] = seed[i];
    ksk_seed_set = true;
    const uint32_t n16 = (uint32_t)cc.N >> 4;
    const uint32_t grid = (uint32_t)((cc.N + 127) / 128);

    const uint32_t n8 = (uint32_t)cc.N >> 3;
    auto expand_one = [&](LimbImpl& l, int digit) {
        STREAM(l).wait(s);
        if (l.index() == U32) {
            const uint32_t p = (uint32_t)cc.precom.constants[id].primes[PRIMEID(l)];
            expandKskA_<<<dim3{grid}, 128, 0, STREAM(l).ptr()>>>(std::get<U32>(l).v.data, sw, digit, p, n16, cc.N);
        } else {
            // SPEC v2 (KSKB): the NATIVE_SIZE=64 chain's limbs, primes < 2^60.
            const uint64_t p = cc.precom.constants[id].primes[PRIMEID(l)];
            expandKskA64_<<<dim3{grid}, 128, 0, STREAM(l).ptr()>>>(std::get<U64>(l).v.data, sw, digit, p, n8, cc.N);
        }
    };
    for (size_t i = 0; i < DECOMPlimb.size(); ++i)
        for (auto& j : DECOMPlimb.at(i))
            expand_one(j, (int)i);
    for (size_t i = 0; i < DIGITlimb.size(); ++i)
        for (auto& j : DIGITlimb.at(i))
            expand_one(j, (int)i);
    CudaCheckErrorModNoSync;

    // The decomp-row pointer mapping the dot kernels read — same as loadDecompDigit's.
    std::vector<void*> cpu_ptr(MAXP, nullptr);
    for (size_t i = 0; i < DECOMPlimb.size(); ++i)
        for (auto& j : DECOMPlimb.at(i))
            for (size_t k = 0; k < meta.size(); ++k)
                if (PRIMEID(j) == meta.at(k).id)
                    cpu_ptr[k] = (j.index() == U32) ? (void*)std::get<U32>(j).v.data
                                                    : (void*)std::get<U64>(j).v.data;
    cudaMemcpyAsync(limbptr.data, cpu_ptr.data(), cpu_ptr.size() * sizeof(void*), cudaMemcpyHostToDevice, s.ptr());

    for (auto& d : DECOMPlimb)
        for (auto& j : d)
            s.wait(STREAM(j));
    for (auto& d : DIGITlimb)
        for (auto& j : d)
            s.wait(STREAM(j));
}

void LimbPartition::generateSpecialLimb(const bool zero_out, const bool for_communication) {
    cudaSetDevice(device);
    if ((for_communication && cc.GPUid.size() > 0 && bufferSPECIAL == nullptr && SPECIALmeta.size() > 0) ||
        (!(for_communication && cc.GPUid.size() > 0) && SPECIALlimb.size() == 0 && SPECIALmeta.size() > 0)) {
        if ((for_communication && cc.GPUid.size() > 0)) {
            assert(SPECIALlimb.size() == 0);
            bufferSPECIALbytes = std::max(1ul, cc.N * SPECIALmeta.size() * 2 * sizeof(uint64_t));
            bufferSPECIALcudaMalloc = true;  // plain cudaMalloc: must be plain cudaFree
            cudaMalloc(&bufferSPECIAL, bufferSPECIALbytes);
            generate(SPECIALmeta, SPECIALlimb, SPECIALlimbptr, (int)SPECIALmeta.size() - 1, &SPECIALauxptr,
                     bufferSPECIAL, 0, bufferSPECIAL, cc.N * SPECIALmeta.size());
        } else {
            assert(bufferSPECIAL == nullptr);
            bufferSPECIALbytes = cc.N * SPECIALmeta.size() * 2 * sizeof(uint64_t);
            bufferSPECIALcudaMalloc = false;
            bufferSPECIAL = (uint64_t*)GPUmalloc(device, (int)bufferSPECIALbytes, s.ptr());
            CudaCheckErrorModNoSync;
            //generate(SPECIALmeta, SPECIALlimb, SPECIALlimbptr, (int)SPECIALmeta.size() - 1, &SPECIALauxptr, nullptr, 0,
            //         nullptr, 0);
            //cudaMalloc(&bufferSPECIAL, std::max(1ul, cc.N * SPECIALmeta.size() * 2 * sizeof(uint64_t)));
            generate(SPECIALmeta, SPECIALlimb, SPECIALlimbptr, (int)SPECIALmeta.size() - 1, &SPECIALauxptr,
                     bufferSPECIAL, 0, bufferSPECIAL, cc.N * SPECIALmeta.size());

            //cudaMallocAsync(&bufferDECOMPandDIGIT, cc.N * SPECIALmeta.size() * 2 * sizeof(uint64_t), s.ptr());
        }
    }
    if (zero_out) {
        if (bufferSPECIAL) {
            if (cc.N * SPECIALmeta.size() * sizeof(uint64_t) > 0)
                cudaMemsetAsync(bufferSPECIAL, 0, cc.N * SPECIALmeta.size() * sizeof(uint64_t), s.ptr());
        } else {
            for (auto& i : SPECIALlimb) {
                if (i.index() == U32) {
                    cudaMemsetAsync(std::get<U32>(i).v.data, 0, cc.N * sizeof(uint32_t), STREAM(i).ptr());
                } else {
                    cudaMemsetAsync(std::get<U64>(i).v.data, 0, cc.N * sizeof(uint64_t), STREAM(i).ptr());
                }
            }
        }
    }
}

template <ALGO algo, NTT_MODE mode>
void LimbPartition::ApplyNTT(int batch, LimbPartition::NTT_fusion_fields fields, std::vector<LimbImpl>& limb,
                             VectorGPU<void*>& limbptr, VectorGPU<void*>& auxptr, ContextData& cc,
                             const int primeid_init, const int limbsize) {
    const int M = (cc.precom.constants[0].type == 0) ? 8 : 4;  // u32 tiles are byte-parity with u64 (kernel M=8): grid must be N/(bd*M*2)

    // The dynamic shared size scales with the limb word size: the kernel lays out
    // `sizeof(T) * blockDim.x * (2*M + 1 + shoup)` bytes, and `32 / M` IS sizeof(T)
    // (M=8 -> 4 for u32, M=4 -> 8 for u64). Same formula at every ApplyNTT/ApplyINTT site.
    const dim3 blockDimFirst{(uint32_t)(1 << ((cc.logN + 1) / 2 - 1))};
    const dim3 blockDimSecond = dim3{(uint32_t)(1 << ((cc.logN) / 2 - 1))};
    const int bytesFirst = (32 / M) * blockDimFirst.x * (2 * M + 1 + (algo == 2 || algo == ALGO_SHOUP ? 1 : 0));
    const int bytesSecond = (32 / M) * blockDimSecond.x * (2 * M + 1 + (algo == 2 || algo == ALGO_SHOUP ? 1 : 0));
    // NTT_RESCALE2: fused composite double drop. The two top limbs are consumed
    // (size = limbsize - 2); stage-1 dat = limbptr + size, so the kernel sees dat[0] = the qb
    // limb and dat[1] = the qa top (both coeff domain); primeid_rescale = the TOP prime (qa).
    const int size = (limbsize != -1 ? limbsize : limb.size()) -
                     (mode == NTT_RESCALE || mode == NTT_MULTPT) - 2 * (mode == NTT_RESCALE2);

    for (int i = 0; i < size; i += batch) {
        uint32_t num_limbs = std::min((uint32_t)batch, (uint32_t)(size - i));

        NTT_<false, algo, mode><<<dim3{cc.N / (blockDimFirst.x * M * 2), num_limbs}, blockDimFirst, bytesFirst,
                                  STREAM(limb.at(i)).ptr()>>>(
            getGlobals(),
            (mode == NTT_RESCALE || mode == NTT_MULTPT || mode == NTT_RESCALE2) ? limbptr.data + size
            : (mode == NTT_MODDOWN)                                             ? fields.op2->limbptr.data + i
                                                                                : limbptr.data + i,
            primeid_init + i, auxptr.data + i, nullptr,
            (mode == NTT_RESCALE || mode == NTT_MULTPT) ? PRIMEID(limb[size])
            : (mode == NTT_RESCALE2)                    ? PRIMEID(limb[size + 1])
                                                        : 0,
            nullptr, nullptr);

        NTT_<true, algo, mode><<<dim3{cc.N / (blockDimSecond.x * M * 2), num_limbs}, blockDimSecond, bytesSecond,
                                 STREAM(limb.at(i)).ptr()>>>(
            getGlobals(), auxptr.data + i, primeid_init + i, limbptr.data + i,
            mode == NTT_MULTPT ? fields.pt->limbptr.data + i : nullptr,
            (mode == NTT_RESCALE || mode == NTT_MULTPT) ? PRIMEID(limb[size])
            : (mode == NTT_RESCALE2)                    ? PRIMEID(limb[size + 1])
                                                        : 0,
            nullptr, nullptr);
    }
}

template <ALGO algo, NTT_MODE mode>
void LimbPartition::NTT(int batch, bool sync, NTT_fusion_fields fields) {
    cudaSetDevice(device);
    int limbsize = getLimbSize(*level);

    if (batch >= 1) {
        if (limbsize > 0) {
            if (sync) {
                for (int i = 0; i < limbsize; i += batch) {
                    STREAM(limb[i]).wait(s);
                }
            }
            ApplyNTT<algo, mode>(batch, fields, limb, limbptr, auxptr, cc, PARTITION(id, 0), limbsize);
            if (sync) {
                for (int i = 0; i < limbsize; i += batch) {
                    s.wait(STREAM(limb[i]));
                }
            }
        }
    } else {
        assert("Invalid NTT batch configuration!");
    }
}

#define YYY(algo, mode) template void LimbPartition::NTT<algo, mode>(int batch, bool sync, NTT_fusion_fields fields);

#include "ntt_types.inc"
#undef YYY

template <ALGO algo, INTT_MODE mode>
void LimbPartition::ApplyINTT(int batch, LimbPartition::INTT_fusion_fields fields, std::vector<LimbImpl>& limb,
                              VectorGPU<void*>& limbptr, VectorGPU<void*>& auxptr, ContextData& cc,
                              const int primeid_init, const int limbsize) {
    const int M = (cc.precom.constants[0].type == 0) ? 8 : 4;  // u32 tiles are byte-parity with u64 (kernel M=8): grid must be N/(bd*M*2)

    dim3 blockDimFirst{(uint32_t)(1 << ((cc.logN - (cc.logN > 13 ? 0 : 0)) / 2 - 1))};
    dim3 blockDimSecond = dim3{(uint32_t)(1 << ((cc.logN + 1 + (cc.logN > 13 ? 0 : 0)) / 2 - 1))};
    int bytesFirst = (32 / M) * blockDimFirst.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));
    int bytesSecond = (32 / M) * blockDimSecond.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));

    for (int i = 0; i < limbsize; i += batch) {
        uint32_t num_limbs = std::min((uint32_t)batch, (uint32_t)(limbsize - i));

        INTT_<false, algo, INTT_NONE>
            <<<dim3{cc.N / (blockDimFirst.x * M * 2), num_limbs}, blockDimFirst, bytesFirst,
               STREAM(limb.at(i)).ptr()>>>(getGlobals(), limbptr.data + i, primeid_init + i, auxptr.data + i);

        INTT_<true, algo, INTT_NONE>
            <<<dim3{cc.N / (blockDimSecond.x * M * 2), num_limbs}, blockDimSecond, bytesSecond,
               STREAM(limb.at(i)).ptr()>>>(getGlobals(), auxptr.data + i, primeid_init + i, limbptr.data + i);
    }
}

template <ALGO algo, INTT_MODE mode>
void LimbPartition::INTT(int batch, bool sync, INTT_fusion_fields fields) {
    cudaSetDevice(device);
    // TODO check level
    const int limbsize = getLimbSize(*level);
    if (batch >= 1) {
        if (limbsize > 0) {
            if (sync) {
                for (int i = 0; i < limbsize; i += batch) {
                    STREAM(limb[i]).wait(s);
                }
            }
            ApplyINTT<algo, mode>(batch, fields, limb, limbptr, auxptr, cc, PARTITION(id, 0), limbsize);
            if (sync) {
                for (int i = 0; i < limbsize; i += batch) {
                    s.wait(STREAM(limb[i]));
                }
            }
        }
    } else {
        assert("Invalid INTT batch configuration!");
    }
}

#define WWW(algo, mode) template void LimbPartition::INTT<algo, mode>(int batch, bool sync, INTT_fusion_fields fields);

#include "ntt_types.inc"
#undef WWW

// The in-place limb add goes through the BYTES-indexed vectorized kernel.
// Forward declaration — the definition sits with the copy dispatch further down this file.
static inline size_t uniform_limb_bytes(const std::vector<LimbRecord>& meta, size_t begin, size_t n, int N);

#ifndef FIDESLIB_ADD_VEC
#define FIDESLIB_ADD_VEC 1
#endif

#ifndef FIDESLIB_ADD_CENSUS
#define FIDESLIB_ADD_CENSUS 0
#endif
#if FIDESLIB_ADD_CENSUS
#include <execinfo.h>
#include <dlfcn.h>
#include <cstdio>
#include <map>
#include <mutex>
#include <vector>
#include <algorithm>
#include <utility>
namespace {
std::mutex g_add_mu;
std::map<void*, long> g_add_sites;
struct AddCensusDump {
    ~AddCensusDump() {
        std::lock_guard<std::mutex> g(g_add_mu);
        long tot = 0;
        for (auto& kv : g_add_sites) tot += kv.second;
        std::fprintf(stderr, "[addcensus] total LimbPartition::add = %ld across %zu sites\n", tot,
                     g_add_sites.size());
        std::vector<std::pair<void*, long>> v(g_add_sites.begin(), g_add_sites.end());
        std::sort(v.begin(), v.end(), [](auto& a, auto& b) { return a.second > b.second; });
        for (auto& kv : v) {
            Dl_info info{};
            size_t off = 0;
            if (dladdr(kv.first, &info) && info.dli_fbase)
                off = (size_t)((char*)kv.first - (char*)info.dli_fbase);
            std::fprintf(stderr, "[addcensus] %6ld  +0x%zx\n", kv.second, off);
        }
    }
} g_add_dump;
#define FIDESLIB_ADD_CENSUS_HIT()                                     \
    do {                                                              \
        void* bt[6];                                                  \
        const int n_ = backtrace(bt, 6);                              \
        void* site = (n_ > 4) ? bt[4] : (n_ > 1 ? bt[n_ - 1] : nullptr); \
        std::lock_guard<std::mutex> g_(g_add_mu);                     \
        g_add_sites[site]++;                                          \
    } while (0)
}  // namespace
#else
#define FIDESLIB_ADD_CENSUS_HIT() ((void)0)
#endif

void LimbPartition::add(const LimbPartition& p, const bool ext) {
    FIDESLIB_ADD_CENSUS_HIT();
    cudaSetDevice(device);
    const int limbsize = getLimbSize(*level);
    s.wait(p.getS());
    for (int i = 0; i < limbsize; i += cc.batch) {
        STREAM(limb[i]).wait(s);
        uint32_t num_limbs = std::min((int)limbsize - i, cc.batch);
        const int add_bpt = fideslibAddBytes();
        const size_t add_bpl = FIDESLIB_ADD_VEC ? uniform_limb_bytes(meta, (size_t)i, (size_t)num_limbs, cc.N) : 0;
        if (add_bpl && (add_bpl % (size_t)(add_bpt * 128)) == 0)
            launchAddBytes(dim3{(uint32_t)(add_bpl / (add_bpt * 128)), num_limbs}, dim3{128}, STREAM(limb[i]).ptr(),
                           limbptr.data + i, p.limbptr.data + i, PARTITION(id, i), add_bpt);
        else
            add_<<<dim3{(uint32_t)cc.N / 128, num_limbs}, 128, 0, STREAM(limb[i]).ptr()>>>(
                limbptr.data + i, p.limbptr.data + i, PARTITION(id, i));
    }
    if (ext) {
        int start = cc.splitSpecialMeta.at(id).at(0).id - (cc.L + 1);
        int num_limbs = cc.splitSpecialMeta.at(id).size();
        for (size_t i = start; i < start + num_limbs; i += cc.batch) {
            STREAM(SPECIALlimb[i]).wait(s);
            uint32_t size = std::min((int)start + num_limbs - (int)i, cc.batch);
            {
                const int sadd_bpt = fideslibAddBytes();
                const size_t sadd_bpl =
                    FIDESLIB_ADD_VEC ? uniform_limb_bytes(SPECIALmeta, (size_t)i, (size_t)size, cc.N) : 0;
                if (sadd_bpl && (sadd_bpl % (size_t)(sadd_bpt * 128)) == 0)
                    launchAddBytes(dim3{(uint32_t)(sadd_bpl / (sadd_bpt * 128)), size}, dim3{128},
                                   STREAM(SPECIALlimb[i]).ptr(), SPECIALlimbptr.data + i, p.SPECIALlimbptr.data + i,
                                   SPECIAL(id, i), sadd_bpt);
                else
                add_<<<dim3{(uint32_t)cc.N / 128, size}, 128, 0, STREAM(SPECIALlimb[i]).ptr()>>>(
                    SPECIALlimbptr.data + i, p.SPECIALlimbptr.data + i,
                    SPECIAL(
                        id,
                        i));  // TODO: have to check if Limbpartition comes from a plaintext, where extension limbs are mapped differently
            }
        }
        for (size_t i = start; i < start + num_limbs; i += cc.batch) {
            s.wait(STREAM(SPECIALlimb[i]));
        }
    }
    for (int i = 0; i < limbsize; i += cc.batch) {
        s.wait(STREAM(limb[i]));
    }
    p.getS().wait(s);
}

void LimbPartition::scaleByP() {
    const int limbsize = getLimbSize(*level);
    cudaSetDevice(device);
    for (int i = 0; i < limbsize; i += cc.batch) {
        STREAM(limb[i]).wait(s);
        uint32_t num_limbs = std::min((int)limbsize - i, cc.batch);
        scaleByP_<<<dim3{(uint32_t)cc.N / 128, num_limbs}, 128, 0, STREAM(limb[i]).ptr()>>>(limbptr.data + i,
                                                                                            PARTITION(id, i));
    }
    for (int i = 0; i < limbsize; i += cc.batch) {
        s.wait(STREAM(limb[i]));
    }
}

void LimbPartition::sub(const LimbPartition& p) {
    cudaSetDevice(device);
    const int limbsize = getLimbSize(*level);
    s.wait(p.getS());
    for (int i = 0; i < limbsize; i += cc.batch) {
        STREAM(limb[i]).wait(s);
        uint32_t num_limbs = std::min((int)limbsize - i, cc.batch);
        const int sub_bpt = fideslibAddBytes();
        const size_t sub_bpl = FIDESLIB_ADD_VEC ? uniform_limb_bytes(meta, (size_t)i, (size_t)num_limbs, cc.N) : 0;
        if (sub_bpl && (sub_bpl % (size_t)(sub_bpt * 128)) == 0)
            launchSubBytes(dim3{(uint32_t)(sub_bpl / (sub_bpt * 128)), num_limbs}, dim3{128}, STREAM(limb[i]).ptr(),
                           limbptr.data + i, p.limbptr.data + i, PARTITION(id, i), sub_bpt);
        else
        sub_<<<dim3{(uint32_t)cc.N / 128, num_limbs}, 128, 0, STREAM(limb[i]).ptr()>>>(
            limbptr.data + i, p.limbptr.data + i, PARTITION(id, i));
    }
    for (int i = 0; i < limbsize; i += cc.batch) {
        s.wait(STREAM(limb[i]));
    }
    p.getS().wait(s);
}

void LimbPartition::multElement(const LimbPartition& p) {
    cudaSetDevice(device);

    int limbsize = getLimbSize(*level);
    assert(limbsize <= (int)p.limb.size());

    s.wait(p.getS());
    for (int i = 0; i < limbsize; i += cc.batch) {
        STREAM(limb[i]).wait(s);
        uint32_t num_limbs = std::min((int)limbsize - i, cc.batch);
        Mult_<<<dim3{(uint32_t)cc.N / 128, num_limbs}, 128, 0, STREAM(limb[i]).ptr()>>>(
            limbptr.data + i, limbptr.data + i, p.limbptr.data + i, PARTITION(id, i));
    }
    for (int i = 0; i < limbsize; i += cc.batch) {
        s.wait(STREAM(limb[i]));
    }
    p.getS().wait(s);
}

void LimbPartition::multElement(const LimbPartition& partition1, const LimbPartition& partition2) {
    cudaSetDevice(device);
    int limbsize = getLimbSize(*level);
    assert(limbsize <= partition1.limb.size());
    assert(limbsize <= partition2.limb.size());

    s.wait(partition1.getS());
    s.wait(partition2.getS());
    for (int i = 0; i < limbsize; i += cc.batch) {
        STREAM(limb[i]).wait(s);
        uint32_t num_limbs = std::min((int)limbsize - i, cc.batch);
        Mult_<<<dim3{(uint32_t)cc.N / 128, num_limbs}, 128, 0, STREAM(limb[i]).ptr()>>>(
            (void**)limbptr.data + i, (void**)partition1.limbptr.data + i, (void**)partition2.limbptr.data + i,
            PARTITION(id, i));
    }
    for (int i = 0; i < limbsize; i += cc.batch) {
        s.wait(STREAM(limb[i]));
    }
    partition1.getS().wait(s);
    partition2.getS().wait(s);
}

void LimbPartition::rescale() {
    const int limbsize = getLimbSize(*level);
    assert(cc.GPUid.size() > 1 || limbsize > 1);
    if (limbsize == 0)
        return;

    cudaSetDevice(device);

    LimbImpl& top = limb.at(limbsize - 1);

    int aux_size;
    SWITCH_RET(top, aux.size, aux_size);
    if (aux_size == 0) {
        {
            constexpr ALGO algo = ALGO_SHOUP;
            const int M = (cc.precom.constants[0].type == 0) ? 8 : 4;  // u32 tiles are byte-parity with u64 (kernel M=8): grid must be N/(bd*M*2)

            dim3 blockDimFirst{(uint32_t)(1 << ((cc.logN) / 2 - 1))};
            dim3 blockDimSecond = dim3{(uint32_t)(1 << ((cc.logN + 1) / 2 - 1))};
            int bytesFirst = (32 / M) * blockDimFirst.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));
            int bytesSecond = (32 / M) * blockDimSecond.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));

            int start = 0;
            for (int i = limbsize - 1; i < limbsize; i += cc.batch) {
                cc.top_limb_stream.at(id).wait(s);
                uint32_t num_limbs = 1;

                INTT_<false, algo, INTT_NONE><<<dim3{cc.N / (blockDimFirst.x * M * 2), num_limbs}, blockDimFirst,
                                                bytesFirst, cc.top_limb_stream.at(id).ptr()>>>(
                    getGlobals(), limbptr.data + start + i, PARTITION(id, start + i), cc.top_limbptr.at(id).data);

                INTT_<true, algo, INTT_NONE><<<dim3{cc.N / (blockDimSecond.x * M * 2), num_limbs}, blockDimSecond,
                                               bytesSecond, STREAM(limb.at(start + i)).ptr()>>>(
                    getGlobals(), cc.top_limbptr.at(id).data, PARTITION(id, start + i), limbptr.data + start + i);
            }
            s.wait(cc.top_limb_stream.at(id));
        }
    } else {
        STREAM(top).wait(s);
        SWITCH(top, INTT<ALGO_SHOUP>());
    }
    if (aux_size == 0) {
        auto& auxLimbs = cc.getModdownAux(0).GPU.at(id);
        s.wait(auxLimbs.getS());
        if (limbsize > 0) {
            constexpr ALGO algo = ALGO_SHOUP;
            const int M = (cc.precom.constants[0].type == 0) ? 8 : 4;  // u32 tiles are byte-parity with u64 (kernel M=8): grid must be N/(bd*M*2)

            dim3 blockDimFirst{(uint32_t)(1 << ((cc.logN) / 2 - 1))};
            dim3 blockDimSecond = dim3{(uint32_t)(1 << ((cc.logN + 1) / 2 - 1))};
            int bytesFirst = (32 / M) * blockDimFirst.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));
            int bytesSecond = (32 / M) * blockDimSecond.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));

            {
                NTT_<false, algo, NTT_RESCALE>
                    <<<dim3{cc.N / (blockDimFirst.x * M * 2), static_cast<unsigned int>(limbsize - 1)}, blockDimFirst, bytesFirst, s.ptr()>>>(
                        getGlobals(), limbptr.data + limbsize - 1, PARTITION(id, 0), auxLimbs.limbptr.data,
                        nullptr /*limbptr.data*/, PRIMEID(top));

                NTT_<true, algo, NTT_RESCALE>
                    <<<dim3{cc.N / (blockDimSecond.x * M * 2), static_cast<unsigned int>(limbsize - 1)}, blockDimSecond, bytesSecond, s.ptr()>>>(
                        getGlobals(), auxLimbs.limbptr.data, PARTITION(id, 0), limbptr.data, nullptr, PRIMEID(top));
            }
        }
        auxLimbs.s.wait(s);
    } else {
        for (size_t i = 0; i < limbsize - 1; i += cc.batch) {
            STREAM(limb[i]).wait(STREAM(top));
        }
        NTT<ALGO_SHOUP, NTT_RESCALE>(cc.batch, false, NTT_fusion_fields{});
        for (size_t i = 0; i < limbsize - 1; i += cc.batch) {
            STREAM(top).wait(STREAM(limb[i]));
        }

        s.wait(STREAM(top));
    }
    //while (bufferLIMB == nullptr && limb.size() > limbsize - 1) {
    //    STREAM(limb.back()).wait(s);
    //    limb.pop_back();
    //}
}

// Fused composite DOUBLE prime drop (compositeDegree()==2 chains). One gy=2
// top-pair INTT + one NTT_RESCALE2 pass replace TWO full sequential rescale passes (each with
// its own gy=1 top INTT + gy=(L-1) NTT_RESCALE pair) — ~half the rescale kernel work.
// Sequential drop semantics are preserved exactly inside the kernel (rescale2_combine,
// NTT.cu), so the result is BIT-IDENTICAL to two rescale() calls. Returns false (caller falls
// back to the two-pass loop) when the shape doesn't fit: fewer than 3 limbs, non-consecutive
// top primeids (multi-GPU interleaving), or aux-less (constant) limbs.
bool LimbPartition::rescale2() {
    const int limbsize = getLimbSize(*level);
    if (limbsize < 3)
        return false;
    LimbImpl& top = limb.at(limbsize - 1);
    LimbImpl& top2 = limb.at(limbsize - 2);
    if (PRIMEID(top2) != PRIMEID(top) - 1)
        return false;
    int aux_size;
    SWITCH_RET(top, aux.size, aux_size);
    if (aux_size == 0)
        return false;

    cudaSetDevice(device);
    constexpr ALGO algo = ALGO_SHOUP;
    const int M = (cc.precom.constants[0].type == 0) ? 8 : 4;  // u32 tiles are byte-parity with u64 (kernel M=8)

    // 1. top-pair INTT (gy=2), in place via the limbs' own aux staging, on the partition stream.
    {
        dim3 blockDimFirst{(uint32_t)(1 << ((cc.logN) / 2 - 1))};
        dim3 blockDimSecond = dim3{(uint32_t)(1 << ((cc.logN + 1) / 2 - 1))};
        int bytesFirst = (32 / M) * blockDimFirst.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));
        int bytesSecond = (32 / M) * blockDimSecond.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));

        INTT_<false, algo, INTT_NONE><<<dim3{cc.N / (blockDimFirst.x * M * 2), 2}, blockDimFirst, bytesFirst,
                                        s.ptr()>>>(getGlobals(), limbptr.data + limbsize - 2,
                                                   PARTITION(id, limbsize - 2), auxptr.data + limbsize - 2);
        INTT_<true, algo, INTT_NONE><<<dim3{cc.N / (blockDimSecond.x * M * 2), 2}, blockDimSecond, bytesSecond,
                                       s.ptr()>>>(getGlobals(), auxptr.data + limbsize - 2,
                                                  PARTITION(id, limbsize - 2), limbptr.data + limbsize - 2);

    }

    // 2. the fused double-drop pass (gy = limbsize-2), also on the partition stream.
    {
        dim3 blockDimFirst{(uint32_t)(1 << ((cc.logN + 1) / 2 - 1))};
        dim3 blockDimSecond = dim3{(uint32_t)(1 << ((cc.logN) / 2 - 1))};
        int bytesFirst = (32 / M) * blockDimFirst.x * (2 * M + 1 + (algo == 2 || algo == ALGO_SHOUP ? 1 : 0));
        int bytesSecond = (32 / M) * blockDimSecond.x * (2 * M + 1 + (algo == 2 || algo == ALGO_SHOUP ? 1 : 0));
        const int size = limbsize - 2;

        NTT_<false, algo, NTT_RESCALE2><<<dim3{cc.N / (blockDimFirst.x * M * 2), (uint32_t)size}, blockDimFirst,
                                          bytesFirst, s.ptr()>>>(getGlobals(), limbptr.data + size, PARTITION(id, 0),
                                                                 auxptr.data, nullptr, PRIMEID(top));
        NTT_<true, algo, NTT_RESCALE2><<<dim3{cc.N / (blockDimSecond.x * M * 2), (uint32_t)size}, blockDimSecond,
                                         bytesSecond, s.ptr()>>>(getGlobals(), auxptr.data, PARTITION(id, 0),
                                                                 limbptr.data, nullptr, PRIMEID(top));
    }
    return true;
}

void LimbPartition::multPt(const LimbPartition& p) {
    const int limbsize = getLimbSize(*level);
    // assert(SPECIALlimb.size() == 0 && p.SPECIALlimb.size() == 0);
    assert(limbsize <= p.limb.size());
    assert(limbsize > 1);
    cudaSetDevice(device);

    constexpr bool capture = false;
    static std::map<int, cudaGraphExec_t> exec_map;

    {
        LimbImpl& top = limb.back();

        cudaGraphExec_t& exec = exec_map[limbsize];

        run_in_graph<capture>(exec, s, [&]() {
            STREAM(top).wait(s);
            SWITCH(top, mult(p.limb.back()));
            SWITCH(top, INTT<ALGO_SHOUP>());

            for (size_t i = 0; i < limbsize - 1; i += cc.batch) {
                STREAM(limb.at(i)).wait(STREAM(top));
            }
            if (limbsize > 1)
                NTT<ALGO_SHOUP, NTT_MULTPT>(cc.batch, false, NTT_fusion_fields{.pt = &p});
            for (size_t i = 0; i < limbsize - 1; i += cc.batch) {
                STREAM(top).wait(STREAM(limb.at(i)));
            }

            s.wait(STREAM(top));
        });

        //while (bufferLIMB == nullptr && limb.size() > limbsize - 1) {
        //    STREAM(limb.back()).wait(s);
        //    limb.pop_back();
        //}
    }
}

void LimbPartition::modup(LimbPartition& aux_partition) {

    constexpr ALGO algo = ALGO_SHOUP;
    constexpr bool PRINT = false;
    //assert(SPECIALlimb.empty());
    cudaSetDevice(device);

    const int limbsize = *level + 1;
    generateAllDecompAndDigit(false);
    s.wait(aux_partition.getS());

    if constexpr (PRINT) {
        std::cout << "Before modup ";
        for (auto& i : limb) {
            SWITCH(i, printThisLimb(2));
        }
        std::cout << std::endl;
    }

    // ONE wide INTT over all source limbs instead of dnum per-digit launches (small per-digit
    // grids under-fill the GPU). The merged launch (gy = limbsize) is byte- and primeid-identical:
    // stage 2 scatters through DECOMPALLptr, whose entry (start_d + i) IS DECOMPlimbptr[d][i].
    // Per-digit conv/NTT below still fan out on the digit streams; each s_d.wait(s) picks up the
    // merged INTT's completion.
    {
        const int M = (cc.precom.constants[0].type == 0) ? 8 : 4;  // u32 tiles are byte-parity with u64 (kernel M=8): grid must be N/(bd*M*2)

        dim3 blockDimFirst{(uint32_t)(1 << ((cc.logN) / 2 - 1))};
        dim3 blockDimSecond = dim3{(uint32_t)(1 << ((cc.logN + 1) / 2 - 1))};
        int bytesFirst = (32 / M) * blockDimFirst.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));
        int bytesSecond = (32 / M) * blockDimSecond.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));

        for (int i = 0; i < limbsize; i += cc.batch) {
            STREAM(limb.at(i)).wait(s);
            uint32_t num_limbs = std::min((uint32_t)cc.batch, (uint32_t)(limbsize - i));

            INTT_<false, algo, INTT_NONE><<<dim3{cc.N / (blockDimFirst.x * M * 2), num_limbs}, blockDimFirst,
                                            bytesFirst, STREAM(limb.at(i)).ptr()>>>(
                getGlobals(), limbptr.data + i, PARTITION(id, i), auxptr.data + i);

            INTT_<true, algo, INTT_NONE><<<dim3{cc.N / (blockDimSecond.x * M * 2), num_limbs}, blockDimSecond,
                                           bytesSecond, STREAM(limb.at(i)).ptr()>>>(
                getGlobals(), auxptr.data + i, PARTITION(id, i), DECOMPALLptr.data + i);

        }
        for (int i = 0; i < limbsize; i += cc.batch) {
            s.wait(STREAM(limb.at(i)));
        }
    }

    for (size_t d = 0; d < DECOMPlimb.size(); ++d) {

        int start = 0;
        for (int j = 0; j < d; ++j)
            start += DECOMPlimb.at(j).size();
        int size = std::min((int)DECOMPlimb.at(d).size(), limbsize - start);
        if (size <= 0)
            break;

        Stream& s_d = cc.digitStream.at(d).at(id);
        s_d.wait(s);

        if constexpr (PRINT) {
            cudaDeviceSynchronize();
            std::cout << "After INTT ";
            for (auto& i : DECOMPlimb[d]) {
                SWITCH(i, printThisLimb(2));
            }
            std::cout << std::endl;
            cudaDeviceSynchronize();
        }

        {
            dim3 blockSize{64, 2};
            dim3 gridSize{(uint32_t)cc.N / blockSize.x};
            int shared_bytes = sizeof(uint64_t) * (size /*DECOMPlimb[d].size()*/) * blockSize.x;
            DecompAndModUpConv<algo><<<gridSize, blockSize, shared_bytes, s_d.ptr()>>>(
                DECOMPlimbptr[d].data, *level + 1, DIGITlimbptr[d].data, digitid[d], getGlobals());
        }
        if constexpr (PRINT) {
            cudaDeviceSynchronize();
            std::cout << "After conv ";
            for (auto& i : DIGITlimb[d]) {
                SWITCH(i, printThisLimb(2));
            }
            std::cout << std::endl;
            cudaDeviceSynchronize();
        }

        const int digitsize = cc.precom.constants[id].num_primeid_digit_to[digitid.at(d)][*level];
        for (size_t i = 0; i < digitsize; i += cc.batch) {
            STREAM(DIGITlimb.at(d).at(i)).wait(s_d);
        }
        ApplyNTT<algo, NTT_NONE>(cc.batch, NTT_fusion_fields{}, DIGITlimb.at(d), DIGITlimbptr.at(d),
                                 aux_partition.DIGITlimbptr.at(d), cc, DIGIT(digitid.at(d), 0), digitsize);

        for (size_t i = 0; i < digitsize; i += cc.batch) {
            s_d.wait(STREAM(DIGITlimb.at(d).at(i)));
        }

        if constexpr (PRINT) {
            cudaDeviceSynchronize();
            std::cout << "After NTT ";
            for (auto& i : DIGITlimb[d]) {
                SWITCH(i, printThisLimb(2));
            }
            std::cout << std::endl;
            cudaDeviceSynchronize();
        }
    }
    for (size_t d = 0; d < DECOMPlimb.size(); ++d) {
        s.wait(cc.digitStream[d][id]);
    }

    aux_partition.getS().wait(s);
}

void LimbPartition::freeSpecialLimbs() {
    cudaSetDevice(device);
    for (size_t i = 0; i < SPECIALlimb.size(); ++i) {
        STREAM(SPECIALlimb.at(i)).wait(s);
    }
    SPECIALlimb.clear();
    freeSpecialBuffer();
}

/* Hand GPUfree the byte count GPUmalloc was given, and match the ROUTE. GPUfree derives the
 * free-list bucket from `bytes` (0 files a large block in the 1 KB bucket, stranding it), and
 * generateSpecialLimb's `for_communication` arm uses plain cudaMalloc, which must not reach cudaFreeAsync. */
void LimbPartition::freeSpecialBuffer() {
    if (bufferSPECIAL == nullptr)
        return;
    if (bufferSPECIALcudaMalloc) {
        cudaFree(bufferSPECIAL);
    } else {
        assert(bufferSPECIALbytes > 0 && "special buffer freed without its GPUmalloc byte count");
        GPUfree(bufferSPECIAL, id, (int)bufferSPECIALbytes, s.ptr());
    }
    bufferSPECIAL = nullptr;
    bufferSPECIALbytes = 0;
    bufferSPECIALcudaMalloc = false;
}

/* FALLBACK limb-copy path. The default copy is the type-unaware copy_bytes_ (dispatch below);
 * copy_v4_ and scalar copy_ survive only for when the limb width is unknown or non-uniform, or the
 * limb does not tile the per-thread width. Neither takes a length, so the grid must cover N exactly.
 * bytes_per_limb == 0 means "unknown or non-uniform widths" -> the typed kernels branch per limb. */
#ifndef FIDESLIB_COPY_ABLATE
#define FIDESLIB_COPY_ABLATE 0
#endif

static inline void launch_copy_limbs(uint32_t N, uint32_t nlimbs, cudaStream_t stream, void** src, void** dst,
                                     size_t bytes_per_limb = 0) {
#if FIDESLIB_COPY_ABLATE
    // DIAGNOSTIC (default 0; produces incorrect results by design): skip the limb copy entirely,
    // to bound what eliminating copies could save.
    return;
#endif
    if (nlimbs == 0)
        return;
    /* Type-unaware byte copy at 16 B/thread, the path when the width is known: same geometry as copy_v4_
     * without the per-element ISU64 branch, and correct for mixed-width chains by construction. 16 B/thread
     * is the production optimum: wider per-thread work shrinks the grid and loses under co-scheduling. */
    if (bytes_per_limb) {
        constexpr int req = 16;
        for (int bpt = req; bpt >= 16; bpt >>= 1) {
            const size_t tile = (size_t)bpt * 128;
            if (bytes_per_limb % tile != 0)
                continue;
            launchCopyBytes(dim3{(uint32_t)(bytes_per_limb / tile), nlimbs}, dim3{128}, stream, src, dst, bpt);
            return;
        }
    }
    if ((N % 512) == 0)
        copy_v4_<<<dim3{N / 512, nlimbs}, 128, 0, stream>>>(src, dst);
    else
        copy_<<<dim3{N / 128, nlimbs}, 128, 0, stream>>>(src, dst);
}

/* Bytes per limb IF every limb in [begin, begin+n) has the same element width; 0 otherwise,
 * which routes the caller to the typed fallback. Cheap (n <= ~54) and called once per copy. */
static inline size_t uniform_limb_bytes(const std::vector<LimbRecord>& meta, size_t begin, size_t n, int N) {
    if (n == 0 || begin + n > meta.size())
        return 0;
    const auto t = meta[begin].type;
    for (size_t k = begin + 1; k < begin + n; ++k)
        if (meta[k].type != t)
            return 0;
    return (size_t)N * (t == U32 ? 4u : 8u);
}

void LimbPartition::copyLimb(const LimbPartition& partition) {
    cudaSetDevice(device);
    s.wait(partition.getS());
    int limbsize = getLimbSize(*level);
    assert(*level == *partition.level);
    //std::cout << "GPU: " << id << " copy " << limbsize << "limbs" << std::endl;
    if (limbsize > 0)
        launch_copy_limbs((uint32_t)cc.N, (uint32_t)limbsize, s.ptr(), partition.limbptr.data, limbptr.data,
                          uniform_limb_bytes(meta, 0, (size_t)limbsize, cc.N));
    /*
    for (size_t i = 0; i < partition.limb.size(); ++i) {
        STREAM(limb.at(i)).wait(s);
        SWITCH(limb.at(i), copyV(partition.limb.at(i)));
    }
    for (size_t i = 0; i < partition.limb.size(); ++i) {
        s.wait(STREAM(limb.at(i)));
    }
    */
    partition.getS().wait(s);
}
void LimbPartition::copySpecialLimb(const LimbPartition& p) {
    cudaSetDevice(device);
    this->generateSpecialLimb(false, false);
    s.wait(p.getS());
    assert(*level == *p.level);
    int start = cc.splitSpecialMeta.at(id).at(0).id - (cc.L + 1);
    int num_limbs = cc.splitSpecialMeta.at(id).size();
    for (size_t i = start; i < start + num_limbs; i += cc.batch) {
        STREAM(SPECIALlimb[i - (SPECIALmeta.size() > SPECIALlimb.size()) * start]).wait(s);
        uint32_t size = std::min((int)start + num_limbs - (int)i, cc.batch);

        launch_copy_limbs(
            (uint32_t)cc.N, size, STREAM(SPECIALlimb[i - (SPECIALmeta.size() > SPECIALlimb.size()) * start]).ptr(),
            p.SPECIALlimbptr.data + i - (SPECIALmeta.size() > p.SPECIALlimb.size()) * start,
            SPECIALlimbptr.data + i - (SPECIALmeta.size() > SPECIALlimb.size()) * start,
            uniform_limb_bytes(SPECIALmeta, (size_t)i, (size_t)size, cc.N));
    }
    for (size_t i = start; i < start + num_limbs; i += cc.batch) {
        s.wait(STREAM(SPECIALlimb[i - (SPECIALmeta.size() > SPECIALlimb.size()) * start]));
    }
    p.getS().wait(s);
}

void LimbPartition::generateAllDecompAndDigit(bool iskey, int q_band) {
    cudaSetDevice(device);
    if ((!(iskey || cc.GPUid.size() == 1) && bufferGATHER == nullptr) ||
        ((iskey || cc.GPUid.size() == 1) && DECOMPlimb[0].size() == 0)) {
        int decomp_limbs = 0;
        for (auto& d : DECOMPmeta)
            decomp_limbs += d.size();
        int digit_limbs = 0;
        for (auto& d : DIGITmeta)
            digit_limbs += d.size();

        size_t size = cc.N * (/*decomp_limbs +*/ digit_limbs);
        if (cc.GPUid.size() == 1 || iskey) {
            //bufferDECOMPandDIGIT = (uint64_t*)GPUmalloc(device, std::max(1ul, size) * sizeof(uint64_t), s.ptr());
            //cudaMallocAsync(&bufferDECOMPandDIGIT, std::max(1ul, size) * sizeof(uint64_t), s.ptr());
        } else {
            // TODO do not do for key switching keys

#ifdef NCCL
            /*
            cudaStreamSynchronize(s.ptr());
            NCCLCHECK(ncclMemAlloc((void**)&bufferDECOMPandDIGIT, std::max(1ul, size) * sizeof(uint64_t)));
            NCCLCHECK(ncclCommRegister(rank, bufferDECOMPandDIGIT, std::max(1ul, size) * sizeof(uint64_t),
                                       &bufferDECOMPandDIGIT_handle));
            if (bufferDECOMPandDIGIT_handle == nullptr)
                bufferDECOMPandDIGIT_handle = (void*)-1;
            cudaDeviceSynchronize();
            */
#else
            assert(false);
#endif
        }
        //generateAllDecompLimb(bufferDECOMPandDIGIT, 0);
        generateGatherLimb(iskey);
        DECOMPlimb.resize(DECOMPmeta.size());
        // Also assemble the digit-major concatenation of all DECOMP staging
        // pointers (DECOMPALLptr) so modup can INTT every source limb in ONE wide launch.
        std::vector<void*> all_ptr;
        for (size_t i = 0; i < DECOMPmeta.size(); ++i) {
            for (size_t j = 0; j < DECOMPmeta.at(i).size(); ++j) {
                int pos = 0;
                for (size_t k = 0; k < cc.meta.size(); ++k) {
                    for (size_t l = 0; l < cc.meta[k].size(); ++l) {
                        if (cc.meta[k][l].id == DECOMPmeta[i][j].id) {
                            generate(DECOMPmeta[i], DECOMPlimb[i], DECOMPlimbptr[i], (int)j, nullptr, bufferGATHER,
                                     pos * cc.N, nullptr, 0, true);
                        }
                        pos++;
                    }
                }
            }
            std::vector<void*> cpu_ptr(DECOMPmeta.at(i).size(), nullptr);
            for (int j = 0; j < DECOMPlimb[i].size(); ++j) {
                cpu_ptr[j] = DECOMPlimb[i][j].index() == U32 ? (void*)std::get<U32>(DECOMPlimb[i][j]).v.data
                                                             : (void*)std::get<U64>(DECOMPlimb[i][j]).v.data;
            }

            if (DECOMPmeta.at(i).size() * sizeof(void*) > 0)
                cudaMemcpyAsync(DECOMPlimbptr[i].data, cpu_ptr.data(), DECOMPmeta.at(i).size() * sizeof(void*),
                                cudaMemcpyHostToDevice, s.ptr());
            all_ptr.insert(all_ptr.end(), cpu_ptr.begin(), cpu_ptr.end());
        }
        if (!all_ptr.empty()) {
            assert((int)all_ptr.size() <= MAXP);
            // NOTE: cudaMemcpyAsync from pageable host memory is staged synchronously by the
            // driver, so all_ptr going out of scope right after is safe (same idiom as the
            // per-digit copies above).
            cudaMemcpyAsync(DECOMPALLptr.data, all_ptr.data(), all_ptr.size() * sizeof(void*),
                            cudaMemcpyHostToDevice, s.ptr());
        }
        generateGatherLimb(iskey);
        generateAllDigitLimb(bufferDECOMPandDIGIT, 0 /*cc.N * decomp_limbs*/, q_band);
        if (q_band >= 0)
            key_q_band = q_band;
    }
}

void LimbPartition::mult1AddMult23Add4(const LimbPartition& partition1, const LimbPartition& partition2,
                                       const LimbPartition& partition3, const LimbPartition& partition4) {
    const int limbsize = getLimbSize(*level);
    cudaSetDevice(device);
    assert(limbsize <= partition1.limb.size());
    assert(limbsize <= partition2.limb.size());
    assert(limbsize <= partition3.limb.size());
    assert(limbsize <= partition4.limb.size());

    s.wait(partition1.getS());
    s.wait(partition2.getS());
    s.wait(partition3.getS());
    s.wait(partition4.getS());

    for (int i = 0; i < limbsize; i += cc.batch) {
        STREAM(limb[i]).wait(s);
        uint32_t num_limbs = std::min((int)limbsize - i, cc.batch);
        mult1AddMult23Add4_<<<dim3{(uint32_t)cc.N / 128, num_limbs}, 128, 0, STREAM(limb[i]).ptr()>>>(
            PARTITION(id, i), limbptr.data + i, partition1.limbptr.data + i, partition2.limbptr.data + i,
            partition3.limbptr.data + i, partition4.limbptr.data + i);
    }
    for (int i = 0; i < limbsize; i += cc.batch) {
        s.wait(STREAM(limb[i]));
    }

    partition1.getS().wait(s);
    partition2.getS().wait(s);
    partition3.getS().wait(s);
    partition4.getS().wait(s);
}

void LimbPartition::multNoModdownEnd(LimbPartition& c0, const LimbPartition& bc0, const LimbPartition& bc1,
                                     const LimbPartition& in, const LimbPartition& aux) {
    const int limbsize = getLimbSize(*level);
    cudaSetDevice(device);
    assert(limbsize <= c0.limb.size());
    assert(limbsize <= bc0.limb.size());
    assert(limbsize <= bc1.limb.size());
    assert(limbsize <= in.limb.size());
    assert(limbsize <= aux.limb.size());

    s.wait(c0.getS());
    s.wait(bc0.getS());
    s.wait(bc1.getS());
    s.wait(in.getS());
    s.wait(aux.getS());
    c0.getS().wait(aux.getS());

    for (int i = 0; i < limbsize; i += cc.batch) {
        STREAM(limb[i]).wait(s);
        uint32_t num_limbs = std::min((int)limbsize - i, cc.batch);
        multnomoddownend_<<<dim3{(uint32_t)cc.N / 128, num_limbs}, 128, 0, STREAM(limb[i]).ptr()>>>(
            PARTITION(id, i), limbptr.data + i, c0.limbptr.data + i, bc0.limbptr.data + i, bc1.limbptr.data + i,
            in.limbptr.data + i, aux.limbptr.data + i);
    }
    this->copySpecialLimb(in);
    c0.copySpecialLimb(aux);
    for (int i = 0; i < limbsize; i += cc.batch) {
        s.wait(STREAM(limb[i]));
    }

    aux.getS().wait(c0.getS());
    aux.getS().wait(s);
    in.getS().wait(s);
    bc1.getS().wait(s);
    bc0.getS().wait(s);
    c0.getS().wait(s);
}

void LimbPartition::mult1Add2(const LimbPartition& partition1, const LimbPartition& partition2) {
    const int limbsize = getLimbSize(*level);
    cudaSetDevice(device);
    assert(limbsize <= partition1.limb.size());
    assert(limbsize <= partition2.limb.size());

    s.wait(partition1.getS());
    s.wait(partition2.getS());

    for (int i = 0; i < limbsize; i += cc.batch) {
        STREAM(limb[i]).wait(s);
        uint32_t num_limbs = std::min((int)limbsize - i, cc.batch);
        mult1Add2_<<<dim3{(uint32_t)cc.N / 128, num_limbs}, 128, 0, STREAM(limb[i]).ptr()>>>(
            PARTITION(id, i), limbptr.data + i, partition1.limbptr.data + i, partition2.limbptr.data + i);
    }
    for (int i = 0; i < limbsize; i += cc.batch) {
        s.wait(STREAM(limb[i]));
    }

    partition1.getS().wait(s);
    partition2.getS().wait(s);
}

void LimbPartition::generateLimbSingleMalloc() {
    cudaSetDevice(device);
    const int limbsize = meta.size();

    assert(limbsize <= meta.size());
    if (bufferLIMB == nullptr) {
        assert(limb.size() == 0);

        bufferLIMBbytes = cc.N * limbsize * 2 * sizeof(uint64_t);
        bufferLIMB = (uint64_t*)GPUmalloc(device, (int)bufferLIMBbytes, s.ptr());
        //cudaMallocAsync(&bufferLIMB, std::max(1ul, cc.N * limbsize * 2 * sizeof(uint64_t)), s.ptr());
    }

    limb.clear();

    //generate(meta, limb, limbptr, (int)limbsize - 1, &auxptr, bufferLIMB, 0, bufferLIMB, cc.N * (limbsize));
    generate(meta, limb, limbptr, (int)limbsize - 1, &auxptr, nullptr, 0, nullptr, cc.N * (limbsize));
}

void LimbPartition::generateLimbConstant() {
    cudaSetDevice(device);

    const int limbsize = getLimbSize(*level);
    assert(limb.size() == 0);
    assert(limbsize <= meta.size());

    if (bufferLIMB != nullptr) {
        GPUfree(bufferLIMB, id, (int)bufferLIMBbytes, s.ptr());
        bufferLIMB = nullptr;
        bufferLIMBbytes = 0;
    }

    //limb.clear();
    generate(meta, limb, limbptr, (int)limbsize - 1, nullptr /*&auxptr*/, bufferLIMB, 0, nullptr, 0);
}

void LimbPartition::loadDecompDigit(const std::vector<std::vector<std::vector<uint64_t>>>& data,
                                    const std::vector<std::vector<uint64_t>>& moduli) {
    cudaSetDevice(device);
    int limb_size = getLimbSize(*level);

    if (cc.GPUid.size() == 1) {

        for (size_t i = 0; i < DECOMPmeta.size(); ++i) {
            for (auto& j : DECOMPlimb.at(i)) {
                for (size_t k = 0; k < data.at(i).size(); ++k) {
                    if (cc.precom.constants[id].primes[PRIMEID(j)] == moduli.at(i).at(k)) {
                        STREAM(j).wait(s);
                        SWITCH(j, load(data.at(i).at(k)));
                        k = data.at(i).size();
                    }
                }
            }
        }
        std::vector<void*> cpu_ptr(MAXP, nullptr);
        for (size_t i = 0; i < DECOMPmeta.size(); ++i) {
            for (auto& j : DECOMPlimb.at(i)) {
                for (size_t k = 0; k < meta.size(); ++k) {
                    if (PRIMEID(j) == meta.at(k).id) {
                        if (j.index() == U64) {
                            cpu_ptr[k] = std::get<U64>(j).v.data;
                        } else {
                            cpu_ptr[k] = std::get<U32>(j).v.data;
                        }
                    }
                }
            }
        }
        cudaMemcpyAsync(limbptr.data, cpu_ptr.data(), cpu_ptr.size() * sizeof(void*), cudaMemcpyHostToDevice, s.ptr());
    } else {
        for (size_t i = 0; i < DECOMPmeta.size(); ++i) {
            for (size_t j = 0; j < limb_size; ++j) {
                if (meta[j].digit == i) {
                    for (size_t k = 0; k < data.at(i).size(); ++k) {
                        if (cc.precom.constants[id].primes[PRIMEID(limb[j])] == moduli.at(i).at(k)) {
                            STREAM(limb[j]).wait(s);
                            SWITCH(limb[j], load(data.at(i).at(k)));
                            k = data.at(i).size();
                        }
                    }
                }
            }
        }
    }

    for (size_t i = 0; i < DECOMPmeta.size(); ++i) {
        for (auto& j : DIGITlimb.at(i)) {
            for (size_t k = 0; k < data.at(i).size(); ++k) {
                if (cc.precom.constants[id].primes[PRIMEID(j)] == moduli.at(i).at(k)) {
                    STREAM(j).wait(s);
                    SWITCH(j, load(data.at(i).at(k)));
                    k = data.at(i).size();
                }
            }
        }
    }
}

/** TODO: deprecate towards fused version */

void LimbPartition::dotKSK(const LimbPartition& src, const LimbPartition& ksk, const bool inplace,
                           const LimbPartition* limbsrc) {
    cudaSetDevice(device);
    constexpr bool PRINT = false;
    s.wait(src.getS());
    s.wait(ksk.getS());
    const int limbsize = *level + 1;
    assert(limbsize <= limb.size());
    assert(limbsize <= src.limb.size());
    // banded keys (rotation-key limb pruning) must not be used above their band
    if (ksk.key_q_band >= 0 && limbsize > ksk.key_q_band + 1)
        throw std::runtime_error("dotKSK: banded key (band " + std::to_string(ksk.key_q_band) +
                                 ") used at limbsize " + std::to_string(limbsize));
    if (ksk.key_pack_bits)
        throw std::runtime_error("dotKSK: per-limb path is not packed-key aware (FIDESLIB_KSK_PACK=0 to disable)");

    if constexpr (0) {
        std::map<int, int> used;
        for (size_t i = 0; i < src.DIGITlimb.size(); ++i) {

            {
                int start = 0;
                for (int j = 0; j < i; ++j)
                    start += src.DECOMPlimb[j].size();
                int size = std::min((int)src.DECOMPlimb[i].size(), (int)limbsize - start);
                if (size <= 0)
                    break;
            }

            for (size_t j = 0; j < ksk.DECOMPlimb.at(i).size(); ++j) {

                int primeid = PRIMEID(ksk.DECOMPlimb.at(i).at(j));

                for (size_t k = 0; k < limbsize; ++k) {
                    auto& l = src.limb.at(k);
                    if (PRIMEID(l) == primeid) {
                        //STREAM(limb.at(k)).wait(s);
                        //STREAM(limb.at(k)).wait(STREAM(ksk.DECOMPlimb.at(i).at(j)));
                        //STREAM(limb.at(k)).wait(STREAM(l));

                        if (!used[primeid]) {
                            SWITCH(limb.at(k), mult(l, ksk.DECOMPlimb.at(i).at(j), inplace));
                            used[primeid]++;
                            if constexpr (PRINT)
                                std::cout << "Init " << primeid;  //<< std::endl;
                        } else {
                            SWITCH(limb.at(k), addMult(l, ksk.DECOMPlimb.at(i).at(j), inplace));
                            if constexpr (PRINT)
                                std::cout << "Acc " << primeid;  // << std::endl;
                        }
                        if constexpr (PRINT)
                            SWITCH(limb.at(k), printThisLimb(1));
                        if constexpr (PRINT)
                            SWITCH(l, printThisLimb(1));
                    }
                }

                if constexpr (PRINT)
                    SWITCH(ksk.DECOMPlimb.at(i).at(j), printThisLimb(1));
            }

            CudaCheckErrorModNoSync;
            for (size_t j = 0; j < src.DIGITlimb.at(i).size(); ++j) {
                int primeid = PRIMEID(src.DIGITlimb.at(i).at(j));

                if (primeid < cc.precom.constants[id].L) {
                    for (auto& l : limb) {
                        if (PRIMEID(l) == primeid) {
                            //STREAM(l).wait(s);
                            //STREAM(l).wait(STREAM(src.DIGITlimb.at(i).at(j)));
                            //STREAM(l).wait(STREAM(ksk.DIGITlimb.at(i).at(j)));

                            if (!used[primeid]) {
                                SWITCH(l, mult(src.DIGITlimb.at(i).at(j), ksk.DIGITlimb.at(i).at(j), inplace));
                                used[primeid]++;
                                if constexpr (PRINT)
                                    std::cout << "Init2 " << primeid;  // << std::endl;
                            } else

                            {
                                SWITCH(l, addMult(src.DIGITlimb.at(i).at(j), ksk.DIGITlimb.at(i).at(j), inplace));
                                if constexpr (PRINT)
                                    std::cout << "Acc2 " << primeid;  // << std::endl;
                            }

                            if constexpr (PRINT)
                                SWITCH(l, printThisLimb(1));
                        }
                    }
                } else {

                    for (auto& l : SPECIALlimb) {
                        if (PRIMEID(l) == primeid) {
                            //STREAM(l).wait(s);
                            //STREAM(l).wait(STREAM(src.DIGITlimb.at(i).at(j)));
                            //STREAM(l).wait(STREAM(ksk.DIGITlimb.at(i).at(j)));
                            if (!used[primeid]) {
                                SWITCH(l, mult(src.DIGITlimb.at(i).at(j), ksk.DIGITlimb.at(i).at(j), inplace));
                                used[primeid]++;
                                if constexpr (PRINT)
                                    std::cout << "Init3 " << primeid;  // << std::endl;
                            } else {
                                SWITCH(l, addMult(src.DIGITlimb.at(i).at(j), ksk.DIGITlimb.at(i).at(j), inplace));
                                if constexpr (PRINT)
                                    std::cout << "Acc3 " << primeid;  // << std::endl;
                            }
                            if constexpr (PRINT)
                                SWITCH(l, printThisLimb(1));
                        }
                    }
                }
                if constexpr (PRINT)
                    SWITCH(src.DIGITlimb.at(i).at(j), printThisLimb(1));
                if constexpr (PRINT)
                    SWITCH(ksk.DIGITlimb.at(i).at(j), printThisLimb(1));
            }

            if constexpr (PRINT) {
                for (auto& i : limb) {
                    if constexpr (PRINT)
                        SWITCH(i, printThisLimb(1));
                }
                for (auto& i : SPECIALlimb) {
                    if constexpr (PRINT)
                        SWITCH(i, printThisLimb(1));
                }
            }
        }

        for (auto& l : limb)
            s.wait(STREAM(l));
        for (auto& l : SPECIALlimb)
            s.wait(STREAM(l));
    } else {

        int start = 0;
        int special = SPECIALmeta.size();

        for (int i = 0; i < DECOMPmeta.size(); ++i) {
            int size = std::min((int)src.DECOMPlimb[i].size(), (int)limbsize - start);
            if (size <= 0) {
                //  std::cout << "Out on " << i << std::endl;
                break;
            }
            Mult_<<<{(uint32_t)cc.N / 128, (uint32_t)size}, 128, 0, s.ptr()>>>(
                inplace ? auxptr.data + start : limbptr.data + start, ksk.DECOMPlimbptr[i].data,
                limbsrc ? limbsrc->limbptr.data + start : src.limbptr.data + start, start);
            start += DECOMPmeta[i].size();
        }

        start = 0;
        for (int i = 0; i < DIGITmeta.size(); ++i) {
            if (start >= limbsize) {
                // std::cout << "Out on " << i << std::endl;
                break;
            }
            if (start > 0) {
                int size = start;
                addMult_<<<{(uint32_t)cc.N / 128, (uint32_t)size}, 128, 0, s.ptr()>>>(
                    inplace ? auxptr.data : limbptr.data, ksk.DIGITlimbptr[i].data + special,
                    src.DIGITlimbptr[i].data + special, 0);
            }
            start += DECOMPmeta[i].size();
            if (start < limbsize) {
                int size = limbsize - start;
                addMult_<<<{(uint32_t)cc.N / 128, (uint32_t)size}, 128, 0, s.ptr()>>>(
                    inplace ? auxptr.data + start : limbptr.data + start,
                    ksk.DIGITlimbptr[i].data + special + start - DECOMPmeta[i].size(),
                    src.DIGITlimbptr[i].data + special + start - DECOMPmeta[i].size(), start);
            }
        }

        start = 0;
        for (int i = 0; i < DIGITmeta.size(); ++i) {
            if (start >= limbsize)
                break;
            start += DECOMPmeta.at(i).size();
            if (i == 0) {
                Mult_<<<{(uint32_t)cc.N / 128, (uint32_t)special}, 128, 0, s.ptr()>>>(
                    inplace ? SPECIALauxptr.data : SPECIALlimbptr.data, ksk.DIGITlimbptr[i].data,
                    src.DIGITlimbptr[i].data, SPECIAL(id, 0));
            } else {
                addMult_<<<{(uint32_t)cc.N / 128, (uint32_t)special}, 128, 0, s.ptr()>>>(
                    inplace ? SPECIALauxptr.data : SPECIALlimbptr.data, ksk.DIGITlimbptr[i].data,
                    src.DIGITlimbptr[i].data, SPECIAL(id, 0));
            }
        }
    }

    src.getS().wait(s);
    ksk.getS().wait(s);

    if (inplace) {
        for (auto& l : limb) {
            if (l.index() == U32) {
                std::swap(std::get<U32>(l).v.data, std::get<U32>(l).aux.data);
            } else {
                std::swap(std::get<U64>(l).v.data, std::get<U64>(l).aux.data);
            }
        }
        std::swap(limbptr.data, auxptr.data);

        for (auto& l : SPECIALlimb) {
            if (l.index() == U32) {
                std::swap(std::get<U32>(l).v.data, std::get<U32>(l).aux.data);
            } else {
                std::swap(std::get<U64>(l).v.data, std::get<U64>(l).aux.data);
            }
        }
        std::swap(SPECIALlimbptr.data, SPECIALauxptr.data);
    }

    if constexpr (PRINT) {
        for (auto& i : limb) {
            SWITCH(i, printThisLimb(1));
        }
        std::cout << std::endl;
        for (auto& i : SPECIALlimb) {
            SWITCH(i, printThisLimb(1));
        }
        std::cout << std::endl;
    }
}

void LimbPartition::multModupDotKSK(LimbPartition& c1, const LimbPartition& c1tilde, LimbPartition& c0,
                                    const LimbPartition& c0tilde, const LimbPartition& ksk_a,
                                    const LimbPartition& ksk_b) {

    const int level_plus_1 = *level + 1;
    constexpr ALGO algo = ALGO_SHOUP;
    constexpr bool PRINT = false;
    assert(c0.SPECIALlimb.size() == SPECIALmeta.size());
    assert(c1.SPECIALlimb.size() == SPECIALmeta.size());
    if (ksk_a.key_pack_bits)
        throw std::runtime_error(
            "*ModupDotKSK: NTT_KSK_DOT paths are not packed-key aware (FIDESLIB_KSK_PACK=0 to disable)");
    cudaSetDevice(device);

    //std::map<int, int> used;
    s.wait(c0.getS());
    s.wait(c1.getS());
    s.wait(c0tilde.getS());
    s.wait(c1tilde.getS());
    s.wait(ksk_a.getS());
    s.wait(ksk_b.getS());

    for (size_t d = 0; d < DECOMPlimb.size(); ++d) {

        int start = 0;
        for (int j = 0; j < d; ++j)
            start += DECOMPlimb[j].size();
        int size = std::min((int)DECOMPlimb[d].size(), level_plus_1 - start);
        if (size <= 0)
            break;

        if constexpr (PRINT)
            if (d == 0) {
                std::cout << cc.precom.constants[id].primes[PRIMEID(limb[0])] << ": ";
                SWITCH(limb[0], printThisLimb());
            }

        if constexpr (1) {  // Batched
            const int M = (cc.precom.constants[0].type == 0) ? 8 : 4;  // u32 tiles are byte-parity with u64 (kernel M=8): grid must be N/(bd*M*2)

            dim3 blockDimFirst{(uint32_t)(1 << ((cc.logN) / 2 - 1))};
            dim3 blockDimSecond = dim3{(uint32_t)(1 << ((cc.logN + 1) / 2 - 1))};
            int bytesFirst = (32 / M) * blockDimFirst.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));
            int bytesSecond = (32 / M) * blockDimSecond.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));

            for (int i = 0; i < size; i += cc.batch) {
                STREAM(limb.at(start + i)).wait(s);
                uint32_t num_limbs = std::min((uint32_t)cc.batch, (uint32_t)(size - i));

                INTT_<false, algo, INTT_MULT_AND_SAVE><<<dim3{cc.N / (blockDimFirst.x * M * 2), num_limbs},
                                                         blockDimFirst, bytesFirst, STREAM(limb.at(start + i)).ptr()>>>(
                    getGlobals(), c1.limbptr.data + start + i, start + i, c1.auxptr.data + start + i,
                    c1tilde.limbptr.data + start + i, c0.limbptr.data + start + i, c1.limbptr.data + start + i,
                    ksk_a.DECOMPlimbptr[d].data + i, ksk_b.DECOMPlimbptr[d].data + i, c0.limbptr.data + start + i,
                    c0tilde.limbptr.data + start + i);

                INTT_<true, algo, INTT_NONE><<<dim3{cc.N / (blockDimSecond.x * M * 2), num_limbs}, blockDimSecond,
                                               bytesSecond, STREAM(limb.at(start + i)).ptr()>>>(
                    getGlobals(), c1.auxptr.data + start + i, start + i, DECOMPlimbptr[d].data + i);
            }
            for (size_t i = 0; i < size; i += cc.batch) {
                s.wait(STREAM(limb.at(start + i)));
            }
        }

        if constexpr (PRINT) {
            std::cout << cc.precom.constants[id].primes[PRIMEID(DECOMPlimb[d][0])] << ": ";
            SWITCH(DECOMPlimb[d][0], printThisLimb());
        }

        {
            dim3 blockSize{64, 2};
            dim3 gridSize{(uint32_t)cc.N / blockSize.x};
            int shared_bytes = sizeof(uint64_t) * (DECOMPlimb[d].size()) * blockSize.x;
            DecompAndModUpConv<algo><<<gridSize, blockSize, shared_bytes, s.ptr()>>>(
                DECOMPlimbptr[d].data, level_plus_1, DIGITlimbptr[d].data, digitid[d], getGlobals());
        }

        if constexpr (1) {  // Batched
            const int M = (cc.precom.constants[0].type == 0) ? 8 : 4;  // u32 tiles are byte-parity with u64 (kernel M=8): grid must be N/(bd*M*2)

            dim3 blockDimFirst{(uint32_t)(1 << ((cc.logN) / 2 - 1))};
            dim3 blockDimSecond = dim3{(uint32_t)(1 << ((cc.logN + 1) / 2 - 1))};
            int bytesFirst = (32 / M) * blockDimFirst.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));
            int bytesSecond = (32 / M) * blockDimSecond.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));

            int size = c0.SPECIALlimb.size();
            for (int i = 0; i < size; i += cc.batch) {
                STREAM(c0.SPECIALlimb.at(i)).wait(s);
                uint32_t num_limbs = std::min((uint32_t)cc.batch, (uint32_t)(size - i));

                NTT_<false, algo, NTT_NONE>
                    <<<dim3{cc.N / (blockDimFirst.x * M * 2), num_limbs}, blockDimFirst, bytesFirst,
                       STREAM(c0.SPECIALlimb.at(i)).ptr()>>>(getGlobals(), DIGITlimbptr[d].data + i, SPECIAL(id, i),
                                                             c1.SPECIALauxptr.data + i, nullptr, 0, nullptr, nullptr);

                if (d == 0) {
                    NTT_<true, algo, NTT_KSK_DOT><<<dim3{cc.N / (blockDimSecond.x * M * 2), num_limbs}, blockDimSecond,
                                                    bytesSecond, STREAM(c0.SPECIALlimb.at(i)).ptr()>>>(
                        getGlobals(), c1.SPECIALauxptr.data + i, SPECIAL(id, i), c0.SPECIALlimbptr.data + i,
                        ksk_a.DIGITlimbptr[d].data + i, 0, c1.SPECIALlimbptr.data + i, ksk_b.DIGITlimbptr[d].data + i);
                } else {
                    NTT_<true, algo, NTT_KSK_DOT_ACC>
                        <<<dim3{cc.N / (blockDimSecond.x * M * 2), num_limbs}, blockDimSecond, bytesSecond,
                           STREAM(c0.SPECIALlimb.at(i)).ptr()>>>(
                            getGlobals(), c1.SPECIALauxptr.data + i, SPECIAL(id, i), c0.SPECIALlimbptr.data + i,
                            ksk_a.DIGITlimbptr[d].data + i, 0, c1.SPECIALlimbptr.data + i,
                            ksk_b.DIGITlimbptr[d].data + i);
                }
            }
        }
    }

    //cudaDeviceSynchronize();

    for (size_t d = 0; d < DECOMPlimb.size(); ++d) {
        int start = 0;
        for (int j = 0; j < d; ++j)
            start += DECOMPlimb[j].size();
        int size = std::min((int)DECOMPlimb[d].size(), level_plus_1 - start);
        if (size <= 0)
            break;

        if constexpr (PRINT)
            for (auto& i : DIGITlimb[d]) {
                SWITCH(i, printThisLimb(2));
            }

        if constexpr (1)  // batched
        {
            int start = 0;
            for (size_t j = 0; j < DECOMPlimb.size(); ++j) {
                if (j == d)
                    continue;

                int Dstart = start + c0.SPECIALlimb.size();
                int Lstart = start + (j > d ? DECOMPlimb[d].size() : 0);

                int size = std::min((int)DECOMPlimb[j].size(), (int)level_plus_1 - Lstart);
                if (size <= 0)
                    break;

                const int M = (cc.precom.constants[0].type == 0) ? 8 : 4;  // u32 tiles are byte-parity with u64 (kernel M=8): grid must be N/(bd*M*2)

                dim3 blockDimFirst{(uint32_t)(1 << ((cc.logN) / 2 - 1))};
                dim3 blockDimSecond = dim3{(uint32_t)(1 << ((cc.logN + 1) / 2 - 1))};
                int bytesFirst = (32 / M) * blockDimFirst.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));
                int bytesSecond = (32 / M) * blockDimSecond.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));

                for (int i = 0; i < size; i += cc.batch) {
                    STREAM(c0.limb.at(Lstart + i)).wait(s);
                    uint32_t num_limbs = std::min((uint32_t)cc.batch, (uint32_t)(size - i));

                    NTT_<false, algo, NTT_NONE><<<dim3{cc.N / (blockDimFirst.x * M * 2), num_limbs}, blockDimFirst,
                                                  bytesFirst, STREAM(c0.limb.at(Lstart + i)).ptr()>>>(
                        getGlobals(), DIGITlimbptr[d].data + Dstart + i, Lstart + i, c1.auxptr.data + Lstart + i,
                        nullptr, 0, nullptr, nullptr);

                    NTT_<true, algo, NTT_KSK_DOT_ACC>
                        <<<dim3{cc.N / (blockDimSecond.x * M * 2), num_limbs}, blockDimSecond, bytesSecond,
                           STREAM(c0.limb.at(Lstart + i)).ptr()>>>(
                            getGlobals(), c1.auxptr.data + Lstart + i, Lstart + i, c0.limbptr.data + Lstart + i,
                            ksk_a.DIGITlimbptr[d].data + Dstart + i, 0, c1.limbptr.data + Lstart + i,
                            ksk_b.DIGITlimbptr[d].data + Dstart + i);
                }

                start += DECOMPlimb[j].size();
            }
        }

        if constexpr (PRINT)
            for (auto& i : DIGITlimb[d]) {
                SWITCH(i, printThisLimb(2));
            }
    }
    for (auto& l : c0.limb)
        s.wait(STREAM(l));
    for (auto& l : c0.SPECIALlimb)
        s.wait(STREAM(l));

    c0.getS().wait(s);
    c1.getS().wait(s);
    c0tilde.getS().wait(s);
    c1tilde.getS().wait(s);
    ksk_a.getS().wait(s);
    ksk_b.getS().wait(s);
}

int LimbPartition::getLimbSize(int level) const {
    int size = 0;
    while (size < meta.size() && meta[size].id <= level) {
        //assert(limb.size() > size);
        ++size;
    }
    return size;
}

void LimbPartition::rotateModupDotKSK(LimbPartition& c1, LimbPartition& c0, const LimbPartition& ksk_a,
                                      const LimbPartition& ksk_b) {

    const int level_plus_1 = *level + 1;
    constexpr ALGO algo = ALGO_SHOUP;
    constexpr bool PRINT = false;
    assert(c0.SPECIALlimb.size() == SPECIALmeta.size());
    assert(c1.SPECIALlimb.size() == SPECIALmeta.size());
    if (ksk_a.key_pack_bits)
        throw std::runtime_error(
            "*ModupDotKSK: NTT_KSK_DOT paths are not packed-key aware (FIDESLIB_KSK_PACK=0 to disable)");
    cudaSetDevice(device);

    //std::map<int, int> used;
    s.wait(c0.getS());
    s.wait(c1.getS());
    s.wait(ksk_a.getS());
    s.wait(ksk_b.getS());

    for (size_t d = 0; d < DECOMPlimb.size(); ++d) {

        int start = 0;
        for (int j = 0; j < d; ++j)
            start += DECOMPlimb[j].size();
        int size = std::min((int)DECOMPlimb[d].size(), level_plus_1 - start);
        if (size <= 0)
            break;

        if constexpr (PRINT)
            if (d == 0) {
                std::cout << cc.precom.constants[id].primes[PRIMEID(limb[0])] << ": ";
                SWITCH(limb[0], printThisLimb());
            }

        if constexpr (1) {  // Batched
            const int M = (cc.precom.constants[0].type == 0) ? 8 : 4;  // u32 tiles are byte-parity with u64 (kernel M=8): grid must be N/(bd*M*2)

            dim3 blockDimFirst{(uint32_t)(1 << ((cc.logN) / 2 - 1))};
            dim3 blockDimSecond = dim3{(uint32_t)(1 << ((cc.logN + 1) / 2 - 1))};
            int bytesFirst = (32 / M) * blockDimFirst.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));
            int bytesSecond = (32 / M) * blockDimSecond.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));

            for (int i = 0; i < size; i += cc.batch) {
                STREAM(limb.at(start + i)).wait(s);
                uint32_t num_limbs = std::min((uint32_t)cc.batch, (uint32_t)(size - i));

                INTT_<false, algo, INTT_ROTATE_AND_SAVE>
                    <<<dim3{cc.N / (blockDimFirst.x * M * 2), num_limbs}, blockDimFirst, bytesFirst,
                       STREAM(limb.at(start + i)).ptr()>>>(
                        getGlobals(), c1.limbptr.data + start + i, start + i, c1.auxptr.data + start + i, nullptr,
                        c0.limbptr.data + start + i, c1.limbptr.data + start + i, ksk_a.DECOMPlimbptr[d].data + i,
                        ksk_b.DECOMPlimbptr[d].data + i, c0.limbptr.data + start + i, nullptr);

                INTT_<true, algo, INTT_NONE><<<dim3{cc.N / (blockDimSecond.x * M * 2), num_limbs}, blockDimSecond,
                                               bytesSecond, STREAM(limb.at(start + i)).ptr()>>>(
                    getGlobals(), c1.auxptr.data + start + i, start + i, DECOMPlimbptr[d].data + i);
            }
            for (size_t i = 0; i < size; i += cc.batch) {
                s.wait(STREAM(limb.at(start + i)));
            }
        }

        if constexpr (PRINT) {
            std::cout << cc.precom.constants[id].primes[PRIMEID(DECOMPlimb[d][0])] << ": ";
            SWITCH(DECOMPlimb[d][0], printThisLimb());
        }

        {
            dim3 blockSize{64, 2};
            dim3 gridSize{(uint32_t)cc.N / blockSize.x};
            int shared_bytes = sizeof(uint64_t) * (DECOMPlimb[d].size()) * blockSize.x;
            DecompAndModUpConv<algo><<<gridSize, blockSize, shared_bytes, s.ptr()>>>(
                DECOMPlimbptr[d].data, level_plus_1, DIGITlimbptr[d].data, digitid[d], getGlobals());
        }

        if constexpr (1) {  // Batched
            const int M = (cc.precom.constants[0].type == 0) ? 8 : 4;  // u32 tiles are byte-parity with u64 (kernel M=8): grid must be N/(bd*M*2)

            dim3 blockDimFirst{(uint32_t)(1 << ((cc.logN) / 2 - 1))};
            dim3 blockDimSecond = dim3{(uint32_t)(1 << ((cc.logN + 1) / 2 - 1))};
            int bytesFirst = (32 / M) * blockDimFirst.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));
            int bytesSecond = (32 / M) * blockDimSecond.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));

            int size = c0.SPECIALlimb.size();
            for (int i = 0; i < size; i += cc.batch) {
                STREAM(c0.SPECIALlimb.at(i)).wait(s);
                uint32_t num_limbs = std::min((uint32_t)cc.batch, (uint32_t)(size - i));

                NTT_<false, algo, NTT_NONE>
                    <<<dim3{cc.N / (blockDimFirst.x * M * 2), num_limbs}, blockDimFirst, bytesFirst,
                       STREAM(c0.SPECIALlimb.at(i)).ptr()>>>(getGlobals(), DIGITlimbptr[d].data + i, SPECIAL(id, i),
                                                             c1.SPECIALauxptr.data + i, nullptr, 0, nullptr, nullptr);

                if (d == 0) {
                    NTT_<true, algo, NTT_KSK_DOT><<<dim3{cc.N / (blockDimSecond.x * M * 2), num_limbs}, blockDimSecond,
                                                    bytesSecond, STREAM(c0.SPECIALlimb.at(i)).ptr()>>>(
                        getGlobals(), c1.SPECIALauxptr.data + i, SPECIAL(id, i), c0.SPECIALlimbptr.data + i,
                        ksk_a.DIGITlimbptr[d].data + i, 0, c1.SPECIALlimbptr.data + i, ksk_b.DIGITlimbptr[d].data + i);
                } else {
                    NTT_<true, algo, NTT_KSK_DOT_ACC>
                        <<<dim3{cc.N / (blockDimSecond.x * M * 2), num_limbs}, blockDimSecond, bytesSecond,
                           STREAM(c0.SPECIALlimb.at(i)).ptr()>>>(
                            getGlobals(), c1.SPECIALauxptr.data + i, SPECIAL(id, i), c0.SPECIALlimbptr.data + i,
                            ksk_a.DIGITlimbptr[d].data + i, 0, c1.SPECIALlimbptr.data + i,
                            ksk_b.DIGITlimbptr[d].data + i);
                }
            }
        }
    }

    for (size_t d = 0; d < DECOMPlimb.size(); ++d) {
        int start = 0;
        for (int j = 0; j < d; ++j)
            start += DECOMPlimb[j].size();
        int size = std::min((int)DECOMPlimb[d].size(), level_plus_1 - start);
        if (size <= 0)
            break;

        if constexpr (PRINT)
            for (auto& i : DIGITlimb[d]) {
                SWITCH(i, printThisLimb(2));
            }

        if constexpr (1)  // batched
        {
            int start = 0;
            for (size_t j = 0; j < DECOMPlimb.size(); ++j) {
                if (j == d)
                    continue;

                int Dstart = start + c0.SPECIALlimb.size();
                int Lstart = start + (j > d ? DECOMPlimb[d].size() : 0);

                int size = std::min((int)DECOMPlimb[j].size(), (int)level_plus_1 - Lstart);
                if (size <= 0)
                    break;

                const int M = (cc.precom.constants[0].type == 0) ? 8 : 4;  // u32 tiles are byte-parity with u64 (kernel M=8): grid must be N/(bd*M*2)

                dim3 blockDimFirst{(uint32_t)(1 << ((cc.logN) / 2 - 1))};
                dim3 blockDimSecond = dim3{(uint32_t)(1 << ((cc.logN + 1) / 2 - 1))};
                int bytesFirst = (32 / M) * blockDimFirst.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));
                int bytesSecond = (32 / M) * blockDimSecond.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));

                for (int i = 0; i < size; i += cc.batch) {
                    STREAM(c0.limb.at(Lstart + i)).wait(s);
                    uint32_t num_limbs = std::min((uint32_t)cc.batch, (uint32_t)(size - i));

                    NTT_<false, algo, NTT_NONE><<<dim3{cc.N / (blockDimFirst.x * M * 2), num_limbs}, blockDimFirst,
                                                  bytesFirst, STREAM(c0.limb.at(Lstart + i)).ptr()>>>(
                        getGlobals(), DIGITlimbptr[d].data + Dstart + i, Lstart + i, c1.auxptr.data + Lstart + i,
                        nullptr, 0, nullptr, nullptr);

                    NTT_<true, algo, NTT_KSK_DOT_ACC>
                        <<<dim3{cc.N / (blockDimSecond.x * M * 2), num_limbs}, blockDimSecond, bytesSecond,
                           STREAM(c0.limb.at(Lstart + i)).ptr()>>>(
                            getGlobals(), c1.auxptr.data + Lstart + i, Lstart + i, c0.limbptr.data + Lstart + i,
                            ksk_a.DIGITlimbptr[d].data + Dstart + i, 0, c1.limbptr.data + Lstart + i,
                            ksk_b.DIGITlimbptr[d].data + Dstart + i);
                }

                start += DECOMPlimb[j].size();
            }
        }

        if constexpr (PRINT)
            for (auto& i : DIGITlimb[d]) {
                SWITCH(i, printThisLimb(2));
            }
    }

    for (auto& l : c0.limb)
        s.wait(STREAM(l));
    for (auto& l : c0.SPECIALlimb)
        s.wait(STREAM(l));

    c0.getS().wait(s);
    c1.getS().wait(s);
    ksk_a.getS().wait(s);
    ksk_b.getS().wait(s);
}

void LimbPartition::squareModupDotKSK(LimbPartition& c1, LimbPartition& c0, const LimbPartition& ksk_a,
                                      const LimbPartition& ksk_b) {

    const int level_plus_1 = *level + 1;
    constexpr ALGO algo = ALGO_SHOUP;
    constexpr bool PRINT = false;
    assert(c0.SPECIALlimb.size() == SPECIALmeta.size());
    assert(c1.SPECIALlimb.size() == SPECIALmeta.size());
    if (ksk_a.key_pack_bits)
        throw std::runtime_error(
            "*ModupDotKSK: NTT_KSK_DOT paths are not packed-key aware (FIDESLIB_KSK_PACK=0 to disable)");
    cudaSetDevice(device);

    //std::map<int, int> used;
    s.wait(c0.getS());
    s.wait(c1.getS());
    s.wait(ksk_a.getS());
    s.wait(ksk_b.getS());

    for (size_t d = 0; d < DECOMPlimb.size(); ++d) {

        int start = 0;
        for (int j = 0; j < d; ++j)
            start += DECOMPlimb[j].size();
        int size = std::min((int)DECOMPlimb[d].size(), level_plus_1 - start);
        if (size <= 0)
            break;

        if constexpr (PRINT)
            if (d == 0) {
                std::cout << cc.precom.constants[id].primes[PRIMEID(limb[0])] << ": ";
                SWITCH(limb[0], printThisLimb());
            }

        if constexpr (1) {  // Batched
            const int M = (cc.precom.constants[0].type == 0) ? 8 : 4;  // u32 tiles are byte-parity with u64 (kernel M=8): grid must be N/(bd*M*2)

            dim3 blockDimFirst{(uint32_t)(1 << ((cc.logN) / 2 - 1))};
            dim3 blockDimSecond = dim3{(uint32_t)(1 << ((cc.logN + 1) / 2 - 1))};
            int bytesFirst = (32 / M) * blockDimFirst.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));
            int bytesSecond = (32 / M) * blockDimSecond.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));

            for (int i = 0; i < size; i += cc.batch) {
                STREAM(limb.at(start + i)).wait(s);
                uint32_t num_limbs = std::min((uint32_t)cc.batch, (uint32_t)(size - i));

                INTT_<false, algo, INTT_SQUARE_AND_SAVE>
                    <<<dim3{cc.N / (blockDimFirst.x * M * 2), num_limbs}, blockDimFirst, bytesFirst,
                       STREAM(limb.at(start + i)).ptr()>>>(
                        getGlobals(), c1.limbptr.data + start + i, start + i, c1.auxptr.data + start + i, nullptr,
                        c0.limbptr.data + start + i, c1.limbptr.data + start + i, ksk_a.DECOMPlimbptr[d].data + i,
                        ksk_b.DECOMPlimbptr[d].data + i, c0.limbptr.data + start + i, nullptr);

                INTT_<true, algo, INTT_NONE><<<dim3{cc.N / (blockDimSecond.x * M * 2), num_limbs}, blockDimSecond,
                                               bytesSecond, STREAM(limb.at(start + i)).ptr()>>>(
                    getGlobals(), c1.auxptr.data + start + i, start + i, DECOMPlimbptr[d].data + i);
            }
            for (size_t i = 0; i < size; i += cc.batch) {
                s.wait(STREAM(limb.at(start + i)));
            }
        }

        if constexpr (PRINT) {
            std::cout << cc.precom.constants[id].primes[PRIMEID(DECOMPlimb[d][0])] << ": ";
            SWITCH(DECOMPlimb[d][0], printThisLimb());
        }

        {
            dim3 blockSize{64, 2};
            dim3 gridSize{(uint32_t)cc.N / blockSize.x};
            int shared_bytes = sizeof(uint64_t) * (DECOMPlimb[d].size()) * blockSize.x;
            DecompAndModUpConv<algo><<<gridSize, blockSize, shared_bytes, s.ptr()>>>(
                DECOMPlimbptr[d].data, level_plus_1, DIGITlimbptr[d].data, digitid[d], getGlobals());
        }

        if constexpr (1) {  // Batched
            const int M = (cc.precom.constants[0].type == 0) ? 8 : 4;  // u32 tiles are byte-parity with u64 (kernel M=8): grid must be N/(bd*M*2)

            dim3 blockDimFirst{(uint32_t)(1 << ((cc.logN) / 2 - 1))};
            dim3 blockDimSecond = dim3{(uint32_t)(1 << ((cc.logN + 1) / 2 - 1))};
            int bytesFirst = (32 / M) * blockDimFirst.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));
            int bytesSecond = (32 / M) * blockDimSecond.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));

            int size = c0.SPECIALlimb.size();
            for (int i = 0; i < size; i += cc.batch) {
                STREAM(c0.SPECIALlimb.at(i)).wait(s);
                uint32_t num_limbs = std::min((uint32_t)cc.batch, (uint32_t)(size - i));

                NTT_<false, algo, NTT_NONE>
                    <<<dim3{cc.N / (blockDimFirst.x * M * 2), num_limbs}, blockDimFirst, bytesFirst,
                       STREAM(c0.SPECIALlimb.at(i)).ptr()>>>(getGlobals(), DIGITlimbptr[d].data + i, SPECIAL(id, i),
                                                             c1.SPECIALauxptr.data + i, nullptr, 0, nullptr, nullptr);

                if (d == 0) {
                    NTT_<true, algo, NTT_KSK_DOT><<<dim3{cc.N / (blockDimSecond.x * M * 2), num_limbs}, blockDimSecond,
                                                    bytesSecond, STREAM(c0.SPECIALlimb.at(i)).ptr()>>>(
                        getGlobals(), c1.SPECIALauxptr.data + i, SPECIAL(id, i), c0.SPECIALlimbptr.data + i,
                        ksk_a.DIGITlimbptr[d].data + i, 0, c1.SPECIALlimbptr.data + i, ksk_b.DIGITlimbptr[d].data + i);
                } else {
                    NTT_<true, algo, NTT_KSK_DOT_ACC>
                        <<<dim3{cc.N / (blockDimSecond.x * M * 2), num_limbs}, blockDimSecond, bytesSecond,
                           STREAM(c0.SPECIALlimb.at(i)).ptr()>>>(
                            getGlobals(), c1.SPECIALauxptr.data + i, SPECIAL(id, i), c0.SPECIALlimbptr.data + i,
                            ksk_a.DIGITlimbptr[d].data + i, 0, c1.SPECIALlimbptr.data + i,
                            ksk_b.DIGITlimbptr[d].data + i);
                }
            }
        }
    }

    for (size_t d = 0; d < DECOMPlimb.size(); ++d) {
        int start = 0;
        for (int j = 0; j < d; ++j)
            start += DECOMPlimb[j].size();
        int size = std::min((int)DECOMPlimb[d].size(), level_plus_1 - start);
        if (size <= 0)
            break;

        if constexpr (PRINT)
            for (auto& i : DIGITlimb[d]) {
                SWITCH(i, printThisLimb(2));
            }

        if constexpr (1)  // batched
        {
            int start = 0;
            for (size_t j = 0; j < DECOMPlimb.size(); ++j) {
                if (j == d)
                    continue;

                int Dstart = start + c0.SPECIALlimb.size();
                int Lstart = start + (j > d ? DECOMPlimb[d].size() : 0);

                int size = std::min((int)DECOMPlimb[j].size(), (int)level_plus_1 - Lstart);
                if (size <= 0)
                    break;

                const int M = (cc.precom.constants[0].type == 0) ? 8 : 4;  // u32 tiles are byte-parity with u64 (kernel M=8): grid must be N/(bd*M*2)

                dim3 blockDimFirst{(uint32_t)(1 << ((cc.logN) / 2 - 1))};
                dim3 blockDimSecond = dim3{(uint32_t)(1 << ((cc.logN + 1) / 2 - 1))};
                int bytesFirst = (32 / M) * blockDimFirst.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));
                int bytesSecond = (32 / M) * blockDimSecond.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));

                for (int i = 0; i < size; i += cc.batch) {
                    STREAM(c0.limb.at(Lstart + i)).wait(s);
                    uint32_t num_limbs = std::min((uint32_t)cc.batch, (uint32_t)(size - i));

                    NTT_<false, algo, NTT_NONE><<<dim3{cc.N / (blockDimFirst.x * M * 2), num_limbs}, blockDimFirst,
                                                  bytesFirst, STREAM(c0.limb.at(Lstart + i)).ptr()>>>(
                        getGlobals(), DIGITlimbptr[d].data + Dstart + i, Lstart + i, c1.auxptr.data + Lstart + i,
                        nullptr, 0, nullptr, nullptr);

                    NTT_<true, algo, NTT_KSK_DOT_ACC>
                        <<<dim3{cc.N / (blockDimSecond.x * M * 2), num_limbs}, blockDimSecond, bytesSecond,
                           STREAM(c0.limb.at(Lstart + i)).ptr()>>>(
                            getGlobals(), c1.auxptr.data + Lstart + i, Lstart + i, c0.limbptr.data + Lstart + i,
                            ksk_a.DIGITlimbptr[d].data + Dstart + i, 0, c1.limbptr.data + Lstart + i,
                            ksk_b.DIGITlimbptr[d].data + Dstart + i);
                }

                start += DECOMPlimb[j].size();
            }
        }

        if constexpr (PRINT)
            for (auto& i : DIGITlimb[d]) {
                SWITCH(i, printThisLimb(2));
            }
    }

    for (auto& l : c0.limb)
        s.wait(STREAM(l));
    for (auto& l : c0.SPECIALlimb)
        s.wait(STREAM(l));

    c0.getS().wait(s);
    c1.getS().wait(s);
    ksk_a.getS().wait(s);
    ksk_b.getS().wait(s);
}

template <ALGO algo>
void LimbPartition::moddown(LimbPartition& auxLimbs, bool ntt, bool free_special_limbs) {
    assert(SPECIALlimb.size() == SPECIALmeta.size());
    const int limbsize = *level + 1;
    cudaSetDevice(device);
    constexpr bool PRINT = false;

    s.wait(auxLimbs.getS());
    {
        if constexpr (PRINT) {
            std::cout << "pre INTT Special GPU ";
            for (auto& i : SPECIALlimb) {
                SWITCH(i, printThisLimb(2));
            }
        }

        if (ntt) {
            for (size_t i = 0; i < SPECIALlimb.size(); i += cc.batch) {
                STREAM(SPECIALlimb[i]).wait(s);
            }
            ApplyINTT<algo, INTT_NONE>(cc.batch, INTT_fusion_fields{}, SPECIALlimb, SPECIALlimbptr, SPECIALauxptr, cc,
                                       SPECIAL(id, 0), SPECIALlimb.size());
            for (size_t i = 0; i < SPECIALlimb.size(); i += cc.batch) {
                s.wait(STREAM(SPECIALlimb[i]));
            }
        }

        if constexpr (PRINT) {
            std::cout << "post INTT Special GPU ";
            for (auto& i : SPECIALlimb) {
                SWITCH(i, printThisLimb(2));
            }
        }

        s.wait(auxLimbs.getS());

        {
            dim3 blockSize{64, 2};  // blockSize.x * blockSize.y * blockSize.z <= 1024, blockSize.x a multiple of 32

            dim3 gridSize{(uint32_t)cc.N / blockSize.x};
            int shared_bytes = sizeof(uint64_t) * (SPECIALlimb.size()) * blockSize.x;

            ModDown2<algo><<<gridSize, blockSize, shared_bytes, s.ptr()>>>(
                auxLimbs.limbptr.data, limbsize, SPECIALlimbptr.data, PARTITION(id, 0), getGlobals());
        }

        if constexpr (PRINT) {
            std::cout << "Output ModDown ";
            for (auto& i : auxLimbs.limb) {
                SWITCH(i, printThisLimb(2));
            }
        }

        for (int i = 0; i < limbsize; i += cc.batch) {
            STREAM(limb.at(i)).wait(s);
        }
        if (limbsize > 0)
            NTT<algo, NTT_MODDOWN>(cc.batch, false, NTT_fusion_fields{.op2 = &auxLimbs});


        if constexpr (PRINT) {
            std::cout << "Output ModDown after sub mult.";
            for (auto& i : limb) {
                SWITCH(i, printThisLimb(2));
            }
        }

        for (int i = 0; i < limbsize; i += cc.batch) {
            s.wait(STREAM(limb.at(i)));
        }
    }
    auxLimbs.getS().wait(s);

    if (free_special_limbs) {
        freeSpecialLimbs();
    }
}

#define YY(algo) \
    template void LimbPartition::moddown<algo>(LimbPartition & auxLimbs, bool ntt, bool free_special_limbs);
#include "ntt_types.inc"

#undef YY

//////// SEYDA /////////

void LimbPartition::automorph(const int index, const int br, LimbPartition* src, const bool ext) {
    cudaSetDevice(device);
    int limbsize = getLimbSize(*level);

    if (src)
        s.wait(src->getS());
    for (int i = 0; i < limbsize; i += cc.batch) {
        STREAM(limb[i]).wait(s);
        uint32_t num_limbs = std::min((int)limbsize - i, cc.batch);
        automorph_multi_<<<dim3{(uint32_t)cc.N / 128, num_limbs}, 128, 0, STREAM(limb[i]).ptr()>>>(
            src ? src->limbptr.data + i : limbptr.data + i, src ? limbptr.data + i : auxptr.data + i, index, br,
            PARTITION(id, i));
    }
    if (ext) {
        int start = cc.splitSpecialMeta.at(id).at(0).id - (cc.L + 1);
        int num_limbs = cc.splitSpecialMeta.at(id).size();

        for (size_t i = start; i < start + num_limbs; i += 1 /*cc.batch*/) {
            STREAM(SPECIALlimb[i - (SPECIALlimb.size() < cc.specialMeta.at(id).size()) * start]).wait(s);
            uint32_t size = std::min((int)start + num_limbs - (int)i, 1 /*cc.batch*/);
            automorph_multi_<<<dim3{(uint32_t)cc.N / 128, size}, 128, 0,
                               STREAM(SPECIALlimb[i - (SPECIALlimb.size() < SPECIALmeta.size()) * start]).ptr()>>>(
                src ? src->SPECIALlimbptr.data + i - (src->SPECIALlimb.size() < SPECIALmeta.size()) * start
                    : SPECIALlimbptr.data + i - (SPECIALlimb.size() < SPECIALmeta.size()) * start,
                (src ? SPECIALlimbptr.data + i : SPECIALauxptr.data + i) -
                    (SPECIALlimb.size() < SPECIALmeta.size()) * start,
                index, br, SPECIAL(0, i));
        }
        for (size_t i = start; i < start + num_limbs; i += 1 /*cc.batch*/) {
            s.wait(STREAM(SPECIALlimb[i - (SPECIALlimb.size() < SPECIALmeta.size()) * start]));
        }

        if (!src) {
            for (auto& i : SPECIALlimb) {
                if (i.index() == U32) {
                    std::swap(std::get<U32>(i).v.data, std::get<U32>(i).aux.data);
                } else {
                    std::swap(std::get<U64>(i).v.data, std::get<U64>(i).aux.data);
                }
            }
            std::swap(SPECIALlimbptr.data, SPECIALauxptr.data);
        }
    }
    if (!src) {
        for (auto& i : limb) {
            if (i.index() == U32) {
                std::swap(std::get<U32>(i).v.data, std::get<U32>(i).aux.data);
            } else {
                std::swap(std::get<U64>(i).v.data, std::get<U64>(i).aux.data);
            }
        }
        std::swap(limbptr.data, auxptr.data);
    }
    for (int i = 0; i < limbsize; i += cc.batch) {
        s.wait(STREAM(limb[i]));
    }
    if (src)
        src->getS().wait(s);
}

void LimbPartition::modupInto(LimbPartition& partition, LimbPartition& aux_partition) {
    constexpr ALGO algo = ALGO_SHOUP;
    constexpr bool PRINT = false;
    //assert(SPECIALlimb.empty());
    cudaSetDevice(device);

    const int limbsize = *level + 1;

    s.wait(partition.getS());
    s.wait(aux_partition.getS());

    if constexpr (PRINT)
        for (auto& i : limb) {
            SWITCH(i, printThisLimb(2));
        }

    for (size_t d = 0; d < DECOMPmeta.size(); ++d) {

        int start = 0;
        for (int j = 0; j < d; ++j)
            start += DECOMPmeta[j].size();
        int size = std::min((int)DECOMPmeta[d].size(), limbsize - start);
        if (size <= 0)
            break;

        const int M = (cc.precom.constants[0].type == 0) ? 8 : 4;  // u32 tiles are byte-parity with u64 (kernel M=8): grid must be N/(bd*M*2)

        dim3 blockDimFirst{(uint32_t)(1 << ((cc.logN) / 2 - 1))};
        dim3 blockDimSecond = dim3{(uint32_t)(1 << ((cc.logN + 1) / 2 - 1))};
        int bytesFirst = (32 / M) * blockDimFirst.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));
        int bytesSecond = (32 / M) * blockDimSecond.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));

        for (int i = 0; i < size; i += cc.batch) {
            STREAM(limb.at(start + i)).wait(s);
            uint32_t num_limbs = std::min((uint32_t)cc.batch, (uint32_t)(size - i));

            INTT_<false, algo, INTT_NONE><<<dim3{cc.N / (blockDimFirst.x * M * 2), num_limbs}, blockDimFirst,
                                            bytesFirst, STREAM(limb.at(start + i)).ptr()>>>(
                getGlobals(), limbptr.data + start + i, PARTITION(id, start + i), auxptr.data + start + i);

            INTT_<true, algo, INTT_NONE><<<dim3{cc.N / (blockDimSecond.x * M * 2), num_limbs}, blockDimSecond,
                                           bytesSecond, STREAM(limb.at(start + i)).ptr()>>>(
                getGlobals(), auxptr.data + start + i, PARTITION(id, start + i),
                partition.DECOMPlimbptr[d].data + i);

        }
        for (size_t i = 0; i < size; i += cc.batch) {
            s.wait(STREAM(limb.at(start + i)));
        }
        if constexpr (PRINT)
            for (auto& i : partition.DECOMPlimb[d]) {
                SWITCH(i, printThisLimb(2));
            }

        {
            dim3 blockSize{64, 2};
            dim3 gridSize{(uint32_t)cc.N / blockSize.x};
            int shared_bytes = sizeof(uint64_t) * (size /*DECOMPlimb[d].size()*/) * blockSize.x;
            DecompAndModUpConv<algo><<<gridSize, blockSize, shared_bytes, s.ptr()>>>(
                partition.DECOMPlimbptr[d].data, *level + 1, partition.DIGITlimbptr[d].data, digitid[d], getGlobals());
        }

        if constexpr (PRINT)
            for (auto& i : partition.DIGITlimb[d]) {
                SWITCH(i, printThisLimb(2));
            }

        const int digitsize = cc.precom.constants[id].num_primeid_digit_to[digitid.at(d)][*level];

        for (size_t i = 0; i < digitsize; i += cc.batch) {
            STREAM(partition.DIGITlimb.at(d).at(i)).wait(s);
        }

        ApplyNTT<algo, NTT_NONE>(cc.batch, NTT_fusion_fields{}, partition.DIGITlimb.at(d), partition.DIGITlimbptr.at(d),
                                 aux_partition.DIGITlimbptr.at(d), cc, DIGIT(digitid.at(d), 0), digitsize);

        for (size_t i = 0; i < digitsize; i += cc.batch) {
            s.wait(STREAM(partition.DIGITlimb.at(d).at(i)));
        }

        if constexpr (PRINT)
            for (auto& i : partition.DIGITlimb[d]) {
                SWITCH(i, printThisLimb(2));
            }
    }

    aux_partition.getS().wait(s);
    partition.getS().wait(s);
}

void LimbPartition::multScalar(std::vector<uint64_t>& vector) {
    cudaSetDevice(device);
    /*
    cudaDeviceSynchronize();
    for (auto& l : limb) {
        if (l.index() == U64) {
            scalar_mult_<uint64_t, ALGO_BARRETT>
                <<<cc.N / 128, 128, 0, STREAM(l).ptr()>>>(std::get<U64>(l).v.data, vector[PRIMEID(l)], PRIMEID(l));
        } else {
            scalar_mult_<uint32_t, ALGO_BARRETT>
                <<<cc.N / 128, 128, 0, STREAM(l).ptr()>>>(std::get<U32>(l).v.data, vector[PRIMEID(l)], PRIMEID(l));
        }
    }
    cudaDeviceSynchronize();
     */
    const int limbsize = getLimbSize(*level);

    // Persistent operand buffer (scratch slot); falls back to a per-call allocation if the slot is unavailable.
    const size_t elems_bytes = vector.size() * sizeof(uint64_t);
    uint64_t* elems = (uint64_t*)scratchGet(SC_SCALAR_MULT, elems_bytes);
    const bool elems_persist = elems != nullptr;
    if (!elems_persist)
        cudaMallocAsync((void**)&elems, elems_bytes, s.ptr());
    cudaMemcpyAsync(elems, vector.data(), vector.size() * sizeof(uint64_t), cudaMemcpyDefault, s.ptr());

    for (int i = 0; i < limbsize; i += cc.batch) {
        STREAM(limb[i]).wait(s);
        uint32_t num_limbs = std::min((int)limbsize - i, cc.batch);
        const int smul_bpt = fideslibAddBytes();
        const size_t smul_bpl = FIDESLIB_ADD_VEC ? uniform_limb_bytes(meta, (size_t)i, (size_t)num_limbs, cc.N) : 0;
        if (smul_bpl && smul_bpt >= 16 && (smul_bpl % (size_t)(smul_bpt * 128)) == 0)
            launchScalarMultBytes(dim3{(uint32_t)(smul_bpl / (smul_bpt * 128)), num_limbs}, dim3{128},
                                  STREAM(limb[i]).ptr(), limbptr.data + i, elems, PARTITION(id, i), nullptr, smul_bpt);
        else
            Scalar_mult_<ALGO_BARRETT><<<dim3{(uint32_t)cc.N / 128, num_limbs}, 128, 0, STREAM(limb[i]).ptr()>>>(
                limbptr.data + i, elems, PARTITION(id, i), nullptr);
    }
    for (int i = 0; i < limbsize; i += cc.batch) {
        s.wait(STREAM(limb[i]));
    }
    if (!elems_persist)
        cudaFreeAsync(elems, s.ptr());
}

void LimbPartition::addScalar(std::vector<uint64_t>& vector) {
    const int limbsize = getLimbSize(*level);
    cudaSetDevice(device);
    // Persistent operand buffer (scratch slot); falls back to a per-call allocation if the slot is unavailable.
    const size_t elems_bytes = vector.size() * sizeof(uint64_t);
    uint64_t* elems = (uint64_t*)scratchGet(SC_SCALAR_ADD, elems_bytes);
    const bool elems_persist = elems != nullptr;
    if (!elems_persist)
        cudaMallocAsync((void**)&elems, elems_bytes, s.ptr());
    cudaMemcpyAsync(elems, vector.data(), vector.size() * sizeof(uint64_t), cudaMemcpyDefault, s.ptr());
    for (int i = 0; i < limbsize; i += cc.batch) {
        STREAM(limb[i]).wait(s);
        uint32_t num_limbs = std::min((int)limbsize - i, cc.batch);
        int primeid_init = PARTITION(id, i);
        const int scalar_add_bpt = fideslibAddBytes();
        const size_t scalar_add_bpl = FIDESLIB_ADD_VEC ? uniform_limb_bytes(meta, (size_t)i, (size_t)num_limbs, cc.N) : 0;
        if (scalar_add_bpl && (scalar_add_bpl % (size_t)(scalar_add_bpt * 128)) == 0)
            launchScalarAddSubBytes(dim3{(uint32_t)(scalar_add_bpl / (scalar_add_bpt * 128)), num_limbs}, dim3{128},
                                    STREAM(limb[i]).ptr(), limbptr.data + i, elems, PARTITION(id, i), scalar_add_bpt, false);
        else
        scalar_add_<<<dim3{(uint32_t)cc.N / 128, num_limbs}, 128, 0, STREAM(limb[i]).ptr()>>>(limbptr.data + i, elems,
                                                                                              primeid_init);
    }
    for (int i = 0; i < limbsize; i += cc.batch) {
        s.wait(STREAM(limb[i]));
    }
    if (!elems_persist)
        cudaFreeAsync(elems, s.ptr());
}

void LimbPartition::subScalar(std::vector<uint64_t>& vector) {
    const int limbsize = getLimbSize(*level);
    cudaSetDevice(device);
    // Persistent operand buffer (scratch slot); falls back to a per-call allocation if the slot is unavailable.
    const size_t elems_bytes = vector.size() * sizeof(uint64_t);
    uint64_t* elems = (uint64_t*)scratchGet(SC_SCALAR_SUB, elems_bytes);
    const bool elems_persist = elems != nullptr;
    if (!elems_persist)
        cudaMallocAsync((void**)&elems, elems_bytes, s.ptr());
    cudaMemcpyAsync(elems, vector.data(), vector.size() * sizeof(uint64_t), cudaMemcpyDefault, s.ptr());
    for (int i = 0; i < limbsize; i += cc.batch) {
        STREAM(limb[i]).wait(s);
        uint32_t num_limbs = std::min((int)limbsize - i, cc.batch);
        const int scalar_sub_bpt = fideslibAddBytes();
        const size_t scalar_sub_bpl = FIDESLIB_ADD_VEC ? uniform_limb_bytes(meta, (size_t)i, (size_t)num_limbs, cc.N) : 0;
        if (scalar_sub_bpl && (scalar_sub_bpl % (size_t)(scalar_sub_bpt * 128)) == 0)
            launchScalarAddSubBytes(dim3{(uint32_t)(scalar_sub_bpl / (scalar_sub_bpt * 128)), num_limbs}, dim3{128},
                                    STREAM(limb[i]).ptr(), limbptr.data + i, elems, PARTITION(id, i), scalar_sub_bpt, true);
        else
        scalar_sub_<<<dim3{(uint32_t)cc.N / 128, num_limbs}, 128, 0, STREAM(limb[i]).ptr()>>>(limbptr.data + i, elems,
                                                                                              PARTITION(id, i));
    }
    for (int i = 0; i < limbsize; i += cc.batch) {
        s.wait(STREAM(limb[i]));
    }
    if (!elems_persist)
        cudaFreeAsync(elems, s.ptr());
}

void LimbPartition::add(const LimbPartition& a, const LimbPartition& b, const bool ext_a, const bool ext_b) {
    FIDESLIB_ADD_CENSUS_HIT();
    cudaSetDevice(device);
    s.wait(a.getS());
    s.wait(b.getS());

    const int limbsize = getLimbSize(*level);

    for (int i = 0; i < limbsize; i += cc.batch) {
        STREAM(limb[i]).wait(s);
        uint32_t num_limbs = std::min((int)limbsize - i, cc.batch);
        if (!ext_a && ext_b) {
            addScaleB_<<<dim3{(uint32_t)cc.N / 128, num_limbs}, 128, 0, STREAM(limb[i]).ptr()>>>(
                limbptr.data + i, a.limbptr.data + i, b.limbptr.data + i, PARTITION(id, i));
        } else if (!ext_b && ext_a) {
            addScaleB_<<<dim3{(uint32_t)cc.N / 128, num_limbs}, 128, 0, STREAM(limb[i]).ptr()>>>(
                limbptr.data + i, b.limbptr.data + i, a.limbptr.data + i, PARTITION(id, i));
        } else {
            add_<<<dim3{(uint32_t)cc.N / 128, num_limbs}, 128, 0, STREAM(limb[i]).ptr()>>>(
                limbptr.data + i, a.limbptr.data + i, b.limbptr.data + i, PARTITION(id, i));
        }
    }

    if (ext_a || ext_b) {
        int start = cc.splitSpecialMeta.at(id).at(0).id - (cc.L + 1);
        int num_limbs = cc.splitSpecialMeta.at(id).size();
        for (size_t i = start; i < start + num_limbs; i += cc.batch) {
            STREAM(SPECIALlimb[i]).wait(s);
            uint32_t size = std::min((int)start + num_limbs - (int)i, cc.batch);

            if (!ext_a && ext_b) {
                // TODO: have to check if Limbpartition comes from a plaintext, where extension limbs are mapped differently
                launch_copy_limbs((uint32_t)cc.N, size, STREAM(SPECIALlimb[i]).ptr(), b.SPECIALlimbptr.data + i,
                                  SPECIALlimbptr.data + i,
                                  uniform_limb_bytes(SPECIALmeta, (size_t)i, (size_t)size, cc.N));
            } else if (!ext_b && ext_a) {
                // TODO: have to check if Limbpartition comes from a plaintext, where extension limbs are mapped differently
                launch_copy_limbs((uint32_t)cc.N, size, STREAM(SPECIALlimb[i]).ptr(), a.SPECIALlimbptr.data + i,
                                  SPECIALlimbptr.data + i,
                                  uniform_limb_bytes(SPECIALmeta, (size_t)i, (size_t)size, cc.N));
            } else {
                add_<<<dim3{(uint32_t)cc.N / 128, size}, 128, 0, STREAM(SPECIALlimb[i]).ptr()>>>(
                    SPECIALlimbptr.data + i, a.SPECIALlimbptr.data + i, b.SPECIALlimbptr.data + i,
                    SPECIAL(
                        id,
                        i));  // TODO: have to check if Limbpartition comes from a plaintext, where extension limbs are mapped differently
            }
        }
        for (size_t i = start; i < start + num_limbs; i += cc.batch) {
            s.wait(STREAM(SPECIALlimb[i]));
        }
    }

    for (int i = 0; i < limbsize; i += cc.batch) {
        s.wait(STREAM(limb[i]));
    }

    a.getS().wait(s);
    b.getS().wait(s);
}
void LimbPartition::squareElement(const LimbPartition& p) {
    const int limbsize = getLimbSize(*level);
    cudaSetDevice(device);
    s.wait(p.getS());
    int size = std::min(limbsize, (int)p.limb.size());
    for (int i = 0; i < size; i += cc.batch) {
        STREAM(limb[i]).wait(s);
        uint32_t num_limbs = std::min((int)size - i, cc.batch);
        square_<<<dim3{(uint32_t)cc.N / 128, num_limbs}, 128, 0, STREAM(limb[i]).ptr()>>>(
            limbptr.data + i, p.limbptr.data + i, PARTITION(id, i));
    }
    for (int i = 0; i < size; i += cc.batch) {
        s.wait(STREAM(limb[i]));
    }
    p.getS().wait(s);
}
void LimbPartition::binomialSquareFold(LimbPartition& c0_res, const LimbPartition& c2_key_switched_0,
                                       const LimbPartition& c2_key_switched_1) {
    const int limbsize = getLimbSize(*level);
    cudaSetDevice(device);
    s.wait(c0_res.getS());
    s.wait(c2_key_switched_0.getS());
    s.wait(c2_key_switched_1.getS());
    for (int i = 0; i < limbsize; i += cc.batch) {
        STREAM(limb[i]).wait(s);
        uint32_t num_limbs = std::min((int)limbsize - i, cc.batch);
        binomial_square_fold_<<<dim3{(uint32_t)cc.N / 128, num_limbs}, 128, 0, STREAM(limb[i]).ptr()>>>(
            c0_res.limbptr.data + i, c2_key_switched_0.limbptr.data + i, limbptr.data + i,
            c2_key_switched_1.limbptr.data + i, PARTITION(id, i));
    }
    for (int i = 0; i < limbsize; i += cc.batch) {
        s.wait(STREAM(limb[i]));
    }
    c0_res.getS().wait(s);
    c2_key_switched_0.getS().wait(s);
    c2_key_switched_1.getS().wait(s);
}
void LimbPartition::dropLimb() {
    cudaSetDevice(device);

    STREAM(limb.back()).wait(s);
    limb.pop_back();
}
void LimbPartition::addMult(const LimbPartition& a, const LimbPartition& b) {
    const int limbsize = getLimbSize(*level);
    assert(a.limb.size() >= limbsize);
    assert(b.limb.size() >= limbsize);
    cudaSetDevice(device);
    s.wait(a.getS());
    s.wait(b.getS());
    for (int i = 0; i < limbsize; i += cc.batch) {
        STREAM(limb[i]).wait(s);
        uint32_t num_limbs = std::min((int)limbsize - i, cc.batch);
        addMult_<<<dim3{(uint32_t)cc.N / 128, num_limbs}, 128, 0, STREAM(limb[i]).ptr()>>>(
            limbptr.data + i, a.limbptr.data + i, b.limbptr.data + i, PARTITION(id, i));
    }
    for (int i = 0; i < limbsize; i += cc.batch) {
        s.wait(STREAM(limb[i]));
    }
    a.getS().wait(s);
    b.getS().wait(s);
}
void LimbPartition::broadcastLimb0() {
    const int limbsize = getLimbSize(*level);
    cudaSetDevice(device);
    assert(limbsize - 1 > 0);
    broadcastLimb0_<<<dim3{(uint32_t)cc.N / 128, (uint32_t)limbsize - 1}, 128, 0, s.ptr()>>>(limbptr.data);
}

void LimbPartition::compositeModRaise(const int d, const std::vector<uint64_t>& qhatinv,
                                      const std::vector<uint64_t>& qhat) {
    const int limbsize = getLimbSize(*level);
    cudaSetDevice(device);
    assert(limbsize > d);
    assert((int)qhatinv.size() >= d && (int)qhat.size() >= d * limbsize);

    // The kernel overwrites every limb, including the d source limbs — snapshot the sources
    // first. Slots are 8*N bytes each regardless of limb width (raw byte copies; the kernel
    // re-reads them at prime k's width via ISU64).
    const size_t slot = (size_t)cc.N * sizeof(uint64_t);
    uint8_t* snap;
    cudaMallocAsync(
        (void**)&snap, (size_t)d * slot + d * sizeof(void*) + (qhatinv.size() + qhat.size()) * sizeof(uint64_t),
        s.ptr());
    void** srcptrs = (void**)(snap + (size_t)d * slot);
    uint64_t* dev_qhatinv = (uint64_t*)(srcptrs + d);
    uint64_t* dev_qhat = dev_qhatinv + qhatinv.size();

    std::vector<void*> hostptrs(d);
    for (int k = 0; k < d; ++k) {
        hostptrs[k] = snap + (size_t)k * slot;
        void* v = nullptr;
        SWITCH_RET(limb.at(k), v.data, v);
        const size_t bytes = (size_t)cc.N * (limb.at(k).index() == U64 ? sizeof(uint64_t) : sizeof(uint32_t));
        cudaMemcpyAsync(hostptrs[k], v, bytes, cudaMemcpyDeviceToDevice, s.ptr());
    }
    cudaMemcpyAsync(srcptrs, hostptrs.data(), d * sizeof(void*),
                    cudaMemcpyHostToDevice, s.ptr());
    cudaMemcpyAsync(dev_qhatinv, qhatinv.data(),
                    qhatinv.size() * sizeof(uint64_t), cudaMemcpyHostToDevice, s.ptr());
    cudaMemcpyAsync(dev_qhat, qhat.data(),
                    qhat.size() * sizeof(uint64_t), cudaMemcpyHostToDevice, s.ptr());

    compositeModRaise_<<<dim3{(uint32_t)cc.N / 128, (uint32_t)limbsize}, 128, 0, s.ptr()>>>(limbptr.data, srcptrs, d,
                                                                                           dev_qhatinv, dev_qhat);
    cudaFreeAsync(snap, s.ptr());
}
void LimbPartition::coeffLiftCentered(const uint64_t q0, const uint64_t q1,
                                      const uint64_t q0inv_mod_q1, const uint64_t Qhalf,
                                      const std::vector<uint64_t>& Q0_mod_qi) {
    const int limbsize = getLimbSize(*level);
    cudaSetDevice(device);
    assert(limbsize > 2);
    assert((int)Q0_mod_qi.size() >= limbsize);

    // Same snapshot discipline as compositeModRaise: the kernel overwrites EVERY limb including
    // the two sources, so copy them aside first. Slots are 8*N bytes regardless of limb width.
    const size_t slot = (size_t)cc.N * sizeof(uint64_t);
    uint8_t* snap;
    cudaMallocAsync((void**)&snap,
                                     2 * slot + 2 * sizeof(void*) + Q0_mod_qi.size() * sizeof(uint64_t), s.ptr());
    void** srcptrs = (void**)(snap + 2 * slot);
    uint64_t* dev_Q0mod = (uint64_t*)(srcptrs + 2);

    std::vector<void*> hostptrs(2);
    for (int k = 0; k < 2; ++k) {
        hostptrs[k] = snap + (size_t)k * slot;
        void* v = nullptr;
        SWITCH_RET(limb.at(k), v.data, v);
        const size_t bytes = (size_t)cc.N * (limb.at(k).index() == U64 ? sizeof(uint64_t) : sizeof(uint32_t));
        cudaMemcpyAsync(hostptrs[k], v, bytes, cudaMemcpyDeviceToDevice, s.ptr());
    }
    cudaMemcpyAsync(srcptrs, hostptrs.data(), 2 * sizeof(void*),
                    cudaMemcpyHostToDevice, s.ptr());
    cudaMemcpyAsync(dev_Q0mod, Q0_mod_qi.data(),
                    Q0_mod_qi.size() * sizeof(uint64_t), cudaMemcpyHostToDevice, s.ptr());

    coeffLiftCentered2_<<<dim3{(uint32_t)cc.N / 128, (uint32_t)limbsize}, 128, 0, s.ptr()>>>(
        limbptr.data, srcptrs, q0, q1, q0inv_mod_q1, Qhalf, dev_Q0mod);
    cudaFreeAsync(snap, s.ptr());
}
void LimbPartition::evalLinearWSum(uint32_t n, std::vector<const LimbPartition*> ps, std::vector<uint64_t>& weights) {
    const int limbsize = getLimbSize(*level);
    cudaSetDevice(device);
    for (int i = 0; i < n; ++i) {
        s.wait(ps[i]->getS());
    }

    // Persistent operand buffer (scratch slot); falls back to a per-call allocation if the slot is unavailable.
    const size_t elems_bytes = weights.size() * sizeof(uint64_t);
    uint64_t* elems = (uint64_t*)scratchGet(SC_LINWSUM_W, elems_bytes);
    const bool elems_persist = elems != nullptr;
    if (!elems_persist)
        cudaMallocAsync((void**)&elems, elems_bytes, s.ptr());
    cudaMemcpyAsync(elems, weights.data(), elems_bytes, cudaMemcpyDefault, s.ptr());
    std::vector<void**> psptr(n, nullptr);
    for (int i = 0; i < n; ++i) {
        psptr[i] = ps[i]->limbptr.data;
        assert(ps[i]->limb.size() >= limbsize);
    }
    const size_t psptr_bytes = psptr.size() * sizeof(void**);
    void*** d_psptr = (void***)scratchGet(SC_LINWSUM_PS, psptr_bytes);
    const bool psptr_persist = d_psptr != nullptr;
    if (!psptr_persist)
        cudaMallocAsync((void**)&d_psptr, psptr_bytes, s.ptr());
    cudaMemcpyAsync(d_psptr, psptr.data(), psptr_bytes, cudaMemcpyDefault, s.ptr());

    {
        const int elws_bpt = fideslibAddBytes();
        const size_t elws_bpl = FIDESLIB_ADD_VEC ? uniform_limb_bytes(meta, 0, (size_t)limbsize, cc.N) : 0;
        if (!limb.empty() && elws_bpl && elws_bpt >= 16 && (elws_bpl % (size_t)(elws_bpt * 128)) == 0)
            launchEvalLinearWSumBytes(dim3{(uint32_t)(elws_bpl / (elws_bpt * 128)), (uint32_t)limbsize}, dim3{128},
                                      s.ptr(), n, limbptr.data, d_psptr, elems, PARTITION(id, 0), elws_bpt);
        else if (!limb.empty())
            eval_linear_w_sum_<<<dim3{(uint32_t)cc.N / 128, (uint32_t)limbsize}, 128, 0, s.ptr()>>>(
                n, limbptr.data, d_psptr, elems, PARTITION(id, 0));
    }
    if (!elems_persist)
        cudaFreeAsync(elems, s.ptr());
    if (!psptr_persist)
        cudaFreeAsync(d_psptr, s.ptr());
    for (uint32_t i = 0; i < n; ++i) {
        ps[i]->getS().wait(s);
    }
}

/**
  Only for MGPU key generation and extended limb partitions
 */
void LimbPartition::generatePartialSpecialLimb() {
    cudaSetDevice(device);
    if (SPECIALlimb.size() == 0 && cc.splitSpecialMeta.at(id).size() > 0 /*&& bufferSPECIAL == nullptr*/) {
        //if (bufferSPECIAL)
        //    GPUfree(bufferSPECIAL, id, std::max(1ul, cc.N * cc.splitSpecialMeta.at(id).size() * sizeof(uint64_t)),
        //            s.ptr());

        //bufferSPECIAL = (uint64_t*)GPUmalloc(
        //    id, std::max(1ul, cc.N * cc.splitSpecialMeta.at(id).size() * sizeof(uint64_t)), s.ptr());
        //cudaMallocAsync(&bufferSPECIAL, std::max(1ul, cc.N * cc.splitSpecialMeta.at(id).size() * sizeof(uint64_t)),
        //                s.ptr());
        //generate(cc.splitSpecialMeta.at(id), SPECIALlimb, SPECIALlimbptr, (int)cc.splitSpecialMeta.at(id).size() - 1,
        //         nullptr, bufferSPECIAL, 0);
        generate(cc.splitSpecialMeta.at(id), SPECIALlimb, SPECIALlimbptr, (int)cc.splitSpecialMeta.at(id).size() - 1,
                 nullptr, nullptr, 0);
        // for (auto& l : SPECIALlimb)
        //     STREAM(l).wait(s);
    }
}
void LimbPartition::dotProductPt(LimbPartition& c1, const std::vector<const LimbPartition*>& c0s,
                                 const std::vector<const LimbPartition*>& c1s,
                                 const std::vector<const LimbPartition*>& pts, const bool ext) {

    const int limbsize = getLimbSize(*level);
    cudaSetDevice(device);
    int n = c0s.size();
    std::vector<void**> h_data(n * 3 * (1 + ext), nullptr);

    for (size_t i = 0; i < n; ++i) {
        assert(c0s[i]);
        assert(c1s[i]);
        assert(pts[i]);
        h_data[i] = c0s[i]->limbptr.data;
        h_data[i + n] = c1s[i]->limbptr.data;
        h_data[i + 2 * n] = pts[i]->limbptr.data;
        s.wait(c0s[i]->getS());
        s.wait(c1s[i]->getS());
        s.wait(pts[i]->getS());
        assert(c0s[i]->limb.size() >= limbsize);
        assert(c1s[i]->limb.size() >= limbsize);
        assert(pts[i]->limb.size() >= limbsize);
        if (ext) {
            int start = cc.splitSpecialMeta.at(id).at(0).id - cc.precom.constants[id].L;
            int num_limbs = cc.splitSpecialMeta.at(id).size();
            assert(c0s[i]->SPECIALlimb.size() >= this->SPECIALlimb.size());
            assert(c1s[i]->SPECIALlimb.size() >= this->SPECIALlimb.size());
            assert(pts[i]->SPECIALlimb.size() >= 0);
            h_data[i + 3 * n] = c0s[i]->SPECIALlimbptr.data + start;
            h_data[i + 4 * n] = c1s[i]->SPECIALlimbptr.data + start;
            h_data[i + 5 * n] = pts[i]->SPECIALlimbptr.data;
        }
    }

    s.wait(c1.getS());

    VectorGPU<void**> data(s, n * 3 * (1 + ext), device, h_data.data());

    for (size_t i = 0; i < limbsize; i += cc.batch) {
        STREAM(limb[i]).wait(s);
        int size = std::min((int)limbsize - (int)i, cc.batch);
        dotProductPt_<<<dim3{(uint32_t)cc.N / 128, (uint32_t)size}, 128, 0, STREAM(limb[i]).ptr()>>>(
            limbptr.data, c1.limbptr.data, data.data, i, PARTITION(id, i), n);
    }

    if (ext) {
        int start = cc.splitSpecialMeta.at(id).at(0).id - (cc.L + 1);
        int num_limbs = cc.splitSpecialMeta.at(id).size();
        for (size_t i = start; i < start + num_limbs; i += cc.batch) {
            STREAM(SPECIALlimb[i]).wait(s);
            int size = std::min((int)start + num_limbs - (int)i, cc.batch);
            dotProductPt_<<<dim3{(uint32_t)cc.N / 128, (uint32_t)size}, 128, 0, STREAM(SPECIALlimb[i]).ptr()>>>(
                SPECIALlimbptr.data + start, c1.SPECIALlimbptr.data + start, data.data + 3 * n, i - start,
                SPECIAL(id, i), n);
        }
        for (size_t i = start; i < start + num_limbs; i += cc.batch) {
            STREAM(SPECIALlimb[i]).wait(s);
        }
    }
    for (size_t i = 0; i < limbsize; i += cc.batch) {
        s.wait(STREAM(limb[i]));
    }
    c1.getS().wait(s);
    for (size_t i = 0; i < c0s.size(); ++i) {
        c0s[i]->getS().wait(s);
        c1s[i]->getS().wait(s);
        pts[i]->getS().wait(s);
    }
    data.free(s);
}

void LimbPartition::binomialDotProduct(LimbPartition& c1, LimbPartition& c2,
                                       const std::vector<const LimbPartition*>& c0s,
                                       const std::vector<const LimbPartition*>& c1s,
                                       const std::vector<const LimbPartition*>& d0s,
                                       const std::vector<const LimbPartition*>& d1s, const bool ext) {
    const int limbsize = getLimbSize(*level);
    cudaSetDevice(device);
    int n = c0s.size();
    std::vector<void**> h_data((n * 4 + 3) * (1 + ext), nullptr);

    for (size_t i = 0; i < n; ++i) {
        assert(c0s[i]);
        assert(c1s[i]);
        assert(d0s[i]);
        assert(d1s[i]);
        h_data[i] = c0s[i]->limbptr.data;
        h_data[i + n] = c1s[i]->limbptr.data;
        h_data[i + 2 * n] = d0s[i]->limbptr.data;
        h_data[i + 3 * n] = d1s[i]->limbptr.data;
        s.wait(c0s[i]->getS());
        s.wait(c1s[i]->getS());
        s.wait(d0s[i]->getS());
        s.wait(d1s[i]->getS());

        assert(c0s[i]->limb.size() >= limbsize);
        assert(c1s[i]->limb.size() >= limbsize);
        assert(d0s[i]->limb.size() >= limbsize);
        assert(d1s[i]->limb.size() >= limbsize);
        if (ext) {
            int start = cc.splitSpecialMeta.at(id).at(0).id - cc.precom.constants[id].L;
            int num_limbs = cc.splitSpecialMeta.at(id).size();
            assert(c0s[i]->SPECIALlimb.size() >= this->SPECIALlimb.size());
            assert(c1s[i]->SPECIALlimb.size() >= this->SPECIALlimb.size());
            assert(d0s[i]->SPECIALlimb.size() >= this->SPECIALlimb.size());
            assert(d1s[i]->SPECIALlimb.size() >= this->SPECIALlimb.size());
            h_data[i + 4 * n] = c0s[i]->SPECIALlimbptr.data + start;
            h_data[i + 5 * n] = c1s[i]->SPECIALlimbptr.data + start;
            h_data[i + 6 * n] = d0s[i]->SPECIALlimbptr.data + start;
            h_data[i + 7 * n] = d1s[i]->SPECIALlimbptr.data + start;
        }
    }

    h_data[n * 4 * (1 + ext)] = limbptr.data;
    h_data[n * 4 * (1 + ext) + 1] = c1.limbptr.data;
    h_data[n * 4 * (1 + ext) + 2] = c2.limbptr.data;
    if (ext) {
        h_data[n * 4 * (1 + ext) + 3] = SPECIALlimbptr.data;
        h_data[n * 4 * (1 + ext) + 4] = c1.SPECIALlimbptr.data;
        h_data[n * 4 * (1 + ext) + 5] = c2.SPECIALlimbptr.data;
    }

    s.wait(c1.getS());
    s.wait(c2.getS());
    for (size_t i = 0; i < c0s.size(); ++i) {
        s.wait(c0s[i]->getS());
        s.wait(c1s[i]->getS());
        s.wait(d0s[i]->getS());
        s.wait(d1s[i]->getS());
    }

    VectorGPU<void**> data(s, (n * 4 + 3) * (1 + ext), device, h_data.data());

    for (size_t i = 0; i < limbsize; i += cc.batch) {
        STREAM(limb[i]).wait(s);
        int size = std::min((int)limbsize - (int)i, cc.batch);
        // dotProductPt_<<<dim3{(uint32_t)cc.N / 128, (uint32_t)size}, 128, 0, STREAM(limb[i]).ptr()>>>(
        //     limbptr.data, c1.limbptr.data, data.data, i, PARTITION(id, i), n);

        binomialDotProdBatched___<<<dim3{(uint32_t)cc.N / 128, (uint32_t)size}, 128, 0, STREAM(limb[i]).ptr()>>>(
            PARTITION(id, i), data.data + 0, data.data + n, data.data + 2 * n, data.data + 3 * n,
            data.data + n * 4 * (1 + ext), data.data + n * 4 * (1 + ext) + 1, data.data + n * 4 * (1 + ext) + 2, n, 1);
    }

    if (ext) {
        int start = cc.splitSpecialMeta.at(id).at(0).id - (cc.L + 1);
        int num_limbs = cc.splitSpecialMeta.at(id).size();
        for (size_t i = start; i < start + num_limbs; i += cc.batch) {
            STREAM(SPECIALlimb[i]).wait(s);
            int size = std::min((int)start + num_limbs - (int)i, cc.batch);
            // dotProductPt_<<<dim3{(uint32_t)cc.N / 128, (uint32_t)size}, 128, 0, STREAM(SPECIALlimb[i]).ptr()>>>(
            //     SPECIALlimbptr.data + start, c1.SPECIALlimbptr.data + start, data.data + 3 * n, i - start,
            //    SPECIAL(id, i), n);

            binomialDotProdBatched___<<<dim3{(uint32_t)cc.N / 128, (uint32_t)size}, 128, 0,
                                        STREAM(SPECIALlimb[i]).ptr()>>>(
                SPECIAL(id, i), data.data + 4 * n, data.data + 5 * n, data.data + 6 * n, data.data + 7 * n,
                data.data + n * 4 * (1 + ext) + 3, data.data + n * 4 * (1 + ext) + 4, data.data + n * 4 * (1 + ext) + 5,
                n, 1);
        }
        for (size_t i = start; i < start + num_limbs; i += cc.batch) {
            STREAM(SPECIALlimb[i]).wait(s);
        }
    }
    for (size_t i = 0; i < limbsize; i += cc.batch) {
        s.wait(STREAM(limb[i]));
    }
    c1.getS().wait(s);
    for (size_t i = 0; i < c0s.size(); ++i) {
        c0s[i]->getS().wait(s);
        c1s[i]->getS().wait(s);
        d0s[i]->getS().wait(s);
        d1s[i]->getS().wait(s);
    }
    data.free(s);
}
void LimbPartition::binomialMult(LimbPartition& c1, LimbPartition& c2, const LimbPartition& d0, const LimbPartition& d1,
                                 bool extend_ins, bool square) {
    const int limbsize = getLimbSize(*level);
    cudaSetDevice(device);

    s.wait(c1.getS());
    s.wait(c2.getS());
    if (!square) {
        s.wait(d0.getS());
        s.wait(d1.getS());
    }

    for (size_t i = 0; i < limbsize; i += cc.batch) {
        STREAM(limb[i]).wait(s);
        int size = std::min((int)limbsize - (int)i, cc.batch);
        // dotProductPt_<<<dim3{(uint32_t)cc.N / 128, (uint32_t)size}, 128, 0, STREAM(limb[i]).ptr()>>>(
        //     limbptr.data, c1.limbptr.data, data.data, i, PARTITION(id, i), n);

        if (!square) {
            if (!extend_ins) {
                binomialMult_<<<dim3{(uint32_t)cc.N / 128, (uint32_t)size}, 128, 0, STREAM(limb[i]).ptr()>>>(
                    PARTITION(id, i), this->limbptr.data + i, c1.limbptr.data + i, c2.limbptr.data + i,
                    d0.limbptr.data + i, d1.limbptr.data + i);
            } else {
                binomialMultExtend_<<<dim3{(uint32_t)cc.N / 128, (uint32_t)size}, 128, 0, STREAM(limb[i]).ptr()>>>(
                    PARTITION(id, i), this->limbptr.data + i, c1.limbptr.data + i, c2.limbptr.data + i,
                    d0.limbptr.data + i, d1.limbptr.data + i);
            }
        } else {
            if (!extend_ins) {
                binomialSquare_<<<dim3{(uint32_t)cc.N / 128, (uint32_t)size}, 128, 0, STREAM(limb[i]).ptr()>>>(
                    PARTITION(id, i), this->limbptr.data + i, c1.limbptr.data + i, c2.limbptr.data + i);
            } else {
                binomialSquareExtend_<<<dim3{(uint32_t)cc.N / 128, (uint32_t)size}, 128, 0, STREAM(limb[i]).ptr()>>>(
                    PARTITION(id, i), this->limbptr.data + i, c1.limbptr.data + i, c2.limbptr.data + i);
            }
        }
    }

    for (size_t i = 0; i < limbsize; i += cc.batch) {
        s.wait(STREAM(limb[i]));
    }

    c1.getS().wait(s);
    c2.getS().wait(s);
    if (!square) {
        d0.getS().wait(s);
        d1.getS().wait(s);
    }
}

void LimbPartition::generateGatherLimb(bool iskey) {
    if (bufferGATHER == nullptr) {
        if (cc.GPUid.size() == 1 || iskey) {
            //bufferGATHER =
            //    (uint64_t*)GPUmalloc(device, std::max(1ul, GATHERmeta.size() * sizeof(uint64_t) * cc.N), s.ptr());
            // cudaMallocAsync(&bufferGATHER, std::max(1ul, GATHERmeta.size() * sizeof(uint64_t) * cc.N), s.ptr());

            if (iskey == false && DECOMPlimb.at(0).size() > 0) {
                std::vector<void*> h_gatherptr(GATHERptr.size, nullptr);

                int a = 0;

                for (size_t i = 0; i < DECOMPmeta.size(); ++i) {
                    for (size_t j = 0; j < DECOMPmeta[i].size(); ++j) {

                        void* ptr;
                        SWITCH_RET(DECOMPlimb[i][j], v.data, ptr);
                        h_gatherptr[a] = ptr;
                        a++;
                    }
                }

                if (GATHERptr.size * sizeof(void*) > 0) {
                    cudaMemcpyAsync(GATHERptr.data, h_gatherptr.data(), GATHERptr.size * sizeof(void*),
                                    cudaMemcpyHostToDevice, s.ptr());
                }
            }
        } else {
#ifdef NCCL
            cudaStreamSynchronize(s.ptr());
            NCCLCHECK(ncclMemAlloc((void**)&bufferGATHER, std::max(1ul, GATHERmeta.size() * sizeof(uint64_t) * cc.N)));
            NCCLCHECK(ncclCommRegister(rank, bufferGATHER, std::max(1ul, GATHERmeta.size() * sizeof(uint64_t) * cc.N),
                                       &bufferGATHER_handle));
            if (bufferGATHER_handle == nullptr)
                bufferGATHER_handle = (void*)-1;
            cudaDeviceSynchronize();
#else
            assert(false);
#endif
            std::vector<void*> h_gatherptr(GATHERptr.size, nullptr);
            for (size_t i = 0; i < h_gatherptr.size(); ++i)
                h_gatherptr[i] = (void*)(bufferGATHER + cc.N * i);

            if (GATHERptr.size * sizeof(void*) > 0) {
                cudaMemcpyAsync(GATHERptr.data, h_gatherptr.data(), GATHERptr.size * sizeof(void*),
                                cudaMemcpyHostToDevice, s.ptr());
            }
        }
    }
}

// namespace FIDESlib::CKKS

}  // namespace FIDESlib::CKKS
