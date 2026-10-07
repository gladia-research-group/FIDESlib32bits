//
// Created by carlosad on 1/10/25.
//
#include "CKKS/BootstrapPrecomputation.cuh"
#include <algorithm>
#include <array>
#include <stdexcept>
#include <variant>
#include <vector>

//#include <cstring>
#include "CKKS/Context.cuh"
#include "CKKS/Conv.cuh"
#include "CKKS/ElemenwiseBatchKernels.cuh"
#include "CKKS/LimbPartition.cuh"
#include "LimbUtils.cuh"
#include "NTT.cuh"
#include "Rotation.cuh"
#include "VectorGPU.cuh"

namespace FIDESlib::CKKS {

namespace {
/** One device pointer table per batch call, built from a function-local std::vector into a persistent
 *  scratch slot of the object whose stream drives the op (element [0] / acc0); reuse across calls is
 *  serialised by that stream. Falls back to a per-call cudaMallocAsync/cudaFreeAsync pair. */
struct BatchPtrTable {
    void*** d = nullptr;
    bool persist = false;
    Stream& s;

    BatchPtrTable(LimbPartition& owner, int slot, const std::vector<void**>& h, Stream& stream) : s(stream) {
        const size_t bytes = sizeof(void**) * h.size();
        d = (void***)owner.scratchGet(slot, bytes);
        persist = d != nullptr;
        if (!persist)
            cudaMallocAsync((void**)&d, bytes, s.ptr());
        cudaMemcpyAsync(d, h.data(), bytes, cudaMemcpyHostToDevice, s.ptr());
    }
    ~BatchPtrTable() {
        if (!persist && d)
            cudaFreeAsync(d, s.ptr());   // stream-ordered: end-of-scope == the old free point
    }
    BatchPtrTable(const BatchPtrTable&) = delete;
    BatchPtrTable& operator=(const BatchPtrTable&) = delete;
};
}  // namespace

void LimbPartition::addBatchManyToOne(std::vector<LimbPartition*>& parta, const std::vector<LimbPartition*>& partb,
                                      int stride, double usage, bool sub, bool exta, bool extb) {
    ContextData& cc = parta[0]->cc;
    int limbsize = parta[0]->getLimbSize(*parta[0]->level);
    int slimbsize = extb ? parta[1]->SPECIALlimb.size() : 0;
    int n = parta.size();

    dim3 block = {128u, 1u, 1u};
    int threads = GetTargetThreads(cc.GPUid[parta[0]->id]) * usage;
    dim3 grid = {cc.N / 128u, (uint32_t)limbsize, 1u};
    uint32_t num_parallel_parts = std::max(
        1u, std::min((uint32_t)((threads + block.x * grid.x * grid.y - 1) / (block.x * grid.x * std::max(1u, grid.y))),
                     (uint32_t)parta.size()));
    grid.z = num_parallel_parts;

    int its = stride;
    int split = 1;
    if (num_parallel_parts > partb.size()) {
        split = (num_parallel_parts + partb.size() - 1) / partb.size();

        its = stride / split;
        grid.z = split * partb.size();
    }

    int size =
        its * split * partb.size() + split * partb.size() + extb * (its * split * partb.size() + split * partb.size());
    int offset1 = its * split * partb.size(), offset2 = its * split * partb.size() + split * partb.size(),
        offset3 = its * split * partb.size() + split * partb.size() + its * split * partb.size();
    std::vector<void**> data_ptrs(size, nullptr);
    for (auto i = 0; i < partb.size(); ++i) {

        for (int j = 0; j < stride; ++j) {
            data_ptrs[its * split * i + j] = parta[stride * i + j]->limbptr.data;
        }
        for (int j = 0; j < split; ++j) {
            data_ptrs[offset1 + i * split + j] = partb[i]->limbptr.data;
        }

        if (extb) {
            for (int j = 0; j < stride; ++j) {
                data_ptrs[offset2 + its * split * i + j] = parta[stride * i + j]->SPECIALlimbptr.data;
            }
            for (int j = 0; j < split; ++j) {
                data_ptrs[offset3 + i * split + j] = partb[i]->SPECIALlimbptr.data;
            }
        }
    }

    dim3 sgrid = {grid.x, (uint32_t)slimbsize, grid.z};

    Stream& s = parta[0]->s;

    BatchPtrTable tbl(*parta[0], LimbPartition::SC_BATCH_ADD, data_ptrs, s);
    void*** const data_ptrs_d = tbl.d;
    s.wait(partb[0]->s);
    if (!sub) {
        if (!exta && !extb) {
            if (limbsize > 0)
                add_reuse_b___<<<grid, block, 0, s.ptr()>>>(data_ptrs_d, data_ptrs_d + offset1,
                                                            PARTITION(parta[0]->id, 0), n, its);
        }
        if (exta && extb) {
            if (limbsize > 0)
                add_reuse_b___<<<grid, block, 0, s.ptr()>>>(data_ptrs_d, data_ptrs_d + offset1,
                                                            PARTITION(parta[0]->id, 0), n, its);
            if (slimbsize > 0)
                add_reuse_b___<<<sgrid, block, 0, s.ptr()>>>(data_ptrs_d + offset2, data_ptrs_d + offset3,
                                                             SPECIAL(parta[0]->id, 0), n, its);
        }
        if (exta && !extb) {
            if (limbsize > 0)
                add_reuse_scale_p_b___<<<grid, block, 0, s.ptr()>>>(data_ptrs_d, data_ptrs_d + offset1,
                                                                    PARTITION(parta[0]->id, 0), n, its);
        }
        if (!exta && extb) {
            if (limbsize > 0)
                add_scale_p_reuse_b___<<<grid, block, 0, s.ptr()>>>(data_ptrs_d, data_ptrs_d + offset1,
                                                                    PARTITION(parta[0]->id, 0), n, its);
            if (slimbsize > 0)
                copy_reuse_b___<<<sgrid, block, 0, s.ptr()>>>(data_ptrs_d + offset2, data_ptrs_d + offset3,
                                                              SPECIAL(parta[0]->id, 0), n, its);
        }

    } else {
        if (!exta && !extb) {
            if (limbsize > 0)
                sub_reuse_b___<<<grid, block, 0, s.ptr()>>>(data_ptrs_d, data_ptrs_d + offset1,
                                                            PARTITION(parta[0]->id, 0), n, its);
        }
        if (exta && extb) {
            if (limbsize > 0)
                sub_reuse_b___<<<grid, block, 0, s.ptr()>>>(data_ptrs_d, data_ptrs_d + offset1,
                                                            PARTITION(parta[0]->id, 0), n, its);
            if (slimbsize > 0)
                sub_reuse_b___<<<sgrid, block, 0, s.ptr()>>>(data_ptrs_d + offset2, data_ptrs_d + offset3,
                                                             SPECIAL(parta[0]->id, 0), n, its);
        }
        if (exta && !extb) {
            if (limbsize > 0)
                sub_reuse_scale_p_b___<<<grid, block, 0, s.ptr()>>>(data_ptrs_d, data_ptrs_d + offset1,
                                                                    PARTITION(parta[0]->id, 0), n, its);
        }
        if (!exta && extb) {
            if (limbsize > 0)
                sub_scale_p_reuse_b___<<<grid, block, 0, s.ptr()>>>(data_ptrs_d, data_ptrs_d + offset1,
                                                                    PARTITION(parta[0]->id, 0), n, its);
            if (slimbsize > 0)
                copy_reuse_negative_b___<<<sgrid, block, 0, s.ptr()>>>(data_ptrs_d + offset2, data_ptrs_d + offset3,
                                                                       SPECIAL(parta[0]->id, 0), n, its);
        }
    }
    partb[0]->s.wait(s);
}

void LimbPartition::multPtBatchManyToOne(std::vector<LimbPartition*>& parta, const std::vector<LimbPartition*>& partb,
                                         int stride, double usage) {
    ContextData& cc = parta[0]->cc;
    int limbsize = parta[0]->getLimbSize(*parta[0]->level);
    int n = parta.size();

    dim3 block = {128u, 1u, 1u};
    int threads = GetTargetThreads(cc.GPUid[parta[0]->id]) * usage;
    dim3 grid = {cc.N / 128u, (uint32_t)limbsize, 1u};
    uint32_t num_parallel_parts = std::max(
        1u, std::min((uint32_t)((threads + block.x * grid.x * grid.y - 1) / (block.x * grid.x * std::max(1u, grid.y))),
                     (uint32_t)parta.size()));
    grid.z = num_parallel_parts;

    int its = stride;
    int split = 1;
    if (num_parallel_parts > partb.size()) {
        split = (num_parallel_parts + partb.size() - 1) / partb.size();

        its = stride / split;
        grid.z = split * partb.size();
    }

    int size = its * split * partb.size() + split * partb.size();
    std::vector<void**> data_ptrs(size, nullptr);
    for (auto i = 0; i < partb.size(); ++i) {

        for (int j = 0; j < stride; ++j) {
            data_ptrs[its * split * i + j] = parta[stride * i + j]->limbptr.data;
        }
        for (int j = 0; j < split; ++j) {
            data_ptrs[its * split * partb.size() + i * split + j] = partb[i]->limbptr.data;
        }
    }

    Stream& s = parta[0]->s;

    BatchPtrTable tbl(*parta[0], LimbPartition::SC_BATCH_MULTPT, data_ptrs, s);
    void*** const data_ptrs_d = tbl.d;
    s.wait(partb[0]->s);
    if (limbsize > 0)
        mult_reuse_b___<<<grid, block, 0, s.ptr()>>>(data_ptrs_d, data_ptrs_d + its * split * partb.size(),
                                                     PARTITION(parta[0]->id, 0), n, its);
    partb[0]->s.wait(s);
}

void LimbPartition::addScalarBatchManyToOne(std::vector<LimbPartition*>& parta,
                                            const std::vector<std::vector<unsigned long int>>& vector, int stride,
                                            double usage) {
    ContextData& cc = parta[0]->cc;
    int limbsize = parta[0]->getLimbSize(*parta[0]->level);
    int n = parta.size();

    dim3 block = {128u, 1u, 1u};
    int threads = GetTargetThreads(cc.GPUid[parta[0]->id]) * usage;
    dim3 grid = {cc.N / 128u, (uint32_t)limbsize, 1u};
    uint32_t num_parallel_parts = std::max(
        1u, std::min((uint32_t)((threads + block.x * grid.x * grid.y - 1) / (block.x * grid.x * std::max(1u, grid.y))),
                     (uint32_t)parta.size()));
    grid.z = num_parallel_parts;

    int its = stride;
    int split = 1;
    if (num_parallel_parts > vector.size()) {
        split = (num_parallel_parts + vector.size() - 1) / vector.size();

        its = stride / split;
        grid.z = split * vector.size();
    }

    int size = its * split * vector.size() + split * vector.size() * MAXP;
    std::vector<void**> data_ptrs(size, nullptr);
    for (auto i = 0; i < vector.size(); ++i) {

        for (int j = 0; j < stride; ++j) {
            data_ptrs[its * split * i + j] = parta[stride * i + j]->limbptr.data;
        }
        for (int j = 0; j < split; ++j) {
            std::memcpy(&data_ptrs[its * split * vector.size() + (i * split + j) * MAXP], vector[i].data(),
                        vector[i].size() * sizeof(uint64_t));
            // data_ptrs[its * split * vector.size() + (i * split + j) * MAXP] = vector[i].data;
        }
    }

    Stream& s = parta[0]->s;

    BatchPtrTable tbl(*parta[0], LimbPartition::SC_BATCH_ADDSCALAR, data_ptrs, s);
    void*** const data_ptrs_d = tbl.d;

    if (limbsize > 0)
        add_scalar_reuse_b___<<<grid, block, 0, s.ptr()>>>(data_ptrs_d, data_ptrs_d + its * split * vector.size(),
                                                           PARTITION(parta[0]->id, 0), n, its);
}

void LimbPartition::multScalarBatchManyToOne(std::vector<LimbPartition*>& parta,
                                             const std::vector<std::vector<unsigned long int>>& vector,
                                             const std::vector<std::vector<unsigned long int>>& vector_shoup,
                                             int stride, double usage) {
    ContextData& cc = parta[0]->cc;
    int limbsize = parta[0]->getLimbSize(*parta[0]->level);
    int n = parta.size();

    dim3 block = {128u, 1u, 1u};
    int threads = GetTargetThreads(cc.GPUid[parta[0]->id]) * usage;
    dim3 grid = {cc.N / 128u, (uint32_t)limbsize, 1u};
    uint32_t num_parallel_parts = std::max(
        1u, std::min((uint32_t)((threads + block.x * grid.x * grid.y - 1) / (block.x * grid.x * std::max(1u, grid.y))),
                     (uint32_t)parta.size()));
    grid.z = num_parallel_parts;

    int its = stride;
    int split = 1;
    if (num_parallel_parts > vector.size()) {
        split = (num_parallel_parts + vector.size() - 1) / vector.size();

        its = stride / split;
        grid.z = split * vector.size();
    }

    int size = its * split * vector.size() + split * vector.size() * MAXP * 2;
    std::vector<void**> data_ptrs(size, nullptr);
    for (auto i = 0; i < vector.size(); ++i) {

        for (int j = 0; j < stride; ++j) {
            data_ptrs[its * split * i + j] = parta[stride * i + j]->limbptr.data;
        }
        for (int j = 0; j < split; ++j) {
            std::memcpy(&data_ptrs[its * split * vector.size() + (i * split + j) * MAXP], vector[i].data(),
                        vector[i].size() * sizeof(uint64_t));
        }
        for (int j = 0; j < split; ++j) {
            std::memcpy(&data_ptrs[its * split * vector.size() + split * vector.size() * MAXP + (i * split + j) * MAXP],
                        vector_shoup[i].data(), vector_shoup[i].size() * sizeof(uint64_t));
        }
    }

    Stream& s = parta[0]->s;

    BatchPtrTable tbl(*parta[0], LimbPartition::SC_BATCH_MULTSCALAR, data_ptrs, s);
    void*** const data_ptrs_d = tbl.d;

    if (limbsize > 0)
        mult_scalar_reuse_b___<<<grid, block, 0, s.ptr()>>>(
            data_ptrs_d, data_ptrs_d + its * split * vector.size(),
            data_ptrs_d + its * split * vector.size() + split * vector.size() * MAXP, PARTITION(parta[0]->id, 0), n,
            its);
}

void LimbPartition::binomialMultAccumBatch(LimbPartition& acc0, LimbPartition& acc1, LimbPartition& acc2,
                                           const std::vector<const LimbPartition*>& a0,
                                           const std::vector<const LimbPartition*>& a1,
                                           const std::vector<const LimbPartition*>& b0,
                                           const std::vector<const LimbPartition*>& b1) {
    ContextData& cc = acc0.cc;
    cudaSetDevice(acc0.device);
    const int limbsize = acc0.getLimbSize(*acc0.level);
    const int n = static_cast<int>(a0.size());
    assert((int)a1.size() == n && (int)b0.size() == n && (int)b1.size() == n);
    if (n == 0 || limbsize <= 0)
        return;

    // Pointer tables: 4 arrays of n limb-table pointers each.
    std::vector<void**> data_ptrs(4 * n, nullptr);
    for (int j = 0; j < n; ++j) {
        data_ptrs[0 * n + j] = a0[j]->limbptr.data;
        data_ptrs[1 * n + j] = a1[j]->limbptr.data;
        data_ptrs[2 * n + j] = b0[j]->limbptr.data;
        data_ptrs[3 * n + j] = b1[j]->limbptr.data;
    }

    Stream& s = acc0.s;
    BatchPtrTable tbl(acc0, LimbPartition::SC_BATCH_BINOMIAL, data_ptrs, s);
    void*** const data_ptrs_d = tbl.d;

    s.wait(acc1.s);
    s.wait(acc2.s);
    for (int j = 0; j < n; ++j) {
        s.wait(a0[j]->getS());
        s.wait(a1[j]->getS());
        s.wait(b0[j]->getS());
        s.wait(b1[j]->getS());
    }

    dim3 block = {128u, 1u, 1u};
    dim3 grid = {cc.N / 128u, (uint32_t)limbsize, 1u};
    binomialMultAccum_<<<grid, block, 0, s.ptr()>>>(PARTITION(acc0.id, 0), acc0.limbptr.data, acc1.limbptr.data,
                                                    acc2.limbptr.data, data_ptrs_d + 0 * n, data_ptrs_d + 1 * n,
                                                    data_ptrs_d + 2 * n, data_ptrs_d + 3 * n, n);

    acc1.s.wait(s);
    acc2.s.wait(s);
    for (int j = 0; j < n; ++j) {
        a0[j]->getS().wait(s);
        a1[j]->getS().wait(s);
        b0[j]->getS().wait(s);
        b1[j]->getS().wait(s);
    }
}

uint32_t g_ltPtMask = 0xFFFFFFFFu;

void LimbPartition::LTdotProductPtBatch(std::vector<LimbPartition*>& out, const std::vector<LimbPartition*>& in,
                                        const std::vector<LimbPartition*>& pt, int bStep, int gStep, int stride,
                                        double usage, bool ext, int part, int limb_begin, int limb_count) {

    constexpr bool VER2 = true;
    constexpr bool VER3 = true;

    cudaSetDevice(out[0]->device);
    ContextData& cc = out[0]->cc;
    int limbsize = out[0]->getLimbSize(*out[0]->level);
    int slimbsize = cc.splitSpecialMeta.at(out[0]->id).size();
    int special_start = 0;
    for (int i = 0; i < out[0]->id; ++i) {
        special_start += cc.splitSpecialMeta.at(i).size();
    }
    // Limb-range form (LT chunking). The kernels index every limb table with blockIdx.y and
    // derive the prime from primeid_flattened[primeidInit + blockIdx.y], so a sub-range is the
    // table entries offset by limb_begin plus primeidInit + limb_begin and a smaller grid.y.
    const bool do_regular = part != 1;
    const bool do_special = part != 0 && ext;
    int reg_begin = 0, sp_begin = 0;
    if (part == 0) {
        reg_begin = limb_begin;
        limbsize = limb_count < 0 ? limbsize - limb_begin : std::min(limb_count, limbsize - limb_begin);
        assert(limb_begin >= 0 && limbsize > 0);
    } else if (part == 1) {
        assert(ext);
        sp_begin = limb_begin;
        slimbsize = limb_count < 0 ? slimbsize - limb_begin : std::min(limb_count, slimbsize - limb_begin);
        assert(limb_begin >= 0 && slimbsize > 0);
    }

    dim3 block;
    dim3 grid;
    if constexpr (!VER2) {
        int blockdimx = 16;
        while (bStep * 2 * blockdimx > 256 && blockdimx > 2) {
            blockdimx /= 2;
        }
        block = {(uint32_t)blockdimx, 2, (uint32_t)bStep};
        grid = {(uint32_t)cc.N / blockdimx, (uint32_t)limbsize, 1u};
    } else {
        block = {(uint32_t)16, 2, 4};
        grid = {(uint32_t)cc.N / 64, (uint32_t)limbsize, 1u};
    }
    int threads = GetTargetThreads(cc.GPUid[out[0]->id]) * usage;

    uint32_t num_parallel_parts =
        std::max(1u, std::min((uint32_t)((threads + block.x * block.y * block.z * grid.x * grid.y - 1) /
                                         (block.x * block.y * block.z * grid.x * std::max(1u, grid.y))),
                              (uint32_t)out.size()));
    grid.z = num_parallel_parts;

    dim3 sgrid = grid;
    sgrid.y = slimbsize;

    assert(out.size() % (2 * gStep * stride) == 0);
    assert(in.size() % (2 * bStep * stride) == 0);
    assert(pt.size() % (gStep * bStep) == 0);
    assert(out.size() / (2 * gStep * stride) == in.size() / (2 * bStep * stride));
    assert(out.size() / (2 * gStep * stride) == pt.size() / (gStep * bStep));
    uint32_t shmem_bytes;
    int num_LT;
    if constexpr (!VER2) {
        shmem_bytes = sizeof(uint64_t) * (stride * 2 + 4 + 1) * block.x * block.z;
        num_LT = out.size() / (2 * stride * gStep);
    } else {
        shmem_bytes = sizeof(uint64_t) * (gStep * (VER3 ? 2 : 1)) * block.y * block.x * block.z;
        num_LT = out.size() / (2 * gStep);
    }

    int size = (out.size() + in.size() + pt.size()) * (1 + ext);
    int offset_out_c0 = 0;
    int offset_out_c1 = out.size() / 2;
    int offset_in_c0 = out.size();
    int offset_in_c1 = out.size() + in.size() / 2;
    int offset_pt = out.size() + in.size();
    int soffset_out_c0 = size / 2 + 0;
    int soffset_out_c1 = size / 2 + out.size() / 2;
    int soffset_in_c0 = size / 2 + out.size();
    int soffset_in_c1 = size / 2 + out.size() + in.size() / 2;
    int soffset_pt = size / 2 + out.size() + in.size();

    std::vector<void**> data_ptrs(size, nullptr);
    for (int i = 0; i < num_LT; ++i) {
        for (int j = 0; j < stride; ++j) {
            for (int k = 0; k < gStep; ++k) {
                data_ptrs[offset_out_c0 + i * stride * gStep + j * gStep + k] =
                    out[2 * (i * stride * gStep + j * gStep + k)]->limbptr.data + reg_begin;
            }
        }

        for (int j = 0; j < stride; ++j) {
            for (int k = 0; k < gStep; ++k) {
                data_ptrs[offset_out_c1 + i * stride * gStep + j * gStep + k] =
                    out[2 * (i * stride * gStep + j * gStep + k) + 1]->limbptr.data + reg_begin;
            }
        }

        for (int j = 0; j < stride; ++j) {
            for (int k = 0; k < bStep; ++k) {
                data_ptrs[offset_in_c0 + i * stride * bStep + j * bStep + k] =
                    in[2 * (i * stride * bStep + j * bStep + k)]->limbptr.data + reg_begin;
            }
        }

        for (int j = 0; j < stride; ++j) {
            for (int k = 0; k < bStep; ++k) {
                data_ptrs[offset_in_c1 + i * stride * bStep + j * bStep + k] =
                    in[2 * (i * stride * bStep + j * bStep + k) + 1]->limbptr.data + reg_begin;
            }
        }

        for (int j = 0; j < gStep; ++j) {
            for (int k = 0; k < bStep; ++k) {
                data_ptrs[offset_pt + i * gStep * bStep + j * bStep + k] =
                    pt[i * gStep * bStep + j * bStep + k]
                        ? pt[i * gStep * bStep + j * bStep + k]->limbptr.data + reg_begin
                        : nullptr;
            }
        }

        if (ext) {
            for (int j = 0; j < stride; ++j) {
                for (int k = 0; k < gStep; ++k) {
                    data_ptrs[soffset_out_c0 + i * stride * gStep + j * gStep + k] =
                        out[2 * (i * stride * gStep + j * gStep + k)]->SPECIALlimbptr.data + special_start + sp_begin;
                }
            }

            for (int j = 0; j < stride; ++j) {
                for (int k = 0; k < gStep; ++k) {
                    data_ptrs[soffset_out_c1 + i * stride * gStep + j * gStep + k] =
                        out[2 * (i * stride * gStep + j * gStep + k) + 1]->SPECIALlimbptr.data + special_start +
                        sp_begin;
                }
            }

            for (int j = 0; j < stride; ++j) {
                for (int k = 0; k < bStep; ++k) {
                    data_ptrs[soffset_in_c0 + i * stride * bStep + j * bStep + k] =
                        in[2 * (i * stride * bStep + j * bStep + k)]->SPECIALlimbptr.data + special_start + sp_begin;
                }
            }

            for (int j = 0; j < stride; ++j) {
                for (int k = 0; k < bStep; ++k) {
                    data_ptrs[soffset_in_c1 + i * stride * bStep + j * bStep + k] =
                        in[2 * (i * stride * bStep + j * bStep + k) + 1]->SPECIALlimbptr.data + special_start +
                        sp_begin;
                }
            }

            for (int j = 0; j < gStep; ++j) {
                for (int k = 0; k < bStep; ++k) {
                    data_ptrs[soffset_pt + i * gStep * bStep + j * bStep + k] =
                        pt[i * gStep * bStep + j * bStep + k]
                            ? pt[i * gStep * bStep + j * bStep + k]->SPECIALlimbptr.data + sp_begin
                            : nullptr;
                }
            }
        }
    }

    Stream& s = in[0]->s;

    BatchPtrTable tbl(*in[0], LimbPartition::SC_BATCH_LTDOT, data_ptrs, s);
    void*** const data_ptrs_d = tbl.d;
    s.wait(out[0]->s);
    s.wait(pt[0]->s);

    if constexpr (!VER2) {
        if (do_regular && limbsize > 0) {
            dotProductLtBatchedPt___<<<grid, block, shmem_bytes, s.ptr()>>>(
                data_ptrs_d + offset_out_c0, data_ptrs_d + offset_out_c1, data_ptrs_d + offset_in_c0,
                data_ptrs_d + offset_in_c1, data_ptrs_d + offset_pt, stride, gStep, PARTITION(out[0]->id, 0) + reg_begin, num_LT);
        }
        if (do_special && slimbsize > 0) {
            dotProductLtBatchedPt___<<<sgrid, block, shmem_bytes, s.ptr()>>>(
                data_ptrs_d + soffset_out_c0, data_ptrs_d + soffset_out_c1, data_ptrs_d + soffset_in_c0,
                data_ptrs_d + soffset_in_c1, data_ptrs_d + soffset_pt, stride, gStep,
                SPECIAL(out[0]->id, special_start) + sp_begin, num_LT);
        }
    } else {
        if constexpr (!VER3) {
            if (do_regular && limbsize > 0) {
                dotProductLtBatchedPt2___<<<grid, block, shmem_bytes, s.ptr()>>>(
                    data_ptrs_d + offset_out_c0, data_ptrs_d + offset_out_c1, data_ptrs_d + offset_in_c0,
                    data_ptrs_d + offset_in_c1, data_ptrs_d + offset_pt, bStep, gStep, PARTITION(out[0]->id, 0) + reg_begin,
                    num_LT);
            }
            if (do_special && slimbsize > 0) {
                dotProductLtBatchedPt2___<<<sgrid, block, shmem_bytes, s.ptr()>>>(
                    data_ptrs_d + soffset_out_c0, data_ptrs_d + soffset_out_c1, data_ptrs_d + soffset_in_c0,
                    data_ptrs_d + soffset_in_c1, data_ptrs_d + soffset_pt, bStep, gStep,
                    SPECIAL(out[0]->id, special_start) + sp_begin, num_LT);
            }
        } else {
            if (do_regular && limbsize > 0) {
                dotProductLtBatchedPt3___<<<grid, block, shmem_bytes, s.ptr()>>>(
                    data_ptrs_d + offset_out_c0, data_ptrs_d + offset_out_c1, data_ptrs_d + offset_in_c0,
                    data_ptrs_d + offset_in_c1, data_ptrs_d + offset_pt, bStep, gStep, PARTITION(out[0]->id, 0) + reg_begin,
                    num_LT, g_ltPtMask);
            }
            if (do_special && slimbsize > 0) {
                dotProductLtBatchedPt3___<<<sgrid, block, shmem_bytes, s.ptr()>>>(
                    data_ptrs_d + soffset_out_c0, data_ptrs_d + soffset_out_c1, data_ptrs_d + soffset_in_c0,
                    data_ptrs_d + soffset_in_c1, data_ptrs_d + soffset_pt, bStep, gStep,
                    SPECIAL(out[0]->id, special_start) + sp_begin, num_LT, g_ltPtMask);
            }
        }
    }
    out[0]->s.wait(s);
    pt[0]->s.wait(s);
}

void LimbPartition::fusedHoistedRotateBatch(std::vector<LimbPartition*>& out, const std::vector<LimbPartition*>& in,
                                            const std::vector<LimbPartition*>& ksk_a,
                                            const std::vector<LimbPartition*>& ksk_b, const std::vector<int>& indexes,
                                            int n, int stride, double usage, bool c0_modup) {
    ContextData& cc = out[0]->cc;
    int limbsize = out[0]->getLimbSize(*out[0]->level);
    int slimbsize = out[0]->SPECIALlimb.size();
    int blockdimx = 64;
    while (stride * 2 * blockdimx > 256 && blockdimx > 2) {
        blockdimx /= 2;
    }

    dim3 block = {(uint32_t)blockdimx, 2, (uint32_t)stride};

    int threads = GetTargetThreads(cc.GPUid[out[0]->id]) * usage;
    dim3 grid = {(uint32_t)cc.N / blockdimx, (uint32_t)slimbsize + limbsize, 1u};

    uint32_t num_parallel_parts =
        std::max(1u, std::min((uint32_t)((threads + block.x * block.y * block.z * grid.x * grid.y - 1) /
                                         (block.x * block.y * block.z * grid.x * std::max(1u, grid.y))),
                              (uint32_t)out.size()));
    //grid.z = num_parallel_parts;
    if (num_parallel_parts > 1) {
        std::cerr << "fusedHoistRotateBatch detected underutilization, should optimize" << std::endl;
    }

    dim3 sgrid = grid;
    //sgrid.y = slimbsize;

    size_t num_d = 0;

    for (; num_d < in[1]->DIGITlimb.size(); ++num_d) {
        {
            int start = 0;
            for (int j = 0; j < num_d; ++j)
                start += in[1]->DECOMPmeta[j].size();
            int size = std::min((int)in[1]->DECOMPmeta[num_d].size(), (int)limbsize - start);
            if (size <= 0)
                break;
        }
    }

    uint32_t shmem_bytes = sizeof(uint64_t) * (num_d * block.x * block.z + 2 * num_d * block.x);

    assert(out.size() == in.size() * n);
    assert(ksk_a.size() * in.size() == out.size());
    assert(ksk_b.size() * in.size() == out.size());
    for (auto* k : ksk_a)
        if (k && k->key_pack_bits)
            throw std::runtime_error(
                "fusedHoistedRotateBatch: hoistedRotateDotKSKBatched___ is not packed-key aware "
                "(FIDESLIB_KSK_PACK=0 to disable)");

    int size = 2 * out.size() + in.size() + (num_d + 1) * in.size() / 2 + (num_d + 1) * (ksk_a.size() + ksk_b.size()) +
               (indexes.size() + 1) / 2;
    int offset_out_c0 = 0;
    int offset_out_sc0 = out.size() / 2;
    int offset_out_c1 = out.size();
    int offset_out_sc1 = 3 * out.size() / 2;
    int offset_in_c0 = 2 * out.size();
    int offset_in_sc0 = 2 * out.size() + in.size() / 2;
    int offset_in_c1 = 2 * out.size() + 2 * in.size() / 2;
    int offset_in_dc1 = 2 * out.size() + 3 * in.size() / 2;
    int offset_ksk = 2 * out.size() + (3 + num_d) * in.size() / 2;
    int offset_index = 2 * out.size() + (3 + num_d) * in.size() / 2 + (num_d + 1) * (ksk_a.size() + ksk_b.size());

    std::vector<void**> data_ptrs(size, nullptr);

    for (int j = 0; j < out.size() / 2; ++j) {
        data_ptrs[offset_out_c0 + j] = out[2 * j]->limbptr.data;
        data_ptrs[offset_out_sc0 + j] = out[2 * j]->SPECIALlimbptr.data;
        data_ptrs[offset_out_c1 + j] = out[2 * j + 1]->limbptr.data;
        data_ptrs[offset_out_sc1 + j] = out[2 * j + 1]->SPECIALlimbptr.data;
    }

    for (int j = 0; j < in.size() / 2; ++j) {
        data_ptrs[offset_in_c0 + j] = in[2 * j]->limbptr.data;
        data_ptrs[offset_in_sc0 + j] = in[2 * j]->SPECIALlimbptr.data;
        data_ptrs[offset_in_c1 + j] = in[2 * j + 1]->limbptr.data;
        for (int k = 0; k < num_d; ++k)
            data_ptrs[offset_in_dc1 + num_d * j + k] = in[2 * j + 1]->DIGITlimbptr[k].data;
    }

    for (int j = 0; j < ksk_a.size(); ++j) {
        for (int k = 0; k < num_d; ++k) {
            data_ptrs[offset_ksk + (num_d + 1) * 2 * j + k] = ksk_a[j] ? ksk_a[j]->DIGITlimbptr[k].data : nullptr;
            data_ptrs[offset_ksk + (num_d + 1) * (2 * j + 1) + k] = ksk_b[j] ? ksk_b[j]->DIGITlimbptr[k].data : nullptr;
        }

        data_ptrs[offset_ksk + (num_d + 1) * 2 * j + num_d] = ksk_a[j] ? ksk_a[j]->limbptr.data : nullptr;
        data_ptrs[offset_ksk + (num_d + 1) * (2 * j + 1) + num_d] = ksk_b[j] ? ksk_b[j]->limbptr.data : nullptr;
        ((int*)&data_ptrs[offset_index])[j] = indexes[j];
    }

    Stream& s = in[0]->s;

    BatchPtrTable tbl(*in[0], LimbPartition::SC_BATCH_HOISTROT, data_ptrs, s);
    void*** const data_ptrs_d = tbl.d;
    s.wait(out[0]->s);
    s.wait(ksk_a[0] ? ksk_a[0]->s : ksk_a[1]->s);

    if (limbsize + slimbsize > 0)
        hoistedRotateDotKSKBatched___<<<grid, block, shmem_bytes, s.ptr()>>>(
            data_ptrs_d + offset_in_c1, data_ptrs_d + offset_in_dc1, data_ptrs_d + offset_in_c0,
            data_ptrs_d + offset_in_sc0, data_ptrs_d + offset_out_c1, data_ptrs_d + offset_out_sc1,
            data_ptrs_d + offset_out_c0, data_ptrs_d + offset_out_sc0, n, (int*)(data_ptrs_d + offset_index),
            data_ptrs_d + offset_ksk, num_d, out[0]->id, slimbsize, 0, c0_modup);

    out[0]->s.wait(s);
    (ksk_a[0] ? ksk_a[0]->s : ksk_a[1]->s).wait(s);
}

}  // namespace FIDESlib::CKKS