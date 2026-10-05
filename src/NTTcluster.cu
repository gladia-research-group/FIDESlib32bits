#include "NTTcluster.cuh"
#include <cooperative_groups.h>
#include <cstdlib>
#include "AddSub.cuh"
#include "ConstantsGPU.cuh"
#include "ModMult.cuh"
#include "NTT.cuh"
#include "NTThelper.cuh"
#include "NTTcore.cuh"

namespace cg = cooperative_groups;

namespace FIDESlib {

static int g_ntt_cluster = -1;
int nttClusterFlag() {
    if (g_ntt_cluster < 0) {
        const char* e = std::getenv("FIDESLIB_NTT_CLUSTER");
        g_ntt_cluster = e ? std::atoi(e) : 0;
    }
    return g_ntt_cluster;
}
void setNttCluster(int v) {
    g_ntt_cluster = v;
}

// ---- geometry (mirrors NTT.cu for u32, logN = 16: block 128 threads x M=8 int2 pairs = 2048 elements) ----
namespace ntc {
// ---------------------------------------------------------------------------------------------------
template <bool INVERSE>
__global__ void __cluster_dims__(CL, 1, 1) __launch_bounds__(GROUPS* BD, 2)
    ntt_cluster_(const Global::Globals* Globals, void** __restrict__ dat, const int __grid_constant__ primeid_init,
                 void** __restrict__ res) {
    __shared__ __align__(16) T region[REG_I2 * 2];  // 4 tiles x 2048 u32 = 32 KB: pass-1 output / working tiles
    __shared__ T psi[BD], psi_sh[BD];

    cg::cluster_group cluster = cg::this_cluster();
    const int limb = blockIdx.x / CL;
    const int rank = (int)cluster.block_rank();
    const int grp = threadIdx.x >> LOGBD;
    const int tid = threadIdx.x & (BD - 1);
    const int j = tid << 1;
    const int bx = rank * GROUPS + grp;  // this group's tile (== the two-pass kernel's blockIdx.x)
    T* gbuf = region + grp * TILE_T;
    const T* in = (const T*)dat[limb];
    T* out = (T*)res[limb];
    const bool hole = (in == nullptr);  // whole cluster sees the same row: uniform
    const int primeid = hole ? 0 : C_.primeid_flattened[primeid_init + limb];

    if (!hole) {
        if (threadIdx.x < BD) {
            psi[tid] = ((T*)(INVERSE ? G_->inv_psi[primeid] : G_->psi[primeid]))[tid];
            psi_sh[tid] = ((T*)(INVERSE ? G_->inv_psi_shoup[primeid] : G_->psi_shoup[primeid]))[tid];
        }
        if constexpr (!INVERSE) {
            // ---- pass 1: transposed load of this tile from global, scale, CT stages, EOT store into own slot
            int4 temp[4];
            const int col_init = j & ~2;
            for (int i = 0; i < M / 2; ++i) {
                const int4 v = NTC_LD((const int4*)in + (transp_i2(col_init + i, bx, j) >> 1));
                ((T*)&temp[0])[i] = v.x;
                ((T*)&temp[1])[i] = v.y;
                ((T*)&temp[2])[i] = v.z;
                ((T*)&temp[3])[i] = v.w;
            }
            __syncthreads();  // psi visible
            store_transposed_regs(gbuf, temp, j);
            __syncthreads();
            fwd_negacyclic(gbuf, primeid, psi, psi_sh, Globals, tid, bx);
            ct_stages(gbuf, psi, psi_sh, primeid, tid);
            ntt_eot_store_inplace(gbuf, psi, psi_sh, primeid, Globals, tid, bx);
        } else {
            // ---- pass 1: contiguous load of this tile, GS stages, transposed scatter into the owners' slots
            for (int i = 0; i < M; ++i) {
                int2 v = NTC_LD((const int2*)in + (TILE_I2 * bx + BD * i + tid));
                const int pos = swz_pos<T>(i, j);
                if (pos & 1) {
                    const int t_ = v.x;
                    v.x = v.y;
                    v.y = t_;
                }
                ((int2*)GA(i))[pos >> 1] = v;
            }
            __syncthreads();
            gs_stages(gbuf, psi, psi_sh, primeid, tid);
            __syncthreads();
            int4 outv[M / 2];
            const int col_init = j & ~2;
            for (int i = 0; i < M / 2; ++i) {
                const int pos_res = col_init + i;
                outv[i].x = (int)GAS(2 * (j & 2), pos_res);
                outv[i].y = (int)GAS(2 * (j & 2) + 1, pos_res);
                outv[i].z = (int)GAS(2 * (j & 2) + 2, pos_res);
                outv[i].w = (int)GAS(2 * (j & 2) + 3, pos_res);
            }
            cluster.sync();  // every block is done with its working tiles -> the slots may be overwritten
            for (int i = 0; i < M / 2; ++i) {
                const int q2 = transp_i2(col_init + i, bx, j);  // logical int2 index of this int4
                T* owner = cluster.map_shared_rank(region, q2 / REG_I2);
                ((int4*)owner)[(q2 % REG_I2) >> 1] = outv[i];
            }
        }
    }
    cluster.sync();  // pass-1 results are in place across the cluster
    if (!hole) {
        if constexpr (!INVERSE) {
            // ---- pass 2: gather this tile transposed from the 8 slots, CT stages, contiguous store
            int4 temp[4];
            const int col_init = j & ~2;
            for (int i = 0; i < M / 2; ++i) {
                const int q2 = transp_i2(col_init + i, bx, j);
                const T* owner = cluster.map_shared_rank(region, q2 / REG_I2);
                const int4 v = ((const int4*)owner)[(q2 % REG_I2) >> 1];
                ((T*)&temp[0])[i] = v.x;
                ((T*)&temp[1])[i] = v.y;
                ((T*)&temp[2])[i] = v.z;
                ((T*)&temp[3])[i] = v.w;
            }
            cluster.sync();  // all gathers done: the region is free again
            store_transposed_regs(gbuf, temp, j);
            __syncthreads();
            ct_stages(gbuf, psi, psi_sh, primeid, tid);
            __syncthreads();
            for (int i = 0; i < M; ++i) {
                const int pos = swz_pos<T>(i, j);
                int2 o = ((int2*)GA(i))[pos >> 1];
                if (pos & 1) {
                    const int t_ = o.x;
                    o.x = o.y;
                    o.y = t_;
                }
                ((int2*)out)[TILE_I2 * bx + BD * i + tid] = o;
            }
        } else {
            // ---- pass 2: own slot holds this tile contiguous; re-swizzle in place, EOT, GS, scale, transposed store
            int2 v[M];
            for (int i = 0; i < M; ++i)
                v[i] = ((int2*)gbuf)[BD * i + tid];
            __syncthreads();
            for (int i = 0; i < M; ++i) {
                const int pos = swz_pos<T>(i, j);
                int2 w = v[i];
                if (pos & 1) {
                    const int t_ = w.x;
                    w.x = w.y;
                    w.y = t_;
                }
                ((int2*)GA(i))[pos >> 1] = w;
            }
            __syncthreads();
            intt_eot_premult(gbuf, psi, psi_sh, primeid, Globals, tid, bx);
            gs_stages(gbuf, psi, psi_sh, primeid, tid);
            bwd_negacyclic(gbuf, primeid, psi, psi_sh, Globals, tid, bx);
            __syncthreads();
            const int col_init = j & ~2;
            for (int i = 0; i < M / 2; ++i) {
                const int pos_res = col_init + i;
                int4 o;
                o.x = (int)GAS(2 * (j & 2), pos_res);
                o.y = (int)GAS(2 * (j & 2) + 1, pos_res);
                o.z = (int)GAS(2 * (j & 2) + 2, pos_res);
                o.w = (int)GAS(2 * (j & 2) + 3, pos_res);
                ((int4*)out)[transp_i2(pos_res, bx, j) >> 1] = o;
            }
        }
    } else {
        cluster.sync();  // a hole is a whole limb, so the whole cluster takes this branch: 2 syncs either way
    }
}
#undef GA
#undef GAS
}  // namespace ntc

void launchNTTcluster(const Global::Globals* Globals, bool inverse, void** dat, int primeid_init, void** res, int nlimbs,
                      cudaStream_t s) {
    if (nlimbs <= 0)
        return;
    const dim3 grid((unsigned)(ntc::CL * nlimbs)), block((unsigned)(ntc::GROUPS * ntc::BD));
    if (inverse)
        ntc::ntt_cluster_<true><<<grid, block, 0, s>>>(Globals, dat, primeid_init, res);
    else
        ntc::ntt_cluster_<false><<<grid, block, 0, s>>>(Globals, dat, primeid_init, res);
}

}  // namespace FIDESlib
