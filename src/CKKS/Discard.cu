#include "CKKS/Discard.cuh"
#include "CKKS/LimbPartition.cuh"
#include "CKKS/Context.cuh"
#include "NTT.cuh"
#include <cstdlib>

namespace FIDESlib::CKKS {

int discardD2Flag() {
    static const int v = [] { const char* e = std::getenv("FIDESLIB_DISCARD_D2"); return e ? std::atoi(e) : 0; }();
    return v;
}

static int g_modup_merge = -1;
int modupMergeFlag() {
    if (g_modup_merge < 0) { const char* e = std::getenv("FIDESLIB_MODUP_MERGE"); g_modup_merge = e ? std::atoi(e) : 0; }
    return g_modup_merge;
}
void setModupMerge(int v) { g_modup_merge = v; }
static int g_fused_rescale = -1;
int fusedRescaleFlag() {
    if (g_fused_rescale < 0) { const char* e = std::getenv("FIDESLIB_FUSED_RESCALE"); g_fused_rescale = e ? std::atoi(e) : 0; }
    return g_fused_rescale;
}
void setFusedRescale(int v) { g_fused_rescale = v; }
static int g_pw_fuse = -1;
int pwFuseFlag() {
    if (g_pw_fuse < 0) { const char* e = std::getenv("FIDESLIB_PW_FUSE"); g_pw_fuse = e ? std::atoi(e) : 0; }
    return g_pw_fuse;
}
void setPwFuse(int v) { g_pw_fuse = v; }

// tables[k][begin + limb] for k < ntab, limb < n: one thread per 128-B line
__global__ void discard_tables_(void*** tabs, const int ntab, const int begin, const int n, const size_t lines) {
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < (size_t)ntab * n * lines) {
        const size_t k = idx / (n * lines), rem = idx % (n * lines);
        const char* p = (const char*)tabs[k][begin + rem / lines] + (rem % lines) * 128;
        asm volatile("discard.global.L2 [%0], 128;" ::"l"(p) : "memory");
    }
}

void discardLimbTables(const std::vector<void**>& tables, int begin, int n, size_t bytes, cudaStream_t s) {
    if (tables.empty() || n <= 0 || !discardScratchFlag())
        return;
    void*** d = nullptr;
    cudaMallocAsync((void**)&d, tables.size() * sizeof(void**), s);
    cudaMemcpyAsync(d, tables.data(), tables.size() * sizeof(void**), cudaMemcpyHostToDevice, s);
    const size_t lines = bytes / 128, total = tables.size() * (size_t)n * lines;
    discard_tables_<<<(unsigned)((total + 255) / 256), 256, 0, s>>>(d, (int)tables.size(), begin, n, lines);
    cudaFreeAsync(d, s);
}

__global__ void discard_table_(void** tbl, const int n, const size_t lines) {
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < (size_t)n * lines) {
        const char* p = (const char*)tbl[idx / lines] + (idx % lines) * 128;
        asm volatile("discard.global.L2 [%0], 128;" ::"l"(p) : "memory");
    }
}

void discardLimbTable(void** d_limbptr, int n, size_t bytes, cudaStream_t s) {
    if (n <= 0 || !discardScratchFlag())
        return;
    const size_t lines = bytes / 128, total = (size_t)n * lines;
    discard_table_<<<(unsigned)((total + 255) / 256), 256, 0, s>>>(d_limbptr, n, lines);
}

static size_t limbBytes(const LimbPartition& p) {
    return (size_t)p.cc.N * (p.cc.precom.constants[0].type == 0 ? 4 : 8);
}

void discardRegularLimbs(LimbPartition& p, cudaStream_t s) {
    discardLimbTable(p.limbptr.data, p.getLimbSize(*p.level), limbBytes(p), s);
}

void discardSpecialLimbs(LimbPartition& p, cudaStream_t s) {
    discardLimbTable(p.SPECIALlimbptr.data, (int)p.SPECIALlimb.size(), limbBytes(p), s);
}

void discardDigitLimbs(LimbPartition& p, cudaStream_t s) {
    for (size_t d = 0; d < p.DIGITlimb.size(); ++d)
        discardLimbTable(p.DIGITlimbptr[d].data, (int)p.DIGITlimb[d].size(), limbBytes(p), s);
}

}  // namespace FIDESlib::CKKS
