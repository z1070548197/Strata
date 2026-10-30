// tests/core/tp_graph_test.cu - Phase 2f: two-GPU CUDA-graph tensor-parallel prototype.
//
// Validates the NOVEL machinery that Phase 2a/2b/real-split did not cover:
//   * TWO CUDA graphs, one captured per device, launched CONCURRENTLY.
//   * Each graph runs all 48 layers: row-parallel GEMV (native_mmvq on half the
//     weights) + single-writer allreduce (tp_allreduce) + a SPIN-WAIT FLAG handshake
//     (the engine's proven cross-device sync pattern: a 1-thread kernel that spins on
//     a flag in mapped host memory, captured INSIDE the graph).
//   * Correctness: the two-graph result must match a single-GPU reference that runs
//     the same 48 layers sequentially with the full weight.
//
// Why spin-wait flags (not cudaEvent cross-device): a CUDA graph is single-device.
// The engine already solves cross-device ordering inside a graph with spin-wait flag
// kernels reading mapped host memory (wait_handoff / handoff_publish in
// verify_kernels.cu). We reuse that exact pattern: each graph raises a flag after its
// allreduce and spins on the other graph's flag before its next layer.
//
// Per-layer handshake (row-parallel + allreduce, single-writer on dev_a=0):
//   dev0 stream: gemv0 -> wait(flag_b) -> allreduce -> raise(flag_a) -> quantize+split
//   dev1 stream: gemv1 -> raise(flag_b) -> wait(flag_a) -> quantize+split
//   - dev0's allreduce reads dev1's partial sum via P2P, so it must wait for dev1's
//     gemv1 to finish  => dev0 spins on flag_b (raised by dev1 after gemv1).
//   - after allreduce both buffers hold the full sum; dev1 must wait for the write to
//     land before its next gemv  => dev1 spins on flag_a (raised by dev0 after allreduce).
//
// Quantization: the engine's own quantize_q8_1_rows runs on the FULL vector (both
// devices), then the quantized blocks are split by device (block boundaries align with
// the half boundary). This keeps the activation quantization identical to the reference
// and isolates the test to the graph/allreduce/flag machinery.
//
// Build (manual, not in CMakeLists):
//   nvcc -O2 -std=c++17 --use_fast_math -I include -I src -I third_party/ggml \
//     tests/core/tp_graph_test.cu src/core/tp_allreduce.cu src/kernels/cuda/native_mmvq.cu \
//     src/kernels/cuda/iq_kernels.cu -o build/tp_graph_test -lcuda
//
// PDL stubs: native_mmvq.cu references pdl_scope()/pdl_launch_ok() (defined in
// verify_kernels.cu; PDL is sm_90+ only, never active on sm_75).
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "strata/core/tp_allreduce.hpp"
#include "strata/kernels/iq_kernels.hpp"    // quantize_q8_1_rows
#include "strata/kernels/native_mmvq.hpp"   // native_mmvq
#include "strata/kernels/dp4a.hpp"          // strata_spin_pause

namespace strata::kernels {
// PDL stubs: native_mmvq.cu references these (defined in verify_kernels.cu; PDL is
// sm_90+ only, never active on sm_75).  Provide them so the standalone link resolves.
bool& pdl_scope() { static bool on = false; return on; }
bool pdl_launch_ok(const void*, cudaStream_t) { return false; }
}  // namespace strata::kernels

namespace {

constexpr int DEV_A = 0;   // single-writer allreduce device
constexpr int DEV_B = 1;
constexpr int N_IN = 2560;      // per-layer reduction axis (full)
constexpr int N_OUT = 2560;     // per-layer output dim (= next layer's n_in, so it chains)
constexpr int HALF = N_IN / 2;  // row-parallel split: 1280 + 1280
constexpr int N_LAYERS = 48;    // full model depth
constexpr int Q80_ELEMS = 32;   // Q8_0 block elems
constexpr int Q80_BYTES = 34;   // Q8_0 block bytes
constexpr int Q81_ELEMS = 32;   // Q8_1 block elems
constexpr int Q81_BYTES = 36;   // Q8_1 block bytes
constexpr unsigned long long WAIT_TIMEOUT_NS = 60ull * 1000ull * 1000ull * 1000ull;   // 60 s

// attn_qkv is 10240 rows x 2560 cols (each row = 2560 elems = 80 Q8_0 blocks = 2720 bytes).
// We take the first 2560 rows => a clean 2560 x 2560 weight (N_OUT x N_IN), each row exactly
// N_IN elements. (ssm_out is 2560 x 6144, so its rows are 6144 elems - not a clean N_IN slice.)
const char* W_PATH = "/media/zc/软件1/strata-nvfp4/tests/core/tp_data/blk_0_attn_qkv_weight.q8_0";

[[noreturn]] void die(const std::string& m) { std::fprintf(stderr, "FATAL: %s\n", m.c_str()); std::exit(1); }
void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) die(std::string(what) + ": " + cudaGetErrorString(e));
}
bool check(bool ok, const char* what) {
    std::fprintf(stderr, "  [%s] %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) std::exit(1);
    return ok;
}

std::vector<uint8_t> read_file(const char* p) {
    FILE* f = std::fopen(p, "rb");
    if (!f) die(std::string("cannot open ") + p);
    std::fseek(f, 0, SEEK_END);
    long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> b((size_t) n);
    if (std::fread(b.data(), 1, (size_t) n, f) != (size_t) n) die("short read " + std::string(p));
    std::fclose(f);
    return b;
}

// deterministic activation in [-1, 1]
void prand(float* x, int64_t n, uint32_t seed) {
    uint32_t s = seed;
    for (int64_t i = 0; i < n; ++i) {
        s = s * 1664525u + 1013904223u;
        x[i] = ((float)(s & 0xFFFFFF) / 8388608.0f) - 1.0f;
    }
}

double rel_err(const float* a, const float* b, int64_t n) {
    double num = 0, den = 0;
    for (int64_t i = 0; i < n; ++i) { num += std::fabs((double) a[i] - (double) b[i]); den += std::fabs((double) b[i]); }
    return den > 0 ? num / den : (num > 0 ? 1.0 : 0.0);
}

// ---- spin-wait flag kernels (the engine's cross-device sync pattern, self-contained) ----
// spin_pause = the engine's own strata_spin_pause (dp4a.hpp): __nanosleep(100) on sm_70+.
__device__ __forceinline__ unsigned long long now_ns() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}
// bounded wait: spin until *flag >= value, or set *err = 1 after timeout_ns (0: never).
__global__ void wait_flag_ge_kernel(const volatile uint32_t* flag, uint32_t value, uint32_t* err,
                                    unsigned long long timeout_ns) {
    const unsigned long long t0 = timeout_ns ? now_ns() : 0ull;
    while (*flag < value) {
        strata_spin_pause();
        if (timeout_ns && now_ns() - t0 > timeout_ns) {
            *(volatile uint32_t*) err = 1u;
            __threadfence_system();
            return;
        }
    }
    __threadfence_system();
}
__global__ void raise_flag_kernel(volatile uint32_t* flag, uint32_t value) {
    *flag = value;
    __threadfence_system();
}
void wait_flag_ge(const uint32_t* flag, uint32_t value, uint32_t* err, unsigned long long timeout_ns, cudaStream_t s) {
    wait_flag_ge_kernel<<<1, 1, 0, s>>>(flag, value, err, timeout_ns);
    ck(cudaGetLastError(), "wait_flag_ge launch");
}
void raise_flag(uint32_t* flag, uint32_t value, cudaStream_t s) {
    raise_flag_kernel<<<1, 1, 0, s>>>(flag, value);
    ck(cudaGetLastError(), "raise_flag launch");
}

// ---- device-side "take half": from the FULL quantized xq (N_IN/Q81_ELEMS blocks),
// ---- copy the first/last HALF/Q81_ELEMS blocks to the device's own xq buffer.
__global__ void take_half_q81_kernel(const uint8_t* __restrict__ src, int64_t half_blocks, int which,
                                     uint8_t* __restrict__ dst) {
    const int64_t blk = blockIdx.x;
    if (blk >= half_blocks) return;
    const int64_t src_blk = (which ? half_blocks : 0) + blk;
    std::memcpy(dst + blk * Q81_BYTES, src + src_blk * Q81_BYTES, Q81_BYTES);
}
void take_half_q81(const uint8_t* src, int64_t half_blocks, int which, uint8_t* dst, cudaStream_t s) {
    take_half_q81_kernel<<<(unsigned) half_blocks, 32, 0, s>>>(src, half_blocks, which, dst);
    ck(cudaGetLastError(), "take_half_q81 launch");
}

}  // namespace

int main() {
    using namespace strata::kernels;
    int count = 0;
    ck(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
    if (count < 2) die("need 2 GPUs");

    // ---- load the real Q8_0 weight (ssm_out 6144x2560), slice to N_OUT x N_IN (2560x2560) ----
    // ssm_out is 6144 rows x 2560 cols. Each row = 2560/32 = 80 Q8_0 blocks = 80*34 = 2720 bytes.
    // Take the first 2560 rows => 2560*2720 = 6,963,200 bytes = W (N_OUT x N_IN).
    const int64_t row_bytes = (N_IN / Q80_ELEMS) * Q80_BYTES;   // 2720
    const int64_t w_bytes = (int64_t) N_OUT * row_bytes;        // 6,963,200
    std::vector<uint8_t> wfull = read_file(W_PATH);
    if ((int64_t) wfull.size() < w_bytes) die("ssm_out weight too small for 2560x2560 slice");
    std::vector<uint8_t> W(wfull.begin(), wfull.begin() + w_bytes);

    // row-parallel split of W (N_OUT x N_IN) on the n_in axis at the Q8_0 block boundary:
    //   W0 = W[:, 0:HALF]   (N_OUT x HALF)  -> first HALF/32 blocks per row
    //   W1 = W[:, HALF:N_IN] (N_OUT x HALF) -> last  HALF/32 blocks per row
    const int64_t half_blocks = HALF / Q80_ELEMS;   // 40 blocks
    const int64_t half_row_bytes = half_blocks * Q80_BYTES;   // 1360
    std::vector<uint8_t> W0((size_t) N_OUT * half_row_bytes), W1((size_t) N_OUT * half_row_bytes);
    for (int64_t r = 0; r < N_OUT; ++r) {
        const uint8_t* src = W.data() + r * row_bytes;
        std::memcpy(W0.data() + r * half_row_bytes, src, half_row_bytes);
        std::memcpy(W1.data() + r * half_row_bytes, src + half_row_bytes, half_row_bytes);
    }

    // ---- reference: single GPU0, full weight, N_LAYERS sequential ----
    // Also keep a per-layer reference trace (ref_layer[l] = x after layer l) for debugging.
    const int dbg_layers = N_LAYERS;
    std::vector<std::vector<float>> ref_layer(dbg_layers + 1, std::vector<float>(N_OUT));
    std::vector<float> xref(N_IN);
    prand(xref.data(), N_IN, 0x1234);
    ref_layer[0] = xref;
    {
        cudaSetDevice(DEV_A);
        float* d_x = nullptr;
        ck(cudaMalloc(&d_x, N_IN * sizeof(float)), "ref malloc x");
        float* d_y = nullptr;
        ck(cudaMalloc(&d_y, N_OUT * sizeof(float)), "ref malloc y");
        void* d_xq = nullptr;
        ck(cudaMalloc(&d_xq, (N_IN / Q81_ELEMS) * Q81_BYTES), "ref malloc xq");
        void* d_W = nullptr;
        ck(cudaMalloc(&d_W, W.size()), "ref malloc W");
        ck(cudaMemcpy(d_W, W.data(), W.size(), cudaMemcpyHostToDevice), "ref copy W");
        cudaStream_t s;
        ck(cudaStreamCreate(&s), "ref stream");
        ck(cudaMemcpy(d_x, xref.data(), N_IN * sizeof(float), cudaMemcpyHostToDevice), "ref copy x");
        for (int l = 0; l < N_LAYERS; ++l) {
            quantize_q8_1_rows(d_x, 1, N_IN, d_xq, s);
            native_mmvq(8, d_W, d_xq, d_y, N_IN, N_OUT, 1, s);
            ck(cudaMemcpy(d_x, d_y, N_OUT * sizeof(float), cudaMemcpyDeviceToDevice), "ref d2d");
            if (l < dbg_layers) {
                ck(cudaMemcpy(ref_layer[l + 1].data(), d_y, N_OUT * sizeof(float), cudaMemcpyDeviceToHost), "ref trace");
            }
        }
        ck(cudaStreamSynchronize(s), "ref sync");
        // NOTE: do NOT copy d_x back into xref - xref must stay the ORIGINAL input so the
        // TP can start from the same state. The 48-layer output is already in ref_layer[N_LAYERS].
        cudaFree(d_x); cudaFree(d_y); cudaFree(d_xq); cudaFree(d_W);
        cudaStreamDestroy(s);
    }
    std::fprintf(stderr, "reference: xref[0..3] = %.6f %.6f %.6f %.6f\n", xref[0], xref[1], xref[2], xref[3]);
    for (int l = 1; l <= 4 && l <= dbg_layers; ++l) {
        std::fprintf(stderr, "  ref layer %d: [0..3] = %.6f %.6f %.6f %.6f\n", l,
                     ref_layer[l][0], ref_layer[l][1], ref_layer[l][2], ref_layer[l][3]);
    }

    // ---- MANUAL per-layer TP diagnostic (no graph, host-synced) to find the divergence ----
    {
        strata::core::TpAllreduce ar2;
        std::string aerr2;
        if (!ar2.open(DEV_A, DEV_B, aerr2)) die(aerr2);
        float* mx0 = nullptr, *my0 = nullptr, *mx1 = nullptr, *my1 = nullptr;
        void* mxq0 = nullptr, *mxqf0 = nullptr, *mW0 = nullptr;
        void* mxq1 = nullptr, *mxqf1 = nullptr, *mW1 = nullptr;
        cudaSetDevice(DEV_A);
        ck(cudaMalloc(&mx0, N_IN * sizeof(float)), "m x0");
        ck(cudaMalloc(&my0, N_OUT * sizeof(float)), "m y0");
        ck(cudaMalloc(&mxq0, (HALF / Q81_ELEMS) * Q81_BYTES), "m xq0");
        ck(cudaMalloc(&mxqf0, (N_IN / Q81_ELEMS) * Q81_BYTES), "m xqf0");
        ck(cudaMalloc(&mW0, W0.size()), "m W0");
        ck(cudaMemcpy(mW0, W0.data(), W0.size(), cudaMemcpyHostToDevice), "m copy W0");
        cudaSetDevice(DEV_B);
        ck(cudaMalloc(&mx1, N_IN * sizeof(float)), "m x1");
        ck(cudaMalloc(&my1, N_OUT * sizeof(float)), "m y1");
        ck(cudaMalloc(&mxq1, (HALF / Q81_ELEMS) * Q81_BYTES), "m xq1");
        ck(cudaMalloc(&mxqf1, (N_IN / Q81_ELEMS) * Q81_BYTES), "m xqf1");
        ck(cudaMalloc(&mW1, W1.size()), "m W1");
        ck(cudaMemcpy(mW1, W1.data(), W1.size(), cudaMemcpyHostToDevice), "m copy W1");
        cudaStream_t ms0, ms1;
        cudaSetDevice(DEV_A); ck(cudaStreamCreateWithFlags(&ms0, cudaStreamNonBlocking), "m s0");
        cudaSetDevice(DEV_B); ck(cudaStreamCreateWithFlags(&ms1, cudaStreamNonBlocking), "m s1");
        // init: copy full xref to BOTH devices, quantize independently on each (no P2P copy),
        // then split the quantized blocks. Independent quantization of the same float input is
        // deterministic, so both devices get identical xq blocks.
        cudaSetDevice(DEV_A);
        ck(cudaMemcpy(mx0, xref.data(), N_IN * sizeof(float), cudaMemcpyHostToDevice), "m copy x0");
        quantize_q8_1_rows(mx0, 1, N_IN, mxqf0, ms0);
        take_half_q81((const uint8_t*) mxqf0, half_blocks, 0, (uint8_t*) mxq0, ms0);
        cudaSetDevice(DEV_B);
        ck(cudaMemcpy(mx1, xref.data(), N_IN * sizeof(float), cudaMemcpyHostToDevice), "m copy x1");
        quantize_q8_1_rows(mx1, 1, N_IN, mxqf1, ms1);
        take_half_q81((const uint8_t*) mxqf1, half_blocks, 1, (uint8_t*) mxq1, ms1);
        ck(cudaStreamSynchronize(ms0), "m init s0");
        ck(cudaStreamSynchronize(ms1), "m init s1");
        std::vector<float> y0h(N_OUT), y1h(N_OUT);
        for (int l = 0; l < N_LAYERS; ++l) {
            cudaSetDevice(DEV_A);
            native_mmvq(8, mW0, mxq0, my0, HALF, N_OUT, 1, ms0);
            cudaSetDevice(DEV_B);
            native_mmvq(8, mW1, mxq1, my1, HALF, N_OUT, 1, ms1);
            ck(cudaStreamSynchronize(ms0), "m s0");
            ck(cudaStreamSynchronize(ms1), "m s1");
            std::string rerr;
            cudaSetDevice(DEV_A);
            if (!ar2.allreduce(my0, my1, N_OUT, ms0, ms1, rerr)) die(rerr);
            ck(cudaStreamSynchronize(ms0), "m ar s0");
            ck(cudaMemcpy(y0h.data(), my0, N_OUT * sizeof(float), cudaMemcpyDeviceToHost), "m read y0");
            ck(cudaMemcpy(y1h.data(), my1, N_OUT * sizeof(float), cudaMemcpyDeviceToHost), "m read y1");
            const double e0 = rel_err(y0h.data(), ref_layer[l + 1].data(), N_OUT);
            const double e01 = rel_err(y0h.data(), y1h.data(), N_OUT);
            if (l < 6 || l == N_LAYERS - 1) {
                std::fprintf(stderr, "  manual layer %d: y0[0..3]=%.6f %.6f %.6f %.6f | ref=%.6f %.6f %.6f %.6f | e0=%.3e e01=%.3e\n",
                             l + 1, y0h[0], y0h[1], y0h[2], y0h[3],
                             ref_layer[l + 1][0], ref_layer[l + 1][1], ref_layer[l + 1][2], ref_layer[l + 1][3],
                             e0, e01);
            }
            // next layer input: quantize full y, split
            cudaSetDevice(DEV_A);
            quantize_q8_1_rows(my0, 1, N_OUT, mxqf0, ms0);
            take_half_q81((const uint8_t*) mxqf0, half_blocks, 0, (uint8_t*) mxq0, ms0);
            cudaSetDevice(DEV_B);
            quantize_q8_1_rows(my1, 1, N_OUT, mxqf1, ms1);
            take_half_q81((const uint8_t*) mxqf1, half_blocks, 1, (uint8_t*) mxq1, ms1);
            ck(cudaStreamSynchronize(ms0), "m s0b");
            ck(cudaStreamSynchronize(ms1), "m s1b");
        }
        cudaSetDevice(DEV_A);
        cudaFree(mx0); cudaFree(my0); cudaFree(mxq0); cudaFree(mxqf0); cudaFree(mW0); cudaStreamDestroy(ms0);
        cudaSetDevice(DEV_B);
        cudaFree(mx1); cudaFree(my1); cudaFree(mxq1); cudaFree(mxqf1); cudaFree(mW1); cudaStreamDestroy(ms1);
    }

    // ---- TP: two graphs, one per device, 48 layers each, concurrent ----
    strata::core::TpAllreduce ar;
    std::string aerr;
    if (!ar.open(DEV_A, DEV_B, aerr)) die(aerr);

    // mapped host flags (the spin kernels read these; both devices see the same host words)
    uint32_t* h_flag_a = nullptr;   // raised by dev0 after allreduce, waited on by dev1
    uint32_t* h_flag_b = nullptr;   // raised by dev1 after gemv1, waited on by dev0
    uint32_t* h_err_a = nullptr;    // set by dev0's wait if it times out
    uint32_t* h_err_b = nullptr;    // set by dev1's wait if it times out
    ck(cudaHostAlloc(&h_flag_a, sizeof(uint32_t), cudaHostAllocMapped), "hostalloc flag_a");
    ck(cudaHostAlloc(&h_flag_b, sizeof(uint32_t), cudaHostAllocMapped), "hostalloc flag_b");
    ck(cudaHostAlloc(&h_err_a, sizeof(uint32_t), cudaHostAllocMapped), "hostalloc err_a");
    ck(cudaHostAlloc(&h_err_b, sizeof(uint32_t), cudaHostAllocMapped), "hostalloc err_b");
    *h_flag_a = 0; *h_flag_b = 0; *h_err_a = 0; *h_err_b = 0;
    // mapped host memory is zero-copy: the device pointer is the same for both devices,
    // obtained with device 0 (the engine's pattern: cudaHostGetDevicePointer(&d, h, 0)).
    uint32_t* d_flag_a = nullptr, *d_flag_b = nullptr, *d_err_a = nullptr, *d_err_b = nullptr;
    cudaSetDevice(DEV_A);
    ck(cudaHostGetDevicePointer(&d_flag_a, h_flag_a, 0), "devptr flag_a");
    ck(cudaHostGetDevicePointer(&d_flag_b, h_flag_b, 0), "devptr flag_b");
    ck(cudaHostGetDevicePointer(&d_err_a, h_err_a, 0), "devptr err_a");
    ck(cudaHostGetDevicePointer(&d_err_b, h_err_b, 0), "devptr err_b");

    // per-device buffers (all pre-allocated; NO malloc/free inside graph capture)
    float* d_x0 = nullptr, *d_y0 = nullptr, *d_x1 = nullptr, *d_y1 = nullptr;
    void* d_xq0 = nullptr, *d_xq1 = nullptr, *d_xq_full0 = nullptr, *d_xq_full1 = nullptr;
    void* d_W0 = nullptr, *d_W1 = nullptr;
    cudaStream_t s0 = nullptr, s1 = nullptr;
    {
        cudaSetDevice(DEV_A);
        ck(cudaMalloc(&d_x0, N_IN * sizeof(float)), "tp x0");
        ck(cudaMalloc(&d_y0, N_OUT * sizeof(float)), "tp y0");
        ck(cudaMalloc(&d_xq0, (HALF / Q81_ELEMS) * Q81_BYTES), "tp xq0");
        ck(cudaMalloc(&d_xq_full0, (N_IN / Q81_ELEMS) * Q81_BYTES), "tp xq_full0");
        ck(cudaMalloc(&d_W0, W0.size()), "tp W0");
        ck(cudaMemcpy(d_W0, W0.data(), W0.size(), cudaMemcpyHostToDevice), "tp copy W0");
        ck(cudaStreamCreateWithFlags(&s0, cudaStreamNonBlocking), "tp s0");
        cudaSetDevice(DEV_B);
        ck(cudaMalloc(&d_x1, N_IN * sizeof(float)), "tp x1");
        ck(cudaMalloc(&d_y1, N_OUT * sizeof(float)), "tp y1");
        ck(cudaMalloc(&d_xq1, (HALF / Q81_ELEMS) * Q81_BYTES), "tp xq1");
        ck(cudaMalloc(&d_xq_full1, (N_IN / Q81_ELEMS) * Q81_BYTES), "tp xq_full1");
        ck(cudaMalloc(&d_W1, W1.size()), "tp W1");
        ck(cudaMemcpy(d_W1, W1.data(), W1.size(), cudaMemcpyHostToDevice), "tp copy W1");
        ck(cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking), "tp s1");
    }

    // initial input: copy the FULL xref to BOTH devices, quantize independently on each
    // (deterministic for the same float input), then split the quantized blocks by device.
    // (No P2P device-to-device copy - those are unreliable; quantize locally instead.)
    {
        cudaSetDevice(DEV_A);
        ck(cudaMemcpy(d_x0, xref.data(), N_IN * sizeof(float), cudaMemcpyHostToDevice), "tp copy x0 full");
        quantize_q8_1_rows(d_x0, 1, N_IN, d_xq_full0, s0);
        take_half_q81((const uint8_t*) d_xq_full0, half_blocks, 0, (uint8_t*) d_xq0, s0);
        cudaSetDevice(DEV_B);
        ck(cudaMemcpy(d_x1, xref.data(), N_IN * sizeof(float), cudaMemcpyHostToDevice), "tp copy x1 full");
        quantize_q8_1_rows(d_x1, 1, N_IN, d_xq_full1, s1);
        take_half_q81((const uint8_t*) d_xq_full1, half_blocks, 1, (uint8_t*) d_xq1, s1);
        ck(cudaStreamSynchronize(s0), "tp init sync0");
        ck(cudaStreamSynchronize(s1), "tp init sync1");
    }

    // ---- capture the two graphs ----
    cudaGraph_t g0 = nullptr, g1 = nullptr;
    {
        // graph 0 (dev_a): per layer: gemv0 -> wait(flag_b) -> allreduce -> raise(flag_a) -> quantize+split
        cudaSetDevice(DEV_A);
        ck(cudaStreamBeginCapture(s0, cudaStreamCaptureModeThreadLocal), "capture0 begin");
        for (int l = 0; l < N_LAYERS; ++l) {
            native_mmvq(8, d_W0, d_xq0, d_y0, HALF, N_OUT, 1, s0);   // y0 = W0 . x[0:HALF]
            wait_flag_ge(d_flag_b, (uint32_t) (l + 1), d_err_a, WAIT_TIMEOUT_NS, s0);   // wait for dev1's gemv1
            std::string rerr;
            if (!ar.allreduce(d_y0, d_y1, N_OUT, s0, s1, rerr)) die(rerr);   // y0=y1=y0+y1 (both)
            raise_flag(d_flag_a, (uint32_t) (l + 1), s0);          // signal dev1: allreduce done
            quantize_q8_1_rows(d_y0, 1, N_OUT, d_xq_full0, s0);      // quantize the full y0
            take_half_q81((const uint8_t*) d_xq_full0, half_blocks, 0, (uint8_t*) d_xq0, s0);   // next layer's xq0
        }
        ck(cudaStreamEndCapture(s0, &g0), "capture0 end");
        // graph 1 (dev_b): per layer: gemv1 -> raise(flag_b) -> wait(flag_a) -> quantize+split
        cudaSetDevice(DEV_B);
        ck(cudaStreamBeginCapture(s1, cudaStreamCaptureModeThreadLocal), "capture1 begin");
        for (int l = 0; l < N_LAYERS; ++l) {
            native_mmvq(8, d_W1, d_xq1, d_y1, HALF, N_OUT, 1, s1);   // y1 = W1 . x[HALF:N_IN]
            raise_flag(d_flag_b, (uint32_t) (l + 1), s1);          // signal dev0: gemv1 done
            wait_flag_ge(d_flag_a, (uint32_t) (l + 1), d_err_b, WAIT_TIMEOUT_NS, s1);   // wait for dev0's allreduce
            quantize_q8_1_rows(d_y1, 1, N_OUT, d_xq_full1, s1);      // quantize the full y1
            take_half_q81((const uint8_t*) d_xq_full1, half_blocks, 1, (uint8_t*) d_xq1, s1);   // next layer's xq1
        }
        ck(cudaStreamEndCapture(s1, &g1), "capture1 end");
    }

    cudaGraphExec_t e0 = nullptr, e1 = nullptr;
    ck(cudaGraphInstantiate(&e0, g0, nullptr, nullptr, 0), "instantiate0");
    ck(cudaGraphInstantiate(&e1, g1, nullptr, nullptr, 0), "instantiate1");
    cudaGraphDestroy(g0);
    cudaGraphDestroy(g1);

    // ---- launch both graphs CONCURRENTLY (no host sync between the two launches) ----
    cudaSetDevice(DEV_A);
    ck(cudaGraphLaunch(e0, s0), "launch0");
    cudaSetDevice(DEV_B);
    ck(cudaGraphLaunch(e1, s1), "launch1");
    // wait for both to finish (each stream ends after its final layer's take_half)
    cudaSetDevice(DEV_A);
    ck(cudaStreamSynchronize(s0), "sync s0");
    cudaSetDevice(DEV_B);
    ck(cudaStreamSynchronize(s1), "sync s1");

    // ---- check the timeout error words ----
    const bool timeout_a = *h_err_a != 0;
    const bool timeout_b = *h_err_b != 0;
    if (timeout_a || timeout_b) {
        std::fprintf(stderr, "TIMEOUT: dev_a=%d dev_b=%d (a spin-wait flag never raised)\n", (int) timeout_a, (int) timeout_b);
    }

    // ---- read back both results, compare to reference ----
    std::vector<float> y0h(N_OUT), y1h(N_OUT);
    cudaSetDevice(DEV_A);
    ck(cudaMemcpy(y0h.data(), d_y0, N_OUT * sizeof(float), cudaMemcpyDeviceToHost), "read y0");
    cudaSetDevice(DEV_B);
    ck(cudaMemcpy(y1h.data(), d_y1, N_OUT * sizeof(float), cudaMemcpyDeviceToHost), "read y1");

    // Compare against the 48-layer reference output (ref_layer[N_LAYERS]), NOT xref (the input).
    const double err0 = rel_err(y0h.data(), ref_layer[N_LAYERS].data(), N_OUT);
    const double err1 = rel_err(y1h.data(), ref_layer[N_LAYERS].data(), N_OUT);
    const double err01 = rel_err(y0h.data(), y1h.data(), N_OUT);
    std::fprintf(stderr, "TP result: y0[0..3] = %.6f %.6f %.6f %.6f\n", y0h[0], y0h[1], y0h[2], y0h[3]);
    std::fprintf(stderr, "TP result: y1[0..3] = %.6f %.6f %.6f %.6f\n", y1h[0], y1h[1], y1h[2], y1h[3]);
    std::fprintf(stderr, "rel_err y0 vs ref = %.3e\n", err0);
    std::fprintf(stderr, "rel_err y1 vs ref = %.3e\n", err1);
    std::fprintf(stderr, "rel_err y0 vs y1 = %.3e\n", err01);

    const bool ok = !timeout_a && !timeout_b && err0 < 1e-4 && err1 < 1e-4 && err01 < 1e-6;
    check(ok, "two-graph TP (48 layers, row-parallel + allreduce + spin-wait flags) matches single-GPU reference");

    std::fprintf(stderr, "ALL PASSED\n");
    return 0;
}
