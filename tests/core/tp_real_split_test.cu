// tests/core/tp_real_split_test.cu - Phase 2a/2b REAL validation (dense Q8_0 path).
//
// The float-GEMV prototypes (tp_weight_split_test.cu, tp_gdn_split_test.cu) proved the TP
// math but used synthetic float weights. This test closes the gap: it runs the REAL
// native_mmvq kernel on REAL quantized weights, split for TP, and checks the halves
// reproduce the full-matrix output.
//
//   Test 1: Column-parallel on real Q8_0 attn_qkv (2560 x 10240, split n_out 5120+5120)
//   Test 2: Row-parallel + allreduce on real Q8_0 ssm_out (6144 x 2560, split n_in 3072+3072)
//
// WHY NO NVFP4 EXPERT TEST HERE:
//   The dense attention/GDN/QSA weights are Q8_0 (GGML type 8) and go through native_mmvq.
//   The MoE expert weights are NVFP4 (GGML type 40) but do NOT go through native_mmvq -
//   native_mmvq_supported() rejects type 40. Experts are computed by native_expert_grouped
//   (iq_kernels.cu), a grouped MoE kernel that takes routing tables (grp_ptr/grp_start/
//   n_groups/ent_dst/ent_tok) and per-expert tail scales (s_gate/s_up/s_down). That is a
//   routed, grouped GEMM - not a plain GEMV - so the column/row-parallel GEMV split does
//   not apply. MoE TP is expert-parallelism (each GPU holds half the experts), a separate
//   phase. The dense path (GDN/attention = ~51% of decode FLOPs) is what this test proves.
//
// Build:
//   nvcc -O2 -std=c++17 --use_fast_math -I include -I src \
//     tests/core/tp_real_split_test.cu src/core/tp_allreduce.cu \
//     src/kernels/cuda/native_mmvq.cu src/kernels/cuda/iq_kernels.cu \
//     -o build/tp_real_split_test -lcuda
//
// Data (extracted by /media/zc/软件1/项目/pcai/extract_nvfp4_tensor.py):
//   tests/core/tp_data/blk_0_attn_qkv_weight.q8_0   (2560 x 10240, Q8_0)
//   tests/core/tp_data/blk_0_ssm_out_weight.q8_0    (6144 x 2560, Q8_0)

#include "strata/core/tp_allreduce.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/pdl.hpp"

#include <cuda_runtime.h>

// PDL (programmatic dependent launch) is an sm_90+ optimization; it is never active on the
// sm_75 2080 Ti. native_mmvq.cu references these two host symbols, so provide safe stubs
// (pdl_scope() = false -> ordinary launches; pdl_launch_ok() = false -> never a PDL edge).
namespace strata::kernels {
bool& pdl_scope() {
    static bool on = false;
    return on;
}
bool pdl_launch_ok(const void*, cudaStream_t) { return false; }
}  // namespace strata::kernels

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

static void check(bool ok, const char* what, const std::string& err = "") {
    if (!ok) { std::fprintf(stderr, "FAIL: %s%s%s\n", what, err.empty() ? "" : " - ", err.c_str()); std::exit(1); }
}
static void gpu(cudaError_t e, const char* what) {
    if (e != cudaSuccess) { std::fprintf(stderr, "FAIL: %s: %s\n", what, cudaGetErrorString(e)); std::exit(1); }
}

static std::vector<uint8_t> read_file(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    check((bool) f, ("cannot open " + p).c_str());
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

static void block_geom(int type, int& elems, int& bytes) {
    if (type == 40) { elems = 64; bytes = 36; }        // NVFP4
    else if (type == 8) { elems = 32; bytes = 34; }    // Q8_0
    else { std::fprintf(stderr, "unsupported type %d\n", type); std::exit(1); }
}

// deterministic activation in [-1, 1]
static float prand(int64_t i) {
    uint64_t x = (uint64_t) i * 6364136223846793005ULL + 1442695040888963407ULL;
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL; x ^= x >> 33;
    return (float) (x & 0xFFFFFF) / 8388608.0f - 1.0f;
}

// relative L1 error between two float vectors
static double rel_err(const std::vector<float>& a, const std::vector<float>& b) {
    double num = 0, den = 0;
    for (size_t i = 0; i < a.size(); ++i) { num += std::fabs((double) a[i] - (double) b[i]); den += std::fabs((double) b[i]); }
    return num / (den + 1e-30);
}

// ---------------------------------------------------------------------------
// Column-parallel: split n_out (rows) in half. Each row is independent, so W0 = first
// half of rows (contiguous), W1 = second half. Same activation. y_full[0..n/2) == y0,
// y_full[n/2..n) == y1.
// ---------------------------------------------------------------------------
static void test_column_parallel(const char* name, const std::vector<uint8_t>& w, int type, int n_in, int n_out,
                                 cudaStream_t s0, cudaStream_t s1) {
    std::fprintf(stderr, "=== %s: Column-parallel (%d x %d, split n_out %d+%d) ===\n", name, n_in, n_out, n_out / 2, n_out / 2);
    int elems, bytes;
    block_geom(type, elems, bytes);
    const int blocks_per_row = n_in / elems;
    const size_t row_bytes = (size_t) blocks_per_row * bytes;
    check(w.size() == (size_t) n_out * row_bytes, "weight size mismatch");
    const int n_half = n_out / 2;

    // activation (host)
    std::vector<float> x(n_in);
    for (int k = 0; k < n_in; ++k) x[k] = prand(k);

    // full reference on GPU0
    float* dx = nullptr;
    void* xq = nullptr;
    void* dw = nullptr;
    float* dy = nullptr;
    gpu(cudaSetDevice(0), "set 0");
    gpu(cudaMalloc(&dx, n_in * 4), "dx");
    gpu(cudaMalloc(&xq, (size_t) n_in / 32 * 36), "xq");
    gpu(cudaMalloc(&dw, w.size()), "dw");
    gpu(cudaMalloc(&dy, (size_t) n_out * 4), "dy");
    gpu(cudaMemcpyAsync(dx, x.data(), n_in * 4, cudaMemcpyHostToDevice, s0), "copy x");
    strata::kernels::quantize_q8_1_rows(dx, 1, n_in, xq, s0);
    gpu(cudaMemcpyAsync(dw, w.data(), w.size(), cudaMemcpyHostToDevice, s0), "copy W");
    strata::kernels::native_mmvq(type, dw, xq, dy, n_in, n_out, 1, s0);
    gpu(cudaStreamSynchronize(s0), "sync full");
    std::vector<float> y_full(n_out);
    gpu(cudaMemcpy(y_full.data(), dy, n_out * 4, cudaMemcpyDeviceToHost), "read y_full");

    // split: W0 = first n_half rows (contiguous), W1 = last n_half rows.
    // Each GPU gets its own activation copy (TP: both cards hold the input).
    const size_t half_bytes = (size_t) n_half * row_bytes;
    float* dx0 = nullptr, * dx1 = nullptr;
    void* xq0 = nullptr, * xq1 = nullptr;
    void* dw0 = nullptr, * dw1 = nullptr;
    float* dy0 = nullptr, * dy1 = nullptr;
    gpu(cudaSetDevice(0), "set 0");
    gpu(cudaMalloc(&dx0, n_in * 4), "dx0");
    gpu(cudaMalloc(&xq0, (size_t) n_in / 32 * 36), "xq0");
    gpu(cudaMalloc(&dw0, half_bytes), "dw0");
    gpu(cudaMalloc(&dy0, (size_t) n_half * 4), "dy0");
    gpu(cudaSetDevice(1), "set 1");
    gpu(cudaMalloc(&dx1, n_in * 4), "dx1");
    gpu(cudaMalloc(&xq1, (size_t) n_in / 32 * 36), "xq1");
    gpu(cudaMalloc(&dw1, half_bytes), "dw1");
    gpu(cudaMalloc(&dy1, (size_t) n_half * 4), "dy1");
    // uploads (explicit streams; current device must match the stream's device)
    gpu(cudaSetDevice(0), "set 0");
    gpu(cudaMemcpyAsync(dx0, x.data(), n_in * 4, cudaMemcpyHostToDevice, s0), "copy x0");
    strata::kernels::quantize_q8_1_rows(dx0, 1, n_in, xq0, s0);
    gpu(cudaMemcpyAsync(dw0, w.data(), half_bytes, cudaMemcpyHostToDevice, s0), "copy W0");
    gpu(cudaSetDevice(1), "set 1");
    gpu(cudaMemcpyAsync(dx1, x.data(), n_in * 4, cudaMemcpyHostToDevice, s1), "copy x1");
    strata::kernels::quantize_q8_1_rows(dx1, 1, n_in, xq1, s1);
    gpu(cudaMemcpyAsync(dw1, w.data() + half_bytes, half_bytes, cudaMemcpyHostToDevice, s1), "copy W1");
    // kernels (current device must match the stream's device)
    gpu(cudaSetDevice(0), "set 0");
    strata::kernels::native_mmvq(type, dw0, xq0, dy0, n_in, n_half, 1, s0);
    gpu(cudaSetDevice(1), "set 1");
    strata::kernels::native_mmvq(type, dw1, xq1, dy1, n_in, n_half, 1, s1);
    gpu(cudaStreamSynchronize(s0), "sync 0");
    gpu(cudaStreamSynchronize(s1), "sync 1");
    std::vector<float> y0(n_half), y1(n_half);
    gpu(cudaSetDevice(0), "set 0");
    gpu(cudaMemcpy(y0.data(), dy0, n_half * 4, cudaMemcpyDeviceToHost), "read y0");
    gpu(cudaSetDevice(1), "set 1");
    gpu(cudaMemcpy(y1.data(), dy1, n_half * 4, cudaMemcpyDeviceToHost), "read y1");

    double e0 = rel_err(y0, std::vector<float>(y_full.begin(), y_full.begin() + n_half));
    double e1 = rel_err(y1, std::vector<float>(y_full.begin() + n_half, y_full.end()));
    std::fprintf(stderr, "  GPU0 (rows 0..%d) rel_err = %.3e\n", n_half, e0);
    std::fprintf(stderr, "  GPU1 (rows %d..%d) rel_err = %.3e\n", n_half, n_out, e1);
    check(e0 < 1e-6 && e1 < 1e-6, "column-parallel mismatch");
    std::fprintf(stderr, "  PASS\n");

    gpu(cudaSetDevice(0), "set 0");
    gpu(cudaFree(dx), "free"); gpu(cudaFree(xq), "free");
    gpu(cudaFree(dw), "free"); gpu(cudaFree(dy), "free");
    gpu(cudaFree(dx0), "free"); gpu(cudaFree(xq0), "free");
    gpu(cudaFree(dw0), "free"); gpu(cudaFree(dy0), "free");
    gpu(cudaSetDevice(1), "set 1");
    gpu(cudaFree(dx1), "free"); gpu(cudaFree(xq1), "free");
    gpu(cudaFree(dw1), "free"); gpu(cudaFree(dy1), "free");
    gpu(cudaSetDevice(0), "set 0");
}

// ---------------------------------------------------------------------------
// Row-parallel: split n_in (reduction) in half at a block boundary. For each row, W0 =
// first row_bytes/2 (k in [0, n_in/2)), W1 = second row_bytes/2 (k in [n_in/2, n_in)).
// Each half uses its own activation half. y0 + y1 (allreduce) == y_full.
// ---------------------------------------------------------------------------
static void test_row_parallel(const char* name, const std::vector<uint8_t>& w, int type, int n_in, int n_out,
                              cudaStream_t s0, cudaStream_t s1, strata::core::TpAllreduce& ar) {
    std::fprintf(stderr, "=== %s: Row-parallel + allreduce (%d x %d, split n_in %d+%d) ===\n", name, n_in, n_out, n_in / 2, n_in / 2);
    int elems, bytes;
    block_geom(type, elems, bytes);
    const int blocks_per_row = n_in / elems;
    check(blocks_per_row % 2 == 0, "n_in must be a multiple of 2*block_elems");
    const size_t row_bytes = (size_t) blocks_per_row * bytes;
    const size_t half_row = row_bytes / 2;
    check(w.size() == (size_t) n_out * row_bytes, "weight size mismatch");
    const int n_in_half = n_in / 2;

    // full activation + reference on GPU0
    std::vector<float> x(n_in);
    for (int k = 0; k < n_in; ++k) x[k] = prand(k);
    float* dx = nullptr;
    void* xq = nullptr;
    gpu(cudaSetDevice(0), "set 0");
    gpu(cudaMalloc(&dx, n_in * 4), "dx");
    gpu(cudaMalloc(&xq, (size_t) n_in / 32 * 36), "xq");
    gpu(cudaMemcpyAsync(dx, x.data(), n_in * 4, cudaMemcpyHostToDevice, s0), "copy x");
    strata::kernels::quantize_q8_1_rows(dx, 1, n_in, xq, s0);
    void* dw = nullptr;
    float* dy = nullptr;
    gpu(cudaMalloc(&dw, w.size()), "dw");
    gpu(cudaMalloc(&dy, (size_t) n_out * 4), "dy");
    gpu(cudaMemcpyAsync(dw, w.data(), w.size(), cudaMemcpyHostToDevice, s0), "copy W");
    strata::kernels::native_mmvq(type, dw, xq, dy, n_in, n_out, 1, s0);
    gpu(cudaStreamSynchronize(s0), "sync full");
    std::vector<float> y_full(n_out);
    gpu(cudaMemcpy(y_full.data(), dy, n_out * 4, cudaMemcpyDeviceToHost), "read y_full");

    // build the two strided halves on the host: for each row, first/second half_row bytes
    std::vector<uint8_t> w0((size_t) n_out * half_row), w1((size_t) n_out * half_row);
    for (int r = 0; r < n_out; ++r) {
        const uint8_t* row = w.data() + (size_t) r * row_bytes;
        std::memcpy(w0.data() + (size_t) r * half_row, row, half_row);
        std::memcpy(w1.data() + (size_t) r * half_row, row + half_row, half_row);
    }

    // activation halves (each n_in/2, a multiple of 32)
    std::vector<float> x0(n_in_half), x1(n_in_half);
    for (int k = 0; k < n_in_half; ++k) { x0[k] = x[k]; x1[k] = x[k + n_in_half]; }
    float* dx0 = nullptr, * dx1 = nullptr;
    void* xq0 = nullptr, * xq1 = nullptr;
    gpu(cudaMalloc(&dx0, n_in_half * 4), "dx0");
    gpu(cudaMalloc(&xq0, (size_t) n_in_half / 32 * 36), "xq0");
    gpu(cudaSetDevice(1), "set 1");
    gpu(cudaMalloc(&dx1, n_in_half * 4), "dx1");
    gpu(cudaMalloc(&xq1, (size_t) n_in_half / 32 * 36), "xq1");
    gpu(cudaSetDevice(0), "set 0");
    gpu(cudaMemcpyAsync(dx0, x0.data(), n_in_half * 4, cudaMemcpyHostToDevice, s0), "copy x0");
    strata::kernels::quantize_q8_1_rows(dx0, 1, n_in_half, xq0, s0);
    gpu(cudaSetDevice(1), "set 1");
    gpu(cudaMemcpyAsync(dx1, x1.data(), n_in_half * 4, cudaMemcpyHostToDevice, s1), "copy x1");
    strata::kernels::quantize_q8_1_rows(dx1, 1, n_in_half, xq1, s1);

    // weight halves
    void* dw0 = nullptr, * dw1 = nullptr;
    float* dy0 = nullptr, * dy1 = nullptr;
    gpu(cudaSetDevice(0), "set 0");
    gpu(cudaMalloc(&dw0, w0.size()), "dw0");
    gpu(cudaMalloc(&dy0, (size_t) n_out * 4), "dy0");
    gpu(cudaSetDevice(1), "set 1");
    gpu(cudaMalloc(&dw1, w1.size()), "dw1");
    gpu(cudaMalloc(&dy1, (size_t) n_out * 4), "dy1");
    gpu(cudaSetDevice(0), "set 0");
    gpu(cudaMemcpyAsync(dw0, w0.data(), w0.size(), cudaMemcpyHostToDevice, s0), "copy W0");
    gpu(cudaSetDevice(1), "set 1");
    gpu(cudaMemcpyAsync(dw1, w1.data(), w1.size(), cudaMemcpyHostToDevice, s1), "copy W1");
    // kernels (current device must match the stream's device)
    gpu(cudaSetDevice(0), "set 0");
    strata::kernels::native_mmvq(type, dw0, xq0, dy0, n_in_half, n_out, 1, s0);
    gpu(cudaSetDevice(1), "set 1");
    strata::kernels::native_mmvq(type, dw1, xq1, dy1, n_in_half, n_out, 1, s1);

    // The single-writer reduction reads dy1 over P2P; s0 does not order its producer on s1.
    gpu(cudaStreamSynchronize(s1), "sync peer projection");
    gpu(cudaSetDevice(0), "set 0");
    std::string aerr;
    ar.allreduce(dy0, dy1, n_out, s0, s1, aerr);
    check(aerr.empty(), ("allreduce: " + aerr).c_str());
    gpu(cudaStreamSynchronize(s0), "sync 0");
    gpu(cudaStreamSynchronize(s1), "sync 1");
    std::vector<float> y0(n_out), y1(n_out);
    gpu(cudaMemcpy(y0.data(), dy0, n_out * 4, cudaMemcpyDeviceToHost), "read y0");
    gpu(cudaSetDevice(1), "set 1");
    gpu(cudaMemcpy(y1.data(), dy1, n_out * 4, cudaMemcpyDeviceToHost), "read y1");

    double e0 = rel_err(y0, y_full);
    double e1 = rel_err(y1, y_full);
    std::fprintf(stderr, "  GPU0 (allreduce) rel_err = %.3e\n", e0);
    std::fprintf(stderr, "  GPU1 (allreduce) rel_err = %.3e\n", e1);
    check(e0 < 1e-6 && e1 < 1e-6, "row-parallel mismatch");
    std::fprintf(stderr, "  PASS\n");

    gpu(cudaFree(dw), "free"); gpu(cudaFree(dy), "free");
    gpu(cudaFree(dx), "free"); gpu(cudaFree(xq), "free");
    gpu(cudaFree(dx0), "free"); gpu(cudaFree(xq0), "free");
    gpu(cudaFree(dw0), "free"); gpu(cudaFree(dy0), "free");
    gpu(cudaSetDevice(1), "set 1");
    gpu(cudaFree(dx1), "free"); gpu(cudaFree(xq1), "free");
    gpu(cudaFree(dw1), "free"); gpu(cudaFree(dy1), "free");
    gpu(cudaSetDevice(0), "set 0");
}

int main() {
    int count = 0;
    gpu(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
    if (count < 2) { std::fprintf(stderr, "SKIP: only %d device(s)\n", count); return 0; }
    strata::core::TpAllreduce ar;
    std::string err;
    if (!ar.open(0, 1, err)) { std::fprintf(stderr, "SKIP: no P2P (%s)\n", err.c_str()); return 0; }
    cudaStream_t s0, s1;
    gpu(cudaSetDevice(0), "set 0");
    gpu(cudaStreamCreateWithFlags(&s0, cudaStreamNonBlocking), "s0");
    gpu(cudaSetDevice(1), "set 1");
    gpu(cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking), "s1");

    const std::string dir = "tests/core/tp_data/";

    // Test 1: Q8_0 attn_qkv column-parallel
    {
        auto w = read_file(dir + "blk_0_attn_qkv_weight.q8_0");
        test_column_parallel("Q8_0 attn_qkv", w, 8, 2560, 10240, s0, s1);
    }
    // Test 2: Q8_0 ssm_out row-parallel
    {
        auto w = read_file(dir + "blk_0_ssm_out_weight.q8_0");
        test_row_parallel("Q8_0 ssm_out", w, 8, 6144, 2560, s0, s1, ar);
    }

    std::fprintf(stderr, "\nALL REAL-WEIGHT TP SPLIT TESTS PASSED (dense Q8_0 path)\n");
    return 0;
}
