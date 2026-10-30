// tests/core/tp_split_test.cu - Phase 3a: verify the Q8_0 tensor-parallel split kernels.
//
// Synthetic Q8_0 weights at real model dimensions.  For each split mode the kernel's output is compared
// byte-for-byte against a host reference split:
//   * row-parallel (split n_in, the reduction axis): each row's first / second n_in / 64 blocks
//   * col-parallel (split n_out, the row axis): the first / second n_out / 2 rows
// A Q8_0 block is 34 bytes (2-byte half scale + 32 int8 values); both splits take whole blocks, so a correct
// kernel reproduces the reference bytes exactly.
#include "strata/core/tp_split.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        std::exit(1);
    }
}
static void gpu(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "FAIL: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}
static uint64_t splitmix(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

int main() {
    int count = 0;
    gpu(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
    if (count < 1) {
        std::fprintf(stderr, "SKIP: no GPU\n");
        return 0;
    }
    gpu(cudaSetDevice(0), "set 0");
    cudaStream_t s;
    gpu(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "stream");

    // Row-parallel: ssm_out 6144 x 2560 (n_in = 2560, n_out = 6144).
    {
        const int n_in = 2560, n_out = 6144;
        const int nb = n_in / 32;  // 80 blocks per row
        const int nbh = nb / 2;    // 40 blocks per row per half
        const size_t full_bytes = (size_t)n_out * nb * 34;
        const size_t half_bytes = (size_t)n_out * nbh * 34;
        std::vector<uint8_t> full(full_bytes);
        for (size_t i = 0; i < full.size(); ++i) full[i] = (uint8_t)(splitmix(i + 1) & 0xff);
        std::vector<uint8_t> ref0(half_bytes), ref1(half_bytes);
        for (int r = 0; r < n_out; ++r)
            for (int i = 0; i < nbh; ++i) {
                std::memcpy(&ref0[(size_t)(r * nbh + i) * 34], &full[(size_t)(r * nb + i) * 34], 34);
                std::memcpy(&ref1[(size_t)(r * nbh + i) * 34], &full[(size_t)(r * nb + nbh + i) * 34], 34);
            }
        uint8_t* d_full;
        uint8_t* d0;
        uint8_t* d1;
        gpu(cudaMalloc(&d_full, full_bytes), "d_full");
        gpu(cudaMalloc(&d0, half_bytes), "d0");
        gpu(cudaMalloc(&d1, half_bytes), "d1");
        gpu(cudaMemcpyAsync(d_full, full.data(), full_bytes, cudaMemcpyHostToDevice, s), "up");
        strata::core::tp_split_q8_0_row(d_full, n_in, n_out, 0, d0, s);
        strata::core::tp_split_q8_0_row(d_full, n_in, n_out, 1, d1, s);
        gpu(cudaStreamSynchronize(s), "sync");
        std::vector<uint8_t> g0(half_bytes), g1(half_bytes);
        gpu(cudaMemcpy(g0.data(), d0, half_bytes, cudaMemcpyDeviceToHost), "dn0");
        gpu(cudaMemcpy(g1.data(), d1, half_bytes, cudaMemcpyDeviceToHost), "dn1");
        check(g0 == ref0, "row-parallel half 0 byte-exact");
        check(g1 == ref1, "row-parallel half 1 byte-exact");
        std::fprintf(stderr, "row-parallel (ssm_out %d x %d): PASS\n", n_out, n_in);
        gpu(cudaFree(d_full), "f");
        gpu(cudaFree(d0), "f");
        gpu(cudaFree(d1), "f");
    }

    // Col-parallel: attn_qkv 2560 x 10240 (n_in = 2560, n_out = 10240).
    {
        const int n_in = 2560, n_out = 10240;
        const int nb = n_in / 32;  // 80 blocks per row
        const int nh = n_out / 2;  // 5120 rows per half
        const size_t full_bytes = (size_t)n_out * nb * 34;
        const size_t half_bytes = (size_t)nh * nb * 34;
        std::vector<uint8_t> full(full_bytes);
        for (size_t i = 0; i < full.size(); ++i) full[i] = (uint8_t)(splitmix(i + 7) & 0xff);
        std::vector<uint8_t> ref0(half_bytes), ref1(half_bytes);
        for (int r = 0; r < nh; ++r) {
            std::memcpy(&ref0[(size_t)r * nb * 34], &full[(size_t)r * nb * 34], (size_t)nb * 34);
            std::memcpy(&ref1[(size_t)r * nb * 34], &full[(size_t)(nh + r) * nb * 34], (size_t)nb * 34);
        }
        uint8_t* d_full;
        uint8_t* d0;
        uint8_t* d1;
        gpu(cudaMalloc(&d_full, full_bytes), "d_full");
        gpu(cudaMalloc(&d0, half_bytes), "d0");
        gpu(cudaMalloc(&d1, half_bytes), "d1");
        gpu(cudaMemcpyAsync(d_full, full.data(), full_bytes, cudaMemcpyHostToDevice, s), "up");
        strata::core::tp_split_q8_0_col(d_full, n_in, n_out, 0, d0, s);
        strata::core::tp_split_q8_0_col(d_full, n_in, n_out, 1, d1, s);
        gpu(cudaStreamSynchronize(s), "sync");
        std::vector<uint8_t> g0(half_bytes), g1(half_bytes);
        gpu(cudaMemcpy(g0.data(), d0, half_bytes, cudaMemcpyDeviceToHost), "dn0");
        gpu(cudaMemcpy(g1.data(), d1, half_bytes, cudaMemcpyDeviceToHost), "dn1");
        check(g0 == ref0, "col-parallel half 0 byte-exact");
        check(g1 == ref1, "col-parallel half 1 byte-exact");
        std::fprintf(stderr, "col-parallel (attn_qkv %d x %d): PASS\n", n_out, n_in);
        gpu(cudaFree(d_full), "f");
        gpu(cudaFree(d0), "f");
        gpu(cudaFree(d1), "f");
    }

    gpu(cudaStreamDestroy(s), "destroy");
    std::fprintf(stderr, "tp_split_test: ALL PASS\n");
    return 0;
}
