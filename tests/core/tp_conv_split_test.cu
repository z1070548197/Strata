// tests/core/tp_conv_split_test.cu - Phase 3a: validate the GDN conv weight split
// (tp_local_conv_weights) against an independent host reference.
//
// The GDN conv weight (ssm_conv1d) is 4 taps x 10240 channels, channel-major (full[c*4 + tap]).  The TP split
// keeps the local 7168 channels per card: q 2048 + k 2048 + v_half 3072.  rank 0 = channels [0, 7168)
// (contiguous); rank 1 = channels [0, 4096) + [7168, 10240) (q/k + the second v half).  This test fills the
// full weight with a deterministic pattern, runs the device split, and checks the local weight matches the host
// reference bit-for-bit.
#include "strata/core/tp_split.hpp"

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

using strata::core::tp_local_conv_weights;

namespace {
constexpr int kFull = 10240;
constexpr int kLocal = 7168;
constexpr int kQk = 4096;
} // namespace

int main() {
    std::vector<float> full((size_t) kFull * 4);
    for (int c = 0; c < kFull; ++c)
        for (int t = 0; t < 4; ++t)
            full[(size_t) c * 4 + t] = (float) (c * 4 + t) * 0.001f - 1.0f;

    // host reference for each rank
    std::vector<std::vector<float>> ref(2, std::vector<float>((size_t) kLocal * 4));
    for (int rank = 0; rank < 2; ++rank) {
        for (int c = 0; c < kLocal; ++c) {
            const int full_c = (c < kQk) ? c : (kQk + (c - kQk) + rank * (kLocal - kQk));
            for (int t = 0; t < 4; ++t)
                ref[rank][(size_t) c * 4 + t] = full[(size_t) full_c * 4 + t];
        }
    }

    float* d_full = nullptr;
    float* d_local = nullptr;
    if (cudaMalloc(&d_full, (size_t) kFull * 4 * 4) != cudaSuccess) { std::fprintf(stderr, "alloc full\n"); return 1; }
    if (cudaMalloc(&d_local, (size_t) kLocal * 4 * 4) != cudaSuccess) { std::fprintf(stderr, "alloc local\n"); return 1; }
    cudaMemcpy(d_full, full.data(), (size_t) kFull * 4 * 4, cudaMemcpyHostToDevice);

    int fails = 0;
    for (int rank = 0; rank < 2; ++rank) {
        tp_local_conv_weights(d_full, rank, d_local, 0);
        cudaDeviceSynchronize();
        std::vector<float> got((size_t) kLocal * 4);
        cudaMemcpy(got.data(), d_local, (size_t) kLocal * 4 * 4, cudaMemcpyDeviceToHost);
        int bad = 0;
        for (int i = 0; i < kLocal * 4; ++i)
            if (got[i] != ref[rank][i]) {
                if (bad < 4) std::fprintf(stderr, "  rank %d idx %d: got %f want %f\n", rank, i, got[i], ref[rank][i]);
                ++bad;
            }
        if (bad) { std::fprintf(stderr, "rank %d: %d/%d mismatch\n", rank, bad, kLocal * 4); fails += bad; }
        else std::printf("rank %d: conv split bit-exact (%d channels x 4 taps)\n", rank, kLocal);
    }
    cudaFree(d_full);
    cudaFree(d_local);
    if (fails) { std::fprintf(stderr, "FAIL: %d mismatches\n", fails); return 1; }
    std::printf("ALL PASS\n");
    return 0;
}
