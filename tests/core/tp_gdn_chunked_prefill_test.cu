#include "strata/prefill/kernels.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {
constexpr int S = 128;
constexpr int HK = 16;
constexpr int HV = 48;
constexpr int QK = HK * S;
constexpr int C = 2 * QK + HV * S;
constexpr int HVT = HV / 2;
constexpr int CT = 2 * QK + HVT * S;
constexpr float EPS = 1e-6f;

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "FAIL: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

float value(size_t i, unsigned seed) {
    unsigned x = (unsigned) i ^ seed;
    x = x * 1664525u + 1013904223u;
    return ((float) ((x >> 8) & 0xffff) / 32768.0f - 1.0f) * 0.25f;
}

void extract_local(const std::vector<float>& full, std::vector<float>& local, int T, int rank) {
    for (int t = 0; t < T; ++t) {
        const float* src = full.data() + (size_t) t * C;
        float* dst = local.data() + (size_t) t * CT;
        for (int i = 0; i < 2 * QK; ++i) dst[i] = src[i];
        const int v0 = 2 * QK + rank * HVT * S;
        for (int i = 0; i < HVT * S; ++i) dst[2 * QK + i] = src[v0 + i];
    }
}

void extract_channels(const std::vector<float>& full, std::vector<float>& local, int width, int rank) {
    for (int c = 0; c < 2 * QK; ++c)
        for (int j = 0; j < width; ++j) local[(size_t) c * width + j] = full[(size_t) c * width + j];
    const int v0 = 2 * QK + rank * HVT * S;
    for (int c = 0; c < HVT * S; ++c)
        for (int j = 0; j < width; ++j)
            local[(size_t) (2 * QK + c) * width + j] = full[(size_t) (v0 + c) * width + j];
}

void run_case(int T, int first) {
    std::vector<float> qkv((size_t) T * C), weights((size_t) C * 4), history((size_t) C * 3);
    for (size_t i = 0; i < qkv.size(); ++i) qkv[i] = value(i, 11);
    for (size_t i = 0; i < weights.size(); ++i) weights[i] = value(i, 29);
    for (size_t i = 0; i < history.size(); ++i) history[i] = value(i, 47);

    float *d_qkv = nullptr, *d_weights = nullptr, *d_history = nullptr, *d_out = nullptr;
    ck(cudaMalloc(&d_qkv, qkv.size() * sizeof(float)), "full qkv");
    ck(cudaMalloc(&d_weights, weights.size() * sizeof(float)), "full weights");
    ck(cudaMalloc(&d_history, history.size() * sizeof(float)), "full history");
    ck(cudaMalloc(&d_out, qkv.size() * sizeof(float)), "full out");
    ck(cudaMemcpy(d_qkv, qkv.data(), qkv.size() * sizeof(float), cudaMemcpyHostToDevice), "full qkv h2d");
    ck(cudaMemcpy(d_weights, weights.data(), weights.size() * sizeof(float), cudaMemcpyHostToDevice), "full weights h2d");
    ck(cudaMemcpy(d_history, history.data(), history.size() * sizeof(float), cudaMemcpyHostToDevice), "full history h2d");
    strata::prefill::gdn_conv(d_history, d_qkv, d_weights, d_out, T, EPS, nullptr);
    ck(cudaDeviceSynchronize(), "full sync");
    std::vector<float> full_out(qkv.size()), full_history(history.size());
    ck(cudaMemcpy(full_out.data(), d_out, full_out.size() * sizeof(float), cudaMemcpyDeviceToHost), "full out d2h");
    ck(cudaMemcpy(full_history.data(), d_history, full_history.size() * sizeof(float), cudaMemcpyDeviceToHost), "full history d2h");

    for (int rank = 0; rank < 2; ++rank) {
        std::vector<float> local_qkv((size_t) T * CT), local_weights((size_t) CT * 4), local_history((size_t) CT * 3);
        extract_local(qkv, local_qkv, T, rank);
        extract_channels(weights, local_weights, 4, rank);
        extract_channels(history, local_history, 3, rank);
        float *lq = nullptr, *lw = nullptr, *lh = nullptr, *lo = nullptr;
        ck(cudaMalloc(&lq, local_qkv.size() * sizeof(float)), "local qkv");
        ck(cudaMalloc(&lw, local_weights.size() * sizeof(float)), "local weights");
        ck(cudaMalloc(&lh, local_history.size() * sizeof(float)), "local history");
        ck(cudaMalloc(&lo, local_qkv.size() * sizeof(float)), "local out");
        ck(cudaMemcpy(lq, local_qkv.data(), local_qkv.size() * sizeof(float), cudaMemcpyHostToDevice), "local qkv h2d");
        ck(cudaMemcpy(lw, local_weights.data(), local_weights.size() * sizeof(float), cudaMemcpyHostToDevice), "local weights h2d");
        ck(cudaMemcpy(lh, local_history.data(), local_history.size() * sizeof(float), cudaMemcpyHostToDevice), "local history h2d");
        if (first > 0 && first < T) {
            strata::prefill::gdn_conv_local(lh, lq, lw, lo, CT, 2 * HK, first, EPS, nullptr);
            strata::prefill::gdn_conv_local(lh, lq + (size_t) first * CT, lw, lo + (size_t) first * CT,
                                            CT, 2 * HK, T - first, EPS, nullptr);
        } else {
            strata::prefill::gdn_conv_local(lh, lq, lw, lo, CT, 2 * HK, T, EPS, nullptr);
        }
        ck(cudaDeviceSynchronize(), "local sync");
        std::vector<float> got_out(local_qkv.size()), got_history(local_history.size());
        ck(cudaMemcpy(got_out.data(), lo, got_out.size() * sizeof(float), cudaMemcpyDeviceToHost), "local out d2h");
        ck(cudaMemcpy(got_history.data(), lh, got_history.size() * sizeof(float), cudaMemcpyDeviceToHost), "local history d2h");
        std::vector<float> want_out(local_qkv.size()), want_history(local_history.size());
        extract_local(full_out, want_out, T, rank);
        extract_channels(full_history, want_history, 3, rank);
        for (size_t i = 0; i < got_out.size(); ++i) {
            if (got_out[i] != want_out[i]) {
                std::fprintf(stderr, "FAIL: T=%d split=%d rank=%d out[%zu] %.9g != %.9g\n",
                             T, first, rank, i, got_out[i], want_out[i]);
                std::exit(1);
            }
        }
        for (size_t i = 0; i < got_history.size(); ++i) {
            if (got_history[i] != want_history[i]) {
                std::fprintf(stderr, "FAIL: T=%d split=%d rank=%d history[%zu] %.9g != %.9g\n",
                             T, first, rank, i, got_history[i], want_history[i]);
                std::exit(1);
            }
        }
        cudaFree(lq); cudaFree(lw); cudaFree(lh); cudaFree(lo);
    }
    cudaFree(d_qkv); cudaFree(d_weights); cudaFree(d_history); cudaFree(d_out);
}
}  // namespace

int main() {
    int devices = 0;
    ck(cudaGetDeviceCount(&devices), "device count");
    if (devices < 1) return 77;
    run_case(1, 0);
    run_case(2, 1);
    run_case(7, 3);
    run_case(64, 17);
    run_case(1536, 0);
    std::printf("ALL PASS: TP chunked prefill local conv output and history\n");
    return 0;
}
