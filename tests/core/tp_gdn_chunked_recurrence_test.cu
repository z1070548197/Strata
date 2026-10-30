#include "strata/prefill/kernels.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {
constexpr int S = 128;
constexpr int HK = 16;
constexpr int HV = 48;
constexpr int HVT = HV / 2;
constexpr int C = 2 * HK * S + HV * S;
constexpr int CT = 2 * HK * S + HVT * S;
constexpr int VD = HV * S;
constexpr int VDT = HVT * S;
constexpr float EPS = 1e-6f;

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "FAIL: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}
float value(size_t i, unsigned seed, float scale = 0.25f) {
    unsigned x = ((unsigned) i ^ seed) * 1664525u + 1013904223u;
    return ((float) ((x >> 8) & 0xffff) / 32768.0f - 1.0f) * scale;
}
template <typename T> T* alloc_copy(const std::vector<T>& v) {
    T* p = nullptr;
    ck(cudaMalloc(&p, v.size() * sizeof(T)), "malloc");
    ck(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice), "h2d");
    return p;
}

void extract_local_h(const std::vector<float>& full, std::vector<float>& local, int T, int rank) {
    for (int t = 0; t < T; ++t) {
        const float* src = full.data() + (size_t) t * C;
        float* dst = local.data() + (size_t) t * CT;
        std::memcpy(dst, src, (size_t) 2 * HK * S * sizeof(float));
        std::memcpy(dst + 2 * HK * S, src + 2 * HK * S + rank * VDT, (size_t) VDT * sizeof(float));
    }
}
void extract_local_heads(const std::vector<float>& full, std::vector<float>& local, int T, int width, int rank) {
    for (int t = 0; t < T; ++t)
        std::memcpy(local.data() + (size_t) t * HVT * width,
                    full.data() + (size_t) t * HV * width + (size_t) rank * HVT * width,
                    (size_t) HVT * width * sizeof(float));
}
void extract_state(const std::vector<float>& full, std::vector<float>& local, int rank) {
    for (int row = 0; row < S; ++row)
        std::memcpy(local.data() + (size_t) row * HVT * S,
                    full.data() + (size_t) row * HV * S + (size_t) rank * HVT * S,
                    (size_t) HVT * S * sizeof(float));
}

void run_case(int T, int first) {
    std::vector<float> h((size_t) T * C), ab((size_t) T * 2 * HV), dt(HV), a(HV), z((size_t) T * VD),
                       gamma(S), state((size_t) S * HV * S);
    for (size_t i = 0; i < h.size(); ++i) h[i] = value(i, 3);
    for (size_t i = 0; i < ab.size(); ++i) ab[i] = value(i, 5);
    for (size_t i = 0; i < dt.size(); ++i) dt[i] = value(i, 7);
    for (size_t i = 0; i < a.size(); ++i) a[i] = -0.1f - value(i, 9, 0.08f);
    for (size_t i = 0; i < z.size(); ++i) z[i] = value(i, 11);
    for (size_t i = 0; i < gamma.size(); ++i) gamma[i] = 0.75f + value(i, 13);
    for (size_t i = 0; i < state.size(); ++i) state[i] = value(i, 17, 0.05f);

    float *dh = alloc_copy(h), *dab = alloc_copy(ab), *ddt = alloc_copy(dt), *da = alloc_copy(a),
          *dz = alloc_copy(z), *dgamma = alloc_copy(gamma), *dstate = alloc_copy(state);
    float *dgate = nullptr, *dbeta = nullptr, *dy = nullptr;
    uint16_t* dy16 = nullptr;
    ck(cudaMalloc(&dgate, (size_t) T * HV * sizeof(float)), "gate");
    ck(cudaMalloc(&dbeta, (size_t) T * HV * sizeof(float)), "beta");
    ck(cudaMalloc(&dy, (size_t) T * VD * sizeof(float)), "y");
    ck(cudaMalloc(&dy16, (size_t) T * VD * sizeof(uint16_t)), "y16");
    strata::prefill::gdn_gates(dab, ddt, da, dgate, dbeta, T, nullptr);
    if (first > 0 && first < T) {
        strata::prefill::gdn_recurrence_tp_canonical(dstate, dh, dgate, dbeta, dz, dgamma, EPS,
            dy, dy16, first, nullptr);
        strata::prefill::gdn_recurrence_tp_canonical(dstate, dh + (size_t) first * C,
            dgate + (size_t) first * HV, dbeta + (size_t) first * HV, dz + (size_t) first * VD,
            dgamma, EPS, dy + (size_t) first * VD, dy16 + (size_t) first * VD,
            T - first, nullptr);
    } else {
        strata::prefill::gdn_recurrence_tp_canonical(dstate, dh, dgate, dbeta, dz, dgamma, EPS,
            dy, dy16, T, nullptr);
    }
    ck(cudaDeviceSynchronize(), "full sync");
    std::vector<float> full_gate((size_t) T * HV), full_beta((size_t) T * HV), full_y((size_t) T * VD), full_state(state.size());
    ck(cudaMemcpy(full_gate.data(), dgate, full_gate.size() * 4, cudaMemcpyDeviceToHost), "gate d2h");
    ck(cudaMemcpy(full_beta.data(), dbeta, full_beta.size() * 4, cudaMemcpyDeviceToHost), "beta d2h");
    ck(cudaMemcpy(full_y.data(), dy, full_y.size() * 4, cudaMemcpyDeviceToHost), "y d2h");
    ck(cudaMemcpy(full_state.data(), dstate, full_state.size() * 4, cudaMemcpyDeviceToHost), "state d2h");

    for (int rank = 0; rank < 2; ++rank) {
        std::vector<float> lh((size_t) T * CT), lz((size_t) T * VDT), lstate((size_t) S * HVT * S);
        extract_local_h(h, lh, T, rank);
        extract_local_heads(z, lz, T, S, rank);
        extract_state(state, lstate, rank);
        float *ldh = alloc_copy(lh), *ldz = alloc_copy(lz), *lds = alloc_copy(lstate);
        float *lg = nullptr, *lb = nullptr, *ly = nullptr;
        uint16_t* ly16 = nullptr;
        ck(cudaMalloc(&lg, (size_t) T * HVT * 4), "local gate");
        ck(cudaMalloc(&lb, (size_t) T * HVT * 4), "local beta");
        ck(cudaMalloc(&ly, (size_t) T * VDT * 4), "local y");
        ck(cudaMalloc(&ly16, (size_t) T * VDT * 2), "local y16");
        strata::prefill::gdn_gates_local(dab, ddt, da, lg, lb, HVT, rank * HVT, T, nullptr);
        if (first > 0 && first < T) {
            strata::prefill::gdn_recurrence_local(lds, ldh, CT, lg, lb, ldz, dgamma, EPS, ly, ly16,
                HK, HVT, rank * HVT, first, nullptr);
            strata::prefill::gdn_recurrence_local(lds, ldh + (size_t) first * CT, CT,
                lg + (size_t) first * HVT, lb + (size_t) first * HVT, ldz + (size_t) first * VDT,
                dgamma, EPS, ly + (size_t) first * VDT, ly16 + (size_t) first * VDT,
                HK, HVT, rank * HVT, T - first, nullptr);
        } else {
            strata::prefill::gdn_recurrence_local(lds, ldh, CT, lg, lb, ldz, dgamma, EPS, ly, ly16,
                HK, HVT, rank * HVT, T, nullptr);
        }
        ck(cudaDeviceSynchronize(), "local sync");
        std::vector<float> got_gate((size_t) T * HVT), got_beta((size_t) T * HVT), got_y((size_t) T * VDT), got_state(lstate.size());
        ck(cudaMemcpy(got_gate.data(), lg, got_gate.size() * 4, cudaMemcpyDeviceToHost), "local gate d2h");
        ck(cudaMemcpy(got_beta.data(), lb, got_beta.size() * 4, cudaMemcpyDeviceToHost), "local beta d2h");
        ck(cudaMemcpy(got_y.data(), ly, got_y.size() * 4, cudaMemcpyDeviceToHost), "local y d2h");
        ck(cudaMemcpy(got_state.data(), lds, got_state.size() * 4, cudaMemcpyDeviceToHost), "local state d2h");
        std::vector<float> want_gate(got_gate.size()), want_beta(got_beta.size()), want_y(got_y.size()), want_state(got_state.size());
        extract_local_heads(full_gate, want_gate, T, 1, rank);
        extract_local_heads(full_beta, want_beta, T, 1, rank);
        extract_local_heads(full_y, want_y, T, S, rank);
        extract_state(full_state, want_state, rank);
        const auto same = [&](const std::vector<float>& got, const std::vector<float>& want, const char* what) {
            for (size_t i = 0; i < got.size(); ++i) if (got[i] != want[i]) {
                std::fprintf(stderr, "FAIL: T=%d split=%d rank=%d %s[%zu] %.9g != %.9g\n",
                             T, first, rank, what, i, got[i], want[i]);
                std::exit(1);
            }
        };
        same(got_gate, want_gate, "gate"); same(got_beta, want_beta, "beta");
        same(got_y, want_y, "y"); same(got_state, want_state, "state");
        cudaFree(ldh); cudaFree(ldz); cudaFree(lds); cudaFree(lg); cudaFree(lb); cudaFree(ly); cudaFree(ly16);
    }
    cudaFree(dh); cudaFree(dab); cudaFree(ddt); cudaFree(da); cudaFree(dz); cudaFree(dgamma); cudaFree(dstate);
    cudaFree(dgate); cudaFree(dbeta); cudaFree(dy); cudaFree(dy16);
}
}  // namespace

int main() {
    run_case(1, 0);
    run_case(2, 1);
    run_case(7, 3);
    run_case(64, 17);
    run_case(1536, 0);
    std::printf("ALL PASS: TP chunked prefill local gates, recurrence, and state\n");
    return 0;
}
