#include "strata/kernels/ple.hpp"
#include "strata/kernels/ngram.hpp"
#include "strata/prefill/kernels.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

constexpr int S = 128;
constexpr int HK = 16;
constexpr int HV = 24;
constexpr int C = 2 * HK * S + HV * S;
constexpr int VD = HV * S;
constexpr int PROMPT = 5;
constexpr int TOTAL = PROMPT + 1;

void gpu(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "FAIL: %s: %s\n", what, cudaGetErrorString(status));
        std::exit(1);
    }
}

void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        std::exit(1);
    }
}

float value(size_t i, unsigned seed, float scale = 0.25f) {
    unsigned x = ((unsigned) i ^ seed) * 1664525u + 1013904223u;
    return ((float) ((x >> 8) & 0xffff) / 32768.0f - 1.0f) * scale;
}

template <class T>
T* upload(const std::vector<T>& values) {
    T* device = nullptr;
    gpu(cudaMalloc(&device, values.size() * sizeof(T)), "allocate fixture buffer");
    gpu(cudaMemcpy(device, values.data(), values.size() * sizeof(T), cudaMemcpyHostToDevice),
        "upload fixture buffer");
    return device;
}

void gdn_handoff() {
    std::vector<float> h((size_t) TOTAL * C), ab((size_t) TOTAL * 2 * HV), dt(HV), a(HV),
                       z((size_t) TOTAL * VD), gamma(S), state((size_t) S * HV * S);
    for (size_t i = 0; i < h.size(); ++i) h[i] = value(i, 3);
    for (size_t i = 0; i < ab.size(); ++i) ab[i] = value(i, 5);
    for (size_t i = 0; i < dt.size(); ++i) dt[i] = value(i, 7);
    for (size_t i = 0; i < a.size(); ++i) a[i] = -0.1f - value(i, 9, 0.08f);
    for (size_t i = 0; i < z.size(); ++i) z[i] = value(i, 11);
    for (size_t i = 0; i < gamma.size(); ++i) gamma[i] = 0.75f + value(i, 13);
    for (size_t i = 0; i < state.size(); ++i) state[i] = value(i, 17, 0.05f);

    float *dh = upload(h), *dab = upload(ab), *ddt = upload(dt), *da = upload(a),
          *dz = upload(z), *dgamma = upload(gamma), *continuous = upload(state), *handoff = upload(state);
    float *gate = nullptr, *beta = nullptr, *y_cont = nullptr, *y_handoff = nullptr;
    uint16_t *y16_cont = nullptr, *y16_handoff = nullptr;
    gpu(cudaMalloc(&gate, (size_t) TOTAL * HV * sizeof(float)), "allocate gates");
    gpu(cudaMalloc(&beta, (size_t) TOTAL * HV * sizeof(float)), "allocate betas");
    gpu(cudaMalloc(&y_cont, (size_t) TOTAL * VD * sizeof(float)), "allocate continuous output");
    gpu(cudaMalloc(&y_handoff, (size_t) TOTAL * VD * sizeof(float)), "allocate handoff output");
    gpu(cudaMalloc(&y16_cont, (size_t) TOTAL * VD * sizeof(uint16_t)), "allocate continuous bf16");
    gpu(cudaMalloc(&y16_handoff, (size_t) TOTAL * VD * sizeof(uint16_t)), "allocate handoff bf16");

    strata::prefill::gdn_gates_local(dab, ddt, da, gate, beta, HV, 0, TOTAL, nullptr);
    strata::prefill::gdn_recurrence_local(continuous, dh, C, gate, beta, dz, dgamma, 1e-6f,
                                          y_cont, y16_cont, HK, HV, 0, TOTAL, nullptr);
    strata::prefill::gdn_recurrence_local(handoff, dh, C, gate, beta, dz, dgamma, 1e-6f,
                                          y_handoff, y16_handoff, HK, HV, 0, PROMPT, nullptr);
    strata::prefill::gdn_recurrence_local(
        handoff, dh + (size_t) PROMPT * C, C,
        gate + (size_t) PROMPT * HV, beta + (size_t) PROMPT * HV,
        dz + (size_t) PROMPT * VD, dgamma, 1e-6f,
        y_handoff + (size_t) PROMPT * VD, y16_handoff + (size_t) PROMPT * VD,
        HK, HV, 0, 1, nullptr);
    gpu(cudaDeviceSynchronize(), "finish GDN handoff");

    std::vector<float> state_a(state.size()), state_b(state.size()), out_a(VD), out_b(VD);
    gpu(cudaMemcpy(state_a.data(), continuous, state_a.size() * sizeof(float), cudaMemcpyDeviceToHost),
        "read continuous state");
    gpu(cudaMemcpy(state_b.data(), handoff, state_b.size() * sizeof(float), cudaMemcpyDeviceToHost),
        "read handoff state");
    gpu(cudaMemcpy(out_a.data(), y_cont + (size_t) PROMPT * VD, out_a.size() * sizeof(float),
                   cudaMemcpyDeviceToHost), "read continuous first decode output");
    gpu(cudaMemcpy(out_b.data(), y_handoff + (size_t) PROMPT * VD, out_b.size() * sizeof(float),
                   cudaMemcpyDeviceToHost), "read handoff first decode output");
    check(std::memcmp(state_a.data(), state_b.data(), state_a.size() * sizeof(float)) == 0,
          "GDN prefill-to-next-token state continuity");
    check(std::memcmp(out_a.data(), out_b.data(), out_a.size() * sizeof(float)) == 0,
          "GDN first-token output continuity");

    cudaFree(dh); cudaFree(dab); cudaFree(ddt); cudaFree(da); cudaFree(dz); cudaFree(dgamma);
    cudaFree(continuous); cudaFree(handoff); cudaFree(gate); cudaFree(beta);
    cudaFree(y_cont); cudaFree(y_handoff); cudaFree(y16_cont); cudaFree(y16_handoff);
}

void ple_handoff() {
    constexpr int D = strata::kernels::NG_HC_DIM;
    constexpr int H = strata::kernels::NG_HIST;
    std::vector<float> initial((size_t) H * D), rows((size_t) TOTAL * D);
    for (size_t i = 0; i < initial.size(); ++i) initial[i] = value(i, 23, 0.05f);
    for (size_t i = 0; i < rows.size(); ++i) rows[i] = value(i, 29, 0.2f);
    float* continuous = upload(initial);
    float* handoff = upload(initial);
    float* device_rows = upload(rows);

    for (int t = 0; t < TOTAL; ++t)
        strata::kernels::ple_history_advance(continuous, device_rows + (size_t) t * D, nullptr);
    for (int t = 0; t < PROMPT; ++t)
        strata::kernels::ple_history_advance(handoff, device_rows + (size_t) t * D, nullptr);
    strata::kernels::ple_history_advance(handoff, device_rows + (size_t) PROMPT * D, nullptr);
    gpu(cudaDeviceSynchronize(), "finish PLE handoff");

    std::vector<float> history_a(initial.size()), history_b(initial.size());
    gpu(cudaMemcpy(history_a.data(), continuous, history_a.size() * sizeof(float), cudaMemcpyDeviceToHost),
        "read continuous PLE history");
    gpu(cudaMemcpy(history_b.data(), handoff, history_b.size() * sizeof(float), cudaMemcpyDeviceToHost),
        "read handoff PLE history");
    check(std::memcmp(history_a.data(), history_b.data(), history_a.size() * sizeof(float)) == 0,
          "PLE history continuity");

    const int32_t tokens[TOTAL] = {101, 203, 307, 401, 503, 607};
    int32_t prev_cont[TOTAL * 2] = {};
    int32_t window[2] = {17, 19};
    for (int t = 0; t < TOTAL; ++t) {
        prev_cont[2 * t] = window[0];
        prev_cont[2 * t + 1] = window[1];
        window[0] = window[1];
        window[1] = tokens[t];
    }
    uint32_t rows_cont[TOTAL * strata::kernels::PLE_N_HEADS] = {};
    strata::kernels::ngram_rows(tokens, prev_cont, TOTAL, strata::kernels::ple_artifact_consts(), rows_cont);

    int32_t prompt_prev[PROMPT * 2] = {};
    window[0] = 17; window[1] = 19;
    for (int t = 0; t < PROMPT; ++t) {
        prompt_prev[2 * t] = window[0];
        prompt_prev[2 * t + 1] = window[1];
        window[0] = window[1];
        window[1] = tokens[t];
    }
    uint32_t rows_prompt[PROMPT * strata::kernels::PLE_N_HEADS] = {};
    strata::kernels::ngram_rows(tokens, prompt_prev, PROMPT, strata::kernels::ple_artifact_consts(), rows_prompt);
    const int32_t decode_prev[2] = {window[0], window[1]};
    uint32_t rows_decode[strata::kernels::PLE_N_HEADS] = {};
    strata::kernels::ngram_rows(tokens + PROMPT, decode_prev, 1,
                                strata::kernels::ple_artifact_consts(), rows_decode);
    check(std::memcmp(rows_cont, rows_prompt, sizeof(rows_prompt)) == 0,
          "PLE prompt row continuity");
    check(std::memcmp(rows_cont + PROMPT * strata::kernels::PLE_N_HEADS,
                      rows_decode, sizeof(rows_decode)) == 0,
          "PLE first decode row continuity");
    check(window[0] == tokens[PROMPT - 2] && window[1] == tokens[PROMPT - 1],
          "PLE prompt token window");

    cudaFree(continuous); cudaFree(handoff); cudaFree(device_rows);
}

}  // namespace

int main() {
    int devices = 0;
    gpu(cudaGetDeviceCount(&devices), "device count");
    if (devices < 1) return 77;
    gdn_handoff();
    ple_handoff();
    std::fprintf(stderr, "PASS: tp_state_handoff_test\n");
    return 0;
}
