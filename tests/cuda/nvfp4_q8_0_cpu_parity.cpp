// tests/cuda/nvfp4_q8_0_cpu_parity.cpp - the GPU NVFP4 expert rows against the CPU pool's kernels, bit for bit.
//
//     nvfp4_q8_0_cpu_parity [seed]
//
// The engine runs a routed expert on the CPU pool when it is a miss and on the GPU when it is resident. Those two
// paths must agree to the last bit, or changing which experts are resident changes the output tokens - measured on
// this pack: the first divergence moved from token 23 to token 2 as the affected fraction grew. `nvfp4_256_rows` and
// `nvfp4_256_gu_rows` (src/kernels/cpu/nvfp4_avx2.cpp) are what the pool computes on an AVX2 CPU, and
// nvfp4_avx512_parity.cpp already asserts they are bit-equal to ggml-cpu's AVX2 dot product. So this test compares
// the GPU kernel to them and asserts zero differing rows, not a tolerance.
//
// Shapes cover the engine's widths (n = 2560 for down, 5120 for gate/up) and edge cases (n = 192/640, ROWS up to
// 4096, n_tok up to 8). Activations are quantized by ggml's own from_float for NVFP4's vec_dot_type, so they are
// exactly the blocks the pool would be handed. Weight blocks are filled with raw bytes so every E2M1 code and every
// ue4m3 scale is exercised, 0x7F (read as zero) included.
#include "strata/kernels/cpu/nvfp4_avx2.hpp"
#include "strata/kernels/nvfp4_q8_0_cpu_order.hpp"

#include "ggml.h"
#include "ggml-cpu.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e)); std::exit(1); }
}

}  // namespace

int main(int argc, char** argv) {
    const unsigned base_seed = argc > 1 ? (unsigned) std::atoi(argv[1]) : 20261027u;

    struct Shape {
        int n, rows, ntok;
    };
    const Shape shapes[] = {
        {2560, 1024, 4},  // the down projection's width, a real expert's row count
        {5120, 512, 4},   // gate/up width
        {2560, 2048, 1},  // a single-token window (decode), many rows
        {640, 4096, 4},   // narrow but tall
        {192, 2048, 8},   // small n, a full 8-token window
        {1280, 1024, 3},  // odd n_tok
    };

    const ggml_type_traits_cpu* tw = ggml_get_type_traits_cpu(GGML_TYPE_NVFP4);
    const ggml_type_traits_cpu* ta = ggml_get_type_traits_cpu(tw->vec_dot_type);

    int total_differ = 0, total_cmp = 0;
    int gu_differ = 0, gu_cmp = 0;

    for (const Shape& sh : shapes) {
        const int N = sh.n, ROWS = sh.rows, NT = sh.ntok;
        const size_t ROW_BYTES = (size_t) N / 64 * 36;   // block_nvfp4: 4 ue4m3 bytes + 32 nibble bytes
        const size_t ACT_STRIDE = (size_t) N / 32 * 34;  // block_q8_0: fp16 d + 32 int8

        for (unsigned rep = 0; rep < 3; ++rep) {
            const unsigned seed = base_seed * 7919u + (unsigned) N + rep;
            std::mt19937 rng(seed);
            std::normal_distribution<float> nd(0.f, 1.f);

            std::vector<uint8_t> w((size_t) ROWS * ROW_BYTES);
            for (uint8_t& b : w) b = (uint8_t) (rng() & 0xFF);

            // three activation scales, so the ue4m3 block scales span tiny / normal / large
            std::vector<uint8_t> act((size_t) ACT_STRIDE * NT);
            for (int t = 0; t < NT; ++t) {
                std::vector<float> x(N);
                const float mag = t == 0 ? 1.f : (t == 1 ? 0.02f : 8.f);
                for (float& v : x) v = nd(rng) * mag;
                ta->from_float(x.data(), act.data() + (size_t) t * ACT_STRIDE, N);
            }

            // ---- plain rows ----
            std::vector<std::vector<float>> ref(NT, std::vector<float>(ROWS));
            std::vector<const void*> ap(NT);
            std::vector<float*> rp(NT);
            for (int t = 0; t < NT; ++t) {
                ap[t] = act.data() + (size_t) t * ACT_STRIDE;
                rp[t] = ref[t].data();
            }
            strata::kernels::cpu::nvfp4_256_rows(w.data(), ROW_BYTES, N, ap.data(), NT, rp.data(), 0, ROWS, 1.f);

            uint8_t* d_w = nullptr;
            uint8_t* d_act = nullptr;
            float* d_out = nullptr;
            ck(cudaMalloc(&d_w, w.size()), "cudaMalloc w");
            ck(cudaMalloc(&d_act, act.size()), "cudaMalloc act");
            ck(cudaMalloc(&d_out, (size_t) ROWS * NT * sizeof(float)), "cudaMalloc out");
            ck(cudaMemcpy(d_w, w.data(), w.size(), cudaMemcpyHostToDevice), "copy w");
            ck(cudaMemcpy(d_act, act.data(), act.size(), cudaMemcpyHostToDevice), "copy act");
            strata::kernels::nvfp4_q8_0_rows(d_w, ROW_BYTES, N, d_act, (int) ACT_STRIDE, NT, d_out, 0, ROWS, 1.f);
            ck(cudaDeviceSynchronize(), "sync rows");
            std::vector<float> got((size_t) ROWS * NT);
            ck(cudaMemcpy(got.data(), d_out, got.size() * sizeof(float), cudaMemcpyDeviceToHost), "copy back");

            int differ = 0;
            for (int t = 0; t < NT; ++t)
                for (int r = 0; r < ROWS; ++r)
                    if (std::memcmp(&got[(size_t) t * ROWS + r], &ref[t][r], sizeof(float))) ++differ;
            total_differ += differ;
            total_cmp += ROWS * NT;
            if (differ)
                std::printf("  rows n=%d rows=%d tok=%d seed=%u: %d / %d differ\n", N, ROWS, NT, seed, differ,
                            ROWS * NT);
            // d_act stays alive: the gate/up section below reads it.
            cudaFree(d_w);
            cudaFree(d_out);

            // ---- gate/up + SiLU through the blob tail's scales ----
            const int F = ROWS / 2;
            const size_t BLOB = (size_t) ROWS * ROW_BYTES + 16;
            std::vector<uint8_t> blob(BLOB);
            for (uint8_t& b : blob) b = (uint8_t) (rng() & 0xFF);
            const float s_gate = 0.75f, s_up = 1.25f;
            std::memcpy(blob.data() + BLOB - 16, &s_gate, 4);
            std::memcpy(blob.data() + BLOB - 12, &s_up, 4);
            const size_t UP_OFF = (size_t) F * ROW_BYTES;

            std::vector<std::vector<float>> refg(NT, std::vector<float>(F));
            std::vector<float*> rg(NT);
            for (int t = 0; t < NT; ++t) rg[t] = refg[t].data();
            strata::kernels::cpu::nvfp4_256_gu_rows(blob.data(), ROW_BYTES, UP_OFF, N, ap.data(), NT, rg.data(), 0, F,
                                                    s_gate, s_up);

            uint8_t* d_blob = nullptr;
            float* d_ff = nullptr;
            ck(cudaMalloc(&d_blob, BLOB), "cudaMalloc blob");
            ck(cudaMalloc(&d_ff, (size_t) F * NT * sizeof(float)), "cudaMalloc ff");
            ck(cudaMemcpy(d_blob, blob.data(), BLOB, cudaMemcpyHostToDevice), "copy blob");
            strata::kernels::nvfp4_q8_0_gu_rows(d_blob, ROW_BYTES, UP_OFF, N, d_act, (int) ACT_STRIDE, NT, d_ff, 0, F,
                                                s_gate, s_up);
            ck(cudaDeviceSynchronize(), "sync gu");
            std::vector<float> gg((size_t) F * NT);
            ck(cudaMemcpy(gg.data(), d_ff, gg.size() * sizeof(float), cudaMemcpyDeviceToHost), "copy gu back");

            int gdiffer = 0;
            for (int t = 0; t < NT; ++t)
                for (int r = 0; r < F; ++r)
                    if (std::memcmp(&gg[(size_t) t * F + r], &refg[t][r], sizeof(float))) ++gdiffer;
            gu_differ += gdiffer;
            gu_cmp += F * NT;
            if (gdiffer)
                std::printf("  gu   n=%d rows=%d tok=%d seed=%u: %d / %d differ\n", N, F, NT, seed, gdiffer, F * NT);
            cudaFree(d_blob);
            cudaFree(d_ff);
            cudaFree(d_act);
        }
    }

    std::printf("nvfp4_q8_0 rows : %d / %d differ\n", total_differ, total_cmp);
    std::printf("nvfp4_q8_0 gu   : %d / %d differ\n", gu_differ, gu_cmp);
    const bool ok = total_differ == 0 && gu_differ == 0;
    std::printf("%s\n", ok ? "BIT-EXACT" : "MISMATCH");
    return ok ? 0 : 1;
}
