// tests/cuda/nvfp4_q8_0_entry_parity.cpp - the ENTRY POINT the engine will call, against the CPU pool.
//
//     nvfp4_q8_0_entry_parity
//
// `nvfp4_q8_0_grouped_{gu,down}` are already proved bit-exact on their own (nvfp4_q8_0_cpu_grouped_parity).  This
// covers the layer above them: `native_expert_grouped_q8_0`, which carves the scratch the way the engine does,
// runs the two phases separately, and quantizes the intermediate with `quantize_q8_0` in between.  That handoff is
// the part a tensor-parallel split depends on, and it is the part no existing test exercised.
//
// Phase 1 + phase 2 must equal phase 0, and all three must equal the pool:
//
//   ff[e] = nvfp4_256_gu_rows(blob, ..., act[ent_tok[e]])        -> quantize -> nvfp4_256_rows(...)
#include "strata/kernels/cpu/nvfp4_avx2.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/quantize_act.hpp"

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

void fill_codes(uint8_t* b, size_t codes_bytes, std::mt19937& rng) {
    for (size_t i = 0; i < codes_bytes; i += 36) {
        for (int j = 0; j < 4; ++j) b[i + j] = (uint8_t) (0x28 | (rng() & 7));
        for (int j = 4; j < 36; ++j) b[i + j] = (uint8_t) (rng() & 0xFF);
    }
}

}  // namespace

int main() {
    // The engine's geometry, and the TP rank's half.
    struct Shape { int64_t H, FF; };
    const Shape shapes[] = { {2560, 640}, {2560, 320} };

    const ggml_type_traits_cpu* tw = ggml_get_type_traits_cpu(GGML_TYPE_NVFP4);
    const ggml_type_traits_cpu* ta = ggml_get_type_traits_cpu(tw->vec_dot_type);

    int bad = 0, tot = 0;

    for (const Shape& sh : shapes) {
        const int64_t H = sh.H, FF = sh.FF;
        const strata::kernels::NativeExpertLayout L = strata::kernels::native_expert_layout(40, 40, H, FF);
        const size_t ACT_H = (size_t) (H / 32) * 34;
        const size_t ACT_F = (size_t) (FF / 32) * 34;

        std::mt19937 rng(20261027u + (unsigned) FF);
        std::normal_distribution<float> nd(0.f, 1.f);

        const int NG = 3, NT = 4, NE = 6;
        std::vector<int32_t> start = {0, 2, 3, 6};
        std::vector<int32_t> etok = {0, 1, 1, 1, 2, 3};
        std::vector<int32_t> edst(NE);
        for (int i = 0; i < NE; ++i) edst[i] = (i * 5 + 1) % 8;

        std::vector<float> x((size_t) NT * H);
        for (int t = 0; t < NT; ++t) {
            const float mag = t == 0 ? 1.f : (t == 1 ? 0.03f : 7.f);
            for (int64_t i = 0; i < H; ++i) x[(size_t) t * H + i] = nd(rng) * mag;
        }
        std::vector<uint8_t> act_h((size_t) NT * ACT_H);
        for (int t = 0; t < NT; ++t) ta->from_float(x.data() + (size_t) t * H, act_h.data() + (size_t) t * ACT_H, H);

        std::vector<std::vector<uint8_t>> blobs(NG, std::vector<uint8_t>(L.bytes));
        for (int g = 0; g < NG; ++g) {
            fill_codes(blobs[g].data(), L.tail_off, rng);
            const float s[3] = { 0.75f + 0.1f * (float) g, 1.25f, 1.1f };
            std::memcpy(blobs[g].data() + L.tail_off, s, 12);
        }

        // ---- the pool, per entry ----
        std::vector<std::vector<float>> ref_ff(NE, std::vector<float>(FF));
        std::vector<std::vector<uint8_t>> hq(NE, std::vector<uint8_t>(ACT_F));
        for (int g = 0; g < NG; ++g)
            for (int e = start[g]; e < start[g + 1]; ++e) {
                const void* ap[1] = { act_h.data() + (size_t) etok[e] * ACT_H };
                float* op[1] = { ref_ff[e].data() };
                strata::kernels::cpu::nvfp4_256_gu_rows(blobs[g].data(), L.gu_row, L.up_off, (int) H, ap, 1, op, 0,
                                                        (int) FF, 0.75f + 0.1f * (float) g, 1.25f);
                ta->from_float(ref_ff[e].data(), hq[e].data(), FF);
            }
        std::vector<std::vector<float>> ref_out(NE, std::vector<float>(H));
        for (int g = 0; g < NG; ++g)
            for (int e = start[g]; e < start[g + 1]; ++e) {
                const void* hp[1] = { hq[e].data() };
                float* op[1] = { ref_out[e].data() };
                strata::kernels::cpu::nvfp4_256_rows(blobs[g].data() + L.down_off, L.d_row, (int) FF, hp, 1, op, 0,
                                                     (int) H, 1.1f);
            }

        // ---- device ----
        std::vector<uint8_t> arena((size_t) NG * L.bytes);
        for (int g = 0; g < NG; ++g) std::memcpy(arena.data() + (size_t) g * L.bytes, blobs[g].data(), L.bytes);

        uint8_t* d_blob = nullptr;
        unsigned long long* d_ptr = nullptr;
        int32_t *d_start = nullptr, *d_ng = nullptr, *d_tok = nullptr, *d_dst = nullptr;
        uint8_t* d_act = nullptr;
        float *d_out = nullptr, *d_ff = nullptr;
        void *d_scr = nullptr, *d_scr2 = nullptr;

        ck(cudaMalloc(&d_blob, arena.size()), "blob");
        ck(cudaMalloc(&d_ptr, NG * 8), "ptr");
        ck(cudaMalloc(&d_start, (NG + 1) * 4), "start");
        ck(cudaMalloc(&d_ng, 4), "ng");
        ck(cudaMalloc(&d_tok, NE * 4), "tok");
        ck(cudaMalloc(&d_dst, NE * 4), "dst");
        ck(cudaMalloc(&d_act, act_h.size()), "act");
        ck(cudaMalloc(&d_out, (size_t) 8 * H * 4), "out");
        ck(cudaMalloc(&d_ff, (size_t) NE * FF * 4), "ff");
        ck(cudaMalloc(&d_scr, strata::kernels::native_expert_scratch_bytes(NE, FF)), "scratch");
        ck(cudaMalloc(&d_scr2, strata::kernels::native_expert_scratch_bytes(NE, FF)), "scratch2");

        ck(cudaMemcpy(d_blob, arena.data(), arena.size(), cudaMemcpyHostToDevice), "cp blob");
        ck(cudaMemcpy(d_start, start.data(), (NG + 1) * 4, cudaMemcpyHostToDevice), "cp start");
        ck(cudaMemcpy(d_ng, &NG, 4, cudaMemcpyHostToDevice), "cp ng");
        ck(cudaMemcpy(d_tok, etok.data(), NE * 4, cudaMemcpyHostToDevice), "cp tok");
        ck(cudaMemcpy(d_dst, edst.data(), NE * 4, cudaMemcpyHostToDevice), "cp dst");
        ck(cudaMemcpy(d_act, act_h.data(), act_h.size(), cudaMemcpyHostToDevice), "cp act");
        std::vector<unsigned long long> hp(NG);
        for (int g = 0; g < NG; ++g) hp[g] = (unsigned long long) (uintptr_t) (d_blob + (size_t) g * L.bytes);
        ck(cudaMemcpy(d_ptr, hp.data(), NG * 8, cudaMemcpyHostToDevice), "cp ptr");

        // ---- phase 0: both ----
        strata::kernels::native_expert_grouped_q8_0(L, d_ptr, d_start, d_ng, d_dst, d_tok, NG, NE, d_act, (int) ACT_H,
                                                    d_scr, d_out, nullptr, 0);
        ck(cudaDeviceSynchronize(), "sync full");
        std::vector<float> full((size_t) 8 * H);
        ck(cudaMemcpy(full.data(), d_out, full.size() * 4, cudaMemcpyDeviceToHost), "cp full");

        // ---- the gu output itself, so a phase-2 mismatch can be traced to its input.
        // `native_expert_grouped_q8_0` writes the SwiGLU product into the scratch's own `h` region (scratched at
        // 2*fa, like `native_expert_grouped`), not into `d_ff`, so read it from there.
        {
            const size_t fa = (((size_t) NE * (size_t) FF * sizeof(float)) + 255) & ~(size_t) 255;
            std::vector<float> gff((size_t) NE * FF);
            ck(cudaMemcpy(gff.data(), (const uint8_t*) d_scr + 2 * fa, gff.size() * 4, cudaMemcpyDeviceToHost), "cp h");
            int fb = 0;
            for (int e = 0; e < NE; ++e) {
                int eb = 0;
                for (int64_t r = 0; r < FF; ++r)
                    if (std::memcmp(&gff[(size_t) e * FF + r], &ref_ff[e][(size_t) r], 4)) ++eb;
                if (eb) std::printf("  gu e%d: %d / %lld differ\n", e, eb, (long long) FF);
                fb += eb;
            }
            std::printf("H=%lld FF=%lld  gu vs pool: %d differ\n", (long long) H, (long long) FF, fb);
        }

        // ---- phase 1 then phase 2 (the split's protocol) ----
        strata::kernels::native_expert_grouped_q8_0(L, d_ptr, d_start, d_ng, d_dst, d_tok, NG, NE, d_act, (int) ACT_H,
                                                    d_scr2, d_ff, nullptr, 1);
        strata::kernels::native_expert_grouped_q8_0(L, d_ptr, d_start, d_ng, d_dst, d_tok, NG, NE, d_act, (int) ACT_H,
                                                    d_scr2, d_out, nullptr, 2);
        ck(cudaDeviceSynchronize(), "sync split");
        std::vector<float> split((size_t) 8 * H);
        ck(cudaMemcpy(split.data(), d_out, split.size() * 4, cudaMemcpyDeviceToHost), "cp split");

        int phase_bad = 0;
        for (size_t i = 0; i < full.size(); ++i)
            if (std::memcmp(&full[i], &split[i], 4)) {
                if (phase_bad < 4) std::printf("  phase1+2 vs phase0 at %zu: %.9g vs %.9g\n", i, full[i], split[i]);
                ++phase_bad;
            }
        std::printf("H=%lld FF=%lld  phase1+2 vs phase0: %d / %zu differ\n", (long long) H, (long long) FF, phase_bad,
                    full.size());
        bad += phase_bad;
        tot += (int) full.size();

        // ---- both against the pool ----
        for (int e = 0; e < NE; ++e) {
            int ebad = 0;
            for (int64_t i = 0; i < H; ++i) {
                ++tot;
                const float got = full[(size_t) edst[e] * H + i], want = ref_out[e][(size_t) i];
                if (std::memcmp(&got, &want, 4)) {
                    ++bad;
                    ++ebad;
                    if (bad <= 4) std::printf("  H=%lld e%d i%lld: got %.9g want %.9g\n", (long long) H, e, (long long) i,
                                              got, want);
                }
            }
            if (ebad != 0) std::printf("  e%d (group %d, tok %d): %d / %lld differ\n", e,
                                       e >= start[1] ? (e >= start[2] ? 2 : 1) : 0, etok[e], ebad, (long long) H);
        }

        cudaFree(d_blob); cudaFree(d_ptr); cudaFree(d_start); cudaFree(d_ng); cudaFree(d_tok); cudaFree(d_dst);
        cudaFree(d_act); cudaFree(d_out); cudaFree(d_ff); cudaFree(d_scr); cudaFree(d_scr2);
    }

    std::printf("entry point (full + split) vs CPU pool: %d / %d differ\n", bad, tot);
    const bool ok = bad == 0;
    std::printf("%s\n", ok ? "BIT-EXACT" : "MISMATCH");
    return ok ? 0 : 1;
}
