// tests/cuda/nvfp4_q8_0_cpu_grouped_parity.cpp - the GROUPED native expert path against what the CPU pool computes.
//
//     nvfp4_q8_0_cpu_grouped_parity [seed]
//
// `native_expert_grouped` (iq_kernels.cu) takes a `block_q8_1` activation and reduces each row with a 16- or 8-lane
// sub-warp; the CPU pool takes `block_q8_0` and keeps eight fp32 accumulators in a fixed FMA order. Those two were
// never the same number, which is why a resident expert and a missed one disagreed and why the resident SET changed
// the generated tokens. `nvfp4_q8_0_grouped_gu` / `_down` implement the native grouped SHAPE with the pool's
// ARITHMETIC, so this compares them against the pool kernels themselves:
//
//   gu   : nvfp4_256_gu_rows(blob, gu_row, up_off, n_embd, act[e], 1, ff, 0, n_ff, s_gate, s_up)
//   down : nvfp4_256_rows(blob + down_off, d_row, n_ff, act2[e], 1, out, 0, n_embd, s_down)
//
// with the intermediate quantized by ggml's own Q8_0 from_float, which `quantize_q8_0` reproduces byte for byte.
// Groups hold several entries and entries are deliberately NOT one-per-token, so `ent_tok` is exercised as a real
// indirection (two experts routed to the same token, one routed twice).
#include "strata/kernels/cpu/nvfp4_avx2.hpp"
#include "strata/kernels/nvfp4_q8_0_cpu_order.hpp"
#include "strata/kernels/quantize_act.hpp"

#include "ggml.h"
#include "ggml-cpu.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

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

// A VALID block_nvfp4: four ue4m3 sub-block scales with a sane exponent (0x28|m -> 0.5..0.875) and raw nibbles.
// Raw random scale bytes reach exponent 15 (value 2^8), whose products overflow float over a full row - a fault of
// the fixture, not of either implementation.
// `codes_bytes` is the blob's packed-code region (a whole number of 36-byte blocks), NOT including the 16-byte
// tail - writing the tail in 36-byte strides overruns the blob by 20 bytes and corrupts the heap.
void fill_codes(uint8_t* b, size_t codes_bytes, std::mt19937& rng) {
    for (size_t i = 0; i < codes_bytes; i += 36) {
        b[i + 0] = (uint8_t) (0x28 | (rng() & 7));
        b[i + 1] = (uint8_t) (0x28 | (rng() & 7));
        b[i + 2] = (uint8_t) (0x28 | (rng() & 7));
        b[i + 3] = (uint8_t) (0x28 | (rng() & 7));
        for (int j = 4; j < 36; ++j) b[i + j] = (uint8_t) (rng() & 0xFF);
    }
}

}  // namespace

int main(int argc, char** argv) {
    const unsigned seed = argc > 1 ? (unsigned) std::atoi(argv[1]) : 20261027u;

    // The engine's real geometry: H = 2560, and a TP rank's local n_ff = 640/2 = 320.  Both are multiples of 64
    // (NVFP4's block) and 32 (Q8_0's block).
    struct Shape { int64_t H, FF; };
    const Shape shapes[] = { {2560, 640}, {2560, 320}, {2560, 128} };

    const ggml_type_traits_cpu* tw = ggml_get_type_traits_cpu(GGML_TYPE_NVFP4);
    const ggml_type_traits_cpu* ta = ggml_get_type_traits_cpu(tw->vec_dot_type);

    int gu_bad = 0, gu_tot = 0, dn_bad = 0, dn_tot = 0;

    for (const Shape& sh : shapes) {
        const int64_t H = sh.H, FF = sh.FF;
        const size_t GU_ROW = (size_t) (H / 64) * 36;   // gate/up rows are n_embd wide
        const size_t D_ROW = (size_t) (FF / 64) * 36;   // down rows are n_ff wide
        const size_t UP_OFF = (size_t) FF * GU_ROW;
        const size_t DOWN_OFF = 2 * UP_OFF;
        const size_t BLOB = DOWN_OFF + (size_t) H * D_ROW + 16;
        const size_t TAIL = BLOB - 16;
        const size_t ACT_H = (size_t) (H / 32) * 34;
        const size_t ACT_F = (size_t) (FF / 32) * 34;

        std::mt19937 rng(seed * 7919u + (unsigned) FF);
        std::normal_distribution<float> nd(0.f, 1.f);

        const int NG = 3, NT = 4;
        // Entries: group 0 gets tokens {0,1}, group 1 gets {1}, group 2 gets {1,2,3} - so two groups share token 1
        // and no group's entries are a contiguous token range.
        std::vector<int32_t> start = {0, 2, 3, 6};
        std::vector<int32_t> etok = {0, 1, 1, 1, 2, 3};
        const int NE = 6;
        std::vector<int32_t> edst(NE);
        for (int i = 0; i < NE; ++i) edst[i] = (i * 5 + 1) % 8;   // scattered, and not the entry index

        // Activations: NT tokens of the pool's own Q8_0 blocks, AND the fp32 per-block scales the pool actually
        // multiplies by.  `ta->from_float` writes ggml's block_q8_0, whose multiplier is the fp16 `d`; the pool
        // keeps an fp32 `ActQ::scale` instead, and those two are 4.761e-04 relative apart (pure fp16 rounding).
        // The kernel takes the fp32 array through `sc_lo`, so the test has to produce the same values the engine's
        // `quantize_q8_0_scaled` does: q = round(x / amax * 127), scale = amax / 127 in fp32.
        std::vector<float> x((size_t) NT * H);
        for (int t = 0; t < NT; ++t) {
            const float mag = t == 0 ? 1.f : (t == 1 ? 0.03f : 7.f);
            for (int64_t i = 0; i < H; ++i) x[(size_t) t * H + i] = nd(rng) * mag;
        }
        std::vector<uint8_t> act_h((size_t) NT * ACT_H);
        std::vector<float> sc_h((size_t) NT * (H / 32));
        for (int t = 0; t < NT; ++t) {
            ta->from_float(x.data() + (size_t) t * H, act_h.data() + (size_t) t * ACT_H, H);
            for (int64_t b = 0; b < H / 32; ++b) {
                const float* xs = x.data() + (size_t) t * H + b * 32;
                float amax = 0.f;
                for (int j = 0; j < 32; ++j) amax = std::max(amax, std::fabs(xs[j]));
                sc_h[(size_t) t * (H / 32) + b] = amax / 127.0f;   // the pool's fp32 multiplier
            }
        }

        // Expert blobs, and the tail scales that NVFP4 puts at the blob's end.
        std::vector<std::vector<uint8_t>> blobs(NG, std::vector<uint8_t>(BLOB));
        for (int g = 0; g < NG; ++g) {
            fill_codes(blobs[g].data(), TAIL, rng);
            const float s[3] = { 0.75f + 0.1f * (float) g, 1.25f, 1.1f };
            std::memcpy(blobs[g].data() + TAIL, s, 12);
        }

        // ---- CPU reference, per entry, with the pool's kernels ----
        std::vector<std::vector<float>> ref_ff(NE, std::vector<float>(FF));
        std::vector<std::vector<uint8_t>> hq(NE, std::vector<uint8_t>(ACT_F));
        for (int g = 0; g < NG; ++g) {
            for (int e = start[g]; e < start[g + 1]; ++e) {
                const float sg = 0.75f + 0.1f * (float) g, su = 1.25f;
                const void* ap[1] = { act_h.data() + (size_t) etok[e] * ACT_H };
                float* op[1] = { ref_ff[e].data() };
                strata::kernels::cpu::nvfp4_256_gu_rows(blobs[g].data(), GU_ROW, UP_OFF, (int) H, ap, 1, op, 0,
                                                        (int) FF, sg, su);
                // the pool then quantizes this expert's own intermediate for the down projection
                ta->from_float(ref_ff[e].data(), hq[e].data(), FF);
            }
        }
        std::vector<std::vector<float>> ref_out(NE, std::vector<float>(H));
        for (int g = 0; g < NG; ++g)
            for (int e = start[g]; e < start[g + 1]; ++e) {
                const float sd = 1.1f;
                const void* hp[1] = { hq[e].data() };
                float* op[1] = { ref_out[e].data() };
                strata::kernels::cpu::nvfp4_256_rows(blobs[g].data() + DOWN_OFF, D_ROW, (int) FF, hp, 1, op, 0,
                                                     (int) H, sd);
            }

        // ---- GPU, grouped ----
        unsigned long long* d_ptr = nullptr;
        int32_t *d_start = nullptr, *d_ng = nullptr, *d_tok = nullptr, *d_dst = nullptr;
        uint8_t* d_blob = nullptr, *d_act = nullptr;
        float *d_gate = nullptr, *d_up = nullptr;
        std::vector<unsigned long long> hptr(NG);
        for (int g = 0; g < NG; ++g) hptr[g] = (unsigned long long) (uintptr_t) blobs[g].data();   // patched below
        // One contiguous arena so a single upload covers every blob, and the pointers are device addresses.
        std::vector<uint8_t> arena((size_t) NG * BLOB);
        for (int g = 0; g < NG; ++g) std::memcpy(arena.data() + (size_t) g * BLOB, blobs[g].data(), BLOB);

        ck(cudaMalloc(&d_blob, arena.size()), "malloc blob");
        ck(cudaMalloc(&d_ptr, NG * sizeof(unsigned long long)), "malloc ptr");
        ck(cudaMalloc(&d_start, (NG + 1) * 4), "malloc start");
        ck(cudaMalloc(&d_ng, 4), "malloc ng");
        ck(cudaMalloc(&d_tok, NE * 4), "malloc tok");
        ck(cudaMalloc(&d_dst, NE * 4), "malloc dst");
        ck(cudaMalloc(&d_act, act_h.size()), "malloc act");
        ck(cudaMalloc(&d_gate, (size_t) NE * FF * 4), "malloc ff");
        ck(cudaMalloc(&d_up, (size_t) 8 * H * 4), "malloc out");
        ck(cudaMemcpy(d_blob, arena.data(), arena.size(), cudaMemcpyHostToDevice), "copy blob");
        ck(cudaMemcpy(d_act, act_h.data(), act_h.size(), cudaMemcpyHostToDevice), "copy act");
        ck(cudaMemcpy(d_start, start.data(), (NG + 1) * 4, cudaMemcpyHostToDevice), "copy start");
        ck(cudaMemcpy(d_ng, &NG, 4, cudaMemcpyHostToDevice), "copy ng");
        ck(cudaMemcpy(d_tok, etok.data(), NE * 4, cudaMemcpyHostToDevice), "copy tok");
        ck(cudaMemcpy(d_dst, edst.data(), NE * 4, cudaMemcpyHostToDevice), "copy dst");
        for (int g = 0; g < NG; ++g) hptr[g] = (unsigned long long) (uintptr_t) (d_blob + (size_t) g * BLOB);
        ck(cudaMemcpy(d_ptr, hptr.data(), NG * sizeof(unsigned long long), cudaMemcpyHostToDevice), "copy ptr");

        strata::kernels::nvfp4_q8_0_grouped_gu(d_ptr, d_start, d_ng, d_tok, NG, NE, H, FF, GU_ROW, UP_OFF, TAIL,
                                               d_act, (int) ACT_H, d_gate, nullptr);
        ck(cudaDeviceSynchronize(), "sync gu");
        std::vector<float> g_ff((size_t) NE * FF);
        ck(cudaMemcpy(g_ff.data(), d_gate, g_ff.size() * 4, cudaMemcpyDeviceToHost), "copy ff");

        // ---- the `sc_lo` path: an fp32 scale array must reproduce the SAME numbers when it carries the block's
        // own `d` widened to float.  That is the contract the engine relies on - the pool's NVFP4 path reads
        // `h2f(y[ib].d)` too (cpu/nvfp4_avx2.cpp:88), so on a native pack the fp16 and fp32 multipliers are the
        // same value and this kernel needs no `sc_lo` to match the pool.  Passing it with widened `d` therefore
        // has to be a no-op; if it is not, the indexing into the scale row is wrong.
        float* d_sc = nullptr;
        ck(cudaMalloc(&d_sc, (size_t) NT * (H / 32) * 4), "malloc sc");
        {
            std::vector<float> widen((size_t) NT * (H / 32));
            for (int t = 0; t < NT; ++t)
                for (int64_t b = 0; b < H / 32; ++b) {
                    const uint8_t* blk = act_h.data() + (size_t) t * ACT_H + (size_t) b * 34;
                    const __half dh = *reinterpret_cast<const __half*>(blk);
                    widen[(size_t) t * (H / 32) + b] = __half2float(dh);
                }
            ck(cudaMemcpy(d_sc, widen.data(), widen.size() * 4, cudaMemcpyHostToDevice), "copy sc");
        }
        float* d_gate2 = nullptr;
        ck(cudaMalloc(&d_gate2, (size_t) NE * FF * 4), "malloc ff2");
        strata::kernels::nvfp4_q8_0_grouped_gu(d_ptr, d_start, d_ng, d_tok, NG, NE, H, FF, GU_ROW, UP_OFF, TAIL,
                                               d_act, (int) ACT_H, d_gate2, nullptr, d_sc, (int) (H / 32));
        ck(cudaDeviceSynchronize(), "sync gu sc");
        std::vector<float> g_ff2((size_t) NE * FF);
        ck(cudaMemcpy(g_ff2.data(), d_gate2, g_ff2.size() * 4, cudaMemcpyDeviceToHost), "copy ff2");
        for (size_t i = 0; i < g_ff.size(); ++i)
            if (std::memcmp(&g_ff[i], &g_ff2[i], 4)) {
                std::printf("  gu sc_lo mismatch at %zu: %.9g vs %.9g\n", i, g_ff[i], g_ff2[i]);
                ++gu_bad;
                ++gu_tot;
                break;
            }

        for (int e = 0; e < NE; ++e)
            for (int64_t r = 0; r < FF; ++r) {
                ++gu_tot;
                const float got = g_ff[(size_t) e * FF + r], want = ref_ff[e][(size_t) r];
                if (std::memcmp(&got, &want, 4)) {
                    ++gu_bad;
                    if (gu_bad <= 4)
                        std::printf("  gu FF=%lld e%d r%lld: got %.9g want %.9g\n", (long long) FF, e, (long long) r,
                                    got, want);
                }
            }

        // The down projection reduces n_ff terms, so its activation is the QUANTIZED INTERMEDIATE - one Q8_0 row
        // of ACT_F bytes per entry, not the layer's n_embd-wide input.  Passing d_act here (n_embd wide) would read
        // past every row and produce finite, plausible, wrong values; that is exactly what this test caught first.
        std::vector<uint8_t> hq_all((size_t) NE * ACT_F);
        for (int e = 0; e < NE; ++e) std::memcpy(hq_all.data() + (size_t) e * ACT_F, hq[e].data(), ACT_F);

        // ---- THE TWO-PHASE HANDOFF: `quantize_q8_0` on the GPU must produce the CPU's bytes.
        // A tensor-parallel split runs gate/up on the GPU, leaves a float intermediate, and then has to quantize
        // it before the down can read it.  If the device quantizer and ggml's `from_float` disagreed here, every
        // split down would be wrong no matter how exact the two dot products are.  They should agree - both are
        // ggml's Q8_0 rounding - but that is worth asserting once rather than assuming for a whole split.
        {
            uint8_t* d_q = nullptr;
            float* d_f = nullptr;
            ck(cudaMalloc(&d_q, hq_all.size()), "malloc q");
            ck(cudaMalloc(&d_f, (size_t) NE * FF * 4), "malloc f");
            ck(cudaMemcpy(d_f, g_ff.data(), g_ff.size() * 4, cudaMemcpyHostToDevice), "copy f");
            for (int e = 0; e < NE; ++e)
                strata::kernels::quantize_q8_0(d_f + (size_t) e * FF, d_q + (size_t) e * ACT_F, (int64_t) FF, nullptr);
            ck(cudaDeviceSynchronize(), "sync q");
            std::vector<uint8_t> g_q(hq_all.size());
            ck(cudaMemcpy(g_q.data(), d_q, g_q.size(), cudaMemcpyDeviceToHost), "copy back");
            int qbad = 0;
            for (size_t i = 0; i < hq_all.size(); ++i)
                if (hq_all[i] != g_q[i]) {
                    if (qbad < 4)
                        std::printf("  q8_0 quantizer differs at %zu: cpu %u gpu %u\n", i, (unsigned) hq_all[i],
                                    (unsigned) g_q[i]);
                    ++qbad;
                }
            std::printf("quantize_q8_0 (gpu) vs from_float (cpu): %d / %zu bytes differ\n", qbad, hq_all.size());
            if (qbad != 0) ++gu_bad;   // the two-phase split cannot be exact if the intermediate is not
            cudaFree(d_f);
            cudaFree(d_q);
        }
        uint8_t* d_hq = nullptr;
        ck(cudaMalloc(&d_hq, hq_all.size()), "malloc hq");
        ck(cudaMemcpy(d_hq, hq_all.data(), hq_all.size(), cudaMemcpyHostToDevice), "copy hq");
        // ent_tok for the down call addresses the INTERMEDIATE rows, one per entry: row e is entry e's own
        // intermediate, so the indirection is the identity here.
        std::vector<int32_t> etok_dn(NE);
        for (int e = 0; e < NE; ++e) etok_dn[e] = e;
        int32_t* d_tok_dn = nullptr;
        ck(cudaMalloc(&d_tok_dn, NE * 4), "malloc tok_dn");
        ck(cudaMemcpy(d_tok_dn, etok_dn.data(), NE * 4, cudaMemcpyHostToDevice), "copy tok_dn");

        strata::kernels::nvfp4_q8_0_grouped_down(d_ptr, d_start, d_ng, d_dst, d_tok_dn, NG, NE, H, FF, D_ROW,
                                                 DOWN_OFF, TAIL, d_hq, (int) ACT_F, d_up, nullptr);
        ck(cudaDeviceSynchronize(), "sync down");
        std::vector<float> g_out((size_t) 8 * H);
        ck(cudaMemcpy(g_out.data(), d_up, g_out.size() * 4, cudaMemcpyDeviceToHost), "copy out");

        for (int e = 0; e < NE; ++e)
            for (int64_t i = 0; i < H; ++i) {
                ++dn_tot;
                const float got = g_out[(size_t) edst[e] * H + i], want = ref_out[e][(size_t) i];
                if (std::memcmp(&got, &want, 4)) {
                    ++dn_bad;
                    if (dn_bad <= 4)
                        std::printf("  dn FF=%lld e%d i%lld: got %.9g want %.9g\n", (long long) FF, e, (long long) i,
                                    got, want);
                }
            }

        cudaFree(d_blob); cudaFree(d_ptr); cudaFree(d_start); cudaFree(d_ng);
        cudaFree(d_tok); cudaFree(d_dst); cudaFree(d_act);
        cudaFree(d_gate); cudaFree(d_up); cudaFree(d_hq); cudaFree(d_tok_dn);
        cudaFree(d_sc); cudaFree(d_gate2);
    }

    std::printf("grouped gu   : %d / %d differ\n", gu_bad, gu_tot);
    std::printf("grouped down : %d / %d differ\n", dn_bad, dn_tot);
    const bool ok = gu_bad == 0 && dn_bad == 0;
    std::printf("%s\n", ok ? "BIT-EXACT" : "MISMATCH");
    return ok ? 0 : 1;
}
