// tests/core/tp_gdn_e2e_test.cu - Phase 3b: the FULL GDN TP pipeline against the single-card reference.
//
// tp_gdn_split_test.cu proves the pieces (qkv split + out-proj row-parallel + allreduce).  This test proves
// the whole per-layer GDN chain that record_window_tp actually runs, on REAL Q8_0 weights from the pack:
//
//   quantize(x) -> native_mmvq(wqkv_half)          [column+Qkv split: q/k full, v half]
//               -> gdn_conv_l2_multi(local 7168)   [local conv weights from tp_local_conv_weights]
//               -> gdn_ab_multi_tp(24 local heads) [global-head weight index]
//               -> native_mmvq(wgate_half)         [column split, 3072 of 6144]
//               -> gdn_step_norm_multi_tp          [local 24 v-heads, local state]
//               -> two-way value exchange          [rank 1's y_half up, rank 0's down; flag g / flag c]
//               -> quantize(y_full 6144)          [the assembled full value vector]
//               -> native_mmvq(wout_half)          [COLUMN split: n_in full 6144, n_out half 1280]
//               -> pitch own half of bo, zero the peer half
//               -> allreduce(bo_0, bo_1)           [single-writer over P2P, stitches the disjoint halves]
//
// against the single-card path (full 48 heads, full 10240-channel conv, full state, full 6144->2560 out).
// The out-projection is COLUMN-parallel (n_out split): each card reduces the FULL 6144 terms over its own 1280
// output rows with the same kernel and block layout as the single card, so every element is bit-identical -
// no partial sums are added.  This is a tolerance check, not bit-exact, because the two cards' q8_1 value
// vectors are assembled from the two-way P2P exchange (the exchange is lossless, but the test asserts the
// end-to-end residual against the reference with a tolerance).
//
// Real data used: tests/core/tp_data/blk_0_attn_qkv_weight.q8_0 (2560x10240 Q8_0) and
// tests/core/tp_data/blk_0_ssm_out_weight.q8_0 (6144x2560 Q8_0).  The gate weight, ssm_alpha/beta, dt,
// ssm_a, conv1d and the state are synthesized (their TP handling is what is under test, and their shapes are
// the artifact's).
#include "strata/core/tp_allreduce.hpp"
#include "strata/core/tp_split.hpp"
#include "strata/core/tp_split_plan.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

// ---- the artifact's GDN geometry
constexpr int N = 2560;      // n_embd
constexpr int S = 128;       // ssm_state_size (head dim)
constexpr int HK = 16;       // ssm_k_heads
constexpr int HV = 48;       // ssm_v_heads
constexpr int QK = HK * S;   // 2048
constexpr int VD = HV * S;   // 6144
constexpr int C = 2 * QK + VD;      // 10240 conv channels
constexpr int CT = 2 * QK + VD / 2; // 7168 local conv channels
constexpr int HVT = HV / 2;         // 24 local v-heads
constexpr int VDT = VD / 2;         // 3072 local value dim
constexpr int GATE = VD;            // attn_gate n_out = 6144
constexpr int GATET = GATE / 2;     // 3072 local
constexpr int T = 3;                // tokens in the window
constexpr int DCONV = 4;            // ssm_d_conv
constexpr float EPS = 1e-6f;

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "FAIL: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}
void req(bool ok, const char* what, const std::string& extra = "") {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s%s%s\n", what, extra.empty() ? "" : " - ", extra.c_str());
        std::exit(1);
    }
}

int g_rank = 0;

template <typename T> T* dev(size_t n, int d) {
    T* p = nullptr;
    ck(cudaSetDevice(d), "set device");
    ck(cudaMalloc(&p, n * sizeof(T)), "malloc");
    return p;
}

// load a real Q8_0 tensor from tests/core/tp_data
std::vector<uint8_t> load_q8(const char* path, int n_in, int n_out) {
    std::FILE* f = std::fopen(path, "rb");
    req(f != nullptr, "open", path);
    const size_t want = (size_t) (n_in / 32) * 34u * (size_t) n_out;
    std::vector<uint8_t> v(want);
    const size_t got = std::fread(v.data(), 1, want, f);
    std::fclose(f);
    req(got == want, "short read", path);
    return v;
}

// deterministic pseudo-random fill
void fill(std::vector<float>& v, float lo, float hi, unsigned seed) {
    unsigned s = seed;
    for (auto& x : v) {
        s = s * 1664525u + 1013904223u;
        x = lo + (hi - lo) * ((float) ((s >> 8) & 0xffff) / 65535.0f);
    }
}

int run() {
    int n_dev = 0;
    ck(cudaGetDeviceCount(&n_dev), "device count");
    req(n_dev >= 2, "this test needs two GPUs");
    const int DEV0 = 0, DEV1 = 1;

    strata::core::TpAllreduce ar;
    std::string err;
    req(ar.open(DEV0, DEV1, err), "TpAllreduce::open", err);
    std::printf("=== GDN TP end-to-end vs single-card reference (T=%d) ===\n", T);
    std::printf("  %s\n", ar.p2p() ? "P2P enabled" : "P2P DISABLED (a plain-PCIe fallback is slower, not wrong)");

    // ------------------------------------------------------------------ host data
    // real quantized weights
    const char* qkv_path = "tests/core/tp_data/blk_0_attn_qkv_weight.q8_0";
    const char* out_path = "tests/core/tp_data/blk_0_ssm_out_weight.q8_0";
    std::vector<uint8_t> h_qkv = load_q8(qkv_path, N, C);       // 2560 x 10240
    std::vector<uint8_t> h_out = load_q8(out_path, VD, N);      // 6144 x 2560
    std::printf("  real weights: attn_qkv %zu KiB, ssm_out %zu KiB\n", h_qkv.size() >> 10, h_out.size() >> 10);

    // synthesized: x, gate weight (Q8_0), alpha/beta (BF16), dt, ssm_a, conv1d, norm
    std::vector<float> h_x((size_t) T * N);
    fill(h_x, -1.0f, 1.0f, 12345u);
    // alpha/beta as BF16 pairs of uint16
    std::vector<uint16_t> h_wa((size_t) HV * N), h_wb((size_t) HV * N);
    {
        std::vector<float> f((size_t) HV * N);
        fill(f, -0.05f, 0.05f, 999u);
        for (size_t i = 0; i < f.size(); ++i) {
            uint32_t u;
            std::memcpy(&u, &f[i], 4);
            h_wa[i] = (uint16_t) (u >> 16);
        }
        fill(f, 0.5f, 1.5f, 777u);
        for (size_t i = 0; i < f.size(); ++i) {
            uint32_t u;
            std::memcpy(&u, &f[i], 4);
            h_wb[i] = (uint16_t) (u >> 16);
        }
    }
    std::vector<float> h_dt(HV), h_a(HV);
    fill(h_dt, -1.0f, 1.0f, 55u);
    fill(h_a, -1.0f, -0.1f, 66u);
    std::vector<float> h_conv((size_t) C * DCONV);
    fill(h_conv, -2.0f, 2.0f, 314u);   // channel-major: full[c*4 + tap]
    std::vector<float> h_norm(S);
    fill(h_norm, 0.5f, 1.5f, 271u);
    // the gate weight: Q8_0 2560x6144.  Quantize a synthetic matrix with the device kernel so the real
    // Q8_0 layout and the real tokenizer are exercised.
    std::vector<float> h_gate_f((size_t) N * GATE);
    fill(h_gate_f, -0.02f, 0.02f, 4242u);
    // the initial state (recurrent + conv history), identical on both cards
    const size_t state_recur = (size_t) S * HV * S;
    std::vector<float> h_state(state_recur + (size_t) C * (DCONV - 1));
    fill(h_state, -0.5f, 0.5f, 808u);

    struct Dev {
        int id;
        uint8_t *qkv = nullptr, *gate = nullptr, *out = nullptr;
        uint16_t *wa = nullptr, *wb = nullptr;
        float *dt = nullptr, *a = nullptr, *conv = nullptr, *norm = nullptr;
        float *x = nullptr, *state = nullptr, *qkv_o = nullptr, *h_o = nullptr, *gate_o = nullptr,
              *beta_o = nullptr, *z = nullptr, *y = nullptr, *bo = nullptr;
        uint8_t* xq = nullptr;
    } d[2];
    for (int r = 0; r < 2; ++r) d[r].id = r == 0 ? DEV0 : DEV1;

    // half-weight sizes from the split plan (the same numbers NativeDense::load uploads)
    const strata::core::TpSplit sp_qkv0 = strata::core::tp_split_for("blk.0.attn_qkv.weight", 0);
    const strata::core::TpSplit sp_qkv1 = strata::core::tp_split_for("blk.0.attn_qkv.weight", 1);
    const strata::core::TpSplit sp_out0 = strata::core::tp_split_for("blk.0.ssm_out.weight", 0);
    const strata::core::TpSplit sp_gate0 = strata::core::tp_split_for("blk.0.attn_gate.weight", 0);
    std::printf("  split plan: qkv %d->%d/%d rows, out n_in %d/%d, gate %d/%d rows\n", sp_qkv0.full_n_out, sp_qkv0.n_out,
                sp_qkv1.n_out, sp_out0.n_in, sp_out0.full_n_in, sp_gate0.n_out, sp_gate0.full_n_out);

    // the gate weight quantized on device, then split (this is what NativeDense does for a Q8_0 tensor)
    uint8_t* d_gate_full = nullptr;
    {
        ck(cudaSetDevice(DEV0), "set");
        float* d_gf = nullptr;
        ck(cudaMalloc(&d_gf, h_gate_f.size() * 4), "gate f32");
        ck(cudaMemcpy(d_gf, h_gate_f.data(), h_gate_f.size() * 4, cudaMemcpyHostToDevice), "gate f32 h2d");
        ck(cudaMalloc(&d_gate_full, (size_t) (N / 32) * 34u * GATE), "gate q8");
        strata::kernels::quantize_q8_0(d_gf, d_gate_full, (int64_t) N * GATE, nullptr);   // 6144 rows of 2560
        ck(cudaDeviceSynchronize(), "gate quantize");
        cudaFree(d_gf);
    }
    std::vector<uint8_t> h_gate((size_t) (N / 32) * 34u * GATE);
    ck(cudaMemcpy(h_gate.data(), d_gate_full, h_gate.size(), cudaMemcpyDeviceToHost), "gate d2h");
    cudaFree(d_gate_full);

    // ------------------------------------------------------------------ per-device buffers
    for (int r = 0; r < 2; ++r) {
        ck(cudaSetDevice(d[r].id), "set");
        Dev& v = d[r];
        // the full-size weights live on BOTH cards (the reference path uses them, and the split source is
        // the host copy anyway)
        v.qkv = dev<uint8_t>((size_t) (N / 32) * 34u * C, v.id);
        v.out = dev<uint8_t>((size_t) (VD / 32) * 34u * N, v.id);
        v.gate = dev<uint8_t>((size_t) (N / 32) * 34u * GATE, v.id);
        v.wa = dev<uint16_t>((size_t) HV * N, v.id);
        v.wb = dev<uint16_t>((size_t) HV * N, v.id);
        v.dt = dev<float>(HV, v.id);
        v.a = dev<float>(HV, v.id);
        v.conv = dev<float>((size_t) C * DCONV, v.id);
        v.norm = dev<float>(S, v.id);
        v.x = dev<float>((size_t) T * N, v.id);
        v.state = dev<float>(h_state.size(), v.id);
        v.qkv_o = dev<float>((size_t) T * C, v.id);
        v.h_o = dev<float>((size_t) T * C, v.id);
        v.gate_o = dev<float>((size_t) T * HV, v.id);
        v.beta_o = dev<float>((size_t) T * HV, v.id);
        v.z = dev<float>((size_t) T * VD, v.id);
        v.y = dev<float>((size_t) T * VD, v.id);
        v.bo = dev<float>((size_t) T * N, v.id);
        // sized for the WIDEST quantize (n_in 6144 = the y), not just the N-wide x: the reference path
        // reuses xq for y, and quantize_q8_1(x, ..., 6144, T) would otherwise write past the N-sized buffer.
        v.xq = dev<uint8_t>((size_t) strata::kernels::native_q8_1_bytes(VD, T), v.id);
        ck(cudaMemcpy(v.qkv, h_qkv.data(), h_qkv.size(), cudaMemcpyHostToDevice), "qkv h2d");
        ck(cudaMemcpy(v.out, h_out.data(), h_out.size(), cudaMemcpyHostToDevice), "out h2d");
        ck(cudaMemcpy(v.gate, h_gate.data(), h_gate.size(), cudaMemcpyHostToDevice), "gate h2d");
        ck(cudaMemcpy(v.wa, h_wa.data(), h_wa.size() * 2, cudaMemcpyHostToDevice), "wa h2d");
        ck(cudaMemcpy(v.wb, h_wb.data(), h_wb.size() * 2, cudaMemcpyHostToDevice), "wb h2d");
        ck(cudaMemcpy(v.dt, h_dt.data(), h_dt.size() * 4, cudaMemcpyHostToDevice), "dt h2d");
        ck(cudaMemcpy(v.a, h_a.data(), h_a.size() * 4, cudaMemcpyHostToDevice), "a h2d");
        ck(cudaMemcpy(v.conv, h_conv.data(), h_conv.size() * 4, cudaMemcpyHostToDevice), "conv h2d");
        ck(cudaMemcpy(v.norm, h_norm.data(), h_norm.size() * 4, cudaMemcpyHostToDevice), "norm h2d");
        ck(cudaMemcpy(v.x, h_x.data(), h_x.size() * 4, cudaMemcpyHostToDevice), "x h2d");
        ck(cudaMemcpy(v.state, h_state.data(), h_state.size() * 4, cudaMemcpyHostToDevice), "state h2d");
    }

    // ------------------------------------------------------------------ REFERENCE (single card, CUDA0)
    cudaStream_t s0 = nullptr, s1 = nullptr;
    ck(cudaSetDevice(DEV0), "set");
    ck(cudaStreamCreateWithFlags(&s0, cudaStreamNonBlocking), "s0");
    ck(cudaSetDevice(DEV1), "set");
    ck(cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking), "s1");

    std::vector<float> h_ref((size_t) T * N);
    {
        Dev& v = d[0];
        ck(cudaSetDevice(DEV0), "set");
        strata::kernels::native_quantize_q8_1(v.x, v.xq, N, T, s0);
        strata::kernels::native_mmvq(8, v.qkv, v.xq, v.qkv_o, N, C, T, s0);
        strata::kernels::gdn_conv_l2_multi(v.state + state_recur, v.qkv_o, v.conv, v.h_o, C, 2 * HK, EPS, T, s0, 0,
                                           false);
        strata::kernels::gdn_ab_multi(v.x, v.wa, v.wb, v.dt, v.a, v.gate_o, v.beta_o, N, HV, T, s0);
        strata::kernels::native_mmvq(8, v.gate, v.xq, v.z, N, GATE, T, s0);
        strata::kernels::gdn_step_norm_multi(v.state, v.h_o, C, v.gate_o, v.beta_o, v.z, v.norm, EPS, v.y, HK, HV, T,
                                             nullptr, s0, 0, nullptr);
        strata::kernels::native_quantize_q8_1(v.y, v.xq, VD, T, s0);
        strata::kernels::native_mmvq(8, v.out, v.xq, v.bo, VD, N, T, s0);
        ck(cudaStreamSynchronize(s0), "ref sync");
        ck(cudaMemcpy(h_ref.data(), v.bo, h_ref.size() * 4, cudaMemcpyDeviceToHost), "ref d2h");
    }

    // ------------------------------------------------------------------ TP (the two cards' halves)
    // Each card's half-weights come from the host-split plan (the exact code NativeDense::load runs).
    struct Half {
        uint8_t *qkv = nullptr, *gate = nullptr, *out = nullptr;
        float *convw = nullptr, *state = nullptr, *qkv_o = nullptr, *h_o = nullptr, *gate_o = nullptr,
              *beta_o = nullptr, *z = nullptr, *y = nullptr, *yfull = nullptr, *half_out = nullptr;
        uint8_t *xq = nullptr, *yq = nullptr;
        size_t state_floats = 0;
    } hf[2];
    {
        std::vector<uint8_t> qkv_half[2], gate_half[2], out_half[2];
        for (int r = 0; r < 2; ++r) {
            qkv_half[r].resize(strata::core::tp_q8_0_bytes("blk.0.attn_qkv.weight", r));
            gate_half[r].resize(strata::core::tp_q8_0_bytes("blk.0.attn_gate.weight", r));
            out_half[r].resize(strata::core::tp_q8_0_bytes("blk.0.ssm_out.weight", r));
            const strata::core::TpSplit a = strata::core::tp_split_for("blk.0.attn_qkv.weight", r);
            const strata::core::TpSplit g = strata::core::tp_split_for("blk.0.attn_gate.weight", r);
            const strata::core::TpSplit o = strata::core::tp_split_for("blk.0.ssm_out.weight", r);
            req(strata::core::tp_split_q8_0_host(h_qkv.data(), a.mode, N, C, r, qkv_half[r].data()) ==
                    qkv_half[r].size(),
                "qkv host split");
            req(strata::core::tp_split_q8_0_host(h_gate.data(), g.mode, N, GATE, r, gate_half[r].data()) ==
                    gate_half[r].size(),
                "gate host split");
            req(strata::core::tp_split_q8_0_host(h_out.data(), o.mode, VD, N, r, out_half[r].data()) ==
                    out_half[r].size(),
                "out host split");
        }
        // the local conv weights (rank r's 7168 of the 10240 channels)
        std::vector<float> conv_local[2];
        for (int r = 0; r < 2; ++r) {
            conv_local[r].resize((size_t) CT * DCONV);
            float* d_full = nullptr;
            float* d_loc = nullptr;
            ck(cudaSetDevice(d[r].id), "set");
            ck(cudaMalloc(&d_full, h_conv.size() * 4), "conv full");
            ck(cudaMemcpy(d_full, h_conv.data(), h_conv.size() * 4, cudaMemcpyHostToDevice), "conv h2d");
            ck(cudaMalloc(&d_loc, conv_local[r].size() * 4), "conv loc");
            strata::core::tp_local_conv_weights(d_full, r, d_loc, nullptr);
            ck(cudaDeviceSynchronize(), "conv split");
            ck(cudaMemcpy(conv_local[r].data(), d_loc, conv_local[r].size() * 4, cudaMemcpyDeviceToHost),
               "conv loc d2h");
            cudaFree(d_full);
            cudaFree(d_loc);

            ck(cudaSetDevice(d[r].id), "set");
            Half& x = hf[r];
            x.qkv = dev<uint8_t>(qkv_half[r].size(), d[r].id);
            x.gate = dev<uint8_t>(gate_half[r].size(), d[r].id);
            x.out = dev<uint8_t>(out_half[r].size(), d[r].id);
            x.convw = dev<float>(conv_local[r].size(), d[r].id);
            x.state_floats = (size_t) S * HVT * S + (size_t) CT * (DCONV - 1);
            x.state = dev<float>(x.state_floats, d[r].id);
            x.qkv_o = dev<float>((size_t) T * CT, d[r].id);
            x.h_o = dev<float>((size_t) T * CT, d[r].id);
            x.gate_o = dev<float>((size_t) T * HVT, d[r].id);
            x.beta_o = dev<float>((size_t) T * HVT, d[r].id);
            x.z = dev<float>((size_t) T * VDT, d[r].id);
            x.y = dev<float>((size_t) T * VDT, d[r].id);
            x.xq = dev<uint8_t>((size_t) strata::kernels::native_q8_1_bytes(N, T), d[r].id);
            // the full ZV value vector (the two-way exchange assembles it) + its q8_1, the compact [T][N/2]
            // out-projection output (the kernel's pitch is n_out = N/2, not N), and the own half of bo.
            x.yfull = dev<float>((size_t) T * VD, d[r].id);
            x.yq = dev<uint8_t>((size_t) strata::kernels::native_q8_1_bytes(VD, T), d[r].id);
            x.half_out = dev<float>((size_t) T * (N / 2), d[r].id);
            ck(cudaMemcpy(x.qkv, qkv_half[r].data(), qkv_half[r].size(), cudaMemcpyHostToDevice), "qkv half h2d");
            ck(cudaMemcpy(x.gate, gate_half[r].data(), gate_half[r].size(), cudaMemcpyHostToDevice),
               "gate half h2d");
            ck(cudaMemcpy(x.out, out_half[r].data(), out_half[r].size(), cudaMemcpyHostToDevice), "out half h2d");
            ck(cudaMemcpy(x.convw, conv_local[r].data(), conv_local[r].size() * 4, cudaMemcpyHostToDevice),
               "convw h2d");
            // this rank's state: its 24 v-heads of the recurrent state + its 7168 channels of the conv history.
            // The recurrent state is [row][v_head][col]: head h of this rank is global head (r*24 + h).
            std::vector<float> st(x.state_floats, 0.0f);
            for (int row = 0; row < S; ++row)
                for (int hh = 0; hh < HVT; ++hh) {
                    const int gh = r * HVT + hh;
                    for (int col = 0; col < S; ++col)
                        st[((size_t) row * HVT + hh) * S + col] =
                            h_state[((size_t) row * HV + gh) * S + col];
                }
            for (int c = 0; c < CT; ++c) {
                const int gc = r == 0 ? c : (c < 2 * QK ? c : (c - 2 * QK) + (2 * QK + VDT));
                for (int j = 0; j < DCONV - 1; ++j)
                    st[(size_t) S * HVT * S + (size_t) c * (DCONV - 1) + j] =
                        h_state[(size_t) S * HV * S + (size_t) gc * (DCONV - 1) + j];
            }
            ck(cudaMemcpy(x.state, st.data(), st.size() * 4, cudaMemcpyHostToDevice), "state h2d");
        }
    }

    // The out-projection is COLUMN-parallel (the fixed design, see TP_FINDINGS): each card holds the FULL ZV
    // value vector (assembled by a two-way value exchange), reduces the FULL 6144 terms over its own N/2 output
    // rows (bit-identical to the single card), and pitches into its OWN half of bo.  The two cards' output
    // halves are disjoint, so the final allreduce stitches them (x + 0.0f is bitwise x).  This is NOT the old
    // row-parallel design (each card reduced 3072 of the 6144 terms and the allreduce added the two partials -
    // (a1+a2)+b != a1+(a2+b) in float32).
    //
    // The engine orders the exchange with DEVICE-SPIN flag kernels (flag g / flag c) because the two ranks are
    // concurrent OS processes inside ONE captured CUDA graph.  This test drives both ranks SEQUENTIALLY from a
    // single host process, so a spin-wait on the peer's flag would deadlock (the peer's chain is never
    // launched).  The host syncs below order the same two phases instead; the P2P writes target disjoint
    // regions, so their relative order is irrelevant.
    // STAGE 1: each card's local half-chain (zero bo first: only the own half is pitched in, so the peer half
    // stays 0 for the allreduce's stitch).
    for (int r = 0; r < 2; ++r) {
        ck(cudaSetDevice(d[r].id), "set");
        const cudaStream_t sx = r == 0 ? s0 : s1;
        Half& x = hf[r];
        ck(cudaMemsetAsync(d[r].bo, 0, (size_t) T * N * sizeof(float), sx), "bo zero");
        strata::kernels::native_quantize_q8_1(d[r].x, x.xq, N, T, sx);
        strata::kernels::native_mmvq(8, x.qkv, x.xq, x.qkv_o, N, CT, T, sx);
        strata::kernels::gdn_conv_l2_multi(x.state + (size_t) S * HVT * S, x.qkv_o, x.convw, x.h_o, CT, 2 * HK, EPS, T,
                                           sx, 0, false);
        strata::kernels::gdn_ab_multi_tp(d[r].x, d[r].wa, d[r].wb, d[r].dt, d[r].a, x.gate_o, x.beta_o, N, HVT, T, sx,
                                         r * HVT);
        strata::kernels::native_mmvq(8, x.gate, x.xq, x.z, N, GATET, T, sx);
        strata::kernels::gdn_step_norm_multi_tp(x.state, x.h_o, CT, x.gate_o, x.beta_o, x.z, d[r].norm, EPS, x.y, HK,
                                                HVT, T, nullptr, sx, 0, nullptr, r * HVT);
        if (r == 0)
            // rank 0's own half [0, ZVt) into its yfull_.  Per-token (the pitch is T*ZV, not T*ZVt).
            for (int t = 0; t < T; ++t)
                ck(cudaMemcpyAsync(x.yfull + (size_t) t * VD, x.y + (size_t) t * VDT, (size_t) VDT * 4,
                                   cudaMemcpyDeviceToDevice, sx), "r0 yfull own");
        else
            // rank 1's own half UP into rank 0's yfull_[ZVt, ZV) over P2P AND into its own yfull_[ZVt, ZV)
            // (nobody else can write the latter - without it the quantizer reads stale values).  Per-token.
            for (int t = 0; t < T; ++t) {
                req(ar.broadcast_to_a(x.y + (size_t) t * VDT, (float*) hf[0].yfull + (size_t) t * VD + VDT,
                                      (int64_t) VDT, sx, err), "r1->r0 value half");
                ck(cudaMemcpyAsync(x.yfull + (size_t) t * VD + VDT, x.y + (size_t) t * VDT, (size_t) VDT * 4,
                                   cudaMemcpyDeviceToDevice, sx), "r1 yfull own");
            }
    }
    // host barrier: both cards' local chains (and rank 1's P2P write into rank 0's yfull) are done.
    ck(cudaStreamSynchronize(s0), "stage1 s0");
    ck(cudaStreamSynchronize(s1), "stage1 s1");
    // STAGE 2: rank 0's half DOWN into rank 1's yfull_[0, ZVt) over P2P (one kernel per token, on DEV0's
    // stream, writing DEV1's memory).  Disjoint from rank 1's writes, so it could have run concurrently in the
    // engine; the host sync above just makes the test correct.
    {
        ck(cudaSetDevice(DEV0), "set");
        Half& r1 = hf[1];
        for (int t = 0; t < T; ++t)
            req(ar.broadcast(hf[0].y + (size_t) t * VDT, r1.yfull + (size_t) t * VD, (int64_t) VDT, s0, err),
                "r0->r1 value half");
        ck(cudaStreamSynchronize(s0), "stage2 s0");
        ck(cudaStreamSynchronize(s1), "stage2 s1");
    }
    // STAGE 3: both cards quantize the FULL ZV vector and run the COLUMN-parallel out-projection (n_in full
    // 6144, n_out half 1280), pitching the compact [T][N/2] into bo_'s [T][N] own half per token.
    for (int r = 0; r < 2; ++r) {
        ck(cudaSetDevice(d[r].id), "set");
        const cudaStream_t sx = r == 0 ? s0 : s1;
        Half& x = hf[r];
        strata::kernels::native_quantize_q8_1(x.yfull, x.yq, VD, T, sx);
        strata::kernels::native_mmvq(8, x.out, x.yq, x.half_out, VD, N / 2, T, sx);
        for (int t = 0; t < T; ++t) {
            const size_t dst = (r == 0) ? (size_t) t * N : (size_t) t * N + (N / 2);
            ck(cudaMemcpyAsync(d[r].bo + dst, x.half_out + (size_t) t * (N / 2), (size_t) (N / 2) * 4,
                               cudaMemcpyDeviceToDevice, sx), "bo pitch");
        }
    }
    // host barrier: the allreduce kernel (a single writer on DEV0) READS DEV1's bo over P2P, and that read has
    // no ordering with DEV1's own stream.  In the engine the flag handshakes order this; here the host sync of
    // BOTH streams does (without it the kernel can read bo1 before rank 1's pitch above has landed).
    ck(cudaStreamSynchronize(s0), "stage3 s0");
    ck(cudaStreamSynchronize(s1), "stage3 s1");
    // the allreduce (single writer on DEV0: reads and writes both cards' bo) stitches the two disjoint halves.
    req(ar.allreduce(d[0].bo, d[1].bo, (int64_t) T * N, s0, s1, err), "allreduce", err);
    // the single-writer kernel runs on DEV0's stream; DEV1's bo is written over P2P with no ordering call on
    // DEV1's stream, so syncing s0 (the writer) is what makes both bo_ valid.
    ck(cudaSetDevice(DEV0), "set");
    ck(cudaStreamSynchronize(s0), "s0 sync");

    std::vector<float> h_tp0((size_t) T * N), h_tp1((size_t) T * N);
    ck(cudaMemcpy(h_tp0.data(), d[0].bo, h_tp0.size() * 4, cudaMemcpyDeviceToHost), "tp0 d2h");
    ck(cudaSetDevice(DEV1), "set");
    ck(cudaMemcpy(h_tp1.data(), d[1].bo, h_tp1.size() * 4, cudaMemcpyDeviceToHost), "tp1 d2h");

    // ------------------------------------------------------------------ compare
    auto stats = [&](const std::vector<float>& a, const std::vector<float>& b, const char* what) {
        double max_abs = 0.0, max_ref = 0.0, sum2 = 0.0;
        size_t bad = 0;
        for (size_t i = 0; i < a.size(); ++i) {
            const double d1 = std::fabs((double) a[i] - (double) b[i]);
            max_abs = std::max(max_abs, d1);
            sum2 += d1 * d1;
            max_ref = std::max(max_ref, std::fabs((double) b[i]));
            if (d1 > 1e-4 * (1.0 + std::fabs((double) b[i]))) ++bad;
        }
        const double rms = std::sqrt(sum2 / (double) a.size());
        std::printf("  %-22s max_abs=%.3e  rms=%.3e  max_ref=%.3e  over_tol=%zu/%zu\n", what, max_abs, rms, max_ref, bad,
                    a.size());
        return max_abs <= 1e-3 * (1.0 + max_ref) && bad == 0;
    };
    std::printf("\n--- residuals (TP vs single-card reference) ---\n");
    const bool ok0 = stats(h_tp0, h_ref, "rank 0 bo");
    const bool ok1 = stats(h_tp1, h_ref, "rank 1 bo");
    std::printf("\n--- the two cards must agree exactly (the allreduce writes both) ---\n");
    const bool same = stats(h_tp1, h_tp0, "rank 1 vs rank 0") && [&] {
        return std::memcmp(h_tp0.data(), h_tp1.data(), h_tp0.size() * 4) == 0;
    }();
    std::printf("  bit-identical: %s\n", same ? "yes" : "NO");

    // a crude speed number: the TP half-chain on both cards vs the full chain on one
    {
        cudaEvent_t a = nullptr, b = nullptr;
        // events are created on the CURRENT device's context: DEV0 must be current here, or record() on s0
        // fails with "invalid resource handle"
        ck(cudaSetDevice(DEV0), "set DEV0 for the events");
        ck(cudaEventCreate(&a), "ev a");
        ck(cudaEventCreate(&b), "ev b");
        const int iters = 30;
        ck(cudaSetDevice(DEV0), "set");
        ck(cudaEventRecord(a, s0), "rec");
        for (int i = 0; i < iters; ++i) {
            strata::kernels::native_quantize_q8_1(d[0].x, d[0].xq, N, T, s0);
            strata::kernels::native_mmvq(8, d[0].qkv, d[0].xq, d[0].qkv_o, N, C, T, s0);
            strata::kernels::gdn_conv_l2_multi(d[0].state + state_recur, d[0].qkv_o, d[0].conv, d[0].h_o, C, 2 * HK,
                                               EPS, T, s0, 0, false);
            strata::kernels::gdn_ab_multi(d[0].x, d[0].wa, d[0].wb, d[0].dt, d[0].a, d[0].gate_o, d[0].beta_o, N, HV,
                                          T, s0);
            strata::kernels::native_mmvq(8, d[0].gate, d[0].xq, d[0].z, N, GATE, T, s0);
            strata::kernels::gdn_step_norm_multi(d[0].state, d[0].h_o, C, d[0].gate_o, d[0].beta_o, d[0].z, d[0].norm,
                                                 EPS, d[0].y, HK, HV, T, nullptr, s0, 0, nullptr);
            strata::kernels::native_quantize_q8_1(d[0].y, d[0].xq, VD, T, s0);
            strata::kernels::native_mmvq(8, d[0].out, d[0].xq, d[0].bo, VD, N, T, s0);
        }
        ck(cudaEventRecord(b, s0), "rec");
        ck(cudaEventSynchronize(b), "sync");
        float ms_full = 0;
        ck(cudaEventElapsedTime(&ms_full, a, b), "elapsed");

        // The whole loop is timed from CUDA0's stream: the two cards' streams have no shared clock, so the
        // span is "DEV0 busy from the first launch to the last allreduce + DEV1's tail".  Every device switch
        // inside the loop is undone before the next cudaEventRecord/cudaEventSynchronize.
        ck(cudaSetDevice(DEV0), "set DEV0");
        ck(cudaEventRecord(a, s0), "rec a");
        for (int i = 0; i < iters; ++i) {
            for (int r = 0; r < 2; ++r) {
                ck(cudaSetDevice(d[r].id), "set");
                const cudaStream_t sx = r == 0 ? s0 : s1;
                Half& x = hf[r];
                strata::kernels::native_quantize_q8_1(d[r].x, x.xq, N, T, sx);
                strata::kernels::native_mmvq(8, x.qkv, x.xq, x.qkv_o, N, CT, T, sx);
                strata::kernels::gdn_conv_l2_multi(x.state + (size_t) S * HVT * S, x.qkv_o, x.convw, x.h_o, CT, 2 * HK,
                                                   EPS, T, sx, 0, false);
                strata::kernels::gdn_ab_multi_tp(d[r].x, d[r].wa, d[r].wb, d[r].dt, d[r].a, x.gate_o, x.beta_o, N, HVT,
                                                 T, sx, r * HVT);
                strata::kernels::native_mmvq(8, x.gate, x.xq, x.z, N, GATET, T, sx);
                strata::kernels::gdn_step_norm_multi_tp(x.state, x.h_o, CT, x.gate_o, x.beta_o, x.z, d[r].norm, EPS,
                                                        x.y, HK, HVT, T, nullptr, sx, 0, nullptr, r * HVT);
                // COLUMN-parallel out-projection (the timed compute of the production path): the full ZV value
                // vector (own half from x.y; the peer half's P2P exchange is a 12 KiB copy, not the dominant
                // cost, so the loop times the local half-chain + full-6144 reduction, not the handshake).
                ck(cudaMemcpyAsync(x.yfull + (r == 0 ? (size_t) 0 : (size_t) VDT), x.y, (size_t) T * VDT * 4,
                                   cudaMemcpyDeviceToDevice, sx), "timed yfull");
                strata::kernels::native_quantize_q8_1(x.yfull, x.yq, VD, T, sx);
                strata::kernels::native_mmvq(8, x.out, x.yq, x.half_out, VD, N / 2, T, sx);
            }
            // the allreduce runs on DEV0 and reads/writes DEV1's bo over P2P: DEV1's stream must not be
            // switched away from mid-loop (the single-writer form needs no ordering call on DEV1)
            ck(cudaSetDevice(DEV0), "set DEV0 for the allreduce");
            ar.allreduce(d[0].bo, d[1].bo, (int64_t) T * N, s0, s1, err);
        }
        ck(cudaSetDevice(DEV0), "set DEV0");
        ck(cudaEventRecord(b, s0), "rec b");
        ck(cudaEventSynchronize(b), "sync b");
        float ms_tp = 0;
        ck(cudaEventElapsedTime(&ms_tp, a, b), "elapsed");
        std::printf("\n--- timing (%d iters, one GDN layer) ---\n", iters);
        std::printf("  single card, full 48 heads:        %.1f us/layer\n", 1000.0 * ms_full / iters);
        std::printf("  two cards, 24 heads each + allreduce: %.1f us/layer\n", 1000.0 * ms_tp / iters);
        std::printf("  speedup: %.2fx\n", ms_full / (ms_tp > 0 ? ms_tp : 1));
    }

    const bool all = ok0 && ok1 && same;
    std::printf("\n%s\n", all ? "ALL PASS: the GDN TP pipeline matches the single-card reference"
                              : "FAILED");
    return all ? 0 : 1;
}

}  // namespace

int main() {
    g_rank = 0;
    return run();
}
