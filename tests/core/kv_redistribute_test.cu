// tests/core/kv_redistribute_test.cu - TP_RESULTS §8.3 step 2: the layer-split -> TP state handover.
//
// The routine's contract is "what a LAYER-SPLIT prefill leaves in the state is, after `kv_redistribute`, the
// state TP decode reads".  This fixture pins that three ways, because a copy can be wrong in three places:
//
//   1. WHAT MOVED (bytes).  A layer-split stage holds only its own layers, so a destination rank has to be
//      assembled from both cards.  After the call each rank's QSA pools must hold exactly the source's bytes
//      for the pages the sequence filled, and must NOT have been written past them; the PLE history and its
//      token window must be the ones the sequence actually advanced.
//   2. WHAT SHAPE (the GDN permute).  GDN is full width on a split stage and head-band split on a TP rank, so
//      this half is a permute, not a copy.  The fixture recomputes the expected split state on the HOST from
//      the definition (global head `head_off + h` -> local head `h`; global channel
//      `qk + head_off*S + (c - qk)` -> local channel `c`) and requires bit equality for every layer, on both
//      ranks.  That is an independent implementation of the same arithmetic - it would not catch a mistake
//      that is in both places, which is what part 3 is for.
//   3. WHAT IT MEANS (semantics).  The engines that will actually read the state are `gdn_conv_step` and
//      `gdn_step`, so the fixture runs them on the REDISTRIBUTED state and on the FULL state and requires the
//      outputs to be bit-identical for each rank's own band.  The local weights come from
//      `tp_local_conv_weights` - the production splitter - so this half is not circular: if the copier's
//      channel map disagreed with the weight splitter's, the two conv outputs would differ.  For the
//      recurrence the local k/q rows are rotated by `head_off % h_k`, which is exactly the pairing
//      `gdn_step_kernel` computes as `h % h_k` once the local head is renumbered.
//
// Also checked: the QSA/PLE-only mode (`gdn_state == nullptr`) leaves the GDN arena untouched, a second run
// is idempotent, and four shape mismatches are refused with an error instead of silently copying.
//
// Both KV formats are run: FP16 (the default, 2 spans) and INT8 (4 spans, so `qsa_kv_spans`' order is
// exercised).  Needs two CUDA devices; skips cleanly otherwise.
#include "strata/core/kv_redistribute.hpp"

#include "strata/core/layer.hpp"
#include "strata/core/layout.hpp"
#include "strata/core/session.hpp"
#include "strata/core/tp_split.hpp"
#include "strata/kernels/gdn.hpp"
#include "strata/kernels/ngram.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace strata::core;

namespace {

constexpr int64_t kMaxCells = 64;   // 16 pages: small, but several indexer blocks
constexpr int64_t kTokens = 50;     // 13 pages used, 14 pooled rows used of 18
constexpr int64_t kBatch = 32;

// ---- deterministic patterns, so an expected buffer can be recomputed instead of stored -------------------
inline uint32_t h32(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}
inline uint8_t pat8(uint32_t seed, uint64_t i) { return (uint8_t) h32(seed * 2654435761u + (uint32_t) i * 2246822519u); }
inline float patf(uint32_t seed, uint64_t i) {
    const uint32_t bits = 0x3F000000u | (h32(seed * 2654435761u + (uint32_t) i * 2246822519u) & 0x7FFFFFu);
    float f = 0.0f;
    std::memcpy(&f, &bits, 4);   // [0.5, 1): exact, no denormals, no NaN
    return f;
}
/// The seed a QSA array of global ordinal `q` is filled with; `slot` picks the array within the layer.
inline uint32_t qsa_seed(int64_t q, int slot) { return (uint32_t) (q * 8 + slot) + 0x5EED0000u; }

void fail(const std::string& what) {
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    std::exit(1);
}
void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) fail(std::string(what) + ": " + cudaGetErrorString(e));
}
void req(bool ok, const std::string& what) {
    if (!ok) fail(what);
}

struct Sess {
    void* buf = nullptr;
    uint64_t bytes = 0;
    SessionState st{};
    int dev = 0;
    int64_t lo = 0, hi = 0;
};

Sess make_session(const ModelGeometry& g, int dev, int64_t lo, int64_t hi) {
    Sess s;
    s.dev = dev;
    s.lo = lo;
    s.hi = hi;
    s.bytes = session_bytes(g, kMaxCells, kBatch, lo, hi);
    ck(cudaSetDevice(dev), "cudaSetDevice");
    ck(cudaMalloc(&s.buf, s.bytes), "cudaMalloc(session)");
    const uint64_t used = session_init(g, kMaxCells, kBatch, s.buf, s.st, lo, hi);
    req(used > 0 && used <= s.bytes, "session_init returned an impossible size");
    return s;
}
void free_session(Sess& s) {
    if (s.buf == nullptr) return;
    ck(cudaSetDevice(s.dev), "cudaSetDevice");
    session_release(s.st);
    ck(cudaFree(s.buf), "cudaFree(session)");
    s.buf = nullptr;
}

// ---- device helpers -------------------------------------------------------------------------------------
void poison(int dev, void* p, uint64_t bytes) {
    ck(cudaSetDevice(dev), "cudaSetDevice");
    ck(cudaMemset(p, 0xCD, (size_t) bytes), "memset(poison)");
}
void fill_pat(int dev, void* p, uint64_t bytes, uint32_t seed) {
    std::vector<uint8_t> h((size_t) bytes);
    for (uint64_t i = 0; i < bytes; ++i) h[(size_t) i] = pat8(seed, i);
    ck(cudaSetDevice(dev), "cudaSetDevice");
    ck(cudaMemcpy(p, h.data(), (size_t) bytes, cudaMemcpyHostToDevice), "memcpy(fill)");
}
void fill_rows(int dev, float* p, int64_t n, uint32_t seed) {
    std::vector<float> h((size_t) n);
    for (int64_t i = 0; i < n; ++i) h[(size_t) i] = patf(seed, (uint64_t) i);
    ck(cudaSetDevice(dev), "cudaSetDevice");
    ck(cudaMemcpy(p, h.data(), (size_t) n * 4, cudaMemcpyHostToDevice), "memcpy(fill rows)");
}
std::vector<uint8_t> readback(int dev, const void* p, uint64_t bytes) {
    std::vector<uint8_t> h((size_t) bytes);
    ck(cudaSetDevice(dev), "cudaSetDevice");
    ck(cudaMemcpy(h.data(), p, (size_t) bytes, cudaMemcpyDeviceToHost), "memcpy(readback)");
    return h;
}
void read_floats(int dev, const float* p, int64_t n, std::vector<float>& out) {
    out.resize((size_t) n);
    ck(cudaSetDevice(dev), "cudaSetDevice");
    ck(cudaMemcpy(out.data(), p, (size_t) n * 4, cudaMemcpyDeviceToHost), "memcpy(floats)");
}

/// `[0, n)` must be the pattern and `[n, total)` must still be the poison.
void check_span(int dev, const void* p, uint64_t total, uint64_t n, uint32_t seed, const std::string& what) {
    const std::vector<uint8_t> got = readback(dev, p, total);
    for (uint64_t i = 0; i < n; ++i)
        if (got[(size_t) i] != pat8(seed, i))
            fail(what + ": byte " + std::to_string(i) + " of the copied prefix is wrong (seed " +
                 std::to_string(seed) + ")");
    for (uint64_t i = n; i < total; ++i)
        if (got[(size_t) i] != 0xCD)
            fail(what + ": byte " + std::to_string(i) + " past the used prefix was overwritten");
}

struct Spans {
    void* p[4]{};
    uint64_t b[4]{};
    int n = 0;
};
Spans spans_of(const QsaState& st, const ModelGeometry& g) {
    Spans s;
    s.n = qsa_kv_spans(st, g, s.p, s.b);
    req(s.n >= 2 && s.n <= 4, "qsa_kv_spans returned an impossible count");
    return s;
}

// ---- the reference GDN permute, on the HOST (the definition, written out again) --------------------------
void ref_permute(const GdnSplitPlan& p, const std::vector<float>& full, std::vector<float>& local) {
    const int64_t S = p.S, HV = p.HV, heads = p.heads, taps = p.d_conv - 1;
    const int64_t rec_full = S * HV * S, rec = S * heads * S;
    local.assign((size_t) p.floats(), 0.0f);
    for (int64_t i = 0; i < S; ++i)
        for (int64_t h = 0; h < heads; ++h)
            for (int64_t j = 0; j < S; ++j)
                local[(size_t) ((i * heads + h) * S + j)] = full[(size_t) ((i * HV + p.head_off + h) * S + j)];
    for (int64_t c = 0; c < p.channels; ++c) {
        const int64_t fc = c < p.qk_channels ? c : p.qk_channels + p.head_off * S + (c - p.qk_channels);
        for (int64_t t = 0; t < taps; ++t)
            local[(size_t) (rec + c * taps + t)] = full[(size_t) (rec_full + fc * taps + t)];
    }
}

void expect_eq(const std::vector<float>& got, const std::vector<float>& want, const std::string& what) {
    req(got.size() == want.size(), what + ": size");
    for (size_t i = 0; i < got.size(); ++i)
        if (std::memcmp(&got[i], &want[i], 4) != 0)
            fail(what + ": float " + std::to_string(i) + " differs (" + std::to_string(got[i]) + " vs " +
                 std::to_string(want[i]) + ")");
}

// ---- part 3: do the engines that read this state get the same answer? -----------------------------------
void semantic_conv(const ModelGeometry& g, const GdnSplitPlan& p, int rank, const std::vector<float>& full_row,
                   const std::vector<float>& split_row, uint32_t seed) {
    ck(cudaSetDevice(0), "cudaSetDevice");   // both sides run on one device: same kernel, same code path
    const int64_t C = p.channels + (p.HV - p.heads) * p.S;   // the FULL channel count
    const int64_t Ct = p.channels, D = p.d_conv;
    req(C == g.ssm_conv_channels, "the full channel count");
    const int64_t rec_full = p.S * p.HV * p.S, rec = p.S * p.heads * p.S;

    auto map = [&](int64_t c) { return c < p.qk_channels ? c : p.qk_channels + p.head_off * p.S + (c - p.qk_channels); };

    std::vector<float> wfull((size_t) (C * D)), xfull((size_t) C);
    for (int64_t i = 0; i < C * D; ++i) wfull[(size_t) i] = patf(seed, (uint64_t) i);
    for (int64_t i = 0; i < C; ++i) xfull[(size_t) i] = patf(seed + 1, (uint64_t) i);
    std::vector<float> xlocal((size_t) Ct);
    for (int64_t i = 0; i < Ct; ++i) xlocal[(size_t) i] = xfull[(size_t) map(i)];

    float *dwf = nullptr, *dwl = nullptr, *dxf = nullptr, *dxl = nullptr, *dsf = nullptr, *dsl = nullptr;
    float *dof = nullptr, *dol = nullptr;
    ck(cudaMalloc((void**) &dwf, (size_t) C * D * 4), "malloc");
    ck(cudaMalloc((void**) &dwl, (size_t) Ct * D * 4), "malloc");
    ck(cudaMalloc((void**) &dxf, (size_t) C * 4), "malloc");
    ck(cudaMalloc((void**) &dxl, (size_t) Ct * 4), "malloc");
    ck(cudaMalloc((void**) &dsf, (size_t) (C * (D - 1)) * 4), "malloc");
    ck(cudaMalloc((void**) &dsl, (size_t) (Ct * (D - 1)) * 4), "malloc");
    ck(cudaMalloc((void**) &dof, (size_t) C * 4), "malloc");
    ck(cudaMalloc((void**) &dol, (size_t) Ct * 4), "malloc");

    ck(cudaMemcpy(dwf, wfull.data(), (size_t) C * D * 4, cudaMemcpyHostToDevice), "H2D weights");
    ck(cudaMemcpy(dxf, xfull.data(), (size_t) C * 4, cudaMemcpyHostToDevice), "H2D x");
    ck(cudaMemcpy(dxl, xlocal.data(), (size_t) Ct * 4, cudaMemcpyHostToDevice), "H2D x");
    ck(cudaMemcpy(dsf, full_row.data() + rec_full, (size_t) (C * (D - 1)) * 4, cudaMemcpyHostToDevice), "H2D state");
    ck(cudaMemcpy(dsl, split_row.data() + rec, (size_t) (Ct * (D - 1)) * 4, cudaMemcpyHostToDevice), "H2D state");

    // the production weight splitter: the authority for which channels this rank owns
    strata::core::tp_local_conv_weights(dwf, rank, dwl, nullptr);
    std::vector<float> wlocal((size_t) (Ct * D)), want((size_t) Ct * D);
    read_floats(0, dwl, Ct * D, wlocal);
    for (int64_t c = 0; c < Ct; ++c)
        for (int64_t t = 0; t < D; ++t) want[(size_t) (c * D + t)] = wfull[(size_t) (map(c) * D + t)];
    expect_eq(wlocal, want, "tp_local_conv_weights does not match the documented channel map");

    strata::kernels::gdn_conv_step(dsf, dxf, dwf, dof, C, D, nullptr);
    strata::kernels::gdn_conv_step(dsl, dxl, dwl, dol, Ct, D, nullptr);

    std::vector<float> of, ol, sf, sl;
    read_floats(0, dof, C, of);
    read_floats(0, dol, Ct, ol);
    read_floats(0, dsf, C * (D - 1), sf);
    read_floats(0, dsl, Ct * (D - 1), sl);
    for (int64_t c = 0; c < Ct; ++c) {
        req(std::memcmp(&ol[(size_t) c], &of[(size_t) map(c)], 4) == 0,
            "the redistributed conv state gives a different output at local channel " + std::to_string(c) +
                " (rank " + std::to_string(rank) + ")");
        for (int64_t t = 0; t < D - 1; ++t)
            req(std::memcmp(&sl[(size_t) (c * (D - 1) + t)], &sf[(size_t) (map(c) * (D - 1) + t)], 4) == 0,
                "the redistributed conv state slides differently at local channel " + std::to_string(c));
    }

    cudaFree(dwf); cudaFree(dwl); cudaFree(dxf); cudaFree(dxl);
    cudaFree(dsf); cudaFree(dsl); cudaFree(dof); cudaFree(dol);
}

void semantic_recurrence(const GdnSplitPlan& p, int rank, const std::vector<float>& full_row,
                         const std::vector<float>& split_row, uint32_t seed) {
    ck(cudaSetDevice(0), "cudaSetDevice");
    const int64_t S = p.S, HK = p.qk_channels / (2 * p.S), HV = p.HV, hvt = p.heads;
    const int64_t rec_full = S * HV * S, rec = S * hvt * S;
    // `gdn_step_kernel` pairs head h with k/q row `h % h_k`.  This rank's global heads are head_off + h, so
    // feeding it a k/q whose row s holds global row (s + head_off) % h_k makes that pairing exact.
    const int64_t rot = p.head_off % HK;

    std::vector<float> qf((size_t) (HK * S)), kf((size_t) (HK * S)), ql((size_t) (HK * S)), kl((size_t) (HK * S));
    for (int64_t i = 0; i < HK * S; ++i) {
        qf[(size_t) i] = patf(seed, (uint64_t) i);
        kf[(size_t) i] = patf(seed + 1, (uint64_t) i);
    }
    for (int64_t s = 0; s < HK; ++s) {
        const int64_t src = (s + rot) % HK;
        for (int64_t i = 0; i < S; ++i) {
            ql[(size_t) (s * S + i)] = qf[(size_t) (src * S + i)];
            kl[(size_t) (s * S + i)] = kf[(size_t) (src * S + i)];
        }
    }

    std::vector<float> vf((size_t) (HV * S)), vl((size_t) (hvt * S)), gf((size_t) HV), bl((size_t) hvt), gl((size_t) hvt);
    for (int64_t i = 0; i < HV * S; ++i) vf[(size_t) i] = patf(seed + 2, (uint64_t) i);
    for (int64_t i = 0; i < HV; ++i) gf[(size_t) i] = patf(seed + 3, (uint64_t) i);
    std::vector<float> bf((size_t) HV);
    for (int64_t i = 0; i < HV; ++i) bf[(size_t) i] = patf(seed + 4, (uint64_t) i);
    for (int64_t h = 0; h < hvt; ++h) {
        const int64_t gh = p.head_off + h;
        gl[(size_t) h] = gf[(size_t) gh];
        bl[(size_t) h] = bf[(size_t) gh];
        for (int64_t j = 0; j < S; ++j) vl[(size_t) (h * S + j)] = vf[(size_t) (gh * S + j)];
    }

    float *dq = nullptr, *dk = nullptr, *dv = nullptr, *dg = nullptr, *db = nullptr, *dof = nullptr, *dol = nullptr;
    float *dsf = nullptr, *dsl = nullptr, *dql = nullptr, *dkl = nullptr, *dvl = nullptr, *dgl = nullptr, *dbl = nullptr;
    auto alloc = [](float** p, int64_t n) { ck(cudaMalloc((void**) p, (size_t) n * 4), "malloc"); };
    alloc(&dq, HK * S); alloc(&dk, HK * S); alloc(&dv, HV * S); alloc(&dg, HV); alloc(&db, HV);
    alloc(&dof, HV * S); alloc(&dsf, rec_full);
    alloc(&dql, HK * S); alloc(&dkl, HK * S); alloc(&dvl, hvt * S); alloc(&dgl, hvt); alloc(&dbl, hvt);
    alloc(&dol, hvt * S); alloc(&dsl, rec);

    auto up = [](float* d, const std::vector<float>& h) {
        ck(cudaMemcpy(d, h.data(), h.size() * 4, cudaMemcpyHostToDevice), "H2D");
    };
    up(dq, qf); up(dk, kf); up(dv, vf); up(dg, gf); up(db, bf);
    up(dql, ql); up(dkl, kl); up(dvl, vl); up(dgl, gl); up(dbl, bl);
    ck(cudaMemcpy(dsf, full_row.data(), (size_t) rec_full * 4, cudaMemcpyHostToDevice), "H2D state");
    ck(cudaMemcpy(dsl, split_row.data(), (size_t) rec * 4, cudaMemcpyHostToDevice), "H2D state");

    strata::kernels::GdnShapes fs;
    fs.S = S; fs.h_k = HK; fs.h_v = HV;
    strata::kernels::GdnShapes ls;
    ls.S = S; ls.h_k = HK; ls.h_v = hvt;
    strata::kernels::gdn_step(dsf, dq, dk, dv, dg, db, dof, fs, nullptr);
    strata::kernels::gdn_step(dsl, dql, dkl, dvl, dgl, dbl, dol, ls, nullptr);

    std::vector<float> of, ol, sf, sl;
    read_floats(0, dof, HV * S, of);
    read_floats(0, dol, hvt * S, ol);
    read_floats(0, dsf, rec_full, sf);
    read_floats(0, dsl, rec, sl);
    for (int64_t h = 0; h < hvt; ++h) {
        const int64_t gh = p.head_off + h;
        for (int64_t j = 0; j < S; ++j)
            req(std::memcmp(&ol[(size_t) (h * S + j)], &of[(size_t) (gh * S + j)], 4) == 0,
                "the redistributed recurrence gives a different output at local head " + std::to_string(h) +
                    ", j " + std::to_string(j) + " (rank " + std::to_string(rank) + ")");
        for (int64_t i = 0; i < S; ++i)
            for (int64_t j = 0; j < S; ++j)
                req(std::memcmp(&sl[(size_t) ((i * hvt + h) * S + j)], &sf[(size_t) ((i * HV + gh) * S + j)], 4) == 0,
                    "the redistributed recurrence updates the state differently at local head " +
                        std::to_string(h));
    }

    cudaFree(dq); cudaFree(dk); cudaFree(dv); cudaFree(dg); cudaFree(db); cudaFree(dof); cudaFree(dsf);
    cudaFree(dql); cudaFree(dkl); cudaFree(dvl); cudaFree(dgl); cudaFree(dbl); cudaFree(dol); cudaFree(dsl);
}

// ---- the fixture ----------------------------------------------------------------------------------------
void run_case(const ModelGeometry& g, bool int8) {
    std::printf("== %s ==\n", int8 ? "INT8 K/V (4 spans)" : "FP16 K/V (2 spans)");
    qsa_set_kv_int8(int8);
    qsa_set_kv_q4(false);
    qsa_set_kv_hybrid(false);

    const int64_t n_layers = g.n_layers, mid = n_layers / 2;
    const int64_t page_size = strata::kernels::qsa_real_shapes().page_size;
    const int64_t idx_block = strata::kernels::qsa_real_shapes().idx_block;
    const int64_t used_pages = (kTokens + page_size - 1) / page_size;
    const int64_t full_floats = g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size +
                                g.ssm_conv_channels * (g.ssm_d_conv - 1);
    const int64_t n_gdn = g.n_gdn_layers();

    Sess src0 = make_session(g, 0, 0, mid);
    Sess src1 = make_session(g, 1, mid, n_layers);
    Sess dst0 = make_session(g, 0, 0, n_layers);
    Sess dst1 = make_session(g, 1, 0, n_layers);
    Sess* srcs[2] = {&src0, &src1};

    GdnSplitPlan plan[2] = {gdn_split_plan(g, 0, 2), gdn_split_plan(g, 1, 2)};
    req(plan[0].heads == g.ssm_v_heads / 2 && plan[1].head_off == plan[0].heads, "gdn_split_plan");

    float* arena[2] = {nullptr, nullptr};
    ck(cudaSetDevice(0), "cudaSetDevice");
    ck(cudaMalloc((void**) &arena[0], (size_t) n_gdn * plan[0].floats() * 4), "malloc(arena 0)");
    ck(cudaSetDevice(1), "cudaSetDevice");
    ck(cudaMalloc((void**) &arena[1], (size_t) n_gdn * plan[1].floats() * 4), "malloc(arena 1)");

    // ---- fill the sources ---------------------------------------------------------------------------
    for (Sess* s : srcs) {
        for (int64_t q = s->st.qsa_ord0; q < s->st.qsa_ord0 + s->st.qsa_alloc; ++q) {
            const QsaState& st = s->st.qsa_states[(size_t) q];
            const Spans sp = spans_of(st, g);
            for (int i = 0; i < sp.n; ++i) fill_pat(s->dev, sp.p[i], sp.b[i], qsa_seed(q, i));
            fill_pat(s->dev, st.page_table, (uint64_t) st.n_pages * 4, qsa_seed(q, 4));
            fill_pat(s->dev, st.idx_tail, (uint64_t) (idx_block - 1) * g.idx_key_dim * 4, qsa_seed(q, 5));
            fill_pat(s->dev, st.idx_dead, (uint64_t) g.idx_key_dim * 4, qsa_seed(q, 6));
            fill_pat(s->dev, st.idx_pooled, (uint64_t) st.idx_pooled_rows * g.idx_key_dim * 4, qsa_seed(q, 7));
            fill_pat(s->dev, st.idx_block_pos, 4, qsa_seed(q, 8));
        }
        for (int64_t r = 0; r < s->st.gdn_alloc; ++r) {
            const int64_t ord = s->st.gdn_ord0 + r;
            fill_rows(s->dev, s->st.gdn_state + (size_t) r * full_floats, full_floats, (uint32_t) ord + 0xA5A50000u);
        }
    }
    // the PLE: layer 1 lives on stage 0, so ITS history is the one that must arrive (stage 1's is different)
    fill_rows(src0.dev, src0.st.ple_hist, strata::kernels::NG_HIST * strata::kernels::NG_HC_DIM, 0xBEEF0001u);
    fill_rows(src1.dev, src1.st.ple_hist, strata::kernels::NG_HIST * strata::kernels::NG_HC_DIM, 0xBEEF0002u);
    src0.st.ple_prev[0] = 11; src0.st.ple_prev[1] = 22; src0.st.ple_token = 33;
    src1.st.ple_prev[0] = 44; src1.st.ple_prev[1] = 55; src1.st.ple_token = 66;

    // ---- poison the destinations --------------------------------------------------------------------
    for (Sess* d : {&dst0, &dst1}) {
        for (int64_t q = 0; q < d->st.qsa_alloc; ++q) {
            const QsaState& st = d->st.qsa_states[(size_t) q];
            const Spans sp = spans_of(st, g);
            for (int i = 0; i < sp.n; ++i) poison(d->dev, sp.p[i], sp.b[i]);
            poison(d->dev, st.page_table, (uint64_t) st.n_pages * 4);
            poison(d->dev, st.idx_tail, (uint64_t) (idx_block - 1) * g.idx_key_dim * 4);
            poison(d->dev, st.idx_dead, (uint64_t) g.idx_key_dim * 4);
            poison(d->dev, st.idx_pooled, (uint64_t) st.idx_pooled_rows * g.idx_key_dim * 4);
            poison(d->dev, st.idx_block_pos, 4);
        }
        poison(d->dev, d->st.ple_hist, (uint64_t) strata::kernels::NG_HIST * strata::kernels::NG_HC_DIM * 4);
        d->st.ple_prev[0] = d->st.ple_prev[1] = -7;
        d->st.ple_token = -7;
    }
    poison(0, arena[0], (uint64_t) n_gdn * plan[0].floats() * 4);
    poison(1, arena[1], (uint64_t) n_gdn * plan[1].floats() * 4);

    KvSrcStage src[2] = {{&src0.st, 0}, {&src1.st, 1}};
    KvDstRank dst[2];
    for (int r = 0; r < 2; ++r) {
        dst[r].session = r == 0 ? &dst0.st : &dst1.st;
        dst[r].device = r;
        dst[r].rank = r;
        dst[r].gdn_state = nullptr;   // QSA/PLE only for the first pass
        dst[r].gdn_floats = 0;
    }
    std::string err;
    KvRedistStats st{};

    // ---- pass 1: QSA + PLE only; the GDN arena must be untouched -------------------------------------
    req(kv_redistribute(g, kTokens, src, 2, dst, 2, &st, err), "pass 1 (no GDN): " + err);
    req(st.qsa_layers == g.n_qsa_layers(), "pass 1: qsa_layers");
    req(st.gdn_layers == 0 && st.gdn_bytes == 0, "pass 1 wrote GDN state it was not asked to move");
    req(st.ple_bytes == 2 * (uint64_t) strata::kernels::NG_HIST * strata::kernels::NG_HC_DIM * 4,
        "pass 1: ple_bytes (the total over both ranks)");
    std::printf("pass 1: %d QSA layers, %.2f MB total, %llu B PLE total\n", st.qsa_layers,
                (double) st.qsa_bytes / 1048576.0, (unsigned long long) st.ple_bytes);
    for (int r = 0; r < 2; ++r) {
        const std::vector<uint8_t> got = readback(r, arena[r], (uint64_t) n_gdn * plan[r].floats() * 4);
        for (size_t i = 0; i < got.size(); i += 4096)
            req(got[i] == 0xCD, "pass 1 touched the GDN arena of rank " + std::to_string(r));
    }

    // ---- pass 2: everything --------------------------------------------------------------------------
    for (int r = 0; r < 2; ++r) {
        dst[r].gdn_state = arena[r];
        dst[r].gdn_floats = plan[r].floats();
    }
    req(kv_redistribute(g, kTokens, src, 2, dst, 2, &st, err), "pass 2: " + err);
    req(st.qsa_layers == g.n_qsa_layers(), "pass 2: qsa_layers");
    req(st.gdn_layers == n_gdn, "pass 2: gdn_layers");
    std::printf("pass 2: %d QSA + %d GDN layers, %.2f MB QSA total, %.2f MB GDN total, %llu B PLE total\n",
                st.qsa_layers, st.gdn_layers, (double) st.qsa_bytes / 1048576.0,
                (double) st.gdn_bytes / 1048576.0, (unsigned long long) st.ple_bytes);

    // ---- verify QSA on both ranks (pass 2 also proves the QSA half is idempotent) ---------------------
    auto verify_qsa_ple = [&](const std::string& tag) {
        for (int r = 0; r < 2; ++r) {
            Sess* d = r == 0 ? &dst0 : &dst1;
            for (int64_t q = 0; q < d->st.qsa_alloc; ++q) {
                const QsaState& s = d->st.qsa_states[(size_t) q];
                const Spans sp = spans_of(s, g);
                const std::string where = tag + " rank " + std::to_string(r) + " QSA " + std::to_string(q);
                req(s.n_slots == (kMaxCells + page_size - 1) / page_size,
                    where + ": the fixture's page count assumption");
                for (int i = 0; i < sp.n; ++i) {
                    const uint64_t per_slot = sp.b[i] / (uint64_t) s.n_slots;
                    check_span(r, sp.p[i], sp.b[i], per_slot * (uint64_t) used_pages, qsa_seed(q, i),
                               where + " pool " + std::to_string(i));
                }
                check_span(r, s.page_table, (uint64_t) s.n_pages * 4, (uint64_t) s.n_pages * 4, qsa_seed(q, 4),
                           where + " page_table");
                check_span(r, s.idx_tail, (uint64_t) (idx_block - 1) * g.idx_key_dim * 4,
                           (uint64_t) (idx_block - 1) * g.idx_key_dim * 4, qsa_seed(q, 5), where + " idx_tail");
                check_span(r, s.idx_dead, (uint64_t) g.idx_key_dim * 4, (uint64_t) g.idx_key_dim * 4,
                           qsa_seed(q, 6), where + " idx_dead");
                const int64_t pooled_rows = std::min<int64_t>(kTokens / idx_block + 2, s.idx_pooled_rows);
                check_span(r, s.idx_pooled, (uint64_t) s.idx_pooled_rows * g.idx_key_dim * 4,
                           (uint64_t) pooled_rows * g.idx_key_dim * 4, qsa_seed(q, 7), where + " idx_pooled");
                check_span(r, s.idx_block_pos, 4, 4, qsa_seed(q, 8), where + " idx_block_pos");
            }
            // the PLE must be stage 0's, not stage 1's
            const uint64_t ple = (uint64_t) strata::kernels::NG_HIST * strata::kernels::NG_HC_DIM;
            const std::vector<uint8_t> hist = readback(r, d->st.ple_hist, ple * 4);
            for (uint64_t i = 0; i < ple; ++i) {
                float want = 0.0f;
                const uint32_t bits =
                    0x3F000000u | (h32(0xBEEF0001u * 2654435761u + (uint32_t) i * 2246822519u) & 0x7FFFFFu);
                std::memcpy(&want, &bits, 4);
                req(std::memcmp(&hist[(size_t) (i * 4)], &want, 4) == 0,
                    tag + " rank " + std::to_string(r) + " PLE history row " + std::to_string(i) +
                        " is not stage 0's");
            }
            req(d->st.ple_prev[0] == 11 && d->st.ple_prev[1] == 22 && d->st.ple_token == 33,
                tag + " rank " + std::to_string(r) + " PLE token window");
        }
    };
    verify_qsa_ple("pass 2");
    std::printf("QSA + PLE: byte-exact on both ranks, unused cells untouched\n");

    // ---- verify GDN: the host reference permute, then the engines ------------------------------------
    std::vector<int> src_of_ord((size_t) n_gdn, -1);
    {
        int64_t ord = 0;
        for (int64_t l = 0; l < n_layers; ++l)
            if (!is_qsa_layer(g, l)) src_of_ord[(size_t) ord++] = l < mid ? 0 : 1;
        req(ord == n_gdn, "the GDN ordinal walk");
    }
    std::vector<float> full_row, got, want;
    for (int64_t ord = 0; ord < n_gdn; ++ord) {
        const Sess* s = srcs[src_of_ord[(size_t) ord]];
        const int64_t row = ord - s->st.gdn_ord0;
        read_floats(s->dev, s->st.gdn_state + (size_t) row * full_floats, full_floats, full_row);
        for (int r = 0; r < 2; ++r) {
            read_floats(r, arena[r] + (size_t) ord * plan[r].floats(), plan[r].floats(), got);
            ref_permute(plan[r], full_row, want);
            expect_eq(got, want, "GDN ordinal " + std::to_string(ord) + " rank " + std::to_string(r) +
                                     " does not match the host reference permute");
            semantic_conv(g, plan[r], r, full_row, got, (uint32_t) (ord * 3 + r) + 0xC0DE0000u);
            semantic_recurrence(plan[r], r, full_row, got, (uint32_t) (ord * 5 + r) + 0xF00D0000u);
        }
    }
    std::printf("GDN: %lld layers x 2 ranks bit-exact against the host permute\n", (long long) n_gdn);
    std::printf("GDN: conv + recurrence give the full model's own outputs, bit for bit\n");

    // ---- pass 3: the PHASED call of §8.4 - the sources ARE the destinations --------------------------
    //
    // A phased run shares ONE whole-model session per card between its two roles: the layer-split prefill
    // advances the half it computed straight into the session TP decode will read, and this call brings the
    // other card's half.  The source is then the destination object itself, so
    //   * the layer ownership must be EXPLICIT - two stages sharing a session cannot get it from
    //     `layer_lo/layer_hi`, which must stay 0/n_layers for the destination checks;
    //   * the spans already in place must be SKIPPED, not copied onto themselves.  A self-copy leaves the bytes
    //     identical either way, so `qsa_bytes` is the only observable of it: the phased call has to move about
    //     half of what the sliced-source pass moved.
    const uint64_t pass2_qsa = st.qsa_bytes;
    {
        const int64_t q_split = mid / std::max<int64_t>(g.qsa_interval, 1);   // ordinals >= this belong to card 1
        for (int64_t q = 0; q < dst0.st.qsa_alloc; ++q) {
            Sess* d = q < q_split ? &dst1 : &dst0;   // take away the half the OTHER card computed
            const QsaState& s = d->st.qsa_states[(size_t) q];
            const Spans sp = spans_of(s, g);
            for (int i = 0; i < sp.n; ++i) poison(d->dev, sp.p[i], sp.b[i]);
            poison(d->dev, s.page_table, (uint64_t) s.n_pages * 4);
            poison(d->dev, s.idx_tail, (uint64_t) (idx_block - 1) * g.idx_key_dim * 4);
            poison(d->dev, s.idx_dead, (uint64_t) g.idx_key_dim * 4);
            poison(d->dev, s.idx_pooled, (uint64_t) s.idx_pooled_rows * g.idx_key_dim * 4);
            poison(d->dev, s.idx_block_pos, 4);
        }
        poison(1, dst1.st.ple_hist, (uint64_t) strata::kernels::NG_HIST * strata::kernels::NG_HC_DIM * 4);
        dst1.st.ple_prev[0] = dst1.st.ple_prev[1] = dst1.st.ple_token = -7;

        KvSrcStage phased[2] = {{&dst0.st, 0, 0, (int) mid}, {&dst1.st, 1, (int) mid, (int) n_layers}};
        KvDstRank pdst[2];
        for (int r = 0; r < 2; ++r) {
            pdst[r].session = r == 0 ? &dst0.st : &dst1.st;
            pdst[r].device = r;
            pdst[r].rank = r;
            pdst[r].gdn_state = nullptr;
            pdst[r].gdn_floats = 0;
        }
        std::string perr;
        KvRedistStats pst{};
        req(kv_redistribute(g, kTokens, phased, 2, pdst, 2, &pst, perr), "pass 3 (phased): " + perr);
        req(pst.qsa_layers == g.n_qsa_layers(), "pass 3: qsa_layers");
        req(pst.ple_bytes == (uint64_t) strata::kernels::NG_HIST * strata::kernels::NG_HC_DIM * 4,
            "pass 3: the PLE crossed exactly once (card 0 already had it)");
        req(pst.qsa_bytes * 2 < pass2_qsa * 3 / 2 && pst.qsa_bytes * 2 > pass2_qsa / 2,
            "pass 3 moved " + std::to_string(pst.qsa_bytes) + " bytes where the sliced pass moved " +
                std::to_string(pass2_qsa) + ": a phased call must move about half (self-copies are skipped)");
        verify_qsa_ple("pass 3");
        std::printf("phased: sources ARE the destinations; %.2f MB moved vs pass 2's %.2f MB; states exact\n",
                    (double) pst.qsa_bytes / 1048576.0, (double) pass2_qsa / 1048576.0);

        // the GDN permute reads one arena and writes another, and a phased run is exactly where the two could
        // become the same allocation - which the call must refuse instead of corrupting rows it has not read
        KvDstRank shared[2] = {pdst[0], pdst[1]};
        shared[0].gdn_state = dst0.st.gdn_state;   // the source stage's own full-width arena
        shared[0].gdn_floats = plan[0].floats();
        shared[1].gdn_state = arena[1];
        shared[1].gdn_floats = plan[1].floats();
        err.clear();
        req(!kv_redistribute(g, kTokens, phased, 2, shared, 2, nullptr, err), "a shared GDN arena was accepted");
        req(err.find("share one GDN arena") != std::string::npos, "the shared-arena error: " + err);

        // an explicit range that owns nothing is a caller bug, not a no-op
        KvSrcStage empty[2] = {phased[0], phased[1]};
        empty[1].layer_hi = (int) mid;
        err.clear();
        req(!kv_redistribute(g, kTokens, empty, 2, pdst, 2, nullptr, err), "a source owning no layers was accepted");
        req(err.find("owns no layers") != std::string::npos, "the empty-range error: " + err);
        std::printf("phased negatives: a shared GDN arena and an empty layer range are refused\n");
    }

    // ---- negative cases ------------------------------------------------------------------------------
    Sess bad_range = make_session(g, 0, 0, mid);
    KvDstRank bad[2] = {dst[0], dst[1]};
    bad[0].session = &bad_range.st;
    err.clear();
    req(!kv_redistribute(g, kTokens, src, 2, bad, 2, nullptr, err), "a sliced destination was accepted");
    req(err.find("whole model") != std::string::npos, "the sliced-destination error: " + err);

    KvSrcStage one[1] = {src[0]};
    err.clear();
    req(!kv_redistribute(g, kTokens, one, 1, dst, 2, nullptr, err), "a missing layer owner was accepted");
    req(err.find("no stage runs layer") != std::string::npos, "the missing-owner error: " + err);

    KvDstRank bad_arena[2] = {dst[0], dst[1]};
    bad_arena[0].gdn_floats = plan[0].floats() - 1;
    err.clear();
    req(!kv_redistribute(g, kTokens, src, 2, bad_arena, 2, nullptr, err), "a short GDN arena was accepted");
    req(err.find("the split needs") != std::string::npos, "the short-arena error: " + err);

    Sess wide2;
    wide2.dev = 1;
    wide2.lo = mid;
    wide2.hi = n_layers;
    wide2.bytes = session_bytes(g, kMaxCells * 2, kBatch, mid, n_layers);
    ck(cudaSetDevice(1), "cudaSetDevice");
    ck(cudaMalloc(&wide2.buf, wide2.bytes), "malloc");
    req(session_init(g, kMaxCells * 2, kBatch, wide2.buf, wide2.st, mid, n_layers) > 0, "session_init");
    KvSrcStage mixed[2] = {{&src0.st, 0}, {&wide2.st, 1}};
    err.clear();
    req(!kv_redistribute(g, kTokens, mixed, 2, dst, 2, nullptr, err), "a differently-sized source was accepted");
    req(err.find("different contexts") != std::string::npos, "the context-mismatch error: " + err);
    std::printf("negative cases: 4 shape mismatches refused with an error\n");

    free_session(bad_range);
    free_session(wide2);
    cudaFree(arena[0]);
    cudaFree(arena[1]);
    free_session(dst0);
    free_session(dst1);
    free_session(src0);
    free_session(src1);
}

}  // namespace

int main() {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev < 2) {
        std::printf("SKIP: kv_redistribute needs two CUDA devices with P2P (found %d)\n", ndev);
        return 0;
    }
    int peer = 0;
    cudaDeviceCanAccessPeer(&peer, 0, 1);
    if (!peer) {
        std::printf("SKIP: no peer access between devices 0 and 1\n");
        return 0;
    }
    ModelGeometry g;   // the artifact's own geometry, which is what the layer pattern and the kernels use
    run_case(g, false);
    run_case(g, true);
    qsa_set_kv_int8(false);
    std::printf("PASS: kv_redistribute\n");
    return 0;
}
