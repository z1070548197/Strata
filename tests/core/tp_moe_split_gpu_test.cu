// tests/core/tp_moe_split_gpu_test.cu - the experts' FFN split against the single-card result, on real kernels.
//
// The split is an OUTPUT-ROW split of the down (each card n_embd/2 down rows, the FULL n_ff reduction), so the
// two phases must reproduce the single-card result EXACTLY, not up to summation order:
//   full   : the single-card `native_expert_grouped` (n_ff 640, n_embd 2560)
//   rank r : gate/up at the local width (n_ff/2 = 320 rows) -> the exchange assembles the FULL hq from the two
//            ranks' local halves -> down at n_embd/2 rows (1280) with the full 640-term reduction -> pitched into
//            the full-width output.  This mirrors the two-card window's `gu_split` / `moe_hq_exchange` /
//            `down_split` / `moe_hit_pitch` exactly (one device stands in for the P2P read).
// The contract is `out_full == out_r` for EVERY element (bit-exact): the down's reduction order is unchanged
// from the single card, so a byte-offset mistake in tp_moe_split.hpp, a wrong hq offset, or a wrong pitch row
// shows up as a difference in exactly the columns that rank owns.
//
// The plan's SHAPE matters as much as its offsets.  With one entry per group the per-entry strides of
// `native_expert_grouped` (the `off[c]` arrays, the staged-activation blocks) are never exercised, so a broken
// multi-entry path still passes.  A decode window with T tokens and top-k 4 puts several entries in a group, so
// the cases below sweep 1..4 entries per group (GRP_NC is 4) over 1 and 3 experts and 4 activation rows.
//
// The blob is filled from a deterministic RNG so the test needs no model file.  Every NVFP4 block is given a
// valid scale byte, so the dequantizer's arithmetic is exercised rather than short-circuited by NaNs.
#include "strata/core/tp_moe_split.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using strata::kernels::NativeExpertLayout;
using strata::kernels::native_expert_layout;

#define CK(call)                                                                     \
    do {                                                                             \
        const cudaError_t e_ = (call);                                               \
        if (e_ != cudaSuccess) {                                                     \
            std::fprintf(stderr, "FAIL: %s: %s\n", #call, cudaGetErrorString(e_));    \
            return 1;                                                                \
        }                                                                            \
    } while (0)

namespace {

constexpr int64_t kNEmbd = 2560;
constexpr int64_t kNff = 640;
constexpr int kGu = 40, kDown = 40;   // NVFP4 both ways: the pack this engine serves
constexpr int64_t kTok = 4;           // activation rows: T = 4

// NVFP4 block: 4 UE4M3 scale bytes + 32 packed nibbles, 64 elements.
constexpr size_t kNvBlock = 36;

void fill_blob(std::vector<uint8_t>& blob, std::mt19937& rng) {
    std::uniform_int_distribution<int> byte(0, 255);
    for (auto& b : blob) b = (uint8_t) byte(rng);
    // Every 36-byte block's first four bytes are scales; a UE4M3 of 0x38..0x40 keeps them sane and non-zero.
    // (A zero or a NaN byte would make the block's contribution vanish or poison the sum, hiding a wrong offset.)
    for (size_t off = 0; off + kNvBlock <= blob.size(); off += kNvBlock)
        for (size_t i = 0; i < 4; ++i) blob[off + i] = (uint8_t) (0x38 + (blob[off + i] & 7));
}

// One case's host plan: `n_expert` groups, each holding `per_group` entries.  Entry `g*per_group + e` writes
// output row `g*per_group + e` and reads activation row `(g*per_group + e) % kTok`.
struct Plan {
    std::vector<int32_t> start;   // n_expert + 1
    std::vector<int32_t> dst;     // entries
    std::vector<int32_t> tok;     // entries
    int32_t ngrp = 0;
    int32_t entries = 0;
    int64_t rows = 0;             // the output rows this plan touches
};

Plan make_plan(int n_expert, int per_group) {
    Plan p;
    p.ngrp = (int32_t) n_expert;
    p.entries = (int32_t) (n_expert * per_group);
    p.rows = p.entries;
    p.start.resize((size_t) n_expert + 1);
    // `dst` is a NON-identity permutation (a rotation by one), so entry e's destination row is not e: this
    // exercises the split down's row indexing (`out[dst[e]]`) and the pitch's read-by-row, which an identity
    // `dst` would hide.  `tok` (the activation row) is independent of `dst`.
    for (int g = 0; g < n_expert; ++g) {
        p.start[(size_t) g] = g * per_group;
        for (int e = 0; e < per_group; ++e) {
            const int32_t row = g * per_group + e;
            p.dst.push_back((row + 1) % p.entries);
            p.tok.push_back(row % (int32_t) kTok);
        }
    }
    p.start[(size_t) n_expert] = p.entries;
    return p;
}

struct Dev {
    unsigned long long* ptr = nullptr;
    int32_t* start = nullptr;
    int32_t* ngrp = nullptr;
    int32_t* dst = nullptr;
    int32_t* tok = nullptr;
};

// Upload the plan, pointing every group at `d_blob + ptr_off`.
int upload(const Plan& p, const uint8_t* d_blob, size_t ptr_off, Dev& d, cudaStream_t cs) {
    std::vector<unsigned long long> ptr((size_t) p.ngrp);
    for (int g = 0; g < p.ngrp; ++g) ptr[(size_t) g] = (unsigned long long) (d_blob + ptr_off);
    if (cudaMalloc((void**) &d.ptr, ptr.size() * sizeof(unsigned long long)) != cudaSuccess) return 1;
    if (cudaMalloc((void**) &d.start, p.start.size() * sizeof(int32_t)) != cudaSuccess) return 1;
    if (cudaMalloc((void**) &d.ngrp, sizeof(int32_t)) != cudaSuccess) return 1;
    if (cudaMalloc((void**) &d.dst, (size_t) p.entries * sizeof(int32_t)) != cudaSuccess) return 1;
    if (cudaMalloc((void**) &d.tok, (size_t) p.entries * sizeof(int32_t)) != cudaSuccess) return 1;
    const bool bad =
        cudaMemcpyAsync(d.ptr, ptr.data(), ptr.size() * sizeof(unsigned long long), cudaMemcpyHostToDevice, cs) != cudaSuccess ||
        cudaMemcpyAsync(d.start, p.start.data(), p.start.size() * sizeof(int32_t), cudaMemcpyHostToDevice, cs) != cudaSuccess ||
        cudaMemcpyAsync(d.ngrp, &p.ngrp, sizeof(p.ngrp), cudaMemcpyHostToDevice, cs) != cudaSuccess ||
        cudaMemcpyAsync(d.dst, p.dst.data(), (size_t) p.entries * sizeof(int32_t), cudaMemcpyHostToDevice, cs) != cudaSuccess ||
        cudaMemcpyAsync(d.tok, p.tok.data(), (size_t) p.entries * sizeof(int32_t), cudaMemcpyHostToDevice, cs) != cudaSuccess;
    return bad ? 1 : 0;
}

void free_dev(Dev& d) {
    if (d.ptr != nullptr) cudaFree(d.ptr);
    if (d.start != nullptr) cudaFree(d.start);
    if (d.ngrp != nullptr) cudaFree(d.ngrp);
    if (d.dst != nullptr) cudaFree(d.dst);
    if (d.tok != nullptr) cudaFree(d.tok);
    d = Dev{};
}

int run_case(const NativeExpertLayout& L, void* d_scratch, float* d_out, int64_t cap, int64_t cap_entries, const Dev& d,
             const void* d_xq, cudaStream_t cs) {
    if (cudaMemsetAsync(d_out, 0, (size_t) cap * kNEmbd * sizeof(float), cs) != cudaSuccess) return 1;
    strata::kernels::native_expert_grouped(L, d.ptr, d.start, d.ngrp, d.dst, d.tok, cap, cap_entries, d_xq, d_scratch,
                                          d_out, cs, 0);
    return cudaStreamSynchronize(cs) == cudaSuccess ? 0 : 1;
}

// The scratch geometry the two-phase split uses (mirrors the engine): the gate/up call at the LOCAL width writes
// its hq at `native_expert_hq_off(cap, n_ff_local)` inside the local scratch; the exchange assembles the full
// hq at `native_expert_hq_off(cap, n_ff_full)` inside a separate full-width buffer (the down reads its hq there).
// The local width (n_ff/2) is a whole number of q8_1 blocks (10 at n_ff 640), so the exchange's half_blocks is
// n_ff_local/32.
struct ScratchGeom {
    uint8_t* local_hq;   // this rank's gate/up scratch hq base
    uint8_t* full_hq;    // the assembled full hq base (separate buffer)
    int half_blocks;
    int own_off;         // this rank's block offset in the assembled hq
};
ScratchGeom scratch_geom(const uint8_t* local_scratch, uint8_t* full_scratch, int64_t cap, const strata::core::TpMoeSplit& s,
                         int rank) {
    const int64_t n_ff_local = s.n_ff_local;
    const int64_t n_ff_full = s.layout.n_ff * 2;
    ScratchGeom g;
    g.local_hq = (uint8_t*) local_scratch + strata::kernels::native_expert_hq_off(cap, n_ff_local);
    g.full_hq = full_scratch + strata::kernels::native_expert_hq_off(cap, n_ff_full);
    g.half_blocks = (int) (n_ff_local / 32);
    g.own_off = rank * g.half_blocks;
    return g;
}

}  // namespace

int main() {
    int devs = 0;
    if (cudaGetDeviceCount(&devs) != cudaSuccess || devs < 1) {
        std::fprintf(stderr, "tp_moe_split_gpu: needs one GPU\n");
        return 77;
    }
    std::printf("tp_moe_split_gpu: full(n_ff 640) vs the two halves, on the real grouped expert kernels\n");
    std::string why;
    if (!strata::core::tp_moe_split_ok(kGu, kDown, kNEmbd, kNff, why)) {
        std::printf("tp_moe_split_gpu: the geometry cannot split (%s)\n", why.c_str());
        return 1;
    }
    strata::core::TpMoeSplit half[2];
    for (int r = 0; r < 2; ++r)
        if (!strata::core::tp_moe_split(kGu, kDown, kNEmbd, kNff, r, half[r], why)) {
            std::fprintf(stderr, "FAIL: split rank %d: %s\n", r, why.c_str());
            return 1;
        }
    const NativeExpertLayout full = native_expert_layout(kGu, kDown, kNEmbd, kNff);

    std::mt19937 rng(12345);
    std::vector<uint8_t> blob(full.bytes);
    fill_blob(blob, rng);
    // the NVFP4 tail: {s_gate, s_up, s_down, 0}.  The kernels multiply by these, so they must be plausible.
    {
        float* tail = (float*) (blob.data() + full.tail_off);
        tail[0] = 1.0f;
        tail[1] = 1.0f;
        tail[2] = 1.0f;
        tail[3] = 0.0f;
    }
    // the packed widths must tile the blob exactly (a sanity check on the fixture itself)
    if (full.bytes != (size_t) kNff * full.gu_row * 2 + (size_t) kNEmbd * full.d_row + 16) {
        std::fprintf(stderr, "FAIL: fixture blob size %zu does not match the layout\n", blob.size());
        return 1;
    }

    cudaStream_t cs = nullptr;
    CK(cudaStreamCreate(&cs));
    uint8_t* d_blob = nullptr;
    float* d_x = nullptr;
    uint8_t* d_xq = nullptr;
    uint8_t* d_xq0 = nullptr;   // the pool-order activation (block_q8_0), for the split scan below
    CK(cudaMalloc((void**) &d_blob, blob.size()));
    CK(cudaMemcpyAsync(d_blob, blob.data(), blob.size(), cudaMemcpyHostToDevice, cs));
    CK(cudaMalloc((void**) &d_x, (size_t) kTok * kNEmbd * sizeof(float)));
    CK(cudaMalloc((void**) &d_xq, (size_t) kTok * (kNEmbd / 32) * 36));
    CK(cudaMalloc((void**) &d_xq0, (size_t) kTok * (kNEmbd / 32) * 34));
    {
        std::vector<float> x((size_t) kTok * kNEmbd);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        for (auto& v : x) v = nd(rng);
        CK(cudaMemcpyAsync(d_x, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice, cs));
        strata::kernels::native_quantize_q8_1(d_x, d_xq, (int) kNEmbd, (int) kTok, cs);
        CK(cudaStreamSynchronize(cs));
    }

    // The real prefill window is cap = T*K (T tokens, top-K 10) = ~50 entries at T=5.  The cases above sweep small
    // scales (cap <= 12); the list also covers real scale (n_expert up to 10, per_group up to 4 -> cap up to 40,
    // plus an explicit 5-expert x 10 = cap 50).  A scale-dependent kernel branch (grid/gy, staging, per-group
    // stride) that is bit-exact at small scale but diverges at the real scale shows up here, without a model load.
    const int cases[] = { 1, 1, 1, 2, 3, 1, 3, 2, 5, 1, 5, 2, 10, 1, 10, 2, 10, 4, 5, 10 };
    int fails = 0;
    for (size_t ci = 0; ci + 1 < sizeof(cases) / sizeof(int); ci += 2) {
        const int n_expert = cases[ci], per_group = cases[ci + 1];
        {
            const Plan plan = make_plan(n_expert, per_group);
            const int64_t cap = plan.rows;
            const int64_t entries = plan.entries;
            Dev d;
            if (upload(plan, d_blob, 0, d, cs) != 0) {
                std::fprintf(stderr, "FAIL: upload (%d x %d)\n", n_expert, per_group);
                return 1;
            }
            const size_t local_scratch_bytes = strata::core::tp_moe_split_scratch_bytes(cap, half[0]);
            const size_t full_scratch_bytes = strata::kernels::native_expert_scratch_bytes(cap, kNff);
            void* d_scratch_full = nullptr;
            void* d_scratch_half[2] = {nullptr, nullptr};
            void* d_down_full[2] = {nullptr, nullptr};   // the assembled full hq, one per rank (the down reads here)
            float* d_out_full = nullptr;
            float* d_out_half[2] = {nullptr, nullptr};   // the full-width pitched output, one per rank
            float* d_compact[2] = {nullptr, nullptr};    // the compact [cap][n/2] down output
            if (cudaMalloc(&d_scratch_full, full_scratch_bytes) != cudaSuccess ||
                cudaMalloc(&d_scratch_half[0], local_scratch_bytes) != cudaSuccess ||
                cudaMalloc(&d_scratch_half[1], local_scratch_bytes) != cudaSuccess ||
                cudaMalloc(&d_down_full[0], full_scratch_bytes) != cudaSuccess ||
                cudaMalloc(&d_down_full[1], full_scratch_bytes) != cudaSuccess ||
                cudaMalloc((void**) &d_out_full, (size_t) cap * kNEmbd * sizeof(float)) != cudaSuccess ||
                cudaMalloc((void**) &d_out_half[0], (size_t) cap * kNEmbd * sizeof(float)) != cudaSuccess ||
                cudaMalloc((void**) &d_out_half[1], (size_t) cap * kNEmbd * sizeof(float)) != cudaSuccess ||
                cudaMalloc((void**) &d_compact[0], (size_t) cap * (kNEmbd / 2) * sizeof(float)) != cudaSuccess ||
                cudaMalloc((void**) &d_compact[1], (size_t) cap * (kNEmbd / 2) * sizeof(float)) != cudaSuccess) {
                std::fprintf(stderr, "FAIL: alloc (%d x %d)\n", n_expert, per_group);
                return 1;
            }
            std::vector<float> out_full((size_t) cap * kNEmbd), out_h0(out_full.size()), out_h1(out_full.size());
            // the UNSHIFTED group pointers (all at the blob base): the down uses these (its row offset is in
            // half[r].down.down_off), unlike the gate/up phase which uses the shifted `ptr` above.
            std::vector<unsigned long long> base_ptr((size_t) plan.ngrp, (unsigned long long) d_blob);

            // the single-card result (the reference): full n_ff 640 gate/up + down, one launch, phase 0.
            strata::kernels::native_expert_set_mode(-1, 0);
            if (run_case(full, d_scratch_full, d_out_full, cap, entries, d, d_xq, cs) != 0) {
                std::fprintf(stderr, "FAIL: full launch (%d x %d)\n", n_expert, per_group);
                return 1;
            }
            CK(cudaMemcpyAsync(out_full.data(), d_out_full, out_full.size() * 4, cudaMemcpyDeviceToHost, cs));

            // PHASE 1: BOTH ranks' gate/up at the LOCAL width (n_ff/2 rows), each into its own local scratch.  The
            // group pointers are shifted by the split's gate/up offset; the hq lands at the local-width offset.
            // Running both gu's BEFORE any exchange mirrors the engine's flag handshake (rank r's exchange waits
            // for the peer's flag_h, i.e. the peer's gu is done) - the exchange reads the peer's local hq over
            // "P2P", so that scratch must be populated.
            for (int r = 0; r < 2; ++r) {
                std::vector<unsigned long long> ptr((size_t) plan.ngrp, (unsigned long long) (d_blob + half[r].ptr_off));
                CK(cudaMemcpyAsync(d.ptr, ptr.data(), ptr.size() * sizeof(unsigned long long), cudaMemcpyHostToDevice, cs));
                strata::kernels::native_expert_set_mode(-1, 1);
                strata::kernels::native_expert_grouped(half[r].layout, d.ptr, d.start, d.ngrp, d.dst, d.tok, cap, entries,
                                                      d_xq, d_scratch_half[r], (float*) d_scratch_half[r], cs, 0);
                if (cudaStreamSynchronize(cs) != cudaSuccess) {
                    std::fprintf(stderr, "FAIL: rank %d gu launch\n", r);
                    return 1;
                }
            }
            // PHASE 2: the exchange (assemble the FULL hq from the two now-populated local halves) + the down at
            // the OUTPUT-ROW split (n_embd/2 rows, full n_ff reduction) reading the assembled full hq, writing the
            // compact [cap][n/2] buffer, then pitched into the full-width output's own half.  One device stands in
            // for the P2P read: rank r's "peer" is the OTHER rank's local scratch.  `moe_hq_exchange` reads the
            // peer at `peer_hq` directly (the engine adds the hq offset to the peer's scratch base before calling).
            for (int r = 0; r < 2; ++r) {
                const ScratchGeom geom = scratch_geom((const uint8_t*) d_scratch_half[r], (uint8_t*) d_down_full[r],
                                                      cap, half[r], r);
                const uint8_t* peer_hq = (const uint8_t*) d_scratch_half[1 - r] +
                                         strata::kernels::native_expert_hq_off(cap, half[r].n_ff_local);
                strata::kernels::moe_hq_exchange(geom.local_hq, peer_hq, geom.full_hq, cap, geom.half_blocks, geom.own_off, cs);
                if (cudaStreamSynchronize(cs) != cudaSuccess) {
                    std::fprintf(stderr, "FAIL: rank %d exchange\n", r);
                    return 1;
                }
                // the down at the OUTPUT-ROW split.  The group pointers are UNshifted (the row offset is in
                // half[r].down.down_off); zero the compact buffer first (only the dst rows the down writes are read).
                CK(cudaMemcpyAsync(d.ptr, base_ptr.data(), base_ptr.size() * sizeof(unsigned long long),
                                   cudaMemcpyHostToDevice, cs));
                cudaMemsetAsync(d_compact[r], 0, (size_t) cap * (kNEmbd / 2) * sizeof(float));
                strata::kernels::native_expert_set_mode(-1, 2);
                strata::kernels::native_expert_grouped(half[r].down, d.ptr, d.start, d.ngrp, d.dst, d.tok, cap, entries,
                                                      d_xq, d_down_full[r], d_compact[r], cs, 0);
                if (cudaStreamSynchronize(cs) != cudaSuccess) {
                    std::fprintf(stderr, "FAIL: rank %d down launch\n", r);
                    return 1;
                }
                // pitch the compact half into the full-width output's own half, zeroing the peer's half.
                cudaMemsetAsync(d_out_half[r], 0, (size_t) cap * kNEmbd * sizeof(float));
                strata::kernels::moe_hit_pitch(d_compact[r], d_out_half[r], d.dst, cap, (int) kNEmbd, (int) (kNEmbd / 2),
                                               (int) (r * (kNEmbd / 2)), cs);
                if (cudaStreamSynchronize(cs) != cudaSuccess) {
                    std::fprintf(stderr, "FAIL: rank %d pitch\n", r);
                    return 1;
                }
                std::vector<float>& out_h = (r == 0) ? out_h0 : out_h1;
                CK(cudaMemcpyAsync(out_h.data(), d_out_half[r], out_full.size() * 4, cudaMemcpyDeviceToHost, cs));
            }
            strata::kernels::native_expert_set_mode(-1, 0);
            // the contract: each rank owns exactly half the down output columns (rank r owns [r*n/2, (r+1)*n/2),
            // and zeroes the peer's half so the engine's per-layer allreduce `x + 0.0f` (bitwise `x`) adds the two
            // disjoint halves exactly once.  So `out_h0 + out_h1` must equal the single-card full output
            // BIT-EXACTLY: the down's reduction order is unchanged from the single card, so a wrong byte offset,
            // hq offset, exchange block, or pitch row shows up as a difference in exactly the columns that rank
            // owns.  (The gate/up half IS split on the intermediate axis, but the exchange reassembles the full
            // q8_1 intermediate, so the down - and hence the output - is the single-card computation.)
            int bad = 0;
            int64_t worst = -1;
            double max_abs = 0.0;
            for (int64_t i = 0; i < cap * kNEmbd; ++i) {
                const float got = out_h0[(size_t) i] + out_h1[(size_t) i];
                if (got != out_full[(size_t) i]) {
                    ++bad;
                    const double dd = std::fabs((double) got - (double) out_full[(size_t) i]);
                    if (dd > max_abs) { max_abs = dd; worst = i; }
                }
            }
            std::printf("  experts %d, entries/group %d: %lld element(s) differ (h0+h1 vs full), max |d| = %.6e (row %lld, col %lld)\n",
                        n_expert, per_group, (long long) bad, max_abs, (long long) (worst < 0 ? 0 : worst / kNEmbd),
                        (long long) (worst < 0 ? 0 : worst % kNEmbd));
            if (bad != 0) ++fails;
            cudaFree(d_scratch_full);
            cudaFree(d_scratch_half[0]);
            cudaFree(d_scratch_half[1]);
            cudaFree(d_down_full[0]);
            cudaFree(d_down_full[1]);
            cudaFree(d_out_full);
            cudaFree(d_out_half[0]);
            cudaFree(d_out_half[1]);
            cudaFree(d_compact[0]);
            cudaFree(d_compact[1]);
            free_dev(d);
        }
    }

    // ===================== THE POOL-ORDER SPLIT (STRATA_POOL_ORDER=1's path) =====================
    // The same two-phase protocol, but every stage is the q8_0/pool-order kernel: the reference is
    // `native_expert_grouped_q8_0` phase 0 at full width, the gu runs phase 1 at the local width into the local
    // scratch, the exchange moves 34-byte blocks, and the down is phase 2 over the assembled hq with the
    // identity entry->row map (null `ent_tok`).  The contract is unchanged: h0 + h1 == full, bit-exactly.
    // Without this the split's q8_0 form had NO test at all - the first full-model run would have been the first
    // place a wrong block size, a reused `ent_tok`, or a stale hq offset could have shown up.
    {
        std::printf("tp_moe_split_gpu: the pool-order (q8_0) split, same contract\n");
        strata::kernels::quantize_q8_0(d_x, d_xq0, (int64_t) kTok * kNEmbd, cs);
        CK(cudaStreamSynchronize(cs));
        const int po_cases[] = { 3, 2, 5, 10, 10, 4 };
        for (size_t ci = 0; ci + 1 < sizeof(po_cases) / sizeof(int); ci += 2) {
            const int n_expert = po_cases[ci], per_group = po_cases[ci + 1];
            const Plan plan = make_plan(n_expert, per_group);
            const int64_t cap = plan.rows, entries = plan.entries;
            Dev d;
            if (upload(plan, d_blob, 0, d, cs) != 0) {
                std::fprintf(stderr, "FAIL: po upload (%d x %d)\n", n_expert, per_group);
                return 1;
            }
            std::vector<unsigned long long> base_ptr((size_t) plan.ngrp, (unsigned long long) d_blob);
            const size_t local_scratch_bytes = strata::core::tp_moe_split_scratch_bytes(cap, half[0]);
            const size_t full_scratch_bytes = strata::kernels::native_expert_scratch_bytes(cap, kNff);
            void* d_scratch_full = nullptr;
            void* d_scratch_half[2] = {nullptr, nullptr};
            void* d_down_full[2] = {nullptr, nullptr};
            float *d_out_full = nullptr, *d_out_half[2] = {nullptr, nullptr}, *d_compact[2] = {nullptr, nullptr};
            if (cudaMalloc(&d_scratch_full, full_scratch_bytes) != cudaSuccess ||
                cudaMalloc(&d_scratch_half[0], local_scratch_bytes) != cudaSuccess ||
                cudaMalloc(&d_scratch_half[1], local_scratch_bytes) != cudaSuccess ||
                cudaMalloc(&d_down_full[0], full_scratch_bytes) != cudaSuccess ||
                cudaMalloc(&d_down_full[1], full_scratch_bytes) != cudaSuccess ||
                cudaMalloc((void**) &d_out_full, (size_t) cap * kNEmbd * sizeof(float)) != cudaSuccess ||
                cudaMalloc((void**) &d_out_half[0], (size_t) cap * kNEmbd * sizeof(float)) != cudaSuccess ||
                cudaMalloc((void**) &d_out_half[1], (size_t) cap * kNEmbd * sizeof(float)) != cudaSuccess ||
                cudaMalloc((void**) &d_compact[0], (size_t) cap * (kNEmbd / 2) * sizeof(float)) != cudaSuccess ||
                cudaMalloc((void**) &d_compact[1], (size_t) cap * (kNEmbd / 2) * sizeof(float)) != cudaSuccess) {
                std::fprintf(stderr, "FAIL: po alloc (%d x %d)\n", n_expert, per_group);
                return 1;
            }
            std::vector<float> out_full((size_t) cap * kNEmbd), out_h0(out_full.size()), out_h1(out_full.size());
            // the reference: the pool-order full pipeline (phase 0)
            strata::kernels::native_expert_grouped_q8_0(full, d.ptr, d.start, d.ngrp, d.dst, d.tok, cap, entries,
                                                        d_xq0, (int) ((kNEmbd / 32) * 34), d_scratch_full,
                                                        d_out_full, cs, 0, 0);
            CK(cudaStreamSynchronize(cs));
            CK(cudaMemcpyAsync(out_full.data(), d_out_full, out_full.size() * 4, cudaMemcpyDeviceToHost, cs));
            CK(cudaStreamSynchronize(cs));
            // phase 1: both ranks' gu at the local width; each leaves its hq in its own scratch
            for (int r = 0; r < 2; ++r) {
                std::vector<unsigned long long> ptr((size_t) plan.ngrp, (unsigned long long) (d_blob + half[r].ptr_off));
                CK(cudaMemcpyAsync(d.ptr, ptr.data(), ptr.size() * sizeof(unsigned long long), cudaMemcpyHostToDevice, cs));
                strata::kernels::native_expert_grouped_q8_0(half[r].layout, d.ptr, d.start, d.ngrp, d.dst, d.tok,
                                                            cap, entries, d_xq0, (int) ((kNEmbd / 32) * 34),
                                                            d_scratch_half[r], (float*) d_scratch_half[r], cs, 1, 0);
                CK(cudaStreamSynchronize(cs));
            }
            // phase 2: the 34-byte exchange + the output-row-split down (null ent_tok = the identity map) + pitch
            for (int r = 0; r < 2; ++r) {
                const ScratchGeom geom = scratch_geom((const uint8_t*) d_scratch_half[r], (uint8_t*) d_down_full[r],
                                                      cap, half[r], r);
                const uint8_t* peer_hq = (const uint8_t*) d_scratch_half[1 - r] +
                                         strata::kernels::native_expert_hq_off(cap, half[r].n_ff_local);
                strata::kernels::moe_hq_exchange(geom.local_hq, peer_hq, geom.full_hq, cap, geom.half_blocks,
                                                 geom.own_off, cs, 34);
                CK(cudaStreamSynchronize(cs));
                CK(cudaMemcpyAsync(d.ptr, base_ptr.data(), base_ptr.size() * sizeof(unsigned long long),
                                   cudaMemcpyHostToDevice, cs));
                cudaMemsetAsync(d_compact[r], 0, (size_t) cap * (kNEmbd / 2) * sizeof(float));
                strata::kernels::native_expert_grouped_q8_0(half[r].down, d.ptr, d.start, d.ngrp, d.dst, nullptr,
                                                            cap, entries, nullptr, 0, d_down_full[r], d_compact[r],
                                                            cs, 2, 0);
                CK(cudaStreamSynchronize(cs));
                cudaMemsetAsync(d_out_half[r], 0, (size_t) cap * kNEmbd * sizeof(float));
                strata::kernels::moe_hit_pitch(d_compact[r], d_out_half[r], d.dst, cap, (int) kNEmbd,
                                               (int) (kNEmbd / 2), (int) (r * (kNEmbd / 2)), cs);
                CK(cudaStreamSynchronize(cs));
                std::vector<float>& out_h = (r == 0) ? out_h0 : out_h1;
                CK(cudaMemcpyAsync(out_h.data(), d_out_half[r], out_full.size() * 4, cudaMemcpyDeviceToHost, cs));
                CK(cudaStreamSynchronize(cs));
            }
            int bad = 0;
            int64_t worst = -1;
            double max_abs = 0.0;
            for (int64_t i = 0; i < cap * kNEmbd; ++i) {
                const float got = out_h0[(size_t) i] + out_h1[(size_t) i];
                if (got != out_full[(size_t) i]) {
                    ++bad;
                    const double dd = std::fabs((double) got - (double) out_full[(size_t) i]);
                    if (dd > max_abs) { max_abs = dd; worst = i; }
                }
            }
            std::printf("  po: experts %d, entries/group %d: %lld differ (h0+h1 vs full), max |d| = %.6e\n",
                        n_expert, per_group, (long long) bad, max_abs);
            if (bad != 0) ++fails;
            cudaFree(d_scratch_full); cudaFree(d_scratch_half[0]); cudaFree(d_scratch_half[1]);
            cudaFree(d_down_full[0]); cudaFree(d_down_full[1]); cudaFree(d_out_full);
            cudaFree(d_out_half[0]); cudaFree(d_out_half[1]); cudaFree(d_compact[0]); cudaFree(d_compact[1]);
            free_dev(d);
        }
    }
    // Prefill MMQ packs four 32-value groups as [4 metadata slots][128 q bytes] in each 144-byte block.
    // Verify the dedicated exchange with independently padded local/full rows (320 -> 512, 640 -> 1024).
    {
        constexpr int rows = 7, live = 10, local_stride = 16, full_stride = 32, block_bytes = 36;
        constexpr int subs = 4, meta_bytes = 4, payload_bytes = 32, mmq_bytes = subs * block_bytes;
        const size_t local_bytes = (size_t) local_stride * rows * block_bytes;
        const size_t full_bytes = (size_t) full_stride * rows * block_bytes;
        auto at = [](int b, int r, int j) {
            const int in = j < meta_bytes ? b % subs * meta_bytes + j
                                          : subs * meta_bytes + b % subs * payload_bytes + j - meta_bytes;
            return ((size_t) (b / subs) * rows + r) * mmq_bytes + in;
        };
        std::vector<uint8_t> local[2] = {std::vector<uint8_t>(local_bytes), std::vector<uint8_t>(local_bytes)};
        for (int rank = 0; rank < 2; ++rank)
            for (int b = 0; b < local_stride; ++b)
                for (int r = 0; r < rows; ++r)
                    for (int j = 0; j < block_bytes; ++j)
                        local[rank][at(b, r, j)] = (uint8_t) (1 + rank * 97 + b * 11 + r * 3 + j);
        uint8_t *d_local[2] = {}, *d_full = nullptr;
        CK(cudaMalloc(&d_local[0], local_bytes)); CK(cudaMalloc(&d_local[1], local_bytes));
        CK(cudaMalloc(&d_full, full_bytes));
        CK(cudaMemcpy(d_local[0], local[0].data(), local_bytes, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_local[1], local[1].data(), local_bytes, cudaMemcpyHostToDevice));
        for (int rank = 0; rank < 2; ++rank) {
            strata::kernels::moe_hq_exchange_mmq(d_local[rank], d_local[1 - rank], d_full, rows, live,
                                                  local_stride, full_stride, rank * live, cs, block_bytes);
            CK(cudaStreamSynchronize(cs));
            std::vector<uint8_t> got(full_bytes);
            CK(cudaMemcpy(got.data(), d_full, full_bytes, cudaMemcpyDeviceToHost));
            for (int b = 0; b < full_stride; ++b)
                for (int r = 0; r < rows; ++r)
                    for (int j = 0; j < block_bytes; ++j) {
                        const int owner = b < live ? 0 : (b < 2 * live ? 1 : -1);
                        const int sb = b - (owner < 0 ? 0 : owner * live);
                        const uint8_t want = owner < 0 ? 0 : local[owner][at(sb, r, j)];
                        if (got[at(b, r, j)] != want) ++fails;
                    }
        }
        cudaFree(d_local[0]); cudaFree(d_local[1]); cudaFree(d_full);
    }
    if (fails != 0) return 1;
    cudaFree(d_xq0);
    std::printf("tp_moe_split_gpu: ALL PASS\n");
    return 0;
}
