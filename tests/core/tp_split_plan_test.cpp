// tests/core/tp_split_plan_test.cpp - Phase 3a: verify the TP split plan.
//
// Pure host-side test (no CUDA): for each dense Q8_0 tensor name, check that
//   * tp_split_for returns the expected mode and per-rank n_in/n_out
//   * tp_q8_0_bytes matches (n_in/32)*34*n_out
//   * tp_split_q8_0_host produces a byte-correct half (rank 0 = first half,
//     rank 1 = second half) against an independent host reference.
//
// Build:  g++ -std=c++17 -I include tests/core/tp_split_plan_test.cpp \
//            src/core/tp_split_plan.cpp -o /tmp/tp_split_plan_test
#include "strata/core/tp_split_plan.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using strata::core::TpMode;
using strata::core::TpSplit;
using strata::core::tp_prefill_projection_ok;
using strata::core::tp_projection_for;
using strata::core::tp_q8_0_bytes;
using strata::core::tp_split_for;
using strata::core::tp_split_q8_0_host;

static int g_fail = 0;
static void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++g_fail;
    }
}
static uint64_t splitmix(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

// A full Q8_0 tensor with deterministic per-byte content, so a wrong copy is
// always detectable.
static std::vector<uint8_t> make_full(int n_in, int n_out) {
    const size_t bytes = (size_t) (n_in / 32) * 34 * (size_t) n_out;
    std::vector<uint8_t> v(bytes);
    for (size_t i = 0; i < bytes; ++i) v[i] = (uint8_t) (splitmix(i + 7) & 0xff);
    return v;
}

// Reference: the first (rank 0) or second (rank 1) half of the rows.
static std::vector<uint8_t> ref_col(const std::vector<uint8_t>& full, int n_in, int n_out, int rank) {
    const size_t row = (size_t) (n_in / 32) * 34;
    const int half = n_out / 2;
    std::vector<uint8_t> out((size_t) half * row);
    std::memcpy(out.data(), full.data() + (size_t) rank * half * row, out.size());
    return out;
}
// Reference: per row, the first (rank 0) or second (rank 1) half of the blocks.
static std::vector<uint8_t> ref_row(const std::vector<uint8_t>& full, int n_in, int n_out, int rank) {
    const size_t row = (size_t) (n_in / 32) * 34;
    const size_t half_row = (size_t) (n_in / 64) * 34;
    std::vector<uint8_t> out((size_t) n_out * half_row);
    for (int r = 0; r < n_out; ++r)
        std::memcpy(out.data() + (size_t) r * half_row,
                    full.data() + (size_t) r * row + (size_t) rank * half_row, half_row);
    return out;
}
// Reference: q (full) + k (full) + v_half (first/second half of the v rows).
static std::vector<uint8_t> ref_qkv(const std::vector<uint8_t>& full, int n_in, int n_out, int rank) {
    const size_t row = (size_t) (n_in / 32) * 34;
    const int qd = 2048, kd = 2048, v_start = 4096, v_half = 3072;
    std::vector<uint8_t> out((size_t) (qd + kd + v_half) * row);
    std::memcpy(out.data(), full.data(), (size_t) qd * row);
    std::memcpy(out.data() + (size_t) qd * row, full.data() + (size_t) qd * row, (size_t) kd * row);
    std::memcpy(out.data() + (size_t) (qd + kd) * row,
                full.data() + (size_t) (v_start + rank * v_half) * row, (size_t) v_half * row);
    return out;
}

static void test_one(const std::string& name, TpMode mode, int n_in, int n_out,
                     int per_rank_nin, int per_rank_nout,
                     const std::vector<uint8_t>& full,
                     std::vector<uint8_t> (*ref)(const std::vector<uint8_t>&, int, int, int)) {
    for (int rank = 0; rank < 2; ++rank) {
        const TpSplit s = tp_split_for(name, rank);
        check(s.mode == mode, (name + " mode rank " + std::to_string(rank)).c_str());
        check(s.n_in == per_rank_nin, (name + " n_in rank " + std::to_string(rank)).c_str());
        check(s.n_out == per_rank_nout, (name + " n_out rank " + std::to_string(rank)).c_str());
        const uint64_t bytes = tp_q8_0_bytes(name, rank);
        const uint64_t expect = (uint64_t) (per_rank_nin / 32) * 34 * (uint64_t) per_rank_nout;
        check(bytes == expect, (name + " bytes rank " + std::to_string(rank)).c_str());
        std::vector<uint8_t> half(bytes);
        const size_t written = tp_split_q8_0_host(full.data(), mode, n_in, n_out, rank, half.data());
        check(written == bytes, (name + " written rank " + std::to_string(rank)).c_str());
        const std::vector<uint8_t> r = ref(full, n_in, n_out, rank);
        check(half.size() == r.size(), (name + " size rank " + std::to_string(rank)).c_str());
        check(half == r, (name + " bytes-exact rank " + std::to_string(rank)).c_str());
    }
}

int main() {
    // GDN: attn_qkv 2560 x 10240 (q/k replicated + v split). Per-rank n_out = 7168.
    {
        const int n_in = 2560, n_out = 10240;
        const auto full = make_full(n_in, n_out);
        test_one("blk.0.attn_qkv.weight", TpMode::Qkv, n_in, n_out, n_in, 7168, full, ref_qkv);
    }
    // GDN: attn_gate 2560 x 6144 (column-parallel). Per-rank n_out = 3072.
    {
        const int n_in = 2560, n_out = 6144;
        const auto full = make_full(n_in, n_out);
        test_one("blk.0.attn_gate.weight", TpMode::Col, n_in, n_out, n_in, 3072, full, ref_col);
    }
    // GDN: ssm_out 6144 x 2560 (COLUMN-parallel: half the output rows, the full 6144-term reduction).
    // Splitting n_in instead made each card sum half the reduction and the allreduce add the partials, which
    // is not the single card's sum in float32 - see tests/core/tp_outproj_split_test.cu. Per-rank n_out = 1280.
    {
        const int n_in = 6144, n_out = 2560;
        const auto full = make_full(n_in, n_out);
        test_one("blk.0.ssm_out.weight", TpMode::Col, n_in, n_out, n_in, 1280, full, ref_col);
    }
    // QSA: attn_q 2560 x 12288 (REPLICATED).  The attention kernel's KV-head group is a compile-time
    // constant, so a query-head split would need a new kernel: both cards run the full projection and
    // attention, and the split lives in attn_output's reduction axis (below).
    {
        const int n_in = 2560, n_out = 12288;
        for (int rank = 0; rank < 2; ++rank) {
            const TpSplit s = tp_split_for("blk.3.attn_q.weight", rank);
            check(s.mode == TpMode::Replicate, "attn_q replicate");
            check(tp_q8_0_bytes("blk.3.attn_q.weight", rank) == 0, "attn_q bytes 0");
            check(s.n_in == n_in && s.n_out == n_out, "attn_q full shape");
        }
    }
    // QSA: attn_k / attn_v 2560 x 512 (replicated).
    {
        const int n_in = 2560, n_out = 512;
        const auto full = make_full(n_in, n_out);
        for (int rank = 0; rank < 2; ++rank) {
            const TpSplit s = tp_split_for("blk.3.attn_k.weight", rank);
            check(s.mode == TpMode::Replicate, "attn_k replicate");
            check(tp_q8_0_bytes("blk.3.attn_k.weight", rank) == 0, "attn_k bytes 0");
            const TpSplit sv = tp_split_for("blk.3.attn_v.weight", rank);
            check(sv.mode == TpMode::Replicate, "attn_v replicate");
        }
        (void) full;
    }
    // QSA: attn_output 6144 x 2560 (COLUMN-parallel, the same reasoning as ssm_out above). Per-rank n_out = 1280.
    {
        const int n_in = 6144, n_out = 2560;
        const auto full = make_full(n_in, n_out);
        test_one("blk.3.attn_output.weight", TpMode::Col, n_in, n_out, n_in, 1280, full, ref_col);
    }
    // The large-T prefill view preserves every reduction axis and gives each
    // output-row split a pitched rank-local destination.
    {
        for (int rank = 0; rank < 2; ++rank) {
            const auto qkv = tp_projection_for("blk.0.attn_qkv.weight", rank);
            check(qkv.special_qkv, "prefill qkv special view");
            check(qkv.output_row0 == 0, "prefill qkv output offset");
            check(tp_prefill_projection_ok(qkv), "prefill qkv accepted");

            const auto gdn_out = tp_projection_for("blk.0.ssm_out.weight", rank);
            check(gdn_out.split_output_rows, "prefill gdn out rows split");
            check(gdn_out.output_row0 == rank * 1280, "prefill gdn out offset");
            check(tp_prefill_projection_ok(gdn_out), "prefill gdn out accepted");

            const auto attn_out = tp_projection_for("blk.3.attn_output.weight", rank);
            check(attn_out.split_output_rows, "prefill attn out rows split");
            check(attn_out.output_row0 == rank * 1280, "prefill attn out offset");
            check(tp_prefill_projection_ok(attn_out), "prefill attn out accepted");
        }
        strata::core::TpProjection bad;
        bad.rank = 0;
        bad.split.mode = TpMode::Row;
        bad.split.full_n_in = 6144;
        bad.split.full_n_out = 2560;
        bad.split.n_in = 3072;
        bad.split.n_out = 2560;
        std::string why;
        check(!tp_prefill_projection_ok(bad, &why), "prefill reduction split refused");
        check(!why.empty(), "prefill reduction split reason");
    }
    // Non-dense names replicate (router, shared expert, norms, hc, ple, output_hc).
    {
        const char* names[] = {
            "blk.0.ffn_gate_inp.weight", "blk.0.ffn_gate_shexp.weight",
            "blk.0.ssm_norm.weight", "blk.0.hc_attn_norm.weight",
            "blk.1.ple_key.weight", "output_hc_norm.weight", "token_embd.weight",
        };
        for (const char* n : names) {
            const TpSplit s = tp_split_for(n, 0);
            check(s.mode == TpMode::Replicate, (std::string(n) + " replicate").c_str());
            check(tp_q8_0_bytes(n, 0) == 0, (std::string(n) + " bytes 0").c_str());
            check(tp_prefill_projection_ok(tp_projection_for(n, 0)),
                  (std::string(n) + " prefill replicate").c_str());
        }
    }
    if (g_fail == 0) {
        std::printf("ALL PASS: tp_split_plan\n");
        return 0;
    }
    std::fprintf(stderr, "%d FAILURES\n", g_fail);
    return 1;
}
