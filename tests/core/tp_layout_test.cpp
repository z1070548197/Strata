// tests/core/tp_layout_test.cpp - Phase 3b: the TP buffer/index layout that record_window_tp relies on.
//
// A pure-host test (no CUDA) of the index arithmetic that is easy to get wrong and impossible to see in a
// diff.  tp_conv_split_test.cu already proves the conv WEIGHTS are split bit-exactly; this proves the
// accompanying claims about WHICH channels and WHICH heads each rank owns, and that the two ranks' local
// buffers together cover the full tensor exactly once.
#include "strata/core/tp_split_plan.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

int g_fail = 0;
void ck(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++g_fail;
    }
}

// ---- the artifact's geometry
constexpr int S = 128;      // ssm_state_size (head dim)
constexpr int HK = 16;      // ssm_k_heads
constexpr int HV = 48;      // ssm_v_heads
constexpr int QK = HK * S;  // 2048
constexpr int VD = HV * S;  // 6144
constexpr int C = 2 * QK + VD;        // 10240
constexpr int CT = 2 * QK + VD / 2;   // 7168
constexpr int HVT = HV / 2;           // 24

// The global conv channel this rank's local channel `c` is.  This mirrors both tp_local_conv_kernel and
// record_window_tp's conv-history copy in the e2e test: q/k are replicated at [0, 4096); the v half follows.
int local_to_global(int rank, int c) {
    if (c < 2 * QK) return c;                                   // q + k: replicated
    return (2 * QK + rank * (VD / 2)) + (c - 2 * QK);           // this rank's v rows
}

// The global v-head of this rank's local head.
int local_to_global_head(int rank, int h) { return rank * HVT + h; }

int run() {
    std::printf("=== TP local/global index layout (S=%d HK=%d HV=%d) ===\n", S, HK, HV);

    // ---- claim 1: the two ranks' conv channels cover the full 10240 exactly once
    {
        std::vector<int> seen((size_t) C, 0);
        for (int r = 0; r < 2; ++r)
            for (int c = 0; c < CT; ++c) ++seen[(size_t) local_to_global(r, c)];
        int bad = 0, missing = 0;
        for (int g = 0; g < C; ++g) {
            if (seen[(size_t) g] == 0) ++missing;      // nothing may be lost
            if (seen[(size_t) g] > 2) ++bad;           // and nothing may be counted three times
        }
        std::printf("  conv channels: %d covered, %d missing, %d over-covered\n", C - missing, missing, bad);
        ck(missing == 0, "every conv channel is on at least one card");
        ck(bad == 0, "no conv channel is on more than the two cards");
        // exactly 2*QK channels replicated, the rest split
        int twice = 0;
        for (int g = 0; g < C; ++g) twice += seen[(size_t) g] == 2;
        std::printf("  replicated: %d (expected %d = q+k), split: %d (expected %d = v)\n", twice, 2 * QK, C - twice,
                    VD);
        ck(twice == 2 * QK, "exactly the q/k channels are replicated");
        // and each rank's first 2*QK local channels are the q/k ones
        for (int r = 0; r < 2; ++r)
            for (int c = 0; c < 2 * QK; ++c) ck(local_to_global(r, c) == c, "q/k are local [0, 4096) on both cards");
        // rank 0's v is the first half, rank 1's the second
        for (int c = 0; c < VD / 2; ++c) {
            ck(local_to_global(0, 2 * QK + c) == 2 * QK + c, "rank 0 owns the first v half");
            ck(local_to_global(1, 2 * QK + c) == 2 * QK + VD / 2 + c, "rank 1 owns the second v half");
        }
    }

    // ---- claim 2: the two ranks' v-heads cover all 48 exactly once
    {
        std::vector<int> seen((size_t) HV, 0);
        for (int r = 0; r < 2; ++r)
            for (int h = 0; h < HVT; ++h) ++seen[(size_t) local_to_global_head(r, h)];
        int missing = 0, twice = 0;
        for (int g = 0; g < HV; ++g) {
            if (seen[(size_t) g] == 0) ++missing;
            if (seen[(size_t) g] > 1) ++twice;
        }
        std::printf("  v-heads: %d missing, %d duplicated\n", missing, twice);
        ck(missing == 0, "every v-head is on a card");
        ck(twice == 0, "no v-head is on both cards (the recurrence is split, not replicated)");
    }

    // ---- claim 3: every v-head's q/k head is present on the same card
    // The recurrence pairs v-head h with q/k head (h % HK).  Because q/k are replicated, both cards have it.
    {
        for (int r = 0; r < 2; ++r)
            for (int h = 0; h < HVT; ++h) {
                const int gh = local_to_global_head(r, h);
                const int qh = gh % HK;
                ck(qh >= 0 && qh < HK, "the q/k head of every local v-head is in range");
                // and it is readable locally: the local conv buffer holds q/k at [qh*S, qh*S+S)
                ck(qh * S + S <= 2 * QK, "the q/k head's channels are inside the replicated region");
            }
        std::printf("  q/k pairing (h %% %d) satisfied for all %d local heads on both cards\n", HK, HVT);
    }

    // ---- claim 4: the local GDN state size and the recurrent/conv split
    {
        const int64_t per = (int64_t) S * HVT * S + (int64_t) CT * 3;
        const int64_t recur = (int64_t) S * HVT * S;
        std::printf("  local state: %lld floats = %lld recurrent (24 heads x 128 x 128) + %lld conv (7168 x 3)\n",
                    (long long) per, (long long) recur, (long long) CT * 3);
        ck(recur == (int64_t) 128 * 24 * 128, "local recurrent state is S * (HV/2) * S");
        ck(CT * 3 == 7168 * 3, "local conv history is 7168 channels x 3 taps");
        // the two cards' recurrent states together are exactly the full one
        ck(2 * recur == (int64_t) S * HV * S, "the two local recurrent states are the full one");
    }

    // ---- claim 5: the out-projection splits its OUTPUT ROWS, not its reduction
    // ssm_out is 6144 x 2560 (n_in = VD).  Splitting the reduction (n_in) would make each card sum half the
    // terms and the allreduce add the two partials - not the single card's sum in float32.  Splitting n_out
    // leaves every output row's full 6144-term reduction on one card, so every element is bit-identical.
    {
        const strata::core::TpSplit sp = strata::core::tp_split_for("blk.0.ssm_out.weight", 0);
        std::printf("  ssm_out per-rank n_in = %d, n_out = %d\n", sp.n_in, sp.n_out);
        ck(sp.mode == strata::core::TpMode::Col, "ssm_out is column-parallel (splits n_out)");
        ck(sp.n_in == 6144, "ssm_out keeps the full 6144-term reduction on each card");
        ck(sp.n_out == 1280, "ssm_out's per-rank n_out is half the output rows");
        ck(2 * sp.n_out == 2560, "the two cards' output rows are disjoint and cover all 2560");
        ck(!sp.allreduce(), "a column-parallel out-projection needs no allreduce (the outputs are disjoint)");
    }

    // ---- claim 6: the QSA out-projection splits its OUTPUT ROWS too
    // attn_output is 6144 x 2560 (n_in = 24 heads x 256).  Same argument as ssm_out: split n_out, not n_in.
    {
        constexpr int NH = 24, HD = 256;
        const strata::core::TpSplit sp = strata::core::tp_split_for("blk.3.attn_output.weight", 0);
        std::printf("  attn_output per-rank n_in = %d, n_out = %d\n", sp.n_in, sp.n_out);
        ck(sp.mode == strata::core::TpMode::Col, "attn_output is column-parallel (splits n_out)");
        ck(sp.n_in == NH * HD, "attn_output keeps the full NH*HD reduction on each card");
        ck(sp.n_out == 1280, "attn_output's per-rank n_out is half the output rows");
        ck(2 * sp.n_out == 2560, "the two cards' output rows are disjoint and cover all 2560");
        ck(!sp.allreduce(), "a column-parallel out-projection needs no allreduce");
        // and the q projection must be REPLICATED (the attention runs full on both cards)
        const strata::core::TpSplit q = strata::core::tp_split_for("blk.3.attn_q.weight", 0);
        ck(q.mode == strata::core::TpMode::Replicate, "attn_q is replicated");
        ck(strata::core::tp_q8_0_bytes("blk.3.attn_q.weight", 0) == 0, "a replicated tensor claims no half bytes");
    }

    // ---- claim 7: the qkv split's local row count is the conv channel count
    {
        const strata::core::TpSplit sp = strata::core::tp_split_for("blk.0.attn_qkv.weight", 0);
        std::printf("  attn_qkv per-rank n_out = %d (expected %d)\n", sp.n_out, CT);
        ck(sp.mode == strata::core::TpMode::Qkv, "attn_qkv uses the q/k-replicated mode");
        ck(sp.n_out == CT, "attn_qkv's per-rank rows are the local conv channels");
        ck(sp.n_in == 2560, "attn_qkv's reduction axis is untouched");
        ck(!sp.allreduce(), "attn_qkv is column-parallel: no allreduce");
        for (int r = 0; r < 2; ++r) {
            const strata::core::TpSplit s = strata::core::tp_split_for("blk.0.attn_qkv.weight", r);
            ck(s.n_out == CT, "both ranks produce the same local row count");
            ck(strata::core::tp_q8_0_bytes("blk.0.attn_qkv.weight", r) == (uint64_t) (2560 / 32) * 34u * CT,
               "the half's byte count matches its rows");
        }
    }

    // ---- claim 8: no dense projection needs an allreduce any more
    // The out-projections used to be the one allreduced pair (TpMode::Row).  Splitting their OUTPUT ROWS
    // instead makes the two cards' results disjoint, so nothing in the dense path reduces - which is also why
    // a per-layer handshake no longer sits between the projection and the residual.
    {
        ck(!strata::core::tp_split_for("blk.0.ssm_out.weight", 0).allreduce(), "ssm_out does not allreduce");
        ck(!strata::core::tp_split_for("blk.3.attn_output.weight", 0).allreduce(), "attn_output does not allreduce");
        ck(!strata::core::tp_split_for("blk.0.attn_qkv.weight", 0).allreduce(), "attn_qkv does not");
        ck(!strata::core::tp_split_for("blk.0.attn_gate.weight", 0).allreduce(), "attn_gate does not");
        ck(!strata::core::tp_split_for("blk.3.attn_k.weight", 0).allreduce(), "attn_k does not (replicated)");
        ck(!strata::core::tp_split_for("blk.0.ffn_gate_inp.weight", 0).allreduce(), "the router does not (replicated)");
    }

    if (g_fail == 0) std::printf("\nALL PASS: tp layout\n");
    else std::printf("\n%d FAILURES\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}

}  // namespace

int main() { return run(); }
