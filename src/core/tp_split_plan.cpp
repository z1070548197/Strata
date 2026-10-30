#include "strata/core/tp_split_plan.hpp"

#include <cctype>
#include <cstdlib>
#include <cstring>

namespace strata::core {

namespace {

// The model's dense geometry (authoritative, from the pack index).  These are
// the unsplit dimensions; the per-rank halves are derived from them.
constexpr int kN = 2560;          // n_embd
// GDN
constexpr int kQkvNin = 2560;     // attn_qkv n_in
constexpr int kQkvNout = 10240;   // attn_qkv n_out = q(2048) + k(2048) + v(6144)
constexpr int kQkDim = 2048;      // q and k each (HK * S = 16 * 128)
constexpr int kVDim = 6144;       // v (HV * S = 48 * 128)
constexpr int kGateNin = 2560;    // attn_gate n_in
constexpr int kGateNout = 6144;   // attn_gate n_out
constexpr int kOutNin = 6144;     // ssm_out n_in
constexpr int kOutNout = 2560;    // ssm_out n_out
// QSA
constexpr int kQNin = 2560;       // attn_q n_in
constexpr int kQNout = 12288;     // attn_q n_out = NH * 2 * HD (48 * 2 * 128)
constexpr int kKvNin = 2560;      // attn_k / attn_v n_in
constexpr int kKvNout = 512;      // attn_k / attn_v n_out = NKV * HD (4 * 128)
constexpr int kAttnOutNin = 6144; // attn_output n_in = NH * HD
constexpr int kAttnOutNout = 2560;// attn_output n_out = N

// "blk.<l>.<suffix>" -> the suffix after the second dot.  Returns false when
// the name is not a blk layer tensor.
bool layer_suffix(const std::string& name, std::string& suffix) {
    if (name.rfind("blk.", 0) != 0) return false;
    const size_t d1 = name.find('.', 4);
    if (d1 == std::string::npos) return false;
    suffix = name.substr(d1 + 1);
    return true;
}

} // namespace

TpSplit tp_split_for(const std::string& name, int rank) {
    (void) rank;   // the per-rank half is symmetric (first/second), resolved by the caller
    std::string suffix;
    if (!layer_suffix(name, suffix)) return TpSplit{};   // not a dense projection: replicate

    // ---- GDN layers ----
    if (suffix == "attn_qkv.weight") {
        // q/k replicated (both cards hold all q and k heads), v split by head.
        // Per-rank n_out = q + k + v_half = 2048 + 2048 + 3072 = 7168.
        TpSplit s;
        s.mode = TpMode::Qkv;
        s.full_n_in = kQkvNin;
        s.full_n_out = kQkvNout;
        s.n_in = kQkvNin;
        s.n_out = kQkDim + kQkDim + kVDim / 2;
        return s;
    }
    if (suffix == "attn_gate.weight") {
        // column-parallel: split the 6144 output rows, no allreduce.
        TpSplit s;
        s.mode = TpMode::Col;
        s.full_n_in = kGateNin;
        s.full_n_out = kGateNout;
        s.n_in = kGateNin;
        s.n_out = kGateNout / 2;
        return s;
    }
    if (suffix == "ssm_out.weight") {
        // STRATA_TP_GDN_FULL=1 (EXPERIMENTAL, default off): REPLICATE the GDN out-projection for rank 0.
        // =0 (default): COLUMN-parallel - each card holds half the OUTPUT ROWS and reduces all 6144 terms.
        //
        // The split axis is the whole argument.  Splitting n_in (the old TpMode::Row default) makes each card
        // sum 3072 of the 6144 terms and the allreduce add the two partials: (a1+a2)+b is not a1+(a2+b) in
        // float32, so the result differs from the single card by ULP-level noise that a later q8_1 bucket
        // boundary amplifies into a wrong token.  Splitting n_out leaves every output row's 6144-term
        // reduction to ONE card with the same kernel and the same block layout the single card uses, so every
        // element is bit-identical (tests/core/tp_outproj_split_test.cu asserts this on the real kernels), and
        // the two cards' outputs are disjoint - no allreduce for this projection at all.
        static const bool gdn_full = [] {
            const char* v = std::getenv("STRATA_TP_GDN_FULL");
            return v != nullptr && std::atoi(v) != 0;   // opt-in diagnostic: full width on rank 0
        }();
        TpSplit s;
        s.full_n_in = kOutNin;
        s.full_n_out = kOutNout;
        if (gdn_full) {
            s.mode = TpMode::Replicate;
            s.n_in = kOutNin;
            s.n_out = kOutNout;
            return s;
        }
        // column-parallel: half the output rows, the full 6144-term reduction on each card.
        s.mode = TpMode::Col;
        s.n_in = kOutNin;
        s.n_out = kOutNout / 2;
        return s;
    }

    // ---- QSA layers ----
    if (suffix == "attn_q.weight") {
        // Replicated: the attention runs FULL on both cards (the attention kernel's KV-head group is a
        // compile-time constant, so it cannot be given a query-head subset), so each card needs all 12288
        // q|gate rows.  The split is in the out-projection's reduction axis (attn_output below).
        TpSplit s;
        s.mode = TpMode::Replicate;
        s.full_n_in = kQNin;
        s.full_n_out = kQNout;
        s.n_in = kQNin;
        s.n_out = kQNout;
        return s;
    }
    if (suffix == "attn_k.weight" || suffix == "attn_v.weight") {
        // K/V replicated (both cards hold the full K and V, so the KV cache and
        // the attention over it are identical on both cards).
        TpSplit s;
        s.mode = TpMode::Replicate;
        s.full_n_in = kKvNin;
        s.full_n_out = kKvNout;
        s.n_in = kKvNin;
        s.n_out = kKvNout;
        return s;
    }
    if (suffix == "attn_output.weight") {
        // STRATA_TP_ATTN_FULL=1 (EXPERIMENTAL, default off): REPLICATE the out-projection for rank 0.
        // =0 (default): COLUMN-parallel, the same reasoning as ssm_out above.  Each card holds half the
        // output rows (N/2 each) and reduces the full NH*HD terms, so every output element is bit-identical
        // to the single card's and the two cards' outputs are disjoint.
        static const bool attn_full = [] {
            const char* v = std::getenv("STRATA_TP_ATTN_FULL");
            return v != nullptr && std::atoi(v) != 0;   // opt-in diagnostic: full width on rank 0
        }();
        TpSplit s;
        s.full_n_in = kAttnOutNin;
        s.full_n_out = kAttnOutNout;
        if (attn_full) {
            s.mode = TpMode::Replicate;
            s.n_in = kAttnOutNin;
            s.n_out = kAttnOutNout;
            return s;
        }
        // column-parallel: half the output rows, the full reduction on each card.
        s.mode = TpMode::Col;
        s.n_in = kAttnOutNin;
        s.n_out = kAttnOutNout / 2;
        return s;
    }

    // Everything else (router, shared expert, norms, ssm_alpha/beta, ple_*,
    // output_hc_*, ...) is replicated.
    return TpSplit{};
}

TpProjection tp_projection_for(const std::string& name, int rank) {
    TpProjection projection;
    projection.rank = rank == 0 ? 0 : 1;
    projection.split = tp_split_for(name, projection.rank);
    projection.replicated = projection.split.mode == TpMode::Replicate;
    projection.split_output_rows = projection.split.mode == TpMode::Col;
    projection.special_qkv = projection.split.mode == TpMode::Qkv;
    if (projection.split_output_rows) {
        projection.output_row0 = projection.rank * projection.split.n_out;
    }
    return projection;
}

bool tp_prefill_projection_ok(const TpProjection& projection, std::string* why) {
    const auto refuse = [why](const char* reason) {
        if (why != nullptr) *why = reason;
        return false;
    };
    const TpSplit& split = projection.split;
    if (projection.rank != 0 && projection.rank != 1) return refuse("rank must be 0 or 1");
    if (split.mode == TpMode::Row) return refuse("prefill cannot split the reduction axis");
    if (split.full_n_in < 0 || split.full_n_out < 0 || split.n_in < 0 || split.n_out < 0) {
        return refuse("projection dimensions must be non-negative");
    }
    if (split.mode == TpMode::Replicate) return true;
    if (split.full_n_in == 0 || split.full_n_out == 0 || split.n_in == 0 || split.n_out == 0) {
        return refuse("split projection has an empty dimension");
    }
    if (split.n_in != split.full_n_in) return refuse("prefill split must preserve the full reduction axis");
    if (split.mode == TpMode::Col) {
        if (split.n_out * 2 != split.full_n_out) return refuse("output rows must split evenly");
        if (projection.output_row0 != projection.rank * split.n_out) return refuse("wrong output row offset");
        return true;
    }
    if (split.mode == TpMode::Qkv) {
        if (!projection.special_qkv || projection.output_row0 != 0) return refuse("invalid GDN QKV view");
        return true;
    }
    return refuse("unknown TP projection mode");
}

bool tp_peer_rows_for(const std::string& name, int64_t n_out, TpPeerRows& out, std::string* why) {
    const auto refuse = [why](const char* r) { if (why != nullptr) *why = r; return false; };
    const TpSplit s = tp_split_for(name, /*rank=*/0);
    if (s.mode == TpMode::Replicate) return refuse("not a split dense projection");
    if (s.mode == TpMode::Row) return refuse("row-parallel: the two halves are partial sums, not row ranges");
    if (n_out != (int64_t) s.full_n_out) return refuse("the caller's row count is not the full matrix's");
    TpPeerRows r;
    if (s.mode == TpMode::Qkv) {
        // The peer's upload is [q; k; v_half] with q and k replicated, so only its v_half is new to us: it is the
        // full matrix's trailing rows, and it starts right after the q and k the peer also carries.
        const int64_t v_half = (int64_t) s.full_n_out - (int64_t) kTpGdnConvChannels;
        r.src_row0 = (int64_t) kTpGdnConvChannels - v_half;
        r.n_own = (int64_t) kTpGdnConvChannels;
        r.rows = v_half;
    } else {
        r.src_row0 = 0;
        r.n_own = (int64_t) s.n_out;   // the leading half of the rows
        r.rows = (int64_t) s.full_n_out - r.n_own;
    }
    if (r.n_own + r.rows != n_out || r.rows <= 0) return refuse("the split plan does not cover the full matrix");
    out = r;
    return true;
}

uint64_t tp_q8_0_bytes(const std::string& name, int rank) {
    const TpSplit s = tp_split_for(name, rank);
    if (s.mode == TpMode::Replicate) return 0;   // not a split dense Q8_0 tensor
    // Q8_0: 34 bytes per 32-element block, n_in the contiguous axis.
    // bytes = (n_in / 32) * 34 * n_out.
    return (uint64_t) (s.n_in / 32) * 34u * (uint64_t) s.n_out;
}

size_t tp_split_q8_0_host(const uint8_t* full, TpMode mode, int n_in, int n_out,
                          int rank, uint8_t* half) {
    const int r = (rank != 0) ? 1 : 0;
    const size_t row_bytes = (size_t) (n_in / 32) * 34;   // one full row
    switch (mode) {
        case TpMode::Replicate:
            std::memcpy(half, full, (size_t) n_out * row_bytes);
            return (size_t) n_out * row_bytes;
        case TpMode::Col: {
            // A contiguous half of the rows (rank 0 = first half, rank 1 = second).
            const int half_rows = n_out / 2;
            const size_t off = (size_t) r * half_rows * row_bytes;
            std::memcpy(half, full + off, (size_t) half_rows * row_bytes);
            return (size_t) half_rows * row_bytes;
        }
        case TpMode::Row: {
            // Per row, half of the reduction blocks (rank 0 = first half, rank 1 =
            // second half).  n_in/2 must be a multiple of 32 (a Q8_0 block).
            const int half_blocks = (n_in / 32) / 2;
            const size_t half_row_bytes = (size_t) half_blocks * 34;
            for (int row = 0; row < n_out; ++row) {
                const uint8_t* src =
                    full + (size_t) row * row_bytes + (size_t) r * half_row_bytes;
                std::memcpy(half + (size_t) row * half_row_bytes, src, half_row_bytes);
            }
            return (size_t) n_out * half_row_bytes;
        }
        case TpMode::Qkv: {
            // q (full) + k (full) + v_half.  Full layout: q rows [0, qd), k rows
            // [qd, qd+kd), v rows [qd+kd, n_out).  v_half = first (rank 0) or last
            // (rank 1) half of the v rows.
            const int qd = kQkDim;                 // 2048
            const int kd = kQkDim;                 // 2048
            const int v_start = qd + kd;           // 4096
            const int v_half = kVDim / 2;          // 3072
            std::memcpy(half, full, (size_t) qd * row_bytes);
            std::memcpy(half + (size_t) qd * row_bytes,
                        full + (size_t) qd * row_bytes, (size_t) kd * row_bytes);
            const size_t v_src_off = (size_t) (v_start + r * v_half) * row_bytes;
            std::memcpy(half + (size_t) (qd + kd) * row_bytes,
                        full + v_src_off, (size_t) v_half * row_bytes);
            return (size_t) (qd + kd + v_half) * row_bytes;
        }
    }
    return 0;
}

} // namespace strata::core
