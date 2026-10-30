// src/core/tp_moe_split.cpp - see include/strata/core/tp_moe_split.hpp.
#include "strata/core/tp_moe_split.hpp"

namespace strata::core {

using strata::kernels::NativeExpertLayout;
using strata::kernels::iq_row_bytes;

namespace {
// The block size of each quantized type, in elements: a rank's half must be a whole number of these.
int64_t block_elems(int type) {
    switch (type) {
        case 40: return 64;   // NVFP4: QK_NVFP4 = 64, 36 bytes (4 scale bytes + 32 packed nibbles)
        case 42: return 64;   // Q2_0: QK = 64
        case 20: return 32;   // IQ4_NL
        case 8: return 32;    // Q8_0
        case 7: return 32;    // Q5_1
        case 6: return 32;    // Q5_0
        case 2: return 32;    // Q4_0
        case 3: return 32;    // Q4_1
        case 16: case 17: case 18: case 21: case 22: case 23: case 29: case 11: case 12: case 13: case 14:
            return 256;       // the K-quants and the IQ grid formats
        default: return 0;
    }
}
}  // namespace

bool tp_moe_split_ok(int gu_type, int d_type, int64_t n_embd, int64_t n_ff, std::string& why) {
    if (n_ff <= 0 || (n_ff % 2) != 0) {
        why = "n_ff " + std::to_string((long long) n_ff) + " is not even";
        return false;
    }
    if (n_embd <= 0 || (n_embd % 2) != 0) {
        why = "n_embd " + std::to_string((long long) n_embd) + " is not even";
        return false;
    }
    const int64_t half = n_ff / 2;
    const int64_t half_out = n_embd / 2;
    const int64_t gu_block = block_elems(gu_type);
    const int64_t d_block = block_elems(d_type);
    if (gu_block == 0 || d_block == 0) {
        why = "expert type " + std::to_string(gu_type) + "/" + std::to_string(d_type) + " has no known block size";
        return false;
    }
    // gate/up: rows are n_embd wide, so the row COUNT (half of n_ff) is what changes.  A whole number of gate/up
    // blocks per half row and a multiple of 32 (native_expert_grouped's launch) are required.
    if ((half % gu_block) != 0) {
        why = "gate/up: n_ff/2 = " + std::to_string((long long) half) + " is not a multiple of the gate/up block (" +
              std::to_string((long long) gu_block) + ")";
        return false;
    }
    if ((half % 32) != 0) {
        why = "n_ff/2 = " + std::to_string((long long) half) + " is not a multiple of 32 (native_expert_grouped)";
        return false;
    }
    // down: the split is by OUTPUT ROW (n_embd/2 of n_embd), each row reducing the FULL n_ff terms.  The row
    // offset is a whole number of packed down rows, so no column arithmetic is needed and d_row stays unchanged.
    if ((half_out % 32) != 0) {
        why = "down: n_embd/2 = " + std::to_string((long long) half_out) + " is not a multiple of 32";
        return false;
    }
    return true;
}

bool tp_moe_split(int gu_type, int d_type, int64_t n_embd, int64_t n_ff, int rank, TpMoeSplit& out,
                  std::string& why) {
    out = TpMoeSplit{};
    if (rank != 0 && rank != 1) {
        why = "rank " + std::to_string(rank) + " is not 0 or 1";
        return false;
    }
    if (!tp_moe_split_ok(gu_type, d_type, n_embd, n_ff, why)) return false;
    const NativeExpertLayout full = strata::kernels::native_expert_layout(gu_type, d_type, n_embd, n_ff);
    const int64_t half = n_ff / 2;         // the local intermediate width (gate/up rows)
    const int64_t half_out = n_embd / 2;   // the local output width (down rows)
    const size_t gu_row = full.gu_row;     // iq_row_bytes(gu_type, n_embd) - packed stride, unchanged
    const size_t d_row = full.d_row;       // iq_row_bytes(d_type, n_ff) - packed stride, unchanged

    // gate/up: half the rows, same packed stride.  The call uses the SHIFTED group pointers (blob base moved by
    // ptr_off = rank*half*gu_row).  The gate rows read [0, half*gu_row) from the shifted base -> blob rows
    // [rank*half, (rank+1)*half).  The up rows read [up_off, up_off + half*gu_row) from the shifted base; with
    // the FULL up_off (= n_ff*gu_row) that is blob rows [n_ff + rank*half, n_ff + (rank+1)*half) - rank r's own
    // up half.  So up_off stays the full width's value (NOT re-based), and only the down offset is re-based from
    // the shifted base (unused by the gu call, kept consistent).  The NVFP4 scales are read from the UNSHIFTED
    // group pointer (blob + L.tail_off), so the gu call's tail_off is re-based by ptr_off to still reach the
    // blob's tail.
    out.ptr_off = (size_t) rank * (size_t) half * gu_row;
    out.layout = full;
    out.layout.n_ff = half;
    out.layout.down_off = full.down_off - out.ptr_off;   // not used by the gate/up call, kept consistent
    out.layout.tail_off = full.tail_off != 0 ? full.tail_off - out.ptr_off : 0;

    // down: half the OUTPUT rows, full n_ff reduction.  The row offset goes into down_off (NOT a group-pointer
    // shift), so the kernel's `grp_ptr + down_off + r*d_row` (grp_ptr unshifted, r in [0, n_embd/2)) lands on
    // the rank's own rows [rank*n_embd/2, (rank+1)*n_embd/2) of the full down.  s_down (tail) is read from the
    // unshifted group pointer too, so tail_off is the blob's.
    out.down_ptr_off = 0;
    out.down = full;
    out.down.n_embd = half_out;
    out.down.n_ff = n_ff;                                     // the FULL width (the reduction is not split)
    out.down.down_off = full.down_off + (size_t) rank * (size_t) half_out * d_row;
    out.down.tail_off = full.tail_off;

    out.n_ff_local = half;
    out.n_out_local = half_out;
    return true;
}

size_t tp_moe_split_scratch_bytes(int64_t cap_entries, const TpMoeSplit& s) {
    // The gate/up call runs at the LOCAL width (n_ff/2): gate|up|h stride cap*(n_ff/2) floats and hq holds
    // cap*(n_ff/2/32) blocks.  The down's full-width hq is a SEPARATE buffer (tp_moe_hq_), not part of this
    // scratch.
    return strata::kernels::native_expert_scratch_bytes(cap_entries, s.n_ff_local);
}

}  // namespace strata::core
