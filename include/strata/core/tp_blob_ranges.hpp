// include/strata/core/tp_blob_ranges.hpp - the windows of one expert blob that a TP rank actually reads.
//
// Under `--tp` each rank's kernels address only its own rows of an expert blob (see tp_moe_split.hpp): gate at
// `ptr_off`, up at `up_off + ptr_off`, down at `down_ptr_off + down.down_off`, and the NVFP4 tail.  The other
// rank's rows are never touched.  So when a blob has to cross a PCIe link - the prompt path's streamed experts,
// or decode's per-layer PCIe share - copying the whole blob pays twice for every byte: each rank needs half.
// These are the windows for that copy, at the SAME offsets the kernels address, so a destination that keeps the
// blob's own stride needs no change to any kernel pointer.
//
// ONE definition, used by both moving paths (prefill's stage_copy_tp and verify's fetch_blobs_spans): a window
// that disagrees with the kernel's addressing is silent corruption (it copies the WRONG rows, which still look
// like expert weights) and it has bitten already - the down window needs `down_ptr_off + down.down_off`, not
// `down_off + down_ptr_off`, because the rank's down rows are part of the LAYOUT's down_off, not of the pointer.
//
// A second invariant rides on the same numbers: the two ranks' windows TILE the blob (no holes), which is what
// lets decode's cache slots be filled by both ranks at once (fetch_blobs_spans dst2_mode 2).  Only the NVFP4
// tail is read - and written - by both ranks, at identical offsets, so the overlap is byte-identical.
#pragma once

#include "strata/core/tp_moe_split.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"

#include <cstddef>

namespace strata::core {

struct TpBlobRange {
    size_t off = 0, bytes = 0;
};

/// The windows of one expert blob that rank r READS under --tp.  Returns 0 when the geometry cannot be expressed
/// as windows inside the blob; the caller then copies it whole.
inline int tp_blob_ranges(size_t up_off, size_t tail_off, size_t blob_bytes, const TpMoeSplit& tp,
                          TpBlobRange* out) {
    if (tp.layout.gu_row == 0 || tp.down.d_row == 0 || blob_bytes == 0 || tp.layout.n_ff <= 0 ||
        tp.down.n_embd <= 0)
        return 0;
    int n = 0;
    const size_t gu_bytes = (size_t) tp.layout.n_ff * tp.layout.gu_row;
    out[n++] = TpBlobRange{tp.ptr_off, gu_bytes};
    out[n++] = TpBlobRange{up_off + tp.ptr_off, gu_bytes};
    out[n++] = TpBlobRange{tp.down_ptr_off + tp.down.down_off, (size_t) tp.down.n_embd * tp.down.d_row};
    if (tail_off != 0 && tail_off < blob_bytes) out[n++] = TpBlobRange{tail_off, blob_bytes - tail_off};
    size_t total = 0;
    for (int i = 0; i < n; ++i) {
        if (out[i].bytes == 0 || out[i].off > blob_bytes || out[i].bytes > blob_bytes - out[i].off) return 0;
        total += out[i].bytes;
    }
    // A geometry whose windows add up to the whole blob means the split did not apply; keep the plain copy.
    return total < blob_bytes ? n : 0;
}

/// tp_blob_ranges for one layer of the pack; 0 (copy the whole blob) in every non-TP mode.
inline int tp_blob_ranges_for_layer(const strata::kernels::cpu::ExpertLayout& lay, int64_t l, const TpMoeSplit& tp,
                                    TpBlobRange* out) {
    return tp.n_ff_local > 0 && lay.native && !lay.fmt.empty()
               ? tp_blob_ranges(lay.fmt[(size_t) l].up_off, lay.fmt[(size_t) l].tail_off, (size_t) lay.blob_bytes(l),
                                tp, out)
               : 0;
}

}  // namespace strata::core
