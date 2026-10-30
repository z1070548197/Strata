// tests/core/tp_moe_split_test.cpp - the experts' FFN split's byte arithmetic (tp_moe_split.hpp).
//
// What this pins down: the two ranks' layouts must address EXACTLY the same packed bytes the single-card
// layout would read for that rank's half, and the two halves must be disjoint and cover the whole blob.  A
// mistake here silently reads another expert's weights (or the previous row's tail), which no numeric test of
// the window would catch as anything but "the answer is wrong".
#include "strata/core/tp_blob_ranges.hpp"
#include "strata/core/tp_moe_split.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

using strata::kernels::NativeExpertLayout;
using strata::kernels::iq_row_bytes;
using strata::kernels::native_expert_layout;

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++failures;
    }
}

// The blob's byte span for an expert's projection row (gate/up) or down row, in the SINGLE-CARD layout.
size_t full_gu_row_off(const NativeExpertLayout& L, bool up, int64_t r) {
    return (up ? L.up_off : 0) + (size_t) r * L.gu_row;
}
size_t full_down_row_off(const NativeExpertLayout& L, int64_t r) { return L.down_off + (size_t) r * L.d_row; }

// The same, addressed through a split rank's (layout, ptr_off).
size_t half_gu_row_off(const NativeExpertLayout& L, size_t ptr_off, bool up, int64_t r) {
    return ptr_off + (up ? L.up_off : 0) + (size_t) r * L.gu_row;
}
size_t half_down_row_off(const NativeExpertLayout& L, size_t ptr_off, int64_t r) {
    return ptr_off + L.down_off + (size_t) r * L.d_row;
}

void one_case(int gu_type, int d_type, int64_t n_embd, int64_t n_ff, const char* name) {
    const NativeExpertLayout full = native_expert_layout(gu_type, d_type, n_embd, n_ff);
    std::string why;
    if (!strata::core::tp_moe_split_ok(gu_type, d_type, n_embd, n_ff, why)) {
        std::printf("  %-22s n_ff=%lld: refused (%s)\n", name, (long long) n_ff, why.c_str());
        return;   // a refusal is a valid answer for a geometry that cannot split
    }
    strata::core::TpMoeSplit s[2];
    for (int r = 0; r < 2; ++r)
        check(strata::core::tp_moe_split(gu_type, d_type, n_embd, n_ff, r, s[r], why),
              std::string(name) + ": split of rank " + std::to_string(r) + " (" + why + ")");
    const int64_t half = n_ff / 2;
    check(s[0].layout.n_ff == half && s[1].layout.n_ff == half, std::string(name) + ": local n_ff is n_ff/2");
    check(s[0].ptr_off == 0, std::string(name) + ": rank 0 keeps the blob's base");
    check(s[1].ptr_off == (size_t) half * full.gu_row, std::string(name) + ": rank 1's base is half the gate rows");
    // the packed strides are untouched: the rows still start where the pack put them
    check(s[0].layout.gu_row == full.gu_row && s[1].layout.gu_row == full.gu_row,
          std::string(name) + ": gu_row is the packed stride");
    check(s[0].layout.d_row == full.d_row && s[1].layout.d_row == full.d_row,
          std::string(name) + ": d_row is the packed stride");

    // gate/up: rank r's row i must be the single-card row r*half + i, byte for byte
    for (int r = 0; r < 2; ++r)
        for (int64_t i = 0; i < half; ++i) {
            const int64_t full_r = r * half + i;
            for (int up = 0; up <= 1; ++up) {
                const size_t got = half_gu_row_off(s[r].layout, s[r].ptr_off, up != 0, i);
                const size_t want = full_gu_row_off(full, up != 0, full_r);
                check(got == want, std::string(name) + ": gate/up r" + std::to_string(r) + " i" + std::to_string(i) +
                                       " up" + std::to_string(up) + " offset " + std::to_string(got) + " != " +
                                       std::to_string(want));
            }
        }
        const int64_t half_out = n_embd / 2;
        // down: an OUTPUT-ROW split.  Each rank reads the UNshifted group pointers (down_ptr_off is 0) and its row
        // offset lives in `down.down_off`; local row i is the single-card row r*half_out + i, byte for byte, with
        // the FULL n_ff reduction (the reduction axis is NOT split).  `down.n_ff` is the full width and `down.n_embd`
        // the local output width.
        for (int r = 0; r < 2; ++r) {
            check(s[r].down_ptr_off == 0, std::string(name) + ": down r" + std::to_string(r) + " uses the unshifted pointers");
            check(s[r].down.n_ff == n_ff, std::string(name) + ": down r" + std::to_string(r) + " reduces the full n_ff");
            check(s[r].down.n_embd == half_out, std::string(name) + ": down r" + std::to_string(r) + " has half the output rows");
            for (int64_t i = 0; i < half_out; ++i) {
                const size_t got = s[r].down.down_off + (size_t) i * full.d_row;
                const size_t want = full_down_row_off(full, (int64_t) r * half_out + i);
                check(got == want, std::string(name) + ": down r" + std::to_string(r) + " row" + std::to_string(i) +
                                       " offset " + std::to_string(got) + " != " + std::to_string(want));
            }
        }
    // the NVFP4 tail must still be reachable through each rank's gate/up pointer (the down reads it from the
    // unshifted blob's tail)
    if (full.tail_off != 0)
        for (int r = 0; r < 2; ++r) {
            check(s[r].ptr_off + s[r].layout.tail_off == full.tail_off,
                  std::string(name) + ": rank " + std::to_string(r) + "'s gate/up tail offset");
            check(s[r].down.tail_off == full.tail_off,
                  std::string(name) + ": rank " + std::to_string(r) + "'s down tail offset");
        }
    // the two ranks' down rows are disjoint and cover the whole blob's down
    for (int r = 0; r < 2; ++r) {
        const size_t end = s[r].down.down_off + (size_t) (half_out - 1) * full.d_row + iq_row_bytes(d_type, n_ff);
        check(end <= full.bytes, std::string(name) + ": rank " + std::to_string(r) + "'s last down row ends at " +
                                     std::to_string(end) + ", past the blob's " + std::to_string(full.bytes));
    }
    // The half-fetch's slot fill (fetch_blobs_spans dst2_mode 2) has both ranks write THEIR windows into one cache
    // slot, so those windows must TILE the blob: a hole would leave stale bytes that a later hit reads as expert
    // weights.  Overlaps are harmless - every window copies source -> the same offsets (the NVFP4 tail is read by
    // both ranks, at the same offsets).
    {
        strata::core::TpBlobRange w[2][4];
        int nw[2] = {0, 0};
        bool have = true;
        for (int r = 0; r < 2 && have; ++r) {
            nw[r] = strata::core::tp_blob_ranges(full.up_off, full.tail_off, (size_t) full.bytes, s[r], w[r]);
            have = nw[r] > 0;
        }
        if (have) {
            std::vector<std::pair<size_t, size_t>> iv;
            for (int r = 0; r < 2; ++r)
                for (int j = 0; j < nw[r]; ++j) iv.push_back({w[r][j].off, w[r][j].off + w[r][j].bytes});
            std::sort(iv.begin(), iv.end());
            size_t reached = 0;
            bool tiled = true;
            for (const auto& q : iv) {
                if (q.first > reached) { tiled = false; break; }
                if (q.second > reached) reached = q.second;
            }
            check(tiled && reached == (size_t) full.bytes,
                  std::string(name) + ": the two ranks' read windows tile the blob (dst2_mode 2 needs no holes)");
        }
    }
    std::printf("  %-22s n_ff=%lld -> gu %lld+%lld rows, down %lld+%lld rows (full n_ff)\n", name, (long long) n_ff,
                (long long) half, (long long) half, (long long) half_out, (long long) half_out);
}

}  // namespace

int main() {
    std::printf("tp_moe_split: the experts' FFN split's byte arithmetic\n");
    // the pack this engine serves: NVFP4 gate/up and NVFP4 down, n_embd 2560, n_ff 640
    one_case(40, 40, 2560, 640, "NVFP4/NVFP4 2560x640");
    // the same geometry in the other formats the engine's packs use
    one_case(21, 20, 2560, 640, "IQ3_S/IQ4_NL 2560x640");
    one_case(20, 20, 2560, 640, "IQ4_NL/IQ4_NL 2560x640");
    one_case(42, 42, 2560, 640, "Q2_0/Q2_0 2560x640");
    one_case(40, 40, 2560, 512, "NVFP4/NVFP4 2560x512");
    // a geometry whose half is NOT a whole number of NVFP4 blocks must be refused, not rounded
    {
        std::string why;
        check(!strata::core::tp_moe_split_ok(40, 40, 2560, 600, why), "n_ff 600 is refused for NVFP4");
        std::printf("  NVFP4 n_ff=600 refused: %s\n", why.c_str());
    }
    if (failures == 0) std::printf("tp_moe_split: ALL PASS\n");
    return failures == 0 ? 0 : 1;
}
