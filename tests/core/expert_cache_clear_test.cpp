// TP_RESULTS §8.4: the phase switch.  A layer-split phase and a TP phase share one cache because a slot holds a
// FULL expert blob (the TP MoE paths read their half by pointer offset) - the only thing that differs is which
// (layer, expert) pairs are admitted.  `admit` never evicts, so the switch needs `clear_residency`: the table
// empties, every `slot_of` answers kNotResident, admission restarts at the bottom of each layer's range, and the
// ARENA keeps its address (the captured graphs and the prompt path's loan hold pointers into it).
// Needs a CUDA device (exits 77 without one).
#include "strata/core/expert_cache.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <string>
#include <vector>

using strata::core::ExpertCache;
using strata::core::kNotResident;

namespace {

bool check(bool ok, const char* what, const std::string& detail) {
    if (ok) return true;
    std::fprintf(stderr, "expert cache clear: %s: %s\n", what, detail.c_str());
    return false;
}

/// One configuration: uniform or sized arena, per-layer admission on or off.
bool run(bool sized, bool per_layer) {
    std::string err;
    ExpertCache cache;
    const bool opened = sized ? cache.open_sized(std::vector<int64_t>(6, 1024), 3, 4, err)
                             : cache.open(6, 3, 4, 1024, err);
    if (!opened) {
        std::fprintf(stderr, "open: %s\n", err.c_str());
        return false;
    }
    cache.set_per_layer_admission(per_layer);

    const int64_t n_layers = 3, n_expert = 4;
    // Admit everything that fits; remember the assignment so the second pass can be compared to it.
    std::vector<int32_t> first((size_t) (n_layers * n_expert), kNotResident);
    for (int64_t l = 0; l < n_layers; ++l)
        for (int64_t e = 0; e < n_expert; ++e) first[(size_t) (l * n_expert + e)] = cache.admit(l, e);
    const int64_t resident_before = cache.resident();
    if (!check(resident_before > 0, "nothing was admitted", "the test proves nothing")) return false;
    const uint8_t* slot0 = cache.device_slot(0);
    const int64_t bytes = cache.bytes();

    cache.clear_residency();

    if (!check(cache.resident() == 0, "resident() after clear",
               std::to_string(cache.resident()) + " slots still claimed"))
        return false;
    for (int64_t l = 0; l < n_layers; ++l)
        for (int64_t e = 0; e < n_expert; ++e)
            if (!check(cache.slot_of(l, e) == kNotResident, "slot_of after clear",
                       "layer " + std::to_string(l) + " expert " + std::to_string(e) + " still maps to slot " +
                           std::to_string(cache.slot_of(l, e))))
                return false;
    // The arena is the same memory: the graphs and the loan's pointers stay valid across the switch.
    if (!check(cache.device_slot(0) == slot0, "the arena moved", "the phase switch must not reallocate"))
        return false;
    if (!check(cache.bytes() == bytes, "the arena changed size", "the phase switch must not resize"))
        return false;

    // A second pass assigns exactly what the first did, which is only true when the per-layer cursors went back to
    // each range's bottom (layer 1 to q, not to 0) and not merely to zero.
    for (int64_t l = 0; l < n_layers; ++l)
        for (int64_t e = 0; e < n_expert; ++e) {
            const int32_t slot = cache.admit(l, e);
            if (!check(slot == first[(size_t) (l * n_expert + e)], "the second admission differs",
                       "layer " + std::to_string(l) + " expert " + std::to_string(e) + ": " +
                           std::to_string(slot) + " vs " + std::to_string(first[(size_t) (l * n_expert + e)])))
                return false;
        }
    if (!check(cache.resident() == resident_before, "residency after re-admission",
               std::to_string(cache.resident()) + " vs " + std::to_string(resident_before)))
        return false;
    return true;
}

}  // namespace

int main() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) {
        std::puts("no CUDA device: skipped");
        return 77;
    }
    for (int sized = 0; sized < 2; ++sized)
        for (int per_layer = 0; per_layer < 2; ++per_layer)
            if (!run(sized != 0, per_layer != 0)) return 1;
    std::puts("expert cache clear_residency: uniform/sized x global/per-layer passed");
    return 0;
}
