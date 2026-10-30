// include/strata/core/kernel_switches.hpp - §8.3 step 3d: the kernel-selection switches that change NUMERICS,
// as values a running engine can change.
//
// A phase plan that serves short and long prompts from ONE process needs this: the block scorer that wins at 160K
// (`STRATA_SELECT_SIMT=2`, §5.4-2: the select bucket 39,980 -> 7,244 ms, +19% on its own) costs ~5% at 8K, where
// the select work is too small to pay for FP32 tiles.  Both levels are correct (`=1|2` pick a different summation
// ORDER, not different math), so the choice is a per-REQUEST one - prefill and that request's decode together, so
// no request mixes orders.
//
// The value is seeded from the environment exactly as the kernel used to read it in a `static const`, so a process
// that never calls the setter behaves as before.  A reader takes one load per kernel launch.
#pragma once

#include <atomic>
#include <cstdlib>

namespace strata {

namespace detail {

inline int select_simt_from_env() {
    const char* v = std::getenv("STRATA_SELECT_SIMT");
    return v == nullptr ? 0 : v[0] == '1' ? 1 : v[0] == '2' ? 2 : 0;
}

inline std::atomic<int>& select_simt_cell() {
    static std::atomic<int> cell{select_simt_from_env()};
    return cell;
}

}  // namespace detail

/// The block scorer the next `qsa_select` launch takes: 0 the warp kernel, 1 the FP32 tiled one, 2 its 64x64 twin.
inline int select_simt() { return detail::select_simt_cell().load(std::memory_order_relaxed); }

/// `level` clamped to 0..2.  The next launch reads it; graphs captured before the change keep the old choice until
/// they are dropped (`Verifier::drop_solo_graphs`).
inline void set_select_simt(int level) {
    detail::select_simt_cell().store(level < 0 ? 0 : level > 2 ? 2 : level, std::memory_order_relaxed);
}

/// The level this process started with (its environment): what a request that does not move asks for.
inline int select_simt_env() {
    static const int env = detail::select_simt_from_env();
    return env;
}

}  // namespace strata
