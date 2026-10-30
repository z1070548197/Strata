#pragma once

#include <utility>

// The runtime prompt loan changes cache-tail slots into writable prompt workspace.  Keep the
// lifecycle ordering outside the serve loop so it can be tested without CUDA or a model.
namespace strata::program {

/// Run one participant's loan transition.  `quiesced` is shared by every participant in the
/// request, so external cache/adaptive writers are drained once, before any refill or relayout.
template <class Quiesce, class Refill, class Relayout, class Publish>
bool prompt_loan_transition(bool& quiesced, bool refill_old,
                            Quiesce&& quiesce, Refill&& refill,
                            Relayout&& relayout, Publish&& publish) {
    if (!quiesced) {
        std::forward<Quiesce>(quiesce)();
        quiesced = true;
    }
    if (refill_old && !std::forward<Refill>(refill)()) return false;
    if (!std::forward<Relayout>(relayout)()) return false;
    std::forward<Publish>(publish)();
    return true;
}

}  // namespace strata::program
