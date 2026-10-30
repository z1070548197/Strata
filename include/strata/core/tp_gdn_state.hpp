#pragma once

#include <cstdint>

namespace strata::core {

struct TpGdnStateView {
    float* data = nullptr;
    int64_t per_layer = 0;
    int64_t layers = 0;
    int device = -1;
    int rank = -1;
};

}  // namespace strata::core
