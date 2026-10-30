#include "strata/prefill/prefill.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <string>

namespace {
void check(bool ok, const char* what, const std::string& err = {}) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s%s%s\n", what, err.empty() ? "" : ": ", err.c_str());
        std::exit(1);
    }
}
void gpu(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "FAIL: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}
}  // namespace

int main() {
    int count = 0;
    gpu(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
    if (count == 0) {
        std::fprintf(stderr, "SKIP: no CUDA device\n");
        return 0;
    }

    std::string err;
    strata::prefill::Prefill p;
    check(!p.tp_gdn_state_bound(), "fresh Prefill is unbound");
    check(!p.set_tp_gdn_state({}, err), "empty view rejected");

    gpu(cudaSetDevice(0), "set device 0");
    float* state = nullptr;
    gpu(cudaMalloc(&state, 4096), "allocate external state");
    strata::core::TpGdnStateView view{state, 1024, 36, 0, 0};

    err.clear();
    auto bad_rank = view;
    bad_rank.rank = 2;
    check(!p.set_tp_gdn_state(bad_rank, err), "invalid rank rejected");

    err.clear();
    auto bad_device = view;
    bad_device.device = count > 1 ? 1 : 7;
    check(!p.set_tp_gdn_state(bad_device, err), "wrong device rejected");

    err.clear();
    check(p.set_tp_gdn_state(view, err), "valid external state accepted", err);
    check(p.tp_gdn_state_bound(), "bound state reported");

    p.reset();
    check(!p.tp_gdn_state_bound(), "reset clears the borrowed binding");
    gpu(cudaMemset(state, 0x5a, 4096), "external state remains owned by verifier");
    err.clear();
    check(p.set_tp_gdn_state(view, err), "state can be rebound after reset", err);
    gpu(cudaFree(state), "free external state");
    std::fprintf(stderr, "PASS: tp_prefill_state_binding_test\n");
    return 0;
}
