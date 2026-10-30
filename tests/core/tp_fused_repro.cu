// Minimal repro for the P1 fused handshake.
#include "strata/core/tp_allreduce.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

static void gpu(cudaError_t e, const char* w) {
    if (e != cudaSuccess) { std::fprintf(stderr, "FAIL %s: %s\n", w, cudaGetErrorString(e)); std::exit(1); }
}

int main() {
    int n_dev = 0;
    gpu(cudaGetDeviceCount(&n_dev), "count");
    std::fprintf(stderr, "devices=%d\n", n_dev);
    strata::core::TpAllreduce ar;
    std::string err;
    if (!ar.open(0, 1, err)) { std::fprintf(stderr, "open failed: %s\n", err.c_str()); return 1; }
    std::fprintf(stderr, "p2p ok\n");

    cudaStream_t s0, s1;
    gpu(cudaSetDevice(0), "set0"); gpu(cudaStreamCreateWithFlags(&s0, cudaStreamNonBlocking), "s0");
    gpu(cudaSetDevice(1), "set1"); gpu(cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking), "s1");

    const int64_t n = 2560;
    std::vector<float> ha(n, 1.0f), hb(n, 2.0f);
    float *da = nullptr, *db = nullptr;
    gpu(cudaSetDevice(0), "set0"); gpu(cudaMalloc(&da, n * 4), "da");
    gpu(cudaMemcpy(da, ha.data(), n * 4, cudaMemcpyHostToDevice), "cp a");
    gpu(cudaSetDevice(1), "set1"); gpu(cudaMalloc(&db, n * 4), "db");
    gpu(cudaMemcpy(db, hb.data(), n * 4, cudaMemcpyHostToDevice), "cp b");

    uint32_t* hf = nullptr;
    gpu(cudaHostAlloc((void**)&hf, 2 * 4, cudaHostAllocMapped), "hostflags");
    hf[0] = 0; hf[1] = 0;
    uint32_t *fa0 = nullptr, *fb0 = nullptr, *fa1 = nullptr, *fb1 = nullptr;
    gpu(cudaSetDevice(0), "s0"); gpu(cudaHostGetDevicePointer((void**)&fa0, &hf[0], 0), "fa0");
    gpu(cudaHostGetDevicePointer((void**)&fb0, &hf[1], 0), "fb0");
    gpu(cudaSetDevice(1), "s1"); gpu(cudaHostGetDevicePointer((void**)&fa1, &hf[0], 0), "fa1");
    gpu(cudaHostGetDevicePointer((void**)&fb1, &hf[1], 0), "fb1");
    std::fprintf(stderr, "flags mapped: fa0=%p fb0=%p fa1=%p fb1=%p\n", (void*)fa0, (void*)fb0, (void*)fa1, (void*)fb1);

    const uint32_t want = 7;
    std::fprintf(stderr, "launching publish_wait on dev1...\n"); std::fflush(stderr);
    if (!ar.publish_wait(fb1, want, fa1, want, s1, err)) { std::fprintf(stderr, "pw: %s\n", err.c_str()); return 1; }
    std::fprintf(stderr, "launching allreduce_wait_raise on dev0...\n"); std::fflush(stderr);
    if (!ar.allreduce_wait_raise(da, db, n, s0, fb0, want, fa0, want, err)) { std::fprintf(stderr, "awr: %s\n", err.c_str()); return 1; }
    std::fprintf(stderr, "launched both; syncing...\n"); std::fflush(stderr);
    gpu(cudaSetDevice(0), "s0"); gpu(cudaStreamSynchronize(s0), "sync0");
    std::fprintf(stderr, "sync0 done, hf[0]=%u hf[1]=%u\n", hf[0], hf[1]); std::fflush(stderr);
    gpu(cudaSetDevice(1), "s1"); gpu(cudaStreamSynchronize(s1), "sync1");
    std::fprintf(stderr, "sync1 done, hf[0]=%u hf[1]=%u\n", hf[0], hf[1]);

    std::vector<float> ra(n), rb(n);
    gpu(cudaSetDevice(0), "s0"); gpu(cudaMemcpy(ra.data(), da, n * 4, cudaMemcpyDeviceToHost), "rda");
    gpu(cudaSetDevice(1), "s1"); gpu(cudaMemcpy(rb.data(), db, n * 4, cudaMemcpyDeviceToHost), "rdb");
    std::fprintf(stderr, "ra[0]=%f rb[0]=%f (want 3.0)\n", ra[0], rb[0]);
    return (ra[0] == 3.0f && rb[0] == 3.0f) ? 0 : 1;
}
