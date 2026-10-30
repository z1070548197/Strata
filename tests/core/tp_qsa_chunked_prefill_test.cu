#include "strata/core/layout.hpp"
#include "strata/core/layer.hpp"
#include "strata/kernels/native_qsa_indexer.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/prefill/kernels.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        std::exit(1);
    }
}

void gpu(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "FAIL: %s: %s\n", what, cudaGetErrorString(status));
        std::exit(1);
    }
}

float value(int64_t pos, int64_t i, unsigned seed) {
    unsigned x = (unsigned) pos * 2246822519u ^ (unsigned) i ^ seed;
    x = x * 1664525u + 1013904223u;
    return ((float) ((x >> 8) & 0xffff) / 32768.0f - 1.0f) * 0.25f;
}

template <class T>
void compare_device(const T* a, const T* b, size_t n, const char* what) {
    std::vector<T> ha(n), hb(n);
    gpu(cudaMemcpy(ha.data(), a, n * sizeof(T), cudaMemcpyDeviceToHost), "copy rank 0 state");
    gpu(cudaMemcpy(hb.data(), b, n * sizeof(T), cudaMemcpyDeviceToHost), "copy rank 1 state");
    if (std::memcmp(ha.data(), hb.data(), n * sizeof(T)) != 0) {
        std::fprintf(stderr, "FAIL: %s differs\n", what);
        std::exit(1);
    }
}

struct RankState {
    void* arena = nullptr;
    strata::core::QsaState state;
    cudaStream_t stream = nullptr;
};

}  // namespace

int main() {
    int devices = 0;
    gpu(cudaGetDeviceCount(&devices), "device count");
    if (devices < 1) return 77;

    constexpr int64_t max_cells = 2060;
    const strata::core::ModelGeometry geometry;
    const strata::kernels::QsaShapes shapes = strata::core::qsa_shapes(geometry);
    check(shapes.page_size == 4 && shapes.idx_block == 4 && shapes.idx_top_k == 2048,
          "unexpected QSA geometry");

    strata::core::qsa_set_kv_resident(0);
    strata::core::qsa_set_kv_int8(false);
    strata::core::qsa_set_kv_q4(false);
    strata::core::qsa_set_kv_hybrid(false);

    RankState ranks[2];
    const uint64_t bytes0 = strata::core::qsa_state_bytes(geometry, max_cells, true);
    const uint64_t bytes1 = strata::core::qsa_state_bytes(geometry, max_cells, false);
    gpu(cudaMalloc(&ranks[0].arena, (size_t) bytes0), "allocate rank 0 QSA state");
    gpu(cudaMalloc(&ranks[1].arena, (size_t) bytes1), "allocate rank 1 QSA state");
    check(strata::core::qsa_state_init(geometry, max_cells, ranks[0].arena, ranks[0].state) != 0,
          "initialize rank 0 QSA state");
    check(strata::core::qsa_state_init(geometry, max_cells, ranks[1].arena, ranks[1].state,
                                       &ranks[0].state) != 0,
          "initialize rank 1 QSA state");
    for (RankState& rank : ranks) {
        gpu(cudaStreamCreateWithFlags(&rank.stream, cudaStreamNonBlocking), "create QSA stream");
        strata::core::qsa_state_zero(rank.state, geometry, rank.stream);
    }
    gpu(cudaDeviceSynchronize(), "zero QSA states");

    check(ranks[0].state.k_pool != ranks[1].state.k_pool &&
              ranks[0].state.v_pool != ranks[1].state.v_pool &&
              ranks[0].state.idx_tail != ranks[1].state.idx_tail &&
              ranks[0].state.idx_pooled != ranks[1].state.idx_pooled,
          "rank mutable QSA state aliases");
    check(ranks[0].state.cos_tab == ranks[1].state.cos_tab &&
              ranks[0].state.sin_tab == ranks[1].state.sin_tab,
          "rank 1 did not borrow the read-only RoPE table");

    float* gamma = nullptr;
    gpu(cudaMalloc(&gamma, (size_t) shapes.idx_dim * sizeof(float)), "allocate indexer gamma");
    std::vector<float> h_gamma((size_t) shapes.idx_dim);
    for (int64_t i = 0; i < shapes.idx_dim; ++i) h_gamma[(size_t) i] = 0.75f + 0.001f * (float) i;
    gpu(cudaMemcpy(gamma, h_gamma.data(), h_gamma.size() * sizeof(float), cudaMemcpyHostToDevice),
        "upload indexer gamma");

    const int64_t chunks[] = {3, 2, 2043, 2, 3, 7};
    int64_t pos = 0;
    for (const int64_t count : chunks) {
        std::vector<float> h_k((size_t) count * shapes.n_head_kv * shapes.head_dim);
        std::vector<float> h_v(h_k.size());
        std::vector<float> h_raw((size_t) count * shapes.idx_dim);
        for (int64_t t = 0; t < count; ++t) {
            for (int64_t i = 0; i < shapes.n_head_kv * shapes.head_dim; ++i) {
                h_k[(size_t) t * shapes.n_head_kv * shapes.head_dim + i] = value(pos + t, i, 11);
                h_v[(size_t) t * shapes.n_head_kv * shapes.head_dim + i] = value(pos + t, i, 29);
            }
            for (int64_t i = 0; i < shapes.idx_dim; ++i)
                h_raw[(size_t) t * shapes.idx_dim + i] = value(pos + t, i, 47);
        }

        for (RankState& rank : ranks) {
            float *k = nullptr, *v = nullptr, *raw = nullptr;
            gpu(cudaMalloc(&k, h_k.size() * sizeof(float)), "allocate chunk K");
            gpu(cudaMalloc(&v, h_v.size() * sizeof(float)), "allocate chunk V");
            gpu(cudaMalloc(&raw, h_raw.size() * sizeof(float)), "allocate chunk indexer keys");
            gpu(cudaMemcpyAsync(k, h_k.data(), h_k.size() * sizeof(float), cudaMemcpyHostToDevice, rank.stream),
                "upload chunk K");
            gpu(cudaMemcpyAsync(v, h_v.data(), h_v.size() * sizeof(float), cudaMemcpyHostToDevice, rank.stream),
                "upload chunk V");
            gpu(cudaMemcpyAsync(raw, h_raw.data(), h_raw.size() * sizeof(float), cudaMemcpyHostToDevice, rank.stream),
                "upload chunk indexer keys");
            strata::prefill::kv_append(k, v, count, pos, rank.state.page_table, shapes.page_size,
                                       rank.state.k_pool, rank.state.v_pool, nullptr, nullptr, nullptr, nullptr,
                                       rank.stream);
            const strata::kernels::QsaIndexerBuffers buffers{
                rank.state.idx_tail, rank.state.idx_dead, rank.state.idx_pooled, rank.state.idx_block_pos};
            strata::kernels::native_qsa_indexer_append_batch(
                raw, count, pos, 0, gamma, strata::kernels::qsa_rms_eps(), buffers, shapes,
                max_cells, strata::kernels::rope_scaling(), rank.stream);
            gpu(cudaStreamSynchronize(rank.stream), "finish QSA chunk");
            gpu(cudaFree(k), "free chunk K");
            gpu(cudaFree(v), "free chunk V");
            gpu(cudaFree(raw), "free chunk indexer keys");
        }
        pos += count;

        const size_t pool_values = (size_t) ranks[0].state.n_pages * shapes.n_head_kv *
                                   shapes.page_size * shapes.head_dim;
        compare_device(ranks[0].state.k_pool, ranks[1].state.k_pool, pool_values, "K pool");
        compare_device(ranks[0].state.v_pool, ranks[1].state.v_pool, pool_values, "V pool");
        compare_device(ranks[0].state.idx_tail, ranks[1].state.idx_tail,
                       (size_t) (shapes.idx_block - 1) * shapes.idx_dim, "indexer tail");
        compare_device(ranks[0].state.idx_dead, ranks[1].state.idx_dead,
                       (size_t) shapes.idx_dim, "indexer spare key");
        compare_device(ranks[0].state.idx_pooled, ranks[1].state.idx_pooled,
                       (size_t) ranks[0].state.idx_pooled_rows * shapes.idx_dim, "indexer pooled keys");
        compare_device(ranks[0].state.idx_block_pos, ranks[1].state.idx_block_pos, 1,
                       "indexer block position");

        check(strata::kernels::qsa_selection_width(pos, shapes) ==
                  (pos < 2051 ? pos : 2051),
              "selection width boundary");
    }
    check(pos == max_cells, "chunk plan did not reach the QSA capacity");

    const int64_t width = strata::kernels::qsa_selection_width(pos, shapes);
    std::vector<float> h_query((size_t) shapes.n_head * shapes.head_dim);
    std::vector<int32_t> h_ids((size_t) width);
    for (size_t i = 0; i < h_query.size(); ++i) h_query[i] = value(pos, (int64_t) i, 71);
    for (int64_t i = 0; i < width; ++i) h_ids[(size_t) i] = (int32_t) i;
    const int32_t h_step[strata::kernels::kStepCount] = {
        (int32_t) (pos - 1), (int32_t) pos, (int32_t) (pos / shapes.idx_block), (int32_t) width};
    std::vector<float> decode_output[2];
    for (int rank = 0; rank < 2; ++rank) {
        float *query = nullptr, *scratch = nullptr, *attn = nullptr;
        int32_t *ids = nullptr, *step = nullptr;
        gpu(cudaMalloc(&query, h_query.size() * sizeof(float)), "allocate decode query");
        gpu(cudaMalloc(&ids, h_ids.size() * sizeof(int32_t)), "allocate decode ids");
        gpu(cudaMalloc(&step, sizeof(h_step)), "allocate decode step");
        gpu(cudaMalloc(&scratch, strata::kernels::qsa_decode_attn_scratch_floats(width, shapes) * sizeof(float)),
            "allocate decode attention scratch");
        gpu(cudaMalloc(&attn, h_query.size() * sizeof(float)), "allocate decode attention output");
        gpu(cudaMemcpy(query, h_query.data(), h_query.size() * sizeof(float), cudaMemcpyHostToDevice),
            "upload decode query");
        gpu(cudaMemcpy(ids, h_ids.data(), h_ids.size() * sizeof(int32_t), cudaMemcpyHostToDevice),
            "upload decode ids");
        gpu(cudaMemcpy(step, h_step, sizeof(h_step), cudaMemcpyHostToDevice), "upload decode step");
        strata::kernels::qsa_decode_attn_step(query, strata::core::qsa_attn_pools(ranks[rank].state),
                                              ids, step, width, shapes, scratch, attn,
                                              ranks[rank].stream);
        gpu(cudaStreamSynchronize(ranks[rank].stream), "finish first decode attention");
        decode_output[rank].resize(h_query.size());
        gpu(cudaMemcpy(decode_output[rank].data(), attn, decode_output[rank].size() * sizeof(float),
                       cudaMemcpyDeviceToHost), "read first decode attention");
        gpu(cudaFree(query), "free decode query");
        gpu(cudaFree(ids), "free decode ids");
        gpu(cudaFree(step), "free decode step");
        gpu(cudaFree(scratch), "free decode attention scratch");
        gpu(cudaFree(attn), "free decode attention output");
    }
    check(std::memcmp(decode_output[0].data(), decode_output[1].data(),
                      decode_output[0].size() * sizeof(float)) == 0,
          "QSA first decode attention differs across rank-local states");

    gpu(cudaFree(gamma), "free indexer gamma");
    for (RankState& rank : ranks) {
        if (rank.state.host_step != nullptr) gpu(cudaFreeHost(rank.state.host_step), "free host step");
        if (rank.state.host_pos != nullptr) gpu(cudaFreeHost(rank.state.host_pos), "free host positions");
        gpu(cudaStreamDestroy(rank.stream), "destroy QSA stream");
        gpu(cudaFree(rank.arena), "free QSA state");
    }
    std::fprintf(stderr, "PASS: tp_qsa_chunked_prefill_test\n");
    return 0;
}
