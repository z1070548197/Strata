// tests/cuda/nvfp4_q8_0_grouped_bench.cpp - speed: the bit-exact grouped kernel vs the engine's
// native_expert_grouped, same geometry and same data, so the ratio means something.
//
//     nvfp4_q8_0_grouped_bench [reps]
//
// This is a BENCHMARK, not a test: it asserts nothing.
//
// TWO LAYOUTS, AND WHY BOTH ARE PRINTED.  With the default (10 groups of T entries) the whole expert set is 27.6 MB,
// which fits in L2, so the run measures L2 rather than the DRAM path the engine takes once residency is high.  With
// --distinct every entry gets its own blob and the read has to come from DRAM.  The engine sits between the two
// (single-card residency is ~23%), but the DRAM number is the one that stays honest as residency rises.
//
// WHERE THE TIME WENT (T=8, grouped gu + down against native_expert_grouped):
//
//   15.05x  start
//    8.23x  pure-ALU E2M1 decode instead of a `__constant__ kFp4[16]` table (a divergent constant access on this
//           hardware: 247 GB/s of the same loads without it, 18.7 GB/s with it)
//    6.43x  entry window 32 -> 4: the accumulator arrays had spilled to local memory (STACK:512, 105 LDL/STL)
//    2.83x  decode the weight block ONCE per block and apply it to every entry.  It was being decoded inside the
//           entry loop, so an 8-entry group decoded the same nibbles 8 times.
//    1.86x  load the q8_0 activation as two aligned 16-bit halves instead of four `LDG.E.U8`.  The old `load4` had an
//           alignment test whose fallback path the compiler could not prove, so it emitted four byte loads per word
//           per lane per sub-block - the single largest remaining cost.
//    1.28x  three-shuffle hsum8 (was eight) and the 32-bit magnitude table
//
// What is left is bandwidth, not ALU: at T=8 the down kernel moves 73.8 MB in 0.459 ms = 161 GB/s, and
// native_expert_grouped moves the same bytes at the same rate.  Measured and rejected on the way here, all
// bit-exact and all slower or neutral: coalescing the qs read through shuffles (gu 0.208 -> 0.268 ms), one warp per
// row instead of eight lanes per row (223 vs 289 GB/s on the read alone), and sharing the activation decode between
// gate and up (the compiler had already done it).
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/nvfp4_q8_0_cpu_order.hpp"
#include "strata/kernels/quantize_act.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#define CK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){printf("CUDA %s @%d\n",cudaGetErrorString(e),__LINE__);return 1;}}while(0)
using namespace strata::kernels;
int main(int argc, char** argv){
  const int reps = argc > 1 ? (std::atoi(argv[1]) > 0 ? std::atoi(argv[1]) : 20) : 20;
  // --distinct: one group per entry, so every entry reads a DIFFERENT blob.  The default (G=10 for every T) puts
  // 27.6 MB of experts in front of 80 entries, which fits in L2 and therefore measures L2, not the DRAM path the
  // engine actually takes once residency is high.  Both numbers are worth having: the default is the cheap case,
  // --distinct is the pessimal one.
  bool distinct = false;
  for (int i = 1; i < argc; ++i) if (!std::strcmp(argv[i], "--distinct")) distinct = true;
  const int64_t H=2560, FF=640;
  const NativeExpertLayout L=native_expert_layout(40,40,H,FF);
  printf("H=%lld FF=%lld gu_row=%zu d_row=%zu up=%zu dn=%zu bytes=%zu tail=%zu\n",
    (long long)H,(long long)FF,L.gu_row,L.d_row,L.up_off,L.down_off,L.bytes,L.tail_off);
  // real decode window: K=10 routed experts/layer, T tokens per window
  for(int T : {1,4,8}){
    // default: 10 groups of T entries (weights shared, L2-resident).  --distinct: one group per entry, so every
    // entry reads a different blob and the read has to come from DRAM.
    const int NE=10*T;
    const int G=distinct?NE:10;
    std::mt19937 rng(7); std::normal_distribution<float> nd(0.f,1.f);
    std::vector<uint8_t> arena((size_t)G*L.bytes);
    for(size_t i=0;i+36<=arena.size();i+=36){
      arena[i]=(uint8_t)(0x28|(rng()&7)); arena[i+1]=(uint8_t)(0x28|(rng()&7));
      arena[i+2]=(uint8_t)(0x28|(rng()&7)); arena[i+3]=(uint8_t)(0x28|(rng()&7));
      for(int j=4;j<36;j++) arena[i+j]=(uint8_t)(rng()&255);
    }
    for(int g=0;g<G;g++){ float s[3]={0.8f,1.2f,1.1f}; memcpy(arena.data()+(size_t)g*L.bytes+L.tail_off,s,12); }
    std::vector<float> x((size_t)T*H); for(auto&v:x) v=nd(rng);
    // both activation formats
    uint8_t* d_arena=nullptr; unsigned long long* d_ptr=nullptr; int32_t *d_start=nullptr,*d_ng=nullptr,*d_tok=nullptr,*d_dst=nullptr;
    float *d_x=nullptr,*d_gate=nullptr,*d_up=nullptr,*d_ff=nullptr,*d_out=nullptr;
    void* d_q81=nullptr; uint8_t* d_q80=nullptr; void* d_scr=nullptr; void* d_scr2=nullptr;
    CK(cudaMalloc(&d_arena,arena.size()));
    CK(cudaMalloc(&d_ptr,G*8)); CK(cudaMalloc(&d_start,(G+1)*4)); CK(cudaMalloc(&d_ng,4));
    CK(cudaMalloc(&d_tok,NE*4)); CK(cudaMalloc(&d_dst,NE*4));
    CK(cudaMalloc(&d_x,(size_t)T*H*4));
    CK(cudaMalloc(&d_gate,(size_t)NE*FF*4)); CK(cudaMalloc(&d_up,(size_t)NE*FF*4));
    CK(cudaMalloc(&d_ff,(size_t)NE*FF*4)); CK(cudaMalloc(&d_out,(size_t)NE*H*4));
    CK(cudaMalloc(&d_q81,(size_t)T*(H/32)*36)); CK(cudaMalloc(&d_q80,(size_t)T*(H/32)*34+(size_t)T*(FF/32)*34));
    const size_t scr=native_expert_scratch_bytes(NE,FF); CK(cudaMalloc(&d_scr,scr));
    std::vector<unsigned long long> hp(G); std::vector<int32_t> st(G+1),tk(NE),ds(NE);
    const int per=NE/G;                                 // entries per group: T normally, 1 with --distinct
    for(int g=0;g<G;g++){st[g]=g*per;} st[G]=G*per;
    for(int i=0;i<NE;i++){tk[i]=i%T; ds[i]=i;}
    CK(cudaMemcpy(d_arena,arena.data(),arena.size(),cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_start,st.data(),(G+1)*4,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_ng,&G,4,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_tok,tk.data(),NE*4,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_dst,ds.data(),NE*4,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_x,x.data(),x.size()*4,cudaMemcpyHostToDevice));
    for(int g=0;g<G;g++) hp[g]=(unsigned long long)(uintptr_t)(d_arena+(size_t)g*L.bytes);
    CK(cudaMemcpy(d_ptr,hp.data(),G*8,cudaMemcpyHostToDevice));
    quantize_q8_1_rows(d_x,T,H,d_q81,nullptr);
    quantize_q8_0(d_x,d_q80,H,nullptr);
    // per-token q8_0 rows
    for(int t=0;t<T;t++) quantize_q8_0(d_x+(size_t)t*H,d_q80+(size_t)t*(H/32)*34,H,nullptr);
    CK(cudaDeviceSynchronize());
    cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
    auto bench=[&](const char* name, auto&& fn)->double{
      for(int i=0;i<3;i++) fn();
      CK(cudaDeviceSynchronize());
      cudaEventRecord(a);
      for(int i=0;i<reps;i++) fn();
      cudaEventRecord(b); cudaEventSynchronize(b);
      float ms=0; cudaEventElapsedTime(&ms,a,b);
      printf("  %-34s %8.3f ms\n",name,ms/reps);
      return ms/reps;
    };
    printf("T=%d G=%d NE=%d\n",T,G,NE);
    double cur=bench("native_expert_grouped (q8_1)",[&]{
      native_expert_grouped(L,d_ptr,d_start,d_ng,d_dst,d_tok,G,NE,d_q81,d_scr,d_out,nullptr);
    });
    double mine=bench("nvfp4_q8_0_grouped gu only (q8_0)",[&]{
      nvfp4_q8_0_grouped_gu(d_ptr,d_start,d_ng,d_tok,G,NE,H,FF,L.gu_row,L.up_off,L.tail_off,d_q80,(int)((H/32)*34),d_ff,nullptr);
    });
    double down=bench("nvfp4_q8_0_grouped down only",[&]{
      nvfp4_q8_0_grouped_down(d_ptr,d_start,d_ng,d_dst,d_tok,G,NE,H,FF,L.d_row,L.down_off,L.tail_off,
                              d_q80+(size_t)T*(H/32)*34,(int)((FF/32)*34),d_out,nullptr);
    });
    printf("  -> bit-exact path gu+down ~= %.3f ms vs current %.3f ms (%.2fx)\n", mine+down, cur, (mine+down)/cur);
    cudaFree(d_scr2);
    cudaFree(d_scr); cudaFree(d_q80); cudaFree(d_q81); cudaFree(d_out); cudaFree(d_ff);
    cudaFree(d_up); cudaFree(d_gate); cudaFree(d_x); cudaFree(d_dst); cudaFree(d_tok);
    cudaFree(d_ng); cudaFree(d_start); cudaFree(d_ptr); cudaFree(d_arena);
  }
  return 0;
}
