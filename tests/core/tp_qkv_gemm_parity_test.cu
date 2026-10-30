#include "strata/core/tp_split_plan.hpp"
#include "strata/prefill/gemm.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {
constexpr int64_t K = 2560;
constexpr int64_t QK = 2048;
constexpr int64_t V = 6144;
constexpr int64_t VH = V / 2;
constexpr int64_t N = 2 * QK + V;
constexpr int64_t NL = 2 * QK + VH;
constexpr int64_t GEMM_SCRATCH = 32ll << 20;
constexpr size_t ROW_BYTES = (size_t) (K / 32) * 34;
constexpr size_t FULL_BYTES = (size_t) N * ROW_BYTES;

[[noreturn]] void fail(const char* what) { std::fprintf(stderr, "FAIL: %s\n", what); std::exit(1); }
void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) { std::fprintf(stderr, "FAIL: %s: %s\n", what, cudaGetErrorString(e)); std::exit(1); }
}
std::vector<uint8_t> read_weight(const char* path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in || in.tellg() != (std::streamsize) FULL_BYTES) fail("unexpected qkv weight");
    std::vector<uint8_t> data(FULL_BYTES); in.seekg(0);
    if (!in.read((char*) data.data(), data.size())) fail("read qkv weight");
    return data;
}
void run_case(const std::vector<uint8_t>& full, int64_t T) {
    std::vector<uint8_t> half[2] = {std::vector<uint8_t>((size_t) NL * ROW_BYTES),
                                    std::vector<uint8_t>((size_t) NL * ROW_BYTES)};
    for (int r=0;r<2;++r)
        if (strata::core::tp_split_q8_0_host(full.data(), strata::core::TpMode::Qkv, K, N, r, half[r].data()) != half[r].size())
            fail("qkv split size");
    std::vector<uint16_t> x((size_t) T*K);
    for (size_t i=0;i<x.size();++i) {
        uint32_t v=(uint32_t)i*1664525u+1013904223u; float f=((float)((v>>8)&0xffff)/32768.0f-1.0f)*0.125f;
        __half h=__float2half_rn(f); std::memcpy(&x[i],&h,2);
    }
    cudaStream_t s=nullptr; ck(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking),"stream");
    uint16_t* dx=nullptr; uint8_t *wf=nullptr,*wh[2]={}; float *yf=nullptr,*yh[2]={};
    ck(cudaMalloc(&dx,x.size()*2),"X"); ck(cudaMalloc(&wf,full.size()),"W full"); ck(cudaMalloc(&yf,(size_t)T*N*4),"Y full");
    ck(cudaMemcpyAsync(dx,x.data(),x.size()*2,cudaMemcpyHostToDevice,s),"upload X");
    ck(cudaMemcpyAsync(wf,full.data(),full.size(),cudaMemcpyHostToDevice,s),"upload W full");
    for(int r=0;r<2;++r){ ck(cudaMalloc(&wh[r],half[r].size()),"W half"); ck(cudaMalloc(&yh[r],(size_t)T*NL*4),"Y half"); ck(cudaMemcpyAsync(wh[r],half[r].data(),half[r].size(),cudaMemcpyHostToDevice,s),"upload W half"); }
    std::string err; strata::prefill::Gemm gemm; if(!gemm.init(s,GEMM_SCRATCH,err)){std::fprintf(stderr,"FAIL: init: %s\n",err.c_str());return;}
    gemm.native(dx,8,wf,yf,T,N,K); for(int r=0;r<2;++r) gemm.native(dx,8,wh[r],yh[r],T,NL,K); ck(cudaStreamSynchronize(s),"sync");
    std::vector<float> ref((size_t)T*N),got((size_t)T*NL); ck(cudaMemcpy(ref.data(),yf,ref.size()*4,cudaMemcpyDeviceToHost),"ref");
    for(int r=0;r<2;++r){
        ck(cudaMemcpy(got.data(),yh[r],got.size()*4,cudaMemcpyDeviceToHost),"got"); size_t mism=0;
        for(int64_t t=0;t<T;++t) for(int64_t j=0;j<NL;++j){
            int64_t fj = j < 2*QK ? j : 2*QK + r*VH + (j-2*QK);
            float a=got[(size_t)t*NL+j], b=ref[(size_t)t*N+fj];
            if(std::memcmp(&a,&b,4)!=0 && mism++==0) std::fprintf(stderr,"FAIL: T=%lld rank=%d token=%lld local=%lld full=%lld got=%.9g want=%.9g\n",(long long)T,r,(long long)t,(long long)j,(long long)fj,a,b);
        }
        if(mism){std::fprintf(stderr,"FAIL: T=%lld rank=%d mismatches=%zu/%zu\n",(long long)T,r,mism,got.size());std::exit(1);}
    }
    cudaFree(dx);cudaFree(wf);cudaFree(yf);for(int r=0;r<2;++r){cudaFree(wh[r]);cudaFree(yh[r]);}cudaStreamDestroy(s);
    std::printf("PASS: real QKV full vs TP local exact T=%lld\n",(long long)T);
}
}
int main(int argc,char**argv){if(argc!=2){std::fprintf(stderr,"usage: %s qkv.q8_0\n",argv[0]);return 2;}int n=0;ck(cudaGetDeviceCount(&n),"devices");if(n<1)return 77;auto w=read_weight(argv[1]);run_case(w,64);run_case(w,1536);}
