# 交接：Strata NVFP4 双卡真张量并行（TP）

## 一句话背景

模型是 Qwen3.8-Flash-Next 125B MoE（`n_embd=2560, hc=4, n_layers=48, n_expert=512, n_expert_used=10, n_ff=640`），
跑在 2×2080Ti + NVLink。已有 `--layer-split`（流水线并行）。现在要做**真张量并行**（每卡算每层的一半）。
代码库在 `/media/zc/软件1/strata-nvfp4`（**不在** VS Code workspace 里，所以 `grep_search`/`file_search` 用不了，
必须用 `run_in_terminal` 里的 `grep`/`sed`/`awk`）。

**用户明确的最终要求：两张卡一起算，真正的张量并行，不许把工作压在单卡上。**

## 当前真实状态（务必先读，别重复踩坑）

### 已经跑通的部分（不要重做）
`--tp --serve` 能完整跑完一个请求，`EXIT=0`，无崩溃、无死锁。但**输出 token 与单卡不同，且更慢**。

### 已验证数值正确的部分（有测试，别怀疑）
```
tests/core/tp_gdn_e2e_test.cu    GDN 完整 TP 链路 vs 单卡参考
  → max_abs=4.172e-07, over_tol=0/7680, 两卡逐位相同
  → 单层: 单卡 216.1 us → 双卡 147.9 us (1.46x)
tests/core/tp_layout_test.cpp    索引布局不变量，ALL PASS
tests/core/tp_split_plan_test.cpp 切分计划，ALL PASS
tests/core/tp_conv_split_test.cu  conv 权重切分，两 rank 逐位一致
tests/core/tp_graph_test.cu       双图并发 + 自旋握手，rel_err 1.02e-07
```
编译运行方式（这些测试需要两卡各约 200 MiB 空闲显存）：
```bash
# GDN 端到端
nvcc -O2 -std=c++17 -I include -I src -I third_party/ggml \
  tests/core/tp_gdn_e2e_test.cu src/core/tp_allreduce.cu src/core/tp_split.cu \
  src/core/tp_split_plan.cpp src/kernels/cuda/native_mmvq.cu src/kernels/cuda/iq_kernels.cu \
  src/kernels/cuda/verify_kernels.cu src/kernels/cuda/quantize_act.cu -o /tmp/tp_e2e -lcuda
/tmp/tp_e2e
# 布局
g++ -std=c++17 -I include tests/core/tp_layout_test.cpp src/core/tp_split_plan.cpp -o /tmp/tp_layout && /tmp/tp_layout
```

### 两个尚未解决的问题

**问题 A：输出不正确**
同一输入（`GEN 8 9707 11 1246`）对比：
```
单卡:  79790 316 52933 91 79751 198 262 1992   14.3 tok/s
--tp:  11    220 16    16 16    22  220 16      11.2 tok/s
```
GDN 已验证正确 → 嫌疑在 QSA 的半拷贝，或 `capture_commit_tp` 的状态推进。

**问题 B：更慢（架构性问题）**
当前 MoE **只由 rank 0 计算**（rank 1 的专家缓存根本没创建）。MoE 是每层最重的部分，
所以 rank 1 只分担了 dense 的零头，却要付每层 2 次 allreduce + 3 次握手。这正是为什么没加速。

**问题 B 是用户这次明确点名要解决的。**

## 目标架构（用户要的）

MoE 必须**按 expert 切分到两卡**：
- rank 0 算偶数 expert，rank 1 算奇数（或按 resident 集合切）
- 两卡的 partial parts 相加后 combine
- 每层：dense 半边 allreduce + MoE 切分 allreduce

或者用 **FFN 中间维切分**（Megatron 式）：每个 expert 的 `n_ff=640` 切成 320+320，
gate/up 列并行，down 行并行 + allreduce。这样每个 expert 两卡各算一半，负载天然均衡，
不需要 expert 分配策略。

**先确认哪条路可行**：专家权重是 NVFP4（type 40，64 elem/block，36 bytes/block），
不是 Q8_0。切分 `n_ff` 需要按 NVFP4 块边界切（64 的倍数，640/2=320 是 5 个块，可以）。
但 `native_expert_grouped` 的 `NativeExpertLayout` 是否支持自定义 `n_ff` 必须先查。
如果不行，走 expert 分配路线。

## 当前 TP 实现的关键位置（都在 `src/core/verify.cpp`）

| 函数 | 行号约 | 说明 |
|---|---|---|
| `tp_setup_flags` | 2400 | 分配 mapped 握手标志 a/b/c（三个），P2P 打开。**每个 rank 各自用自己设备解析 `cudaHostGetDevicePointer`** |
| `tp_setup_scratch` | 2470 | 每卡一块 `tp_arena_`（72 MiB）：`qkv_tp_`/`h_tp_`（nG*MT*7168）、`gate_tp_`/`beta_tp_`（nG*MT*24）、`z_tp_`/`y_tp_`（MT*3072）、`attn_tp_`（MT*12*256）、`conv_w_tp_`（nG*7168*4）、`gdn_state_tp_`。**开头必须有 `OnDevice on_self(device_)`** |
| `capture_tp` | 2455 | 镜像 `capture`，存进 `tp_exec_[T]`，调 `record_window_tp`。**有 OnDevice** |
| `record_window_tp` | 2700+ | 默认 decode 路径的 TP 版。GDN 真切分，QSA 复制 attention + 切 attn_output，MoE 只在 rank 0 |
| `capture_commit_tp` | 4047 | 局部 GDN 状态 + 复制 QSA indexer。**有 OnDevice** |
| `run_tp` | 4150 | rank 0 协调者：先启 rank1 图，再启 rank0 图，join 两卡，拷 `h_out_`。**不提交状态**（由 `commit()` 转发） |
| `tp_state_advance` | 4299 | 两卡 commit 图一起启。**有 OnDevice** |
| `tp_share_doorbell` | 3230 | rank 1 共享 rank 0 的 `m_seq_`/`m_flagA_`/`m_flagB_`/`m_plan_`/`sink_`/`staging_` |
| `commit` | 4317 | **TP 时转发到 `tp_state_advance`**（否则用整头状态图会崩） |

`include/strata/core/verify.hpp` 的 TP 成员区在约 600-680 行。
`tp_setup_flags`/`tp_setup_scratch`/`set_tp_bo_peer`/`tp_bo` 必须在 **public**（`generate.cpp` 要用）。

### 每层握手（关键，别改错）

```
rank 0:  ... 行并行 out-proj partial → wait(flag_b, l+1) → allreduce(bo_, peer) → raise(flag_a, l+1)
rank 1:  ... 行并行 out-proj partial → raise(flag_b, l+1) → wait(flag_a, l+1)

MoE 之后：
rank 0:  算完整 MoE（GPU hits + PCIe share + CPU share + combine）
         → broadcast(bo_ → peer) → raise(flag_c, l+1)
rank 1:  wait(flag_c, l+1)          # 它的 bo_ 此时是 rank 0 的结果
```
标志值 = 全局层号 + 1。`run_tp` 启动前清零三个标志。
**一个门铃**：只有 rank 0 调 `doorbell_publish*`，否则 `m_seq_` 每层涨 2，host 永远等不到。

## 已修掉的 9 个 bug（别改回去）

1. `gdn_ab_body_tp` 必须带 `template <int MAX_T, bool EXACT_T>` 头
2. `gdn_ab_body_tp` 里是 `fmaf(w6, xb.z, a)`，不是 `xb.x`
3. `attn_q.weight` 必须是 `TpMode::Replicate`（attention 内核算 `G = n_head/n_head_kv` 是编译期常量，不能切 query head）
4. conv 权重是 **4 个 tap**（`w[c*4+tap]`），不是 `d_conv-1`=3
5. `tp_setup_flags` 里 peer 的标志指针要用 peer 自己的设备解析
6. `tp_setup_scratch`/`capture_tp`/`capture_commit_tp`/`tp_state_advance` 都要 `OnDevice`
7. `--tp` 时 prompt 路径也必须走 `run_tp`（`generate.cpp` 约 14506 行）
8. 只有一个门铃（见上）
9. `commit()` 在 TP 时要转发到 `tp_state_advance`

## 立刻要做的第一件事：定位问题 A

已插入逐层残差 dump（`h_tpdump_`，`STRATA_TP_DUMP=1` 开启）：
- `src/core/verify.cpp:728` 分配（`n_layers * 8` 个 float，mapped）
- `src/core/verify.cpp:2096` 在 `record_window` 的 `post` 末尾写入
- `src/core/verify.cpp:3212` 在 `record_window_tp` 的 `post` 末尾写入
- `src/core/verify.cpp:3904` / `4335` 在 `run` / `run_tp` 结束时打印

用法：
```bash
# 两卡都被别的进程占着先清理
pkill -9 -f "build/strata"; sleep 3
# 单卡
STRATA_TP_DUMP=1 ./build/strata --serve ...（不加 --tp）> /tmp/base.log 2>&1
# TP
STRATA_TP_DUMP=1 ./build/strata --serve --tp ... > /tmp/tp.log 2>&1
# 比较 stderr 里的 dump 行，找第一处偏离的层
```
**注意**：`dump` 在 `post` 里写的是 `bo_`（combine 后）。如果发现某层开始偏离，就是那一层的
mixer（GDN 或 QSA）或 MoE 有问题。QSA 层是 `l % 4 == 3`（3,7,11,...,47）。

跑服务的完整命令（参数必须齐全，缺一个就报 `bad request` 或 `no such option`）：
```bash
( printf 'GEN 8 9707 11 1246\n'; sleep 14 ) | timeout 280 ./build/strata --serve --tp \
 --pack /media/zc/软件1/strata-models/NVFP4-GPTQ/pack \
 --native /media/zc/软件1/strata-models/NVFP4-GPTQ/qwen-nvfp4-gptq-dense.gguf \
 --native-dense-gguf /media/zc/软件1/strata-models/NVFP4-GPTQ/qwen-nvfp4-gptq-dense.gguf \
 --ple-gguf /media/zc/软件1/strata-models/NVFP4-GPTQ/ple-fp8.gguf \
 --embd-gguf /media/zc/软件1/strata-models/NVFP4-GPTQ/token-embd-bf16.gguf \
 --mtp /media/zc/软件1/strata-models/NVFP4-GPTQ/mtp --spec 4 --prefill auto \
 --expert-profile data/expert-profile.bin --expert-cache auto \
 --max-context 4096 --kv int8 --vram-reserve-mib 960
```
请求格式是 `GEN <max_new> <token ids...>`（**不是** JSON）。整机加载约 100 秒，耐心等。

## 环境与注意事项（血泪）

- **不要在 VS Code 里打开大文件**，用户明确说过有时会让 VS Code 崩溃。一律用
  `sed -n`/`awk`/`head`/`tail`/`grep` 读，命令输出也要截断。
- 构建：`cd /media/zc/软件1/strata-nvfp4/build && cmake --build . -j12`
- 测试/运行前先 `pkill -9 -f "build/strata"; sleep 3`，否则显存不够。
- 长命令用 `nohup /tmp/xxx.sh >/dev/null 2>&1 &` 后台跑 + 短命令读日志，
  否则终端会回显历史输出把命令吞掉。
- `compute-sanitizer` 在 CUDA graph 内捕获不到越界（只看到 stall），别浪费时间。
- 修改 `record_window`（非 TP 的 1200 行函数）时务必保证**单卡基线不变**，
  每次改完先跑一次不带 `--tp` 的请求确认 `79790 316 52933 91 ...` 这串 token 没变。

## 交付标准

1. `--tp` 的输出 token 与单卡**完全一致**
2. `--tp` 的 decode 速度**明显快于** 14.3 tok/s
3. `tests/core/` 下所有 TP 测试仍然 ALL PASS
4. 单卡路径无回归
