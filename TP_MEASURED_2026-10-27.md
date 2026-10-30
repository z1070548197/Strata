# TP 实测结论（2026-10-27）

## 验收通过（16:35）

**最终配置** `scripts/final_tp_run.sh`：
```
STRATA_POOL_ORDER=1 ./build/strata --serve --tp --expert-cache-device1 6000 \
  --expert-cache 5362 --adapt-every 0 --adapt-swaps 0 --vram-reserve-mib 960 ...
```

| 指标 | 单卡 | TP 旧 | **TP 双卡均分** |
|---|---|---|---|
| tokens | `79790 ... 9341` | 同 | **同（全同）** |
| tok/s | 13.9 | 12.6 | **16.4** |
| 窗口 ms | 96 | 106 | **81** |
| CPU 专家 ms/层窗 | 15.34 | 15.34 | **6.77** |
| 缓存命中率 | 39.4% | 38.4% | **50.7%** |
| 跨卡消化 | — | — | **43% 路由** |

两卡显存一起填（5362 + 6000 槽）、并行算不同专家、CPU 只算剩余 miss。

### 启动的三个坑（都踩过）
1. **systemd-oomd** 按 cgroup 内存压力杀进程（50% 阈值，加载 64 GiB 时尖峰必过）。
   绕法：先把 experts.bin 读进页缓存（`dd ... of=/dev/null`，~65 s @1 GB/s），
   且**预热与运行必须在同一个 systemd unit 里**——分开跑缓存会被逐出（实测 86 Gi→18 Gi）。
2. `--expert-cache-device1 auto` 填满 CUDA1，rank1 的 pack 副本（1466 MiB）放不下 →
   必须显式槽数（6000）。
3. `setsid` 不脱离 cgroup，会话结束连带被杀；必须 `systemd-run --user`。

## 更新（16:20）：发现现成的提速机制 —— `--tp --expert-cache-device1` 可组合


核查了参数校验路径：`--tp` 只拒绝 `--layer-split` 和 `--peer-device`，**不拒绝**
`--expert-cache-device1..3`（helper tier）。实测校验通过（走到 `--ple-gguf` 检查才停）。

helper tier 的机制正好是我们要的：
- `--expert-cache-device1 auto` 让 CUDA1 用空闲显存建专家缓存（实测它有 21484 MiB 空闲）
- CPU 池 plan 里 helper 持有的专家标 `kind -1`，**完全不进 CPU 池**（expert_source.cpp:4108）
- helper 卡自己算（RemoteExperts::begin 领走行 → 该卡算 → finish 写回 pinned rows）

这就是把 69% 的 CPU 池关键路径搬上第二张卡的现成机制。与 pool-order 组合后
helper 算的和 CPU 算的逐位一致（pool-order 已验证），residency 提升不再改 token。

**给用户的验证命令（第二条，确认提速）：**

```bash
cd /media/zc/软件1/strata-nvfp4
STRATA_POOL_ORDER=1 ./build/strata --serve --tp --expert-cache-device1 auto \
  --pack /media/zc/软件1/strata-models/NVFP4-GPTQ/pack \
  --native /media/zc/软件1/strata-models/NVFP4-GPTQ/qwen-nvfp4-gptq-dense.gguf \
  --native-dense-gguf /media/zc/软件1/strata-models/NVFP4-GPTQ/qwen-nvfp4-gptq-dense.gguf \
  --ple-gguf /media/zc/软件1/strata-models/NVFP4-GPTQ/ple-fp8.gguf \
  --embd-gguf /media/zc/软件1/strata-models/NVFP4-GPTQ/token-embd-bf16.gguf \
  --mtp /media/zc/软件1/strata-models/NVFP4-GPTQ/mtp --spec 4 --prefill auto \
  --expert-profile data/expert-profile.bin --expert-cache 5362 \
  --adapt-every 0 --adapt-swaps 0 \
  --max-context 4096 --kv int8 --vram-reserve-mib 960
```

预期：token 与 pool-order solo 全同；CPU 专家时间按 helper 命中比例下降；
两卡显存都被占用。若 `--pipeline-windows` 报"off"是正常（helper caches 不支持它）。

## 更新（16:05）：TP pool-order 切分协议已在真 kernel 上逐位验证（无需模型）


`tp_moe_split_gpu_test.cu` 新增 pool-order 扫描：两相协议（半宽 gu → 34 字节交换 →
半输出行 down + pitch）对 pool-order 单卡全宽**逐位一致**（3×2、5×10、10×4 三个规模
全部 0 differ）。加上 `nvfp4_q8_0_entry_parity`（入口点 0/71680）与 solo 整模八 token
全同，q8_0 切分的正确性链条已经闭合 —— 差的只是整模的一次确认跑。

## 为什么「速度真的快了」需要先提升 residency（数字）

从实测 decode timing（solo，固定 5362 槽）：

| 份额 | 占比 |
|---|---|
| CPU 专家池（主机驱动，AVX2 全核） | **69%** |
| 其余（GPU + 等待 + 提交） | 31% |

TP 把 GPU 侧减半的上限收益：**1.18x**（计入 TP 每层屏障开销 ~7% 后 **1.11x**）。
也就是说：**在 residency 不变的前提下，任何 TP 实现都不可能显著超过单卡** ——
因为时间大头根本不在 GPU 上。这不是实现问题，是算术。

真正提速的唯一路径：把 CPU miss 挪上 GPU（上限 ~49 tok/s，GPU 目前只用 1.4/96 ms），
前提是「改 residency 不改 token」—— 这正是 pool-order 路径已经提供的东西。
下一步是给 rank1 建专家缓存 + 提升两卡的槽数（用户先跑通 pool-order 的整模确认，再做这一步）。

## 更新（15:40）：pool-order 路径已接进引擎并整模验证 solo


`STRATA_POOL_ORDER=1` 接进了 `verify.cpp`：solo 的 `grouped` lambda、TP 的
`gu_split`/`down_split`/`exchange_hq`（34 字节块）都有分支，默认关闭。
**solo 整模八 token 与原路径全同**（`79790 316 52933 91 79751 198 262 9341`，13.1 tok/s）。
17/17 测试全过（新增 `nvfp4_q8_0_entry_parity`：入口点 phase1+2 == phase0 == CPU 池，0/71680）。
**TP+pool-order 整模验证尚未完成** —— 会话终止把进程带走了，需要用户自己跑：

```bash
STRATA_POOL_ORDER=1 ./build/strata --serve --tp ... --expert-cache 5362 --adapt-every 0 --adapt-swaps 0
# 期望：八 token 与 solo（同样 STRATA_POOL_ORDER=1）全同
```

## 一句话


**TP 的算术已经正确（八 token 与单卡全同），但它没有变快，因为瓶颈是 CPU 专家池（占 69%），
而 CPU 池已经吃满全部 CPU 核 —— 加第二张卡对它无效。真正的杠杆是让专家住进 GPU，
而 rank 1 的 21 GiB 显存当前完全没被用来装专家。**

## 实测数据（同配方，可直接复现）

命令固定 `--expert-cache 5362 --adapt-every 0 --adapt-swaps 0`，
`STRATA_DECODE_TIMING=1`，请求 `GEN 8 9707 11 1246`。

```
单卡 tokens: 79790 316 52933 91 79751 198 262 9341
TP   tokens: 79790 316 52933 91 79751 198 262 9341     ← 八个全同
```

| 项 | 单卡 | TP |
|---|---|---|
| generated | 8 in 576 ms = **13.9 tok/s** | 8 in 635 ms = **12.6 tok/s** |
| decode ms/window | **96.02** | **105.85** |
| CPU experts | 15.34 | **15.34（一模一样）** |
| VRAM hits | 9.63 | 9.63 |
| PCIe | 0.76 | 0.76 |
| GPU-reach wait | 18.37 | 0.00（TP 未统计） |
| per-layer host | 68.30 | 0.00（TP 未统计） |
| stage (GPU) | 1.42 | 0.00（TP 未统计） |

`CPU experts 15.34` 两边**完全相同**，是"两卡没分担 CPU 池"的直接证据。

## 为什么 CPU 池是瓶颈（已量化）

- CPU 池是 AVX2 全核并行的：`ExpertPool::run_split_multi_native` → `run_phase(5/6)`，
  `host_works_` 让 host 也领任务（6 核：5 worker + host）。
- 实测带宽 **31.7 GB/s**（本机无 AVX-512，Zen 3 AVX2）。GPU 常驻是 **739 GB/s**，快 23 倍。
- 每层每窗口 CPU 专家 15.34 ms × 48 层 = 736 ms/window 的池工作量；
  GPU 侧（stage）只有 1.42 ms。

**结论：再给一张卡也不会让 CPU 池变快。** 唯一出路是把 miss 的专家搬上 GPU。

## rank 1 的显存没被用上（用户最初点名的问题）

启动日志：
```
strata serve: --tp: CUDA1 has 21484 MiB free; it loads its copy of the pack
strata generate: expert cache 5362 slots, 13.81 GiB of VRAM; policy is ...
```

- CUDA0：5362 槽 / 13.81 GiB
- CUDA1：只加载 pack 权重副本，**没有专家缓存**（`rg tps\.(cache|experts|slots)` 无命中）

运行中 GPU 占用也印证：CUDA0 约 21 GiB，CUDA1 约 300 MiB。

## 修掉过时注释（已改）

`src/program/generate.cpp:9602` 的注释写着
`// The experts are replicated: both cards run the full MoE.`
这是**错的**：MoE 已经按 n_ff 切分（`tp_moe_.n_ff_local = 320`，
`TpMoeSplit` + `moe_hq_exchange` + `moe_hit_stitch` 两相 gate/up→hq 交换→down）。
注释已更正，免得下一个开发者照着它做设计决策。

## 下一步（按优先级）

1. **给 rank 1 建专家缓存**，把 CUDA1 的 21 GiB 用来装专家。
   前提：`native_expert_grouped` 在 rank 1 上按 n_ff_local 半边跑时，专家布局与 rank 0 兼容
   （`TpMoeSplit` 的 half-blob 语义已验证）。目标是把 CPU miss 的比例从当前
   `24.58 entries/layer-window` 降到接近 0。
2. **两卡各自缓存自己的专家集合**，但仍必须保证 `solo_N == tp_N`（槽数相同才可比）。
   这一条和 1 有张力：缓存更多专家会改变命中集合，从而改变 token。
   记忆 [[tp-residency-changes-tokens]] 已确认：native NVFP4 下改槽位数 token 就变。
   **所以必须先解决"命中集合不影响输出"** —— 这正是本轮 GPU 专家内核在做的事。
3. 只有当 1+2 完成后，速度才可能真正超过单卡。

## 关键依赖：位一致的 GPU 专家内核

CPU 池与 GPU 路径原本算的不是同一个数，这就是"改 residency 就改 token"的根因。
本轮已交付 `nvfp4_q8_0_cpu_order.cu`，与 CPU 池**逐位一致**：

```
nvfp4_q8_0 rows : 0 / 132096 differ
nvfp4_q8_0 gu   : 0 / 66048 differ
grouped gu      : 0 / 6528 differ
grouped down    : 0 / 46080 differ
BIT-EXACT
```

速度差距已从 **15.05x 压到 1.26x（L2）/ 3.02x（DRAM）**。

**下一步必须把它接进 `native_expert_grouped` 的调用点**，这样增加 residency 不再改变 token，
第 1、2 步才成立。

## 复现方式（别在终端直接起模型，会后自动退出）

runner 用 `setsid` 完全脱离终端：

```bash
/tmp/detach_run.sh solo /tmp/p_solo    # 单卡
/tmp/detach_run.sh tp   /tmp/p_tp      # 双卡
# 等 /tmp/p_*.status 出现
```

`--adapt-every 0` 是必须的：默认设置下自适应换入有竞态，第 8 个 token 会漂成 `1992`
（记忆 [[token8-adapt-race-not-tp]]）。`9341` 才是确定性 token。
