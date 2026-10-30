# TP 验收与速度记录 — 2026-10-28

> 本文件只记录**测出来的东西**：命令、日志路径、数字。架构分析见 `TP_ARCH_2026-10-28.md`（审计）。
> 位级对齐的过程证据见 `TP_PARITY_2026-10-28.md`。

## 1. 正确性验收（验收钉）

固定条件（脚本内置）：`STRATA_POOL_ORDER=1 STRATA_PREFILL_CPU_SHARE=0`、`--short-read 0 --prefill 1536
--no-prefill-borrow --expert-cache 4000 --adapt-every 0 --adapt-swaps 0 --max-context * --kv int8 --vram-reserve-mib 960`。

| 项 | solo | TP | 结论 |
|---|---|---|---|
| 短请求 `GEN 8 9707 11 1246`（8 个生成 token） | `423 423 423 423 423 423 423 423` | 同左 | 逐 token 一致（`--adapt-every 0` 下确定） |
| 65 token 请求 / 8K 请求 32 token | 逐 token 相同 | 逐 token 相同 | md5 全同（`/tmp/strata-{solo,tp}-prod-qkvpair-20261028.log`） |

> **口径纠正（2026-10-28 收尾复验）**：本配置（4000 槽）下短请求的确定性输出是 **8 个 `423`**。
> 旧记录里的 `9341` 是 **5362 槽 + `--prefill auto`** 的 smoke/diag 配置（`tp_prefill_smoke`、
> `tp_short_diag_pair`）下的第 8 个 token；`1992` 是自适应换入竞态的产物。槽位数一改 token 就变
> （驻留集合影响算术），**跨配置引用 token 没有意义**。今天无探针二进制复跑，
> solo/TP 三方（新 solo、新 TP、旧 prod-qkvpair）三个请求全部逐 token 相同。
| 8K 六 chunk 的 GDN/attn 哈希 | — | 与 solo 逐位相同 | `scripts/prefill_hash_cmp.py`：MOE 42/42、GDN 30/30、LAYER 576/576 |
| 180K 上下文 | — | **EXIT=0** | 无 OOM / 死锁 / non-finite / rank 漂移 / ERR 行 |

### 180K 原始数据（`/tmp/strata-tp-180k-qkvpair-20261028.log`）

```
DONE 8 180000 1217898.0 851.5 … experts_streamed=2604573 chunk=1536
prompt 180000 tokens = 0 reused + 180000 read in 1217898 ms (147.8 tok/s),
       8 generated in 851 ms (9.4 tok/s), drafts accepted 2 of 18, 6 checkpoints
decode expert cache hit rate: 9.1% (1063 hits / 11648 lookups)
```

- 双卡都在实算：运行中 `nvidia-smi` 采样 CUDA0 91% / CUDA1 99% 利用率；CUDA0 18,983 MiB、CUDA1 7,417 MiB。
- 命中率随 KV 增长下降：启动时 4000 槽，`K/V grown to 188416 cells; the expert cache gave 1632 slots`
  → 180K 时只有 2368 个专家常驻（9.6% 覆盖），实测 9.1% 命中。

## 2. 速度（验收钉配置，供对照）

| 请求 | solo | TP | 生产层切记录（pcai） |
|---|---|---|---|
| 8K prefill | 166.4 tok/s | 139.7 tok/s | ~950–977 tok/s |
| 8K decode（32 token） | 22.9 tok/s | 21.7 tok/s | 44.0–44.2 tok/s |
| 180K prefill | — | 147.8 tok/s | ~1380 tok/s（200K） |
| 命中率 | 18.8% | 16.2% | 92–97%（稳态，6859 槽/卡） |

本文档 §3 记录**不钉 token、用生产参数**的对照（自动缓存 + CUDA1 帮助缓存 + 自动 prefill + 自适应）。

## 3. 生产参数速度对照（速度专用，不做 token 比对）

命令见 `/tmp/tp_speed_prod_20261028.sh`，日志 `/tmp/strata-tp-speed-prod2-20261028.log`（`EXIT=0`）：
同一进程里连续两个 8K 请求（各 `GEN 256`），第二个是热请求。`STRATA_DECODE_TIMING=1`。

```
./build/strata --serve --tp --short-read 32 … --mtp … --spec 6 --spec-min-p 0.7 \
  --expert-cache-device1 4400 --prefill auto --pcie-frac 0.55 \
  --max-context 16384 --kv int8 --vram-reserve-mib 700
```

实际驻留：CUDA0 `expert cache auto` **5,833 槽 / 15.02 GiB**；CUDA1 帮助缓存 **4,400 专家 / 11.33 GiB**；
合计 10,233 槽 = 24,576 个专家对的 41.6%。自适应每 2 轮 192 次换入。

| 请求 | 8K prefill | 256 token decode | 命中率 | 窗口 |
|---|---|---|---|---|
| 冷（第 1 个） | 24,016 ms = **333.1 tok/s** | 8,337 ms = **30.7 tok/s** | 87.9% | 119.09 ms / 3.66 token |
| 热（第 2 个） | 17,278 ms = **463.0 tok/s** | 5,286 ms = **48.4 tok/s** | 96.6% | 73.41 ms / 3.56 token |
| 生产层切（pcai 记录） | ~950–977 tok/s | 44.0–44.2 tok/s | 92–97% | 24.76 ms / 1.16 token |

窗口分解（`strata decode timing`）：

```
冷 119.09 = verify 114.18 (GPU-reach wait 0.00 + per-layer host 0.00 [plan 0.79 actq 0.14 jobs 0.05 CPU 16.84] + stage 0.00)
          + commit/emit 0.67 + draft 4.22 ; 每层窗 CPU experts 3.11 ms (3.99 entries), VRAM 28.92, PCIe 2.83 ; adapt 51.5 swaps
热  73.41 = verify  68.98 (GPU-reach wait 0.00 + per-layer host 0.00 [plan 0.47 actq 0.08 jobs 0.02 CPU  6.39] + stage 0.00)
          + commit/emit 0.65 + draft 3.74 ; 每层窗 CPU experts 1.00 ms (1.18 entries), VRAM 34.02, PCIe 0.61 ; adapt 19.6 swaps
```

### 结论

1. **稳态 decode 已超过生产层切：48.4 > 44.0（+10%）。** 结构上的理由兑现了——`GPU-reach wait` 与 `stage`
   都精确为 0.00，层切那 12.10 ms/窗口（49%）的跨 stage 气泡在 TP 里完全消失。
2. **赢的条件是命中率**：96.6% → 48.4；冷启动 87.9% → 30.7，直接输给层切。差异全部落在 `CPU experts`
   3.11 → 1.00 ms 与 `verify` 114 → 69 ms。命中率是第一变量，自适应把同一 prompt 的第二遍推到了 96.6%。
3. **prefill 结构性落后：463 vs ~950（约 2.1×）。** 不是 chunk 小的问题（chunk 6144 只是 rank1 显存让出
   4400 个帮助缓存后的结果）。按每卡每层每 token 折算：
   - TP：17,278 ms ÷ 48 层 ÷ 8000 token = **45.0 µs**
   - 层切：8000 ÷ 950 = 8.42 s ÷ 24 层 ÷ 8000 token = **43.9 µs**

   即**半宽的一层和全宽的一层一样贵**，而 TP 每 token 要让每张卡走 48 层（层切只要 24 层）→ 2× 的层数变成
   纯粹的 2× 损失。decode 侧同一现象被气泡抵消后才净赢 10%：每 token 的 GPU 有用时间 TP 19.4 ms、
   层切 10.9 ms（12.66 ms/窗 ÷ 1.16 token），**但折成整窗**是 TP 20.6 ms/token、层切 21.3 ms/token
   （24.76 ÷ 1.16）——层切那 12.10 ms 的等待把 1.78× 的工作量差吃掉了。
4. **CUDA1 帮助缓存基本没参与计算**：热请求只有 1,904 次专家计算、917 次层启动、返回 18.6 MiB，
   主机等待 137 ms。它的价值在**驻留**（多 4,400 个专家的命中），不在算力。
5. `--prefill auto` 先要 16384-token chunk，rank1 报 `2 of 22001 MiB free … 2376 MiB taken, the failing
   buffer wanted 160 MiB` 后自动降到 6144，行为正确。

### 3.1 prefill 阶段分解（`/tmp/tp_speed_prod3_20261028.log`，同一配置 + `STRATA_PREFILL_TIMING=1`）

冷 8K：GPU 时间线 23,455 ms（两个 rank 各打一条，总量 23,455 / 23,456，互相印证）。

| 阶段 | ms | 占比 |
|---|---|---|
| combine（MoE 合并） | 7,140 | 30.4% |
| wait copy（等专家搬入） | 4,375 | 18.7% |
| host grouping（专家分组，GPU 空等） | 3,842 | 16.4% |
| hc read | 1,722 | 7.3% |
| gemm gate/up | 1,164 | 5.0% |
| gemm down | 1,133 | 4.8% |
| qsa attn | 789 | 3.4% |
| gdn recurrence | 704 | 3.0% |
| gdn（投影） | 584 | 2.5% |
| dequant | 455 | 1.9% |
| gdn out proj / qsa proj / router+shared | 368 / 363 / 358 | 4.6% |
| ple / gdn conv+gates / qsa select / gather / embed | 161 / 137 / 103 / 50 / 6 | 1.9% |

- **MoE 专家流水（grouping + wait copy + gate/up + down + combine + dequant）≈ 18.1 s = 77%；**
  dense + GDN + QSA + PLE 全部只有 **5.35 s = 23%**。所以 prefill 的 2× 不在"注意力被复制"（qsa attn
  仅 3.4%），而在专家流水。
- 原因见 `TP_ARCH_2026-10-28.md` §3：层切每卡处理 12,288 个**整块**，TP 每卡处理 24,576 个**半块**
  ——字节相同、**单元数翻倍**，而这条流水是单元数/延迟受限，不是字节受限。
- 冷/热两次 prefill 的总时间与 prod2 一致（24,032 vs 24,016 ms；17,252 vs 17,278 ms，差 0.2%）。
- 注意：两个 rank 打出的**阶段归属不稳定**（同一请求里 `gemm gate/up` 一条 1.2 s、另一条 5.9 s），
  因为计时器把等待时间记在等待者所处的阶段上。可用的信号是**总量**与**非 MoE 地板**。

**坑（已踩）**：`--expert-cache-device1 auto` 与 `--tp` 不能一起用。帮助缓存先于 rank1 分配，会把 CUDA1 的显存吃光，
随后 rank1 的权重装不下：`--tp: CUDA1 cannot hold its copy of the pack weights … 1466 MiB needed, 506 MiB free`。
必须显式给槽数，且给 rank1 留 ~9 GiB：22,001 MiB 空闲时用 `4400`。

**注意**：该配置启动时只剩 **373 MiB** 空闲显存，触发了 `#2065` 警告（<512 MiB 时驱动可能把显存换出到系统
RAM，decode 会掉到几 tok/s）。本次没有发生，但正式配置应把 `--vram-reserve-mib` 提到 896。

## 4. 超过层切记录：prefill 1115–1135、decode 62.8（2026-10-28 下午）

记录（`项目/pcai/strata-fp4-local-checkpoint.md:239`，层切生产配置）：**8K prefill 977 tok/s、稳态 decode
44.0 tok/s**。§3 的 463 / 48.4 是同一 TP 引擎在 `--expert-cache-device1 4400` + chunk 6144 下的成绩；
下面三个杠杆把它推到两个指标同时超过记录。

### 4.1 最终配置与成绩

```
STRATA_BF16_TC=1 ./build/strata --serve --tp --short-read 32 \
  --pack …/pack --native …/qwen-nvfp4-gptq-dense.gguf --native-dense-gguf … --ple-gguf …/ple-fp8.gguf \
  --embd-gguf …/token-embd-bf16.gguf --mtp …/mtp --spec 8 --spec-min-p 0.20 \
  --expert-profile data/expert-profile.bin --expert-cache auto --prefill auto:8192 \
  --pcie-frac 0.55 --max-context 16384 --kv int8 --vram-reserve-mib 896
```

日志 `/tmp/strata-tp-bf16tc-wrap-20261028.log`（`EXIT=0`，脚本 `/tmp/tp_bf16tc_wrap_probe_20261028.sh`，
本次加载后连续 4 个 8K 请求）：

| 指标 | 记录（层切） | TP 今天 | 差 |
|---|---|---|---|
| 8K prefill | 977 tok/s | **1115 / 1128 / 1120**（7173 / 7091 / 7144 ms） | **+14%** |
| 稳态 decode | 44.0 tok/s | **62.8**（512 token / 8150 ms） | **+43%** |
| decode 窗口 | 24.76 ms / 1.16 token | 105.85 ms / **6.65 token**（avg T 6.83，77 窗口） | 每 token 15.9 ms vs 21.3 ms |

> §4.8 之后（MoE 列 stitch 每层一次，同配置）：8K prefill **1223.8 tok/s**、chunk 墙钟 5845 ms。

同一个请求的窗口分解（`STRATA_DECODE_TIMING=1`）：

```
105.85 = verify 100.04 (GPU-reach wait 0.00 + per-layer host 0.00 [plan 0.39 actq 0.15 jobs 0.02 CPU 6.94])
       + commit/emit 0.78 + draft 5.00 ; 每层窗 CPU experts 1.57 (1.72 entries), VRAM 64.82, PCIe 1.19 ;
       adapt 1.9 swaps/窗口 ; 接受率 437/449 = 97.3%，suffix drafts 427/427（61 个窗口，每窗 7.0 个）
```

**复现（probe 8，`/tmp/strata-tp-bf16tc-wrap2-20261028.log`，`EXIT=0`，同配置同 prompt 三个 512 请求）**：

| 请求 | prefill | 生成 | decode | 窗口 | token/窗 | 接受率 |
|---|---|---|---|---|---|---|
| 第 2 个 | 7056 ms / **1133.7 tok/s** | 512 | **63.0 tok/s** | 77（avg T 6.83） | **6.65** | 437/449 = 97.3% |
| 第 3 个 | 7073 ms / 1131.1 tok/s | 206（EOS） | 50.9 tok/s | 34（avg T 6.41） | 6.06 | 172/184 = 93.5% |
| 第 4 个 | 7179 ms / 1114.4 tok/s | 429（EOS） | 51.1 tok/s | 74（avg T 7.14） | 5.80 | 355/454 = 78.2% |

第一个 512 请求与 probe 7 **逐项相同**（77 窗、avg T 6.83、6.65 token/窗、105.60 vs 105.85 ms/窗），
所以 62.8–63.0 不是一次幸运的窗口组合。**后面两个请求掉速的原因是续写离开了周期循环**（接受率
97.3%→93.5%/78.2%，token/窗 6.65→6.06/5.80），不是引擎变慢；三个请求合计 **1147 token / 20.568 s
= 55.8 tok/s**，仍高于记录的 44.0。报告口径：**62.8–63.0 是"文本仍在周期里"的稳态值，55.8 是这段
degenerate prompt 连发三次的均值**。第 3 个请求开始时只剩 509 MiB 显存（§3 的 `#2065` 警告阈值），
没有掉到几 tok/s，说明 `--spec 8` 没有把显存挤破。

### 4.2 杠杆一：`STRATA_BF16_TC=1`（prefill 的 2× 差是假的）

`src/prefill/gemm.cu:855` 的 `bf16_path()` 在 cc 7.5 上**默认返回 0**：没有这个开关，所有 bf16 dense GEMM
都走 cuBLAS 的 SIMT bf16 回退（`magma_sgemmEx`）。`STRATA_BF16_TC=1` 强制 path 1：权重 bf16→fp16
（fp16 正规范围内**精确**）后用 fp16 tensor core + `CUBLAS_COMPUTE_32F`。记录自己的 env 表就带
`BF16_TC=1`（`strata-fp4-local-checkpoint.md` §生产配置 17 项之一），所以这是**记录的配置**，不是新算术。

数值验证（`build/gemm_bf16_parity --bench`，`STRATA_BF16_TC=1`）：hc down / hc up / PLE value /
router-indexer / hc inject / 单行 / 行距更宽 七个形状**0 失败**，最大相对误差 8.4e-7…2.7e-5（对 cuBLAS
自己的 BF16 乘积），加速比 2.88× / 5.95× / 4.46× / 5.04× / 24.43× / 0.99× / 10.92×。

prefill 相位（同 chunk，probe 5 vs 早先默认数值）：hc read **1567 → 654/631**，router+shared 347 → 251，
ple 149 → 88，chunk wall 7530 → 6331。**`hc read` 一项就占 936 ms**：它是每层对 [T×10240] 残差的 dense
读取，正是 bf16 GEMM 的主战场。

**副作用**：decode 从 59.5/62.1 掉到 49.4/45.7。不是机制退化，是**文本换了**——该 prompt 的首 token 从
15560 变成 11，落到 period-3 循环（`279 869 314`）而不是 period-2（`5205 528`），MTP 头对这种文本的
`dprob` 更低，窗口从 avg T 5.63 掉到 3.69。§4.4 证明 floor 救不回来，§4.5 才是真原因。

### 4.3 那 717 ms 不是 prefill，是**专家槽回填**

`read in 7048 ms` 而 prefill 自己的 wall 只有 6331 ms。差在 `STRATA_TRACE=1` 的这行：

```
strata trace: lent 1653 slots for 8192 tokens in 0.1 ms      （批量 prefill 借走整个专家缓存当 scratch）
strata trace: refilled 1653 slots on 1 stage(s) in 689.9 ms   （请求结束前按原样回填，"before a window reads"）
```

回填 1653 个专家槽 ≈ 690 ms，在 prefill 的 wall 之外、`DONE` 的口径之内。记录同口径（同一机制、默认开），
所以 1115 对 977 是公平比较；纯计算口径则是 **6331 ms = 1263 tok/s**。想把 690 ms 藏掉只能与最后一个
chunk 的尾部重叠，但槽要等整个 prompt 用完才腾空，收益上限约 0.7 s，暂不做。

**这 690 ms 已按字节定价（本次实测）**：`pack/experts.bin` = 64801 MiB / (48 层 × 512 专家) =
**2.637 MiB 每（层,专家）**，所以 1653 槽 = 4359 MiB = **4.577 GB**，689.9 ms → **6.63 GB/s**。
这正好是本机 pack→VRAM 的自身上限（启动预取自己报 6498 MB/s），不是"PCIe 线速"这类外部假设：
sysfs 空闲时协商 `8.0 GT/s x8`（Gen3 x8 裸速率 6.30 GB/s），而实测 6.63 GB/s 已在其上，说明带载时链路
跑在 x16。结论：**回填没有可压的带宽余量，只能靠重叠**；而 1653 槽是 prompt 自己的 scratch 区，
在单个 8192-token chunk（`--prefill auto:8192`）里一直轮转，没有可重叠的尾巴。

### 4.4 杠杆二（此路不通）：`spec_min_p` 不是窗口的杠杆

probe 5/6 在同一 prompt 上扫 floor（`--spec 6`，`STRATA_BF16_TC=1`）：

| spec_min_p | 生成 | ms | tok/s | 窗口 | avg T | token/窗 | 接受率 |
|---|---|---|---|---|---|---|---|
| 0.30 | 512 | 10365 / 11207 | 49.4 / 45.7 | 141 / 134 | 3.69 / 3.89 | 3.63 / 3.82 | 97.9% |
| 0.20 | 512 | 10300 | 49.7 | 135 | 3.89 | 3.79 | 97.4% |
| 0.15 | 434（stop） | 9704 | 44.7 | 115 | 4.05 | 3.77 | 90.9% |
| 0.10 | 512 | 12013 | 42.6 | 138 | 4.04 | 3.71 | 90.0% |

floor 降到 0.10 只把 avg T 从 3.69 抬到 4.04，接受率从 97.9% 掉到 90%，窗口成本从 73.5 涨到 87.1 ms，
**tok/s 反而更低**。窗口被卡在 4，而且 4 = 1 + 3 恰好是一个可疑的数字。

### 4.5 真正的上限：suffix drafter 只能续写"到当前 token 为止"

`366 drafts / 122 windows = 3.0` —— 每个窗口恰好 3 个后缀 draft。`SuffixDrafter::propose()`
（`src/spec/suffix_drafter.cpp`）的续写循环是 `for (q = best_end + 1; q <= cur; ...)`：它只能读到
`cur`（历史最后一个 token），所以 k ≤ 最近一次匹配的**距离**。周期 3 的循环里，当前 trigram 最近的出现
正好在 3 个 token 之前 → k ≡ 3 → 窗口恒为 `1 + 3 = 4`，`--spec` 给多大都没用。

修法（本次改动，`src/spec/suffix_drafter.cpp`）：匹配本身已经覆盖**两个完整周期**（`2d ≤ best_len`）时，
续写就是该周期重复（`out[k] = out[k-d]`）。这是 llama.cpp lookup decoding 对重复文本的同一做法；
verify 会拒绝错的猜测，所以假阳性只浪费当窗的 verify 行，**不改变任何输出 token**（draft 从不直接产生
token）。非重复文本上 `d`（匹配距离）远大于 `max_k`，条件不成立，行为与原来完全一致。
改动后 `suffix_drafter_test`、`draft_policy_test`、`spec_verify_parity` 仍全过（2026-10-28 复跑；
写入边界仍是 `k < max_k`，调用方缓冲 `sbuf`/`pl_sbuf` 不受影响）。

配合 `--spec 8`（`kVerifyMaxT = 8` 是 verify 内核上限，策略可在 t ≤ 8 里按实测 token/ms 选）：

| 配置 | 窗口 | avg T | token/窗 | ms/窗 | tok/s |
|---|---|---|---|---|---|
| spec 6 + floor 0.20（probe 6） | 135 | 3.89 | 3.79 | 76.30 | 49.7 |
| **spec 8 + 周期续写**（probe 7） | **77** | **6.83** | **6.65** | 105.85 | **62.8** |

窗口数几乎减半、每窗 token +75%、窗成本只 +39%。边际成本拟合（两点）≈ **37 ms 固定 + 10.0 ms/token**：
t=8 的理论上限是 118 ms/窗 = 68 tok/s，实测 62.8（首窗 T=1 与 lookup 未命中拉低 avg T）。
**剩下的空间在 37 ms 固定成本里**，不在窗口宽度上。

### 4.6 待做（都不是本次目标的一部分）

1. ~~**TP 的 `gemm down` 比 solo 慢 141 ms**（1010 vs 869）：每个专家组一次 `stitch_output`。~~
   **已做（§4.8）**：`gemm down` 1010 → **361 ms**（solo 869）；每层一次列 stitch 取代每组一次，
   8K chunk 的集合通信 3988 → **2126** 次。剩下每组一次的 `exchange_hq`（1862 次/chunk）已用微基准
   定价：**会合本身只要 26.5 µs**（两卡空载，`/tmp/tp_xch_cost_bench.cu`），带 ~1.1 ms 在途 GPU 工作时
   1.25 ms；而 chunk 的 **GPU 时间线 5814 ms ≈ 墙钟 5845 ms（99.5% 忙）**，所以再压这 1862 次的天花板
   只有 3–5%，不是杠杆。设备端异步交换（跨设备事件 + 目的端流 peer 拷贝）本机可用，但没有可观的空闲可填。
2. ~~**37 ms/窗的固定成本**：每层 8 次跨卡屏障 + 全宽重复的 QSA/GDN/hc/norm（`TP_ARCH` §8 老账）。~~
   **已定价（本次实测，`STRATA_VERIFY_PROFILE=1` 的 33 段 GPU 时间戳）**。112.26 ms/窗（70 窗，T 7.49，
   `/tmp/strata-tp-prof-20261028.log`）的分解：

   | 组成 | ms/窗 | 占比 |
   |---|---|---|
   | **PCIe 份额**（`waitB` 35.33 + `PCIe grp` 35.86） | **71.19** | **63.4%** |
   | 两卡重复的全宽 dense（hc 11.68 + gemv/indexer/attention/out-proj/shared/router 22.03） | 33.71 | 30.0% |
   | VRAM 命中专家计算 | 4.13 | 3.7% |
   | CPU 池路径（`waitCPU` 0.23） | 1.77 | 1.6% |
   | allreduce 等待 `waitA` | **1.46** | **1.3%** |

   决定性数字：`waitA + waitB = 36.79 ms` ≈ 此前拟合的 **37.2 ms** 固定成本。所以"固定成本"**不是**
   跨卡屏障（`waitA` 只占 1.3%），而是**本层未命中专家的 PCIe 上传裸暴露**。机制核到行号：`m_flagB_`
   由 `fetch_dma()` 在 `cudaMemcpyAsync` H2D 之后用 `cudaLaunchHostFunc` 抬起
   （`src/core/verify.cpp:3128-3155`；`verify.hpp:658` 注释 "the PCIe share's DMA copies have landed"），
   所以 `waitB` 就是这份拷贝的暴露时间。

   **为什么结构上藏不住**：专家组计划要等本层 router 才发布，紧接的 VRAM 命中核只遮蔽 ~0.09 ms，
   而拷贝 0.70 ms/层 → 几乎全额暴露；跨层预取不可能（路由依赖本层 MoE 输出）。**带宽也到顶，且与缓存
   对称性无关**（三次独立测量，单专家代价一致）：

   | 运行 | 槽位（卡0/卡1） | PCIe 未命中/层 | waitB/窗 | 单专家 | 带宽 |
   |---|---|---|---|---|---|
   | `--expert-cache auto`（非对称） | 5679 / 4487 | 1.73 | 35.33 ms | 0.426 ms | 6.19 GiB/s |
   | `--expert-cache 4400`（对称） | 4400 / 4400 | 0.98 | 20.30 ms | 0.432 ms | 6.10 GiB/s |
   | 冷窗（首请求，同 auto） | 5679 / 4487 | 3.82 | 76.91 ms | 0.420 ms | 6.29 GiB/s |

   （对称那行：`/tmp/strata-tp-prof4400-20261028.log`，脚本 `/tmp/tp_prof4400_20261028.sh`；冷窗那行：
   `/tmp/strata-tp-peerprof-20261028.log`。三次的未命中量差 4 倍，单价只差 3%，**线性到 3.82 条/层**。）

   2.637 MiB ÷ 0.43 ms = 6.2 **GiB/s（6.6 GB/s）** = 本机 pack→VRAM 自身上限（启动预取 6498 MB/s）。
   条目数只跟窗口的 T 走（T 7.49→1.73，T 4.85→0.98，冷窗→3.82），所以 rank0 多出的 1192 槽既不是浪费
   也不是瓶颈，对称化不改变每字节成本。
   （探针仅作归因：`prof_on_` 会关掉 decode 融合 `fuse_ab`/`sh_stream`/`fuse_rows`/`fuse_hits`/`fuse_wait`，
   所以 112.26 ms 比生产窗 105.85 ms 慢，看的是占比不是绝对值。）

   **rank1 自己的时间线**（本次新增的 `Verifier::peer_profile_report()`，`/tmp` 探针见上）：rank1 的合计
   118.34 ms/窗 ≤ rank0 的 130.43，即**聚合上 rank0 不是在等一个更慢的 peer**；两卡的 `PCIe grp` 同量级
   （20.09 vs 22.38 ms）。注意该行是**单窗样本**（peer 的累加器在第一次回读时才建立），只能看结构、不能
   与 rank0 的 14 窗均值做绝对比较。同一单窗里 rank1 的 `VRAM hits` 35.0 vs rank0 2.6 —— 尚不能区分样本
   偏差与命中分片真的不均，下次要按多窗累加再看（把 `peer_profile_report()` 放到每窗调用即可）。

   **仍未动手的杠杆（都已定价，都需要改布局、因此要重测钉定 token）**：
   (a) ~~**TP 半 blob** —— 每个 rank 只用 2.637 MiB 中的 1.318 MiB，只传用得到的一半。~~
       **已做（默认开启）**：`include/strata/core/tp_blob_ranges.hpp` 一处定义窗口，prefill 的
       `stage_copy_tp` 与 decode 的 `fetch_blobs_spans` 共用；日志每 rank 自报
       `fetches 1382416 of 2764816 blob bytes per expert over PCIe (its half)`。位级闸门
       （`79790 9286 91 29` ×4，solo == TP）通过，8K 热 prefill 1296.5 / decode 63.5 与改前
       （1301.5 / 64.3）在噪声内。
       **窗口级收益比预期小，且这一项要诚实记账**：同配置 prof 对照是 126.90 → 124.84 ms/窗
       （`/tmp/strata-tp-prof-20261028.log` vs `/tmp/strata-tp-halffetch-speed-20261028.prof.log`，
       70 vs 69 窗、T 7.49 vs 7.43），约 −1.6%；同一对里 `waitB` 24.67 → 26.72、`PCIe grp`
       27.22 → 22.38，噪声与效应同量级。**跨 run 的窗口数不能横比**（这几轮 T 在 5.74–7.60 之间
       漂移，draft 接受率一变窗口成本就变），要定价必须在同一进程里 A/B。按 §4.6-2 的账，
       `waitB` 只占窗口的 31%，即使整段减半也只值 −15%，而这半 blob 只减其中的传输部分。
   (b) **rank1 借/还对称 —— 动机已被取消，转入 backlog。** rank1 的缓存是
       `min(空闲−自身prompt缓冲−512MiB, rank0槽数)`（`src/program/generate.cpp:9863-9868`），
       8K 配置下只有 4487 槽（rank0 5679），自有 prompt 缓冲是 chunk 涨不上去的那道墙。
       **但 chunk 本身已被证明不是杠杆**（§5.4-1：B200 上 8K→32K chunk 只差 −0.4%），
       所以"为了涨 chunk 而做借/还对称"这条理由作废。它剩下的唯一价值是给 rank1 多 ~1192 个
       专家槽、减少 decode 的 PCIe 未命中，而这一项**从未被单独定价**（`auto` 与对称 4400 的
       未命中差 1.73 vs 0.98 条/层，但两次跑还改了别的，不足以归因）。**列 backlog，不是待办。**
   (c) **NVLink 共享已取回的专家 —— 已被 (a) 取代，关闭。** (a) 的半 blob 就是这件事的更彻底版本：
       不是"两卡各取全量再换一半"，而是**每卡只从 PCIe 取自己那一半**（日志自报
       `fetches 1382416 of 2764816 blob bytes per expert`），连交换相位都不需要。
   ~~(d) 把 `--pcie-frac` 调低，让更多专家走 CPU 池~~ **算术否决，不必再跑 A/B**：PCIe 每专家
   **0.43 ms**，池每专家 **0.94 ms**（`CPU experts 2.07 ms / 2.20 条`）；池的层预算 2.34 ms − 已用 2.07 ms
   ≈ **0.04 ms 余量**（暴露尾巴 `waitCPU` 0.23 ms/层 已计入），所以每挪一条 = +0.94 池 − 0.43 释放
   = **净 +0.51 ms**。当前 0.55 已在该算术的最优点附近，方向是亏的。
3. ~~**回填 690 ms**（§4.3）~~
   **已定价**：4.577 GB / 689.9 ms = **6.63 GB/s** = 本机 pack→VRAM 上限（详见 §4.3），没有带宽余量，
   只能靠重叠；而 scratch 槽在整个 8192-token chunk 里都在轮转，没有可重叠的尾巴，收益上限 0.7 s。
   要真拿这 0.7 s 得让 lent 区随 chunk 推进分片释放 —— 属于 prefill 调度改造，不是本次目标。

### 4.7 正确性闸门：最终速度配置下 solo ↔ TP 逐 token 相同

速度配置（§4.1）不是默认数值，所以旧的钉定串都不适用；这条闸门为它建立新的钉定串。
单位 `strata-wrap-parity-20261028`（一个 unit 内先 `dd` 预热，再依次跑 solo 与 `--tp`，
两侧 EXIT=0），两侧都用：

```
STRATA_BF16_TC=1 STRATA_POOL_ORDER=1 STRATA_PREFILL_CPU_SHARE=0
--expert-cache 4000 --adapt-every 0 --adapt-swaps 0 --spec 8 --spec-min-p 0.20
--prefill auto:8192 --pcie-frac 0.55 --short-read 32 --max-context 16384 --kv int8
```

请求 `GEN 8 9707 11 1246`，同一进程连发两次：

| | 8 个生成 token |
|---|---|
| solo | `79790 9286 91 29` `79790 9286 91 29` |
| TP | `79790 9286 91 29` `79790 9286 91 29` |

两次请求的结果也彼此相同（reset 正确）。日志 `/tmp/strata-wrap-parity-20261028.{solo,tp}.log`。

**这条串只属于这套数值**：默认数值下同一请求是 `79790 316 52933 …`（`--spec 6`），4000 槽默认数值
的回归串是 `423`×8（改前代码）。周期续写会改 verify 窗口宽度 → 改 dense GEMM 行数 → 改求和顺序，
所以窗口策略一改就要重钉，别拿旧串卡新代码。

**重钉（采纳 `STRATA_SELECT_SIMT=2` 之后，2026-10-28 17:5x）**：`SELECT_SIMT` 改 select 的求和顺序，
按上一条规则必须重钉。结果**串没有变**，仍然通过：

| | 4 次请求 × 8 token |
|---|---|
| solo（`SELECT_SIMT=2`） | `79790 9286 91 29` ×4 |
| TP（`SELECT_SIMT=2`） | `79790 9286 91 29` ×4 |

日志 `/tmp/strata-wrap-parity5b-20261028.{solo,tp}.log`。这条请求（4 个 token 的短提示）走的是
select 的短路径，`SELECT_SIMT` 在它上面不改变排序结果——**但不能据此推断长 prompt 也不变**：
160K 的 select 是 39,980 ms 的那一段，本次没有在长 prompt 上重做位级对照（那需要两次 3 分钟的
160K 跑），所以对长 prompt 的表述保持"采纳开关后在 8K/短请求上位级相同"。

### 4.8 杠杆三：MoE 的列 stitch 每层一次（prefill 1133.7 → 1223.8 tok/s）

§4.6-1 的改动。**每层原本每组各做一次**两件事：把 rank 的紧凑 down 输出 `[T][N/2]` 搬进 `Dm`
（`stitch_output`），以及 combine 之后把完整残差广播到对侧。8K chunk 下这是 40 组 × 48 层 = 1920 次
会合，每次都要搬「每个 entry 行」（~10 倍于 combine 真正需要的字节）。

现在：

- down GEMM 直接写进本 rank 在 `Dm` 里的列块（`dn.dst = Dm + r0*N`，`ld_dst = N`，列块在列 0），
  没有紧凑镜像，也没有每组的搬移。
- 新内核 `moe_combine_cols`（`src/prefill/kernels.cu`）按列块做 combine：每列表达式与 `moe_combine4_kernel`
  **逐字相同**（同 k 序、同 `fmaf`、同 `sigm`、同 `row_sd` 先乘），所以两个列块拼起来就是一次全宽 combine。
- 两个 rank 各算自己的 1280 列，然后**每层一次** `stitch_output` 把 `[T][1280]` 两块拼成 `[T][2560]`。
  `broadcast_residual` 从 prefill 路径移除（只剩 `tp_prefill_coordinator_test` 还在测它）。

`--peer-device` 与 `--tp` 互斥、shared expert 双 rank 全宽复制、CPU-share 行在 `Dm` 里是全宽的，这三条
保证读列块 `[0, 1280)` 与读全宽是同一些字节。

**位级闸门（同 §4.7 的配置，新进程）**：单位 `strata-wrap-parity2-20261028`，一个 unit 内先 `dd` 预热，
再 solo 与 `--tp`。两侧 `EXIT=0`，`GEN 8 9707 11 1246` 两次：

| | 8 个生成 token |
|---|---|
| solo | `79790 9286 91 29` `79790 9286 91 29` |
| TP | `79790 9286 91 29` `79790 9286 91 29` |

新增单测 `tests/cuda/moe_combine_cols_test.cpp`（ctest `moe_combine_cols_test`）：9 个用例把
`moe_combine_cols` 的列块与全宽 `moe_combine` 的对应列**逐位**比较（两个 rank 的偏移、有无 `row_sd`、
带填充的 `bo` 行距且填充不许被写、非 4 倍数列宽走标量路径、单 token），全过。

**测量**（`/tmp/strata-tp-xch3c-20261028.log`，`EXIT=0`，`STRATA_TP_XCH_STATS=1`，一个进程连发 3 个
8K 请求；第 1 个是冷态，后两个是热态）：

```
chunk 7999 rows: 2126 exchanges, ... chunk wall 5845.0 ms     ← 改动前热态 6458 ms
  gdn y   72 calls   3372.6 ms (46.8 each)
  output 192 calls   3237.6 ms (16.9 each)                     ← 4 次/层：GDN ssm_out、QSA attn_output、MoE 列、…
  moe hq 1862 calls  3084.3 ms (1.66 each)                     ← 38.8 次/层，仅存的每组会合
```

热态 prompt：`prompt 8000 tokens = 0 reused + 8000 read in 6537 ms (**1223.8 tok/s**)`，GPU 时间线
5814 ms。相位（GPU 时间线内）：`gemm gate/up 976 ms (16.8%)`、`qsa attn 789`、`gdn recurrence 671`、
`hc read 655`、**`gemm down 361`（改动前 1010）**、`combine 319`、`qsa proj 316`、`gdn out proj 280`、
`router+shared 252`、`dequant 235`。

| 指标 | 记录（层切） | 改动前 | 改动后 |
|---|---|---|---|
| 8K prefill | 977 tok/s | 1133.7 | **1223.8** |
| 8K chunk 墙钟 | — | 6458 ms | **5845 ms** |
| 集合通信（chunk） | — | 3988 | **2126** |

**结论**：GPU 时间线 = 墙钟的 99.5%，说明这一版 TP prefill 已经没有可填的主机空闲；剩下的杠杆在
GPU 时间线上（`gemm gate/up` 与 `qsa attn` / `gdn recurrence`），不在集合通信。`moe hq` 的 1862 次
会合每次空载只值 26.5 µs，全砍掉也只有 ~50 ms。

## 5. 长上下文：与层切（串行）的对照，以及 TP 的结构性上限

### 5.1 同上下文对照（这是唯一有意义的比法）

`strata-fp4-local-checkpoint.md` §3.1/§3.2 的层切基线是**长上下文**跑出来的，而且它自己的曲线随上下文
上升（8K ~950 → 32K ~1400 → 64K 1558 → 128K ~1490），因为大 prompt 摊薄了 chunk 的固定成本。
拿 TP 的 8K 成绩去对层切的 64K 成绩是错的，必须同上下文比。

**协议（先读这条，它决定下表能不能比）**：层切的"冷预填充"是 **prompt 前缀缓存冷**（每条请求 `0 reused`、
换新内容），但它是在**长跑生产服务**上测的，**专家缓存是热的**。TP 在新进程里跑第一条请求时专家缓存
是空的，那是另一个工况，代价很大：同一 prompt 在同一进程发两次，8K 上 **473.9 → 1255.6**（2.6 倍，
`/tmp/strata-tp-8k-sw-20261028.log`）。里面最大的一项是专家驻留（第一次请求要把 4.577 GB 的 pack 铺进
5550 个槽），可能混有 CUDA 上下文/内存池的首次开销——**不影响结论**：两条请求之间什么都没变，所以
"新进程第一条请求"这个协议本身就不能拿来和长跑服务的数比。

所以下表一律用**热专家**口径（同一进程的第二条请求），TP 只加 `STRATA_SELECT_SIMT=2`：

| 上下文 | 层切 prefill | TP prefill（热） | TP（冷启动） | TP 对层切 | 层切 decode | TP decode |
|---|---|---|---|---|---|---|
| 8K | 950–977 | **1294.7** | 520.7 | **+33%** | 41.7–44.0 | **63.5** |
| 16K | 1114 | **1377.6** | 779.7 | **+24%** | — | 未测 |
| 32K | 1400–1449 | **1270.7** | 935.7 | **−10%** | 39.5–47.0 | 未测 |
| 64K | 1558 | 未测 | — | — | 37.6 | 未测 |
| 128K | ~1490 | 未测 | — | — | 39.6–42.4 | 未测 |
| 160K | ~1450（插值） | **913.9** | 870.3 | **−37%** | ~40 | **32.1**（采纳配置）/ 41.0（全 7 开关）/ 26.2（冷） |

交叉验证（说明这些数不是一次性噪声）：8K 热 1294.7 vs 8K 记录值 1296.5（差 0.1%）；16K 冷 779.7 vs
另一次冷跑 775.3（差 0.6%）；160K 冷 870.3 vs 870.8 vs 861.9（三次，落差 1.0%）。**冷、热两列各自都可复现**。
唯一要打折的是 8K 热数：同一改动后各跑在 **1223.8–1296.5** 之间散布（6%，§5.4-2），所以 8K 对层切
是 **+25% 到 +36%**，不是精确 +33%。

#### 5.1.1 交点：约 26K token（这是分阶段设计的阈值）

两条曲线方向相反，所以有一个真实的交点，而且是算出来的：

| | 16K | 32K | 每 +1K token |
|---|---|---|---|
| TP（热） | 1377.6 | 1270.7 | **−6.7 tok/s** |
| 层切 | 1114 | ~1418（1449/1444/1362） | **+19.0 tok/s** |

解 `1377.6 − 6.7(t−16) = 1114 + 19.0(t−16)` 得 **t ≈ 26K**；层切 32K 取 1400 而非均值 1418 时 t ≈ 26.7K。
**阈值取 24–26K，保守用 24K。**

**TP 的 160K 内部是一条实测出来的衰减曲线。** 日志里 `PP <done> <total> <ms> <累计速率>` 每 chunk
一行，相邻两行之差就是该 chunk 的边际速率。下表是同进程**第二条请求**（专家缓存已热）的序列：

| chunk（位置） | 边际 tok/s | chunk（位置） | 边际 tok/s |
|---|---|---|---|
| 1（0–8K） | **1420** | 11（80–88K） | 860 |
| 2（8–16K） | 1402 | 13（96–104K） | 851 |
| 3（16–24K） | 1270 | 15（112–120K） | 835 |
| 4（24–32K） | 1103 | 17（128–136K） | 821 |
| 5（32–40K） | 977 | 19（144–152K） | 801 |
| 7（48–56K） | 912 | 20（152–160K，4351 token） | **613** |
| 9（64–72K） | 875 | | |

三个可以直接读出来的事实：

1. **TP 的单 chunk 能力是 ~1420**，和 8K 热跑的 1294.7 同量级（8K 只有一个 chunk，还带着每请求的
   固定开销）。**160K 的 914 不是"TP 慢"，是被后面 19 个更贵的 chunk 平均下来的。**
2. 衰减是**两段**：前 5 个 chunk 掉 31%（1420 → 977，每 1K 位置 −14 tok/s），之后缓慢线性
   （977 → 801，每 1K 位置 −1.5 tok/s）。
3. 最后一个 chunk 只有 4351 token 却最慢（613）——收尾比一个整 chunk 还贵。

**我此前写过"KV 复制让 staging 提前一倍、在 58% 处出现拐点"——这条被上面的数据否掉了**：没有拐点，
是平滑的两段衰减。能站住的只有"每 chunk 的成本随位置上升"，而"KV staging 变大"与"attention/indexer
随上下文变贵"在这份数据里分不开；要分开得在 chunk trace 里同时打 staging 字节数和毫秒数。

**层切为什么相反**：层切只付 28/48 层的 dense、KV 也是分摊的，固定成本随 prompt 摊薄，所以平均速率
一路涨到 64K 的峰值 1558。**两条曲线方向相反，就是 26K 交点的来源。**

160K 的几个开关对照（同一台机器、同一 prompt、同一 5550 槽驻留集，都是本进程第一条请求）：

- `870.3`（`/tmp/strata-tp-ctxwarm-160000-20261028-v2.log`，采纳配置、无计时）、
  `870.8`（`...-ctxwarm-160000-...log`）与 `861.9`（`strata-tp-160k-sel-...log`，185,632 ms）：
  采纳配置的三次冷启动测量，落差 1.0%。
- `872.1`：6 个开关、无计时，183,456 ms。只比采纳配置多 1.2%——**另外 5 个开关不值得采纳**。
- `830.9`：全 7 开关 + 有计时，191,437 ms。差额 −7.9 s 中约 4.9 s 归 PA_V2T（§5.4-2 的相位归因），
  其余 ~3 s 是计时器（每层每 chunk 一次同步）。
- 无开关的基准没有无计时测量；扣掉 ~3 s 计时开销后约 722 tok/s，所以 **`SELECT_SIMT` 单独值 +19%**。

**测法上的一个坑（已记进 `TP_HANDOFF` §5-9）**：32K 起引擎会写检查点（32K 写 1 个、160K 写 6 个、
8K/16K 不写），第二条**同 prompt** 请求会走检查点续跑，而 `--tp` 直接拒绝
（`--tp does not resume from a saved conversation checkpoint`）。所以 ≥32K 的热数必须用**同长度、
不同首 token** 的第二个 prompt 来测，不能重复同一条。第一次尝试（32K 942.5 冷、第二条报错）
因此只有冷数有效。

**结论：TP prefill 在 ≤24–26K 赢（8K +33%、16K +24%），32K 起输（−10%），160K 输 1.59 倍（914 vs ~1450）。**
decode 在 8K 大幅赢（63.5 vs 44.0，+44%）；但 160K **不再稳赢**：全 7 开关测得 41.0（与串行 37.6–42.4 持平），
采纳的最小开关集在同一探针里只有 **32.1**（输 15–24%）。两次 160K 的 draft 接受率不同，不能把差额全归给开关，
但 8K 那组 A/B 证明开关集是负的（−4.7%）——**最优点确实随长度移动，长上下文的 decode 优势不是白给的。**

### 5.2 为什么长上下文输：两卡的阶段账（`/tmp/strata-tp-final-160k-20261028.log`）

**方法论警告（先读这条）**：`STRATA_PREFILL_TIMING=1` 在每个 `pt.mark()` 处同步流，会打断两卡的
异步流水。它有两个后果，**大小不同、要分开看**：

1. **绝对时间只多 ~3 s**（160K：191.4 s 计时 vs 183.5 s 无计时，−3 s 里已扣掉 PA_V2T 的 4.9 s；
   8K 冷跑 ~1.5 s）。所以本节的相位表**在 160K 上可以当作真实占比**用。采纳配置（只 `SELECT_SIMT`）
   的 160K 无计时基线是 185.6 s = 861.9 tok/s，比 6 开关的 183.5 s 慢 1.2%——同一噪声量级。
2. **但桶间归属会被打乱**：同步点把"等对方"的时间记在了当时打开的那个桶上。所以
   **桶间分布不可跨 rank 比较**（§5.2 末尾的三条证据），只有每卡合计可用。

另外 `combine` 桶在 8K 冷跑里占 **63.6%**（9,702 / 15,258 ms）不是计时假象：8K 冷启时全部专家
都要从 pack 取，两卡在每层的列 `stitch_output` 处互等，冷启动的未命中成本就全被记到 combine 上。
160K 只有 8.3%，因为那时专家已热。

GPU 时间线 223.4 s，两个 rank 各自的相位合计都等于它（98.5% 以上被计时覆盖）：

| 阶段 | rank0 | rank1 | 是否被并行砍半 |
|---|---|---|---|
| qsa proj | 6.4 s (2.8%) | **59.9 s (26.8%)** | **否**（见下方：桶间偏移，非分工） |
| qsa select | 40.0 s (17.9%) | 37.3 s (16.7%) | **否**——两边都算全量 |
| qsa attn | **69.0 s (30.9%)** | 18.2 s (8.1%) | **否**（同上：两卡各跑全量，桶间偏移） |
| **QSA 合计** | **115.4 s** | **115.3 s** | 两卡各跑**一整份**，合计必然相等 |
| hc read | 13.0 s | 12.6 s | 否（全宽复制） |
| gdn（conv+recurrence+out proj） | 23.1 s | 22.5 s | **部分**——conv/recurrence/out 按 v-head 切，入投影的 q/k 复制 |
| gemm gate/up + down + dequant | 31.7 s | 30.7 s | **是**（n_ff 对半） |
| combine（跨卡） | 20.1 s | 20.1 s | 通信开销 |
| router+shared / embed+steps / ple | 11.0 s | 10.9 s | 否（复制） |

**决定性事实**：占 51.6% 的 QSA **完全没有被并行砍半**——两卡各跑一份完整的 QSA（投影、选择、attention
全都算全量），各忙 115 s。层切引擎同一份工作只做 28/48 = 58.3%（而且每卡只跑自己那些层），所以它在长
上下文必然更快。

**这张表里 proj / attn 两列的悬殊（6.4↔59.9、69.0↔18.2）不是角色分工，是计时归属偏移**，有三条独立证据：

1. **代码里没有 rank 守卫**：QSA 段（`prefill.cpp:3258-3500`）的 k/v/q 投影、indexer、选择、attention
   全部无条件执行；唯一的按 rank 切分是 `attn_output` 的 Col 切 + `stitch_output`
   （`tp_split_plan.cpp:125-147`）。
2. **8K 下两卡完全对称**（`/tmp/strata-tp-speed-prod3-20261028.log`、`/tmp/strata-tp-envbisect-20261028.base.log`）：

   | 8K 跑 | proj | select | attn | QSA 合计 |
   |---|---|---|---|---|
   | prod3 rank0 / rank1 | 363 / 446 | 103 / 101 | 789 / 741 | **1255 / 1288**（差 2.6%） |
   | bisect-base rank0 / rank1 | 304 / 341 | 102 / 101 | 753 / 728 | **1159 / 1170**（差 0.9%） |

   如果真是"一卡投影、一卡 attention"，8K 也必须呈现同样的分裂；它没有。
3. **160K 里三个桶精确守恒**：两卡 QSA 合计 115,370 vs 115,344（0.02%）、87,479 vs 87,526（0.05%）；
   而 select 的差与 (proj+attn) 的差逐跑相等——base 2,729 ↔ 2,703、env 511 ↔ 558。
   一个桶少掉的秒数原封不动出现在另一对桶里，只能是同一段阻塞被记在了不同位置。

结论：**160K 的桶间分布不可跨 rank 比较，只有每卡合计可用。** 每个 rank 承担的是**一整份** QSA。

### 5.3 TP prefill 的结构性上限（算术，不是猜测）

**先说清这一段的历史**：我先推理出"按 query 行切 QSA 能砍半"，随后看到一个"角色切、两卡已配平"的
表象把它否掉了，现在用 8K 对称性和三个桶的守恒律证明**第一次的判断是对的、中间那次纠正是错的**。
保留下面的推导是因为它自己就能验证：任何读到这里的实现者应该先用 8K 的对称性复核一遍。

由于 QSA 是两卡各跑一整份（§5.2），而它的工作**本来就按 query 行可切**——每个 query 只读自己选出的
`cap` 个 cell（这正是 QSA 的定义），不读整段 KV——**行切是真正的 2 倍杠杆**，不需要 Ulysses 式的
KV 重分布（两卡本来就各有一份完整 KV 池），只要在层边界 all-gather `[T][ZV]` 的 attention 输出。

帐是这么算的（**基线 185.6 s = 861.9 tok/s，冷口径**——它和 §5.2 的计时跑同协议，都是本进程的第一条
请求，所以占比与总量能对账；各桶按计时跑的占比等比折算，采纳配置本身没开计时）：

> **热口径换算（必须一起看）**：采纳配置的热数是 **913.9（175.1 s）**，两者差 10.5 s，是冷启动铺
> 专家缓存那一次的开销（§5.1.1）。QSA 不吃专家、绝对秒数不随冷热变，所以热口径下
> `175.1 − 42.4 = 132.7 s → ~1206 tok/s`；再叠 hc 行切 `126.4 s → ~1266`。
> 下面按冷口径把账走完（与 §5.2 同源、可对账），**结论方向不变，只是与层切的差距从 1.30× 收到 1.15–1.20×**。

| 项 | 每卡 | 行切后 |
|---|---|---|
| QSA 合计（proj+select+attn） | 84.8 s (45.7%) | **~42.4 s** |
| hc read（全宽复制） | 12.6 s | 12.6 s（下一步再切） |
| gdn（conv/recurrence/out 已按 v-head 切） | 23.0 s | 23.0 s |
| 专家 gate/up + down + dequant（已对半） | 30.8 s | 30.8 s |
| combine（跨卡） | 15.4 s | ~15.4 s |
| router+shared / embed+ple / 其它 | ~18.4 s | ~18.4 s |
| **合计** | **185.6 s → 862 tok/s** | **~143.2 s → 1117 tok/s** |

再叠一步 hc 行切（12.6 → 6.3 s）约到 **1169 tok/s**（冷口径；热口径 ~1266）。而层切在 160K 是
**~1450 tok/s**（`strata-fp4-local-checkpoint.md` §3.1）。**所以即使把 QSA 做到底，TP 的长上下文
prefill 仍然输，只是从 862 抬到 ~1117（热口径 914 → ~1206），把 1.59–1.68× 的差距缩到 1.15–1.30×。**

为什么抬不过去，一句话：**TP 必须为 48 层付 48 层的 dense 开销（hc + gdn + router/shared + embed/ple
≈ 51 s，行切最多砍掉其中一半），而层切只付 28/48 = 58.3%**。专家的对半省下来的 31.7 s，
抵不过"层数没有分摊"这一项。

结论硬化：**在 2×2080Ti 上，TP 的长上下文 prefill 上限约 1170–1270 tok/s（冷/热口径），层切约 1450。
这不是 kernel 优化问题，是并行维度的选择问题**——想彻底改只能换并行维度（见 §6：业界也是这么分的）。

### 5.4 三个悬而未决的问题（逐条结论）

1. **chunk 弹性 → 降级，不是杠杆。** 所有 TP 长上下文都被钉在 `--prefill auto:8192`，而 bare `auto`
   的扫描上限是 **32768**（`src/program/generate.cpp:2571`），看起来像被白白浪费的 4 倍。但外部实测
   否决了它：B200 上 chunk 8K→32K 吞吐只从 3434 掉到 3366 tok/s（**−0.4%**），因为因果 attention
   的总 FLOP 恒为 ~T²/2，chunk 只改 GEMM 算术强度与峰值激活显存（§6.5）。**不值得再花机时。**
   另外它确实有一道墙：rank0 能借出的上限 `0.85×5679 = 4827` 槽对应 chunk 约 23900 token，
   而 rank1 的缓存是 `min(空闲−自身prompt缓冲−512MiB, rank0槽数)`（`generate.cpp:9863-9868`），
   只有 4487 槽——chunk 16K 会把 rank1 压到 ~1023 槽。这条墙要拆得先做 §4.6-2(b)。
2. **Turing 开关 → 已测，是本次最大的一笔免费收益。** 串行生产配置的 7 个开关从未在任何 TP 跑里
   开过。160K 全开：prefill **712.4 → 830.9**（+16.6%）、decode **23.5 → 41.0**。归因（两跑逐桶对照）：

   | 桶 | 开前 | 开后 | 变化 |
   |---|---|---|---|
   | `qsa select` | 39,980 | 7,244 | **−32.7 s（全部收益）** |
   | `qsa attn` | 69,039 | 73,911 | **+4.9 s（变慢）** |
   | `combine` | 20,086 | 15,939 | −4.1 s |
   | 其余全部 | — | — | 逐毫秒相同 |

   净值 −31.9 s ≈ 时间线 223.4 → 191.4 的 −32.0 s。`SELECT_SIMT` 指向 select、是收益来源；
   `PA_V2T` 是唯一指向 attn 的开关，那 +4.9 s 记在它头上，**丢弃**。

   **但在 8K 上这套开关是负的**（`/tmp/strata-tp-8k-sw-20261028.log`，同 §4.1 配置，只多 6 个开关）：

   | 8K（热） | 记录（无开关） | 6 开关 |
   |---|---|---|
   | prefill | 1296.5 | **1255.6**（−3.2%） |
   | decode | 63.5 | **60.5**（−4.7%） |
   | 冷 prefill | 457.7 | 473.9（+3.5%） |
   | 相位表的 `combine` | 9,702 | **10,704**（66.2% 的时间线） |
   | 相位表的 `qsa select` | 102 | 42 |

   即 `SELECT_SIMT` 省下的 60 ms 被其余开关在冷启动/combine 路径上加回去了。加上 §4.8 自身的
   跑间散布是 1223.8–1296.5（6%），8K 上**不能声称收益**。

   **最终采纳：只加 `STRATA_SELECT_SIMT=2`**，其余 6 个不采纳。理由是它单独的两条证据都指向正：
   160K 的 select −32.7 s（归因链完整），8K 冷跑 G1 组 500.7 → 508.8（+1.6%）。
   采用前必须重跑 §4.7 的位级闸门——`SELECT_SIMT` 改求和顺序，token 会变。
3. **32K 交点 → 已测，约 26K。** TP 热 16K **1377.6**、32K **1270.7**；层切 16K 1114、32K ~1418。
   插值得交点 **≈26K**（§5.1.1），**阈值取 24–26K**。§8 的分阶段设计就用这个数。
   过程中否掉了两个测法错误，都值得记：(a) 把"新进程第一条请求"（专家缓存全冷）当成可比数——
   16K 的 775.3 因此作废，正确口径是同一进程的第二条请求（热）779.7 → 那条只是冷数；
   (b) 用"同一 prompt 发两次"测 ≥32K 的热数会撞上检查点续跑，`--tp` 直接拒绝（`TP_HANDOFF` §5-9），
   必须第二条换成同长度、不同首 token 的 prompt。

## 6. 外部对照：别人怎么分 prefill 和 decode（本次调研）

结论先写：**我们的实测和业界结论一致，没有走偏；错的是"在所有长度上都用 TP 打 prefill"这个方向本身**
（TP 在 ≤26K 是真赢的，见 §5.1.1——业界说的也是"长"上下文，不是所有上下文）。

1. **llama.cpp 官方直接把两种模式按用途分好了**：`-sm layer`（层切/PP）是默认，"mainly for people
   who want fast prefill"，"minimizes data transfers between GPUs but requires many tokens to scale
   well"、"can tolerate slow interconnect"；`-sm row`（已被 `-sm tensor` 取代）"comparatively poor
   performance"、定位是低延迟 decode，只有"大 dense 模型 + 快互联"时 prefill 才追平。
   → 和本机实测（160K 长 prefill：层切 ~1450 vs TP 913.9）**完全同向**。来源
   [docs/multi-gpu.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/multi-gpu.md)
2. **PP 的 bubble 公式** `(p−1)/(m+p−1)`，p=2 时效率 = `m/(m+1)`：m=1 → 50%、m=4 → 80%、
   m=8 → 89%（[Megatron-LM 2104.04473](https://arxiv.org/abs/2104.04473)）。本机的层切用
   `--pipeline-windows 2`（被 `generate.cpp:2325` 夹到 2），在 160K 那就是 ~83% 效率——这也是为什么
   串行不是"两倍"而是 1.6–2 倍。**注意：层切对单请求 prefill 不加速，它只是"不减速"；是 TP 在减速。**
3. **按阶段换并行有先例，但产品级框架都不在同一进程内换**：vLLM/SGLang/TRT-LLM 都是拆成两个实例
   （PD 分离）；vLLM 的 DeepEP 后端本身就按阶段分（`deepep_high_throughput` 面向 prefill、
   `deepep_low_latency` 面向 decode）；NVIDIA 明说 decode "performs best with a wide EP setup"。
   DeepSeek-V3 §3.4 是最明确的分阶段表：prefill 用 attention TP4+SP+DP8 / MoE EP32，decode 用
   attention DP128+SP(TP=1) / MoE EP320——**MoE 两个阶段都用 EP，变的只是 attention 的并行维度**。
   研究系统里有直接先例：**Seesaw** 在同一组 GPU 上让 prefill 用 PP、decode 用 TP，在 PCIe-only
   的 A10 上 geomean 1.45×、最高 1.78×，结论是"慢互联 favor PP、极快互联 favor TP"
   （[2503.06433](https://arxiv.org/abs/2503.06433)）。**这台机器正是慢互联，方向吻合。**
4. **Sequence parallel 不是我以为的东西**：Megatron 的 SP 只切 LayerNorm/dropout/残差这些 token-wise
   算子，**不切 attention**，收益约 6% 且主要是省激活显存；按序列切 attention 属于 context parallel
   （Ulysses/ring），代价是每层一次 all-to-all（[2205.05198](https://arxiv.org/abs/2205.05198)）。
   对我们的影响：**行切分值得做，而且比 Megatron 的情形便宜得多**——我们不需要 Ulysses 的 KV
   重分布，因为两卡本来就各持一份完整 KV 池；QSA 是稀疏的，每个 query 只读自己选出的 `cap` 个 cell，
   所以按 query 行切能真的把读放大砍半。代价只有层边界一次 `[T][ZV]` 输出 all-gather
   （160K 下约 0.6 s 量级）。§5.3 给的是这条路的天花板：~1117（再把 hc 行切掉到 ~1169）。
5. **chunk 大小几乎不是杠杆**（推翻了我原来的假设）：因果 attention 的总 FLOP 恒为 ~T²/2，chunk 只改
   GEMM 算术强度与峰值激活显存。B200 实测 8K→32K chunk 吞吐 3434→3366 tok/s（**−0.4%**）；业界默认
   8K（vLLM 类）/16K（H100 类）。真正有平台期的是**prompt 长度**本身（5090 35B-A3B：1K 11.5k →
   8K 峰值 16.4k → 32K 14.1k → 256K 6.1k tok/s）。→ **本机"chunk 8192 vs 32768"不值得再花机时**。
6. **专家 offload 的两条路线我们已经站在正确那条上**：路线 A = CPU 直接算驻留 CPU 的专家
   （KTransformers / Fiddler），路线 B = 专家缓存 + 预取搬上 GPU（MoE-Infinity / Mixtral-offloading）。
   我们的"CPU 池算 + GPU 专家槽 LRU + 自适应换入"同时具备 A（CPU 池）和 B（槽位缓存）的要素；
   Fiddler 实测旧方案 73% GPU 时间耗在 kernel launch（[2402.07033](https://arxiv.org/abs/2402.07033)），
   而我们的 GPU 时间线已占墙钟 99.5%（§4.8），这一项不是问题。
   MoE-Infinity 用"按历史路由轨迹预测"胜过裸 LRU（[2401.14361](https://arxiv.org/abs/2401.14361)），
   我们是 `--expert-profile` 静态热点 + `STRATA_ADAPT_*` 自适应，方向相同。


## 7. 附录：`--tp` prefill 逐桶审计（每个 `kPf*` 桶两卡到底在做什么）

判定依据：`m.tp_gdn.data != nullptr` 进入所有 TP 分支（`prefill.cpp:3183/3517/3697`），`m.tp_moe_ok`
进入 MoE 切分（`3697`），每层分 half=0（attention）与 half=1（MoE FFN）两趟（`3088-3090`）。

| 桶（打印名） | 两卡各自实际做的事 | 分类 |
|---|---|---|
| embed+steps | 各卡自己 gather 全 T 行 embedding + 卡内 `gr_broadcast`；steps 全 T 填写并上传 | 全复制 |
| hc read | 全宽 D=10240 的 hc down/up/inject GEMM + `gr_mix` 残差写回，**每层两次** | 全复制 |
| gdn（入投影） | `attn_qkv`（q、k 全 + v 半，n_out 7168/10240）、`attn_gate` Col、`ssm_alpha/beta` 全宽 | 混合 |
| gdn conv+gates | 本卡 24 个 v-head（`h_v=24`、`head_offset=rank*24`），conv 覆盖 7168/10240 通道 | 切半 |
| gdn recurrence | 24/48 个 v-head 递推（state 每卡一份），**桶内含 `stitch_y`** | 切半+通信 |
| gdn out proj | `ssm_out` Col（1280/2560 输出行，K=6144 全宽），后接 `stitch_output` | 切半+通信 |
| qsa proj | k/v/q、indexer k/q、rms/rope、KV append、indexer append、`gate_attn` 全宽；`attn_output` Col + `stitch_output` | 混合 |
| qsa indexer | 整 chunk 的 indexer append：全 T、全 4 个 indexer head | 全复制 |
| qsa select | 全 T 行的 block scores + topk（按 `sel_batch` 分窗） | 全复制 |
| qsa attn | 全 T 行 × 24 q-head × 2 KV-head，读本卡自己的全量 KV 池 | 全复制 |
| router+shared | router 全 512 出 + `route`；shared expert gate/up/down **全宽 2560** | 全复制 |
| gather / wait copy / dequant | 只搬本 rank 读的窗口（gate 半 + up 半 + down 半 + tail）；MMQ 下 dequant 是 gather+tail memset | 切半 |
| gemm gate/up | 2·n_ff_local=640 行输出、K=2560 全宽；swiglu→320 宽；**桶内含 `exchange_hq`** | 切半+通信 |
| gemm down | n_out_local=1280 行、K=2·nf=640（两卡各 320 经 `moe_hq_exchange_mmq` 拼回全宽） | 切半 |
| combine | `moe_combine_cols` 只算自己那 1280 列，后接 `stitch_output` | 切半+通信 |
| host grouping | 两卡各自建同样的 group 表（routing 与 residency 无关） | 全复制 |
| ple / kv stage | PLE gather/mix 全 T；staging 池与 host 镜像每卡一份 | 全复制 |

**读法**：想做长上下文 prefill，杠杆只有"把全复制桶变成行切"。按 160K 的量级排序，全复制的桶是
qsa attn + qsa select（~87 s，最大的一个）、hc read（13 s）、router+shared（5 s）、ple（1.7 s）。
前两项正是 §5.3 里 862 → 1117 的来源（热口径 914 → 1206）；后两项加起来不到 7 s，性价比低。

**注意**：这张表是**代码路径**的分类，不是计时归属。160K 下 `qsa proj`/`qsa attn` 两桶的跨 rank
悬殊是同一个阻塞被记在不同桶造成的（§5.2 的三条证据），不要据此反推分工。

## 8. 架构级结论：分阶段换并行（这是唯一能同时在两个指标上超过层切的设计）

前七节的账指向一个结论，且业界已有同名方案（Seesaw，§6.3）：

| 上下文 | prefill 谁赢 | decode 谁赢 |
|---|---|---|
| 8K | **TP**（1294.7 vs 950–977，+33%） | **TP**（63.5 vs 44.0，+44%） |
| 16K | **TP**（1377.6 vs 1114，+24%） | 未测 |
| 32K | **层切**（~1418 vs 1270.7，+12%） | 未测 |
| 160K | **层切**（~1450 vs 914，+59%） | TP 需全 7 开关才持平（41.0）；采纳配置 32.1 输 |

所以"选一个模式跑到底"必然在某一半上输。**正确的设计是按阶段换并行维度**：长 prompt 用层切
prefill，decode 用 TP。这不是折中，是在两个阶段各自用该阶段的赢家。
**交点已实测：≈26K token（§5.1.1），所以 ≤24K 直接全 TP 跑到底，>26K 才需要换相位。**

### 8.1 为什么在本引擎里可行（不是空想，逐项核过）

1. **两套模式已经都在，且共用同一份 pack 与同一套专家流式机制。** 层切与 TP 的区别只是
   "哪些层/哪些半份"在卡上驻留；专家本来就从 pack 按需流式取，所以切换的代价是**一次缓存重热**
   （6.6 GB/s、按 §4.3 的 4.6 GB 计 ~0.7 s），不是重新加载模型。
2. **dense 权重两份可以同时驻留。** 全部 dense 只有 1,992 MiB：TP 要一半（~1.0 GB）、层切要
   20/48（~0.83 GB），合计 1.8 GB，而 VRAM 里 ~18 GB 是专家槽——放得下。
3. **KV 重分布很便宜。** 层切 prefill 后每卡只有自己那些层的 KV；TP decode 要求两卡各有全部 48 层。
   QSA 层 12 个、每 token 每层 K+V = 2 head × 128 × 2 = 512 B（int8），160K token → **~1.0 GB**，
   走 NVLink（2080Ti NVLink2 单向 25 GB/s）约 **0.04–0.08 s**，相对 110 s 的 prefill 可忽略。
   GDN 的 recurrent state 是每序列标量级，PLE history ~0.4 GB，同量级。
4. **数值口径**：层切 prefill 与 TP prefill 的求和顺序不同，token 流会不同。但这是**换实现的
   重新钉定**（像 §4.7 那样建立新的钉定串），不是正确性缺陷——两个模式各自都是逐位确定的。

### 8.2 门槛与预期

切换阈值 = **24–26K token**（§5.1.1 的实测交点；TP 热 16K 1377.6 → 32K 1270.7 向下，层切
16K 1114 → 32K ~1418 向上，插值 26K）。**短 prompt 不需要任何切换**：≤24K 时分阶段 == 全 TP，
所以 8K 的 +33%、16K 的 +24% 原样保留。

预期成绩（160K，按已实测的两个赢家拼起来）：

| 指标 | 串行（记录） | TP today | 分阶段 |
|---|---|---|---|
| prefill | ~1450 | 914 | **~1450**（用层切） |
| decode | 37.6–42.4 | **32.1**（采纳开关集）/ 41.0（全 7 开关） | **41.0**（TP + 全 7 开关——长上下文的最优开关集与短上下文不同，这需要开关可运行时切换；否则只有 32.1，仍输） |

**这就是"必须超过之前串行版本"的答案**：不是在 TP 里把 prefill 调到 1450（§5.3 证明上限只有
~1170–1270），而是让 prefill 阶段根本不用 TP。

### 8.2.1 顺带发现：层切自己还有一个 16.6% 的白给（零代码改动）

串行生产配置是 `--layer-split 20`（48 层里的 20，剩 28），也就是**忙的那张卡做 28/48 = 58.3%**。
两级流水线的吞吐由最忙的一级决定，所以改成 24/24 就是 50%：

```
1450 × 58.3/50 = 1690 tok/s   （理论值，改一个 flag，不改代码）
```

这个数字比本报告里任何 TP 优化都大，而且**它同时也把 TP 打进长上下文 prefill 的门槛抬高到 1690**。
我没有实测它——用户的生产配置不归我改，而且这属于层切那条线的工作。但如果要在长上下文上真正
"超过串行"，必须同时把这一项算进去，否则是在跟一个本身没调好的基线比。

**这条也改变了 §8.1 的第 3 点**：KV 重分布的目标布局要按 24/24 来设计，不是 20/28。

### 8.3 实施顺序（每步都能独立验收）

1. ~~**先补 32K 的 TP 与层切同条件点**，把交点量出来。~~ **已做**：交点 ≈26K（§5.1.1）。
   代价是四轮上下文测量（8K/16K/32K/160K × 冷热），并顺带纠正了两个测法错误（§5.4-3）。
2. ~~**KV 重分布例程**：`layer-shard → replicated`，层切 prefill 结束处调用。~~ **已做**：
   `include/strata/core/kv_redistribute.hpp` + `src/core/kv_redistribute.cu`，出口
   `kv_redistribute(g, tokens, src[], n_src, dst[], n_dst, stats, err)`。三块状态各走各的路：
   - **QSA（复制语义）**：每个目标 rank 都要全部 12 层，而层切的一段只有自己那几层，所以从 owning 段取。
     泳池按 `qsa_kv_spans` 给出的 span 拷**前缀**（`ceil(tokens/page_size)` 页；模式 0 的池是恒等页布局，
     所以前缀 == 单元 `[0, slots)`），再拷 page_table / idx_tail / idx_dead / idx_pooled 的已用行 /
     idx_block_pos。同卡 `cudaMemcpyAsync`，跨卡 `cudaMemcpyPeerAsync`（从目标卡开 peer），源不动。
   - **GDN（置换语义）**：层切是整宽 `S*HV*S + C*(d_conv-1)`，TP 是头带切分
     `S*HVt*S + Ct*(d_conv-1)`，所以不是拷贝而是 permute：`gdn_split_rec_kernel`（全局头
     `head_off+h` → 本地头 `h`）与 `gdn_split_conv_kernel`（全局通道
     `qk + head_off*S + (c-qk)` → 本地 `c`），在**目标卡**上发起、经 peer 读源（和 TP decode 读 peer
     缓冲同一个路子），不走主机中转。
   - **PLE**：`ple_hist`（9×10240）+ `ple_prev[2]` + `ple_token`，从 owning layer 1 的那一段取。
   拒绝：`kv_mode != 0`、弹性/流式状态（其 VRAM 池是槽不是恒等页）、`max_cells`/KV 格式/槽数不一致、
   目标不是整模会话、层段有洞或重叠。全部同步返回，只在相位边界调用一次，不在 token 路径上。

   **fixture `tests/core/kv_redistribute_test.cu`**（ctest 注册，3.1 s，双卡，FP16 与 INT8 两种 K/V 格式
   各跑一遍）验四层：
   - 字节：12 个 QSA 层在两卡上都与源逐字节相同，且**未用单元仍是毒值**（证明没有多拷、前缀算对了）；
     PLE 历史取的是 layer 1 所在那一段的，不是另一段的。
   - 形状：36 个 GDN 层 × 2 卡与**主机独立重算的 permute 逐位相同**。
   - 语义：把 `gdn_conv_step` / `gdn_step` 跑在重分布后的状态上，对本 rank 自己的头带/通道带与整宽状态
     **逐位相同**（本地 conv 权重用生产的 `tp_local_conv_weights` 切，不是测试自己切）。这一项是第三个
     独立意见：前两项在"copier 和主机参考一起错"时不会报警。我故意把 copier 和主机参考同时改成错的
     通道映射，这条在第一个 v 通道（4096）就报错——所以它不是空测试。
   - 负例：整模会话、层覆盖、GDN 竞技场长度、`max_cells` 四种形状不匹配都被拒且带错误信息。

   **量**（50 token / 64 cell 的小 fixture）：GDN 每 rank 56.95 MB，**与上下文长度无关**（36 层 ×
   1.66 MB）；QSA 每 rank 1.32 MB（FP16）/ 0.73 MB（INT8），随 token 线性 ≈ 24.6 KB/单元/rank。
   照此外推 160K 上下文 ≈ **3.9 GB/rank**（估计值，未实测），相对 160K prefill 本身的百秒量级仍很小。

   **这条 fixture 证明的边界（写清楚，免得被当成整模验收）**：它证明的是"层切 prefill 留下的状态，经
   `kv_redistribute` 之后 == TP decode 实际读的那份状态"，包括这些状态喂给生产 kernel 的结果逐位相同。
   它**不**证明整模"第一个 token 相等"——那还要求权重分片、专家驻留槽数、开关集在同一配置（§4.7 的重新
   钉），fixture 无权替它作证。整模验收等第 4 步把相位计划接上之后再做。
3. **双驻留 init**：dense 两份同时常驻；专家缓存按"字节区间"统一索引（TP 用半区间，层切用全区间）。
4. **每请求的相位计划**：`prompt_len > 阈值` → 层切 prefill → KV 重分布 → TP decode；否则全 TP。
   **开关集也要进相位计划**：长上下文 decode 要全 7 开关才持平（41.0 vs 32.1），短上下文多开开关是
   −4.7%，而开关现在是启动期环境变量、进程内不可变——这一步要么给引擎加运行时可切，要么按相位分两个进程。
5. 钉新闸门（solo == TP == 分阶段），并把 §4.7 的字符串替换掉。

第 1、2 步已经完成，所以现在可以动代码了：阈值 24–26K 是实测的，混合的每一处状态迁移都能靠它验收。

### 8.4 第 3、4 步的落点：两套角色怎么在同一进程里并存

**现状（行号都核过）。** 不是"两种开关"，是两套对象图，且互斥是双重的：
- 层切：`stages[i]` 是 `GpuStage`（`generate.cpp:1265-1303`），每段自带 `wt`（自己那些层的**整宽**权重）、
  `ss`（**切片**会话）、`cache`（只装自己层的 (layer, expert)）、`ver`、`sp`；
- TP：`tp_stage` 也是一个 `GpuStage`（`generate.cpp:9679`），但 `lb=0, le=n_layers`、`ss` 是**整模**会话、
  `wt` 在 `--pack` 路径上也覆盖全部 48 层（**不是**每层半宽：半宽只发生在 `--native-dense-gguf`，
  见下）、`cache` 是自己的（`generate.cpp:9862`）。
- CLI 互斥 `generate.cpp:3412`；构建互斥 `generate.cpp:9671`（`n_stages > 1` 直接拒绝）。

**落点：不做第二套对象图，只加两个 `Prefill`。** 下面三条把它推到"零新增状态内存"：dense 不用第二份、
KV/GDN/PLE 不用第二个会话、专家缓存只换驻留表。

所以第 3 步的名义目标（"dense 两份同时常驻"）**在 `--pack` 路径上其实已经成立**，不需要第二份——这一条
改写 §8.1-2 的算法：
- TP 模式下 `trim_asked` 只对层切生效（`stage_trim = trim_asked && multi_gpu && ...`，`generate.cpp:3868`），
  所以 CUDA0 的 arena 是**整份 dense**（`cudaMalloc(&arena, pool_bytes)`，`generate.cpp:3910`）；rank1 用
  **同一个 `pool_bytes`、同一个 `skip`** 再读一份（`generate.cpp:9691-9692`）。也就是说**两卡本来就各持
  全部 48 层的 dense**（`pool_bytes` ≈ 1,992 MiB，§8.1-2），TP 的"半宽"只是**计算**的分片。
- 层切角色要的"自己那 24 层的整宽权重"是这份的**子集**。混相位不需要第二套 dense，只要给层切角色绑一个
  `layer_lo/layer_hi` 视图——`GpuStage::wt` 就是 `WeightTable`，只持 `WeightRef.data` 指针，arena 是
  调用方的裸 `cudaMalloc`（`weights.hpp:175-177`、`weights.cpp:117-135`），没有全局注册表。
- **更正（2026-10-28 夜，实测）**：本节原先写"采纳配置不走 `--native-dense-gguf`，所以第 3 步在 dense 上
  的成本是 0"——**这是错的**。§5.1 的 870.3 等成绩、以及 §9 的全部房间测试，命令行里都同时带着
  `--native` 与 `--native-dense-gguf`，rank 0 启动时打印的 "300 native projection matrices, **1992.19 MiB**
  of weights" 就是**每 rank 半宽之后**的总量。也就是说半宽切分在采纳配置里是**生效的**，层切角色确实要
  一次真正的整宽重读。实测代价（`generate.cpp` 的 `load_stage_weights`，新增计量行）：

  | 卡 | 层切角色的 pack（按层裁剪） | 整宽 native dense（本段 24 层） | 合计 |
  |---|---|---|---|
  | CUDA0 | 771 MiB | 1,474 MiB | **2,245 MiB** |
  | CUDA1 | 771 MiB | 1,474 MiB | **2,245 MiB** |

  两张表都在 `cudaMalloc` 里各占一份独立 arena（pack 与 dense 分别分配），所以 §8.1-2 的"两卡本来就各持
  整份 dense"只对 **pack** 成立，对 native dense 不成立。原"层切角色绑一个 `layer_lo/layer_hi` 视图"的
  想法在 native 路径上**行不通**：同一张表的 ref 已经被半宽 native 指针占满（`native_dense.cpp:380`
  的 "override already attached"），而且 prefill 只认一张 `WeightTable`，混不了半宽与整宽。
- 顺带一个**顺序陷阱**（第一次跑就是这么崩的）：`phased_arena` 按"卡上还剩多少"一次性 malloc 提示区，
  所以整宽表必须在它**之前**装好；而 `tp_stage` 又要等 rank 1 自己的表装完才存在。两者夹在一起，唯一
  正确的位置是 `init_prompt_paths` 的**开头**（首个 prompt 的延迟初始化里），并用一个 once 标志防止
  chunk 重试时重复 malloc。装晚了的症状是 CUDA1 `native dense upload blk.34.ssm_out.weight: out of memory`，
  装错了位置（`tp_stage` 还没建）的症状是 `blk.0.hc_attn_inject.weight is missing`。
- 还有一个**连带后果**：CUDA1 装完两套表后只剩 **3,532 MiB**，放不下 8192 chunk 需要的 4,357 MiB 提示区，
  自动降档到 **4096**（自降逻辑有效；`phased_arena` 的失败串原来不含 "do not fit"，不会触发重试，已改）。
  这会影响长上下文 prefill 的 chunk 大小，见 §9.4 的实测。

**专家缓存不需要改布局，§8.3-3 的措辞可以收紧。** slot 本来就按**全 blob** 分配，TP 的 MoE 路径用指针
偏移读自己那一半（`generate.cpp:9845-9846` 的注释；`open(..., max_blob)` 在 `generate.cpp:6784` 与
`9862`），所以两个相位用的是**同一种 slot 布局**，差别只有两点：(a) admit 哪些 (layer, expert)——层切只
admit 自己那 24 层，TP admit 全部 48 层；(b) 层切从 pack 搬全量、TP 从 PCIe 只搬自己那半（§4.6-2(a)）。
于是**切相位 = 清空驻留表 + 按目标集合重新 admit**，不需要"哪半边有效"的位图。缓存侧唯一要加的代码是
`ExpertCache::clear_residency()`（现在没有；`admit` 不驱逐，表满了就 admit 不进去）：十几行 + 一个单测，
不碰 slot 布局，也不碰任何 kernel。

**第二个会话也不需要——两个角色绑的是同一对整模会话。** rank0 的 TP prefill 是 `sp`（绑 `ss`），rank1 是
`tps.sp`（绑 `tps.ss`）；`Prefill::init` 只**存引用**（`prefill.cpp:1222` 的 `m.ss = &ss`，此后不写会话），
而层切角色跑哪些层由 `set_stage(lb, le, next)` 单独给出（`prefill.hpp:242`），**与会话自己的 `layer_lo/hi`
无关**。整模会话的 ordinals 本来就是全局的（`qsa_ord0 = 0`、`gdn_ord0 = 0`，`session.cpp:94-97`；层循环里
`ss.qsa_states[qsa_index]`、`ss.gdn_state + gdn_index - gdn_ord0` 都是全局序），所以层切 prefill 直接往**同一份**
KV 里写自己那半，`kv_redistribute` 只需把对侧那半搬过来。§8.1-3 的"切片 0.5 GB"因此归零。

**整宽 GDN arena 在 `--tp` 下已经分配且闲置，是白捡的。** `session_init(..., 0, -1)` 按整模切出 36 层整宽
（36 × 3.27 MB ≈ **118 MB**，`session.cpp:100`），而 TP decode 读的是另一块 `gdn_state_tp_`
（`verify.hpp:222-225`）。层切角色读的正是 `ss.gdn_state`（`prefill.cpp:3220`），所以这块 118 MB 不用新增。
PLE 也只有含 layer 1 的那一段做（`prefill.cpp:2588` 的 `LB <= 1 && 1 < LE`）：rank0 的层切 prefill 直接推进
`ss.ple_hist`，rank1 由 `kv_redistribute` 的 PLE 拷贝拿到——已经实现并测过。

**每卡预算（`--expert-cache 4000` 采纳配置）：**

| 项 | 实测 / 估算 | 依据 |
|---|---|---|
| 专家缓存（一份，服务两个相位） | 4000 × 2.637 MiB = **10.30 GiB** | 全 blob 分配 |
| 层切角色的整宽表（**每卡一份**） | pack 771 MiB + native dense 1,474 MiB = **2,245 MiB** | 实测，见上 |
| KV / GDN / PLE | **0 额外** | 同一对整模会话；整宽 GDN arena 本来就在 |
| prompt scratch | **0**（默认借同一块 borrow 区） | 见下 |
| 新代码自己的分配 | `Prefill` 对象自身（staging、事件），**几 MB** | — |

合计约 12.5 GiB，22.5 GiB 的卡在装完 rank 1 自己的半宽表之后只剩 **3,532 MiB**（实测），刚好不够 8192
chunk 的 4,357 MiB 提示区，于是自动降到 4096 chunk。**另一件已核过的好消息**：两个 session / 两个 `Verifier`
同进程是既有能力——两者都只绑定非拥有的 `SessionState*`（`verify.cpp:679`、`prefill.cpp:1222`），多
verifier 由 `g_live[16]` 显式支持（`verify.cpp:427-428`）。

**唯一要证的假设是"两个 `Prefill` 共用一块 borrow 区"。** 默认配置下 prompt 缓冲不是自有的，而是**借**
专家缓存顶部那批 slot（`generate.cpp:9154` 的 `borrow = xcache.device_slot(lend_first)`）：两个 Prefill 先后
`init` 同一块借出区、各自只在自己的相位里跑（相位是顺序的，永不并发），chunk 取同一个值，于是两者的偏移
算术完全一致、都不持有这块内存。这一条要在 3a 里用 fixture 先证明（同一 borrow 区上两个 Prefill 先后跑，
输出与各自独占时逐位相同）。退路：`--no-prefill-borrow` 下每份 scratch 是 2–3 GiB（8192 chunk），两张卡
各多一份也仍然放得下（剩 ~12 GiB），只是缓存少 4000 槽的一半左右——不划算但不失败。

**开关集不能整组进相位计划。** 7 个里 `STRATA_HC_REQ8` 改的是**权重装载布局**（`weights.cpp:535`、
`layer.cpp:1533`），而且它在层切下是**被关掉**的（`generate.cpp:3901-3905` 的 `hc_q8 = hc_req8_env && !multi_gpu`）
——混相位里它会对两个相位一起关掉，所以它不能进运行时开关集，还会改 TP 相位的数值口径；其余多数是
`static const` 一次读死（`qsa_select.cu:1881`、`qsa_prompt_attn.cu:2325`、`moe_fused_iq.cu:1295`）。可行的
最小改造是把"选核"那一类改成运行时读一个原子量（初值来自环境变量）：同一进程内可切，顺带解决 §4.6 反复
强调的"跨 run 不能横比"——同一进程 A/B 才是给开关定价的唯一口径。

**一个实现约束（读代码读出来的，不是猜的）**：`multi_gpu` 现在是层切的总开关，除了选路径还派生别的行为
（上面那条 `hc_q8` 只是其中一例）。混相位不能让它同时代表两种角色，必须另立 `phased` 标志，凡 `multi_gpu`
分支都要逐个判断"这条属于层切角色还是属于进程"，否则 TP 相位会静默继承层切的行为。

**实施顺序（每步都能独立验收，且每一步失败都不伤 `--tp` 单模式）：**
0. **已完成，且不需要整模**：`ExpertCache::clear_residency()`（`expert_cache.hpp`/`.cpp`；单测
   `tests/core/expert_cache_clear_test.cpp` 覆盖 uniform/sized × global/per-layer 四组，验"表清空、arena 地址
   与大小不动、重新 admit 落回同样的槽"——后者只有每层游标回到**自己区间底部**才成立）；`kv_redistribute` 的
   相位化扩展（`KvSrcStage::layer_lo/hi` 显式层范围、同址 span 跳过、共享 GDN arena 拒绝）。`kv_redistribute_test`
   新增第 3 趟正是相位化形态：**源就是目的**，搬的字节恰好是切片源那一趟的**一半**（FP16 1.32 vs 2.65 MB、
   INT8 0.73 vs 1.47 MB），两种 KV 格式下状态都逐位正确；统计口径同时改成"跨目的 rank 的总量 + 层的集合大小"。
1. **3a 两角色并存**：去掉两处互斥（`generate.cpp:3412`、`9671`），在既有 TP 图上加两个"层切角色"的
   `Prefill`：`pf_ls0` 绑 `ss`、`pf_ls1` 绑 `tps.ss`，**都不绑** `tp_gdn`/`gdn_y_exchange`（这样它们读整宽的
   `ss.gdn_state`），用 `set_stage(0, L0, &pf_ls1)` / `set_stage(L0, n_layers, nullptr)` 串成两段流水；
   启动日志打印上表的实测占用。验收 = 不带 `--layer-split` 时行为不变（§4.7 钉定串仍过），
   带 `--layer-split 24 --tp` 能起来且预算对得上。
2. **3b 现场重分布**：层切 prefill 结束处调 `kv_redistribute`，源 = `{ss@dev0 [0,L0), tps.ss@dev1 [L0,n)}`，
   目的 = 同一对会话（QSA/PLE）+ `ver`/`tps.ver` 的 `gdn_state_tp_`（GDN permute）。验收 = 8K prompt 分阶段与
   纯 TP 的 token 相同（两者都 TP decode）；若不同，先归因到 prefill 求和顺序（§8.1-4）。
3. **3c 相位选择**：阈值 + 每请求切换 + `clear_residency` + 重新 admit；验收 = ≤24K 全 TP 成绩不掉，≥26K 走分阶段。
4. **3d 选核开关运行时可切**；验收 = 同进程切换与两次独立进程输出相同。
5. **3e 新闸门**：solo == TP == 分阶段，替换 §4.7 的字符串。

**风险排序：** (1) 两个 `Prefill` 共用一块 borrow 区、同一对会话（新组合，必须先用 fixture 钉住：两个 Impl 的
缓冲不别名、会话不被互相覆盖、输出与各自独占时逐位相同）；(2) 层切 prefill 与 TP prefill 的求和顺序
不同，token 流必须重新钉定，不能与旧串比；(3) 相位切换的 re-warm 时延（0.7 s 是 §8.1-1 的估计，未在混合
进程里实测）；(4) 层切相位里缓存配额按 48 层切（`layer_slot_range`），只用 24 层时有一半配额空着——命中率
问题不是正确性问题，3c 里按实测决定要不要按相位改配额。

## 9. 第 3a–3e 步的实测（2026-10-28 晚）

验收口径：钉定驻留集（`--expert-cache 4000 --adapt-every 0 --adapt-swaps 0`），两段请求
L = 8000 token、S = `9707 11 1246`，各生成 8 token，`--spec 8 --spec-min-p 0.20`，`STRATA_BF16_TC=1`，
其余同 §4.1。门以 **token 序列的 md5** 为准，不以速度为准（速度另列）。脚本
`/tmp/tp_phased3de_run_20261028.sh`（`MODE=a..e`），日志 `/tmp/strata-tp-phased3de-{a..e}.log`。

| 相位 | 配置 | seq md5 | 8K 读 | 8 生成 |
|---|---|---|---|---|
| a（参照） | `--tp` | `f17fe819…` | 15539 ms / 514.8 tok/s | 932 ms |
| b（3a/3b） | `--tp --layer-split 24`（8K < 交点 → TP 角色） | `f17fe819…` **同 a** | 15473 ms / 517.0 | 949 ms |
| c（3b/3c） | 同上 + `STRATA_PHASED_MIN_TOKENS=0`（强制层切角色 + 交接） | 见 §9.2 | | |
| d（3d 对照） | `--tp` + `STRATA_SELECT_SIMT=2` | `T 7 16 8 220 17 15 17 15` | 15547 ms / 514.6 | 1102 ms |
| e（3d 本体） | `--tp` + `STRATA_SELECT_SIMT_LONG_TOKENS=1024`，S,L,S | S=`79790 316 52933 91 79751 198 285 1992`、L=d、S 再=同 | 同 a | — |

### 9.1 3a/3b：两套角色同进程，初始化零成本

b 与 a 的 16 个 token 逐位相同（md5 一致），8000 token 的读只差 66 ms（−0.4%，噪声内），
生成只差 17 ms。也就是说**多出来的两套 `Prefill`（`pf_ls0`/`pf_ls1`）+ CUDA1 上多的一份 prompt
arena（`pf_arena1`）不影响 TP 角色**：相位是严格顺序的，谁都不在对方跑的时候活着。rank1 加载后剩
16388 MiB（4000 槽）。3a 的"两个 Prefill 共用一块 borrow 区"在整模上直接成立（两者都借
`xcache.device_slot(lend_first)` 的同一块，`--short-read 32` 下 8K 请求走同一批 chunk）。

### 9.2 3b/3c：第一次跑出的三个真问题（都已修，不是猜的）

第一次跑 c 直接 `EXIT=1`，日志 `/tmp/strata-tp-phased3de-c-guardbug.{log,status}`：

```
ERR phased: a prompt read that does not start at token 0 cannot take the layer-split role
```

三处，全部是我自己写错的地方，逐条记下来免得以后再犯：

1. **`b0 != n` 这个守卫永远成立。** prompt 读取的最后一个 cut 是 `n - 1`（最后一个 prompt token
   固定留给第一个 decode 窗口：`cuts = {reread_to, root_at, message_at, turn_at, n - 1}`，
   `generate.cpp` 的 `for (const int64_t to : cuts)`），所以 `read_part(a0, b0)` 的 `b0` 是
   `n - 1` 而不是 `n`，守卫必拒。交接的 `tokens` 就该是**这一段**的结尾（`r == b0`），不是 prompt 长度。
2. **交接放错了层。** `phased_handover` 原先写在 `run(q, r)` 里，而 `run` 是**按 piece/chunk**
   被调的（`--batch` 一次一块），一段读被切成 k 块就会交接 k 次，每次都搬一遍前缀。
   改成只在 `r == b0`（这一段的最后一块）交接一次。
3. **第三段会更糟：带对话边界的 prompt 会被切成两段。** `turn_at`（`<|im_start|>`）存在时
   `read_part` 先读 `[0, turn_at)` 再读 `[turn_at, n-1)`，第一段的 `b0` 既不是 `n` 也不是 `n-1`。
   按原来的写法整个请求会直接失败——而这正是**量化 agent 的典型请求**（长 system+历史，然后
   每轮追加几十个 token）。改法是**角色按段定**：`a0 == 0` 的那一段（第一段）走层切角色并在
   段末交接；之后的段走 TP 角色——因为交接已经把状态变成了 TP 读的形状。这样带边界的 agent
   请求第一段仍然吃层切 prefill 的速度，后续小段用 TP（本来也没多少 token）。

顺带堵掉一个会**静默污染结果**的洞：分阶段读不能 BYIELD。被停到槽里的状态只带 QSA
（`copy_to_slot` 走 `conversation_kv_save/restore`，不含 GDN，也不含 verifier 的 split arena），
续读时 `a0 != 0` 会用 TP 角色去接一个层切形状的状态。`give_way` 现在对 `phased_ls` 一律返回
false（读到底，不中途让位）。**注意这是一个 pre-existing 的 `--tp` 洞**：非分阶段的 `--tp`
请求 BYIELD 同样丢 GDN 状态，只是我们的单请求跑没碰过；已在 §10 待办里记下。

### 9.3 3d：选核开关运行时可切，同进程 == 两次独立进程

`STRATA_SELECT_SIMT` 从 `static const` 变成运行时原子量（`kernel_switches.hpp`），3d 让它在**请求边界**
按长度选：≥ `STRATA_SELECT_SIMT_LONG_TOKENS`（默认 26624）用 2，否则用环境给的那一档（没给就是 0）。
`--batch` / `--pipeline-windows` 下不切（一张图服务多个请求或永不重捕），生产配置
`--pipeline-windows 3` 因此**行为不变**（它一直显式 `SELECT_SIMT=2`）。

- d（独立进程、全程 level 2）：`7 16 8 220 17 15 17 15`，8000 读 15547 ms（与 level 0 相同，8K 上
  tiled 选核不多花时间）、8 生成 1102 ms（比 level 0 慢 18%——**8K 的 decode 确实不该用 2**，
  这正是 3d 要按长度选的理由）。
- e（同进程 S→L→S，阈值 1024）：S 出 `79790 …`（= a 的 S）、L 出 `7 16 8 220 17 15 17 15`（= d 的 L）、
  再 S 又出 `79790 …`。两行都打了：`select kernel 0 -> 2 (8000 tokens)`、`select kernel 2 -> 0 (3 tokens)`。
  **两个方向都等于"各自独立进程"的输出**，这就是 3d 的验收。`0 window graphs dropped` 是对的：
  `ev_done_` 只在 `capture_all`（`--pipeline-windows`）里创建，所以 `drop_solo_graphs` 的早退守卫
  只挡住"图不是我们的"那种情况；TP 窗口图解析到 `tp_exec_`，实测没被烤进选核选择（否则 e 的第二
  个 S 会用 L 的图，token 必变）。

顺带清掉一行我早前排查 P2P 时留下、**每个 decode 窗口都往 stderr 打**的调试 `fprintf`
（`verify.cpp` 的 `run_tp` 开头，a 的日志里 14 行、e 里 21 行）。它每次都是一个无缓冲 write 系统调用，
留在里面既脏又会吃掉一点 decode。

### 9.4 3e 与 160K 头号指标（本次要交的成绩）

#### 9.4.1 闸门：solo == TP（已过），分阶段是"重新钉定"（已定性）

同一提示（8000 token，8 generated；短请求 `9707 11 1246`），同一钉定驻留（4000 槽、`--adapt-every 0`），
四种实现各自一份全新进程：

| 实现 | 长提示 8 token 的 md5 | 短请求 8 token |
|---|---|---|
| a：纯 TP（`--tp`，chunk 8192） | `f17fe819…` | 相同 |
| solo：单卡全宽，chunk 8192 | **`f17fe819…`（与 a 逐位相同）** | 相同 |
| solo：单卡全宽，chunk **4096** | `b087b22e…` | 相同 |
| c5：分阶段（层切角色 @4096 + 交接 + TP 解码） | `6e8c9999…` | 相同 |

结论三条，都有数据支撑：
1. **solo == TP 成立**（第 1、2 行逐位相同），这是 3e 真正要证的那一腿。
2. **仅改 chunk 边界就会重新钉定**（第 2、3 行不同）。所以"分阶段的 token 与纯 TP 不同"**不能**当成 bug 的
   证据——本仓库早有同类结论（驻留集一变 token 就变）。
3. 分阶段与 solo@同 chunk 仍不同（第 3、4 行），来源是层切角色的驻留集（只 admit 自己那 24 层）与两段
   流水线求和顺序。**短请求在四种实现下逐位全同**，这是"层切角色本身算对了"的旁证。
   因此分阶段模式和层切模式一样，是**换实现的重新钉定**（§8.1-4 早已写明），需要自己的钉定串。

#### 9.4.2 交接本身（c5 实测，8000 token 提示）

```
strata phased: handover at 7999 tokens: 12 QSA layers 110.7 MiB, 36 GDN layers 113.9 MiB, PLE 360 KiB, in 6.4 ms
```

225 MiB / 6.4 ms ≈ **35 GB/s**，走 NVLink（2080Ti NVLink2），相对任何一次 prefill 都是噪声。
"7999" 是设计使然：最后一段的终点是 `n-1`，最后一个 token 由解码角色的首个窗口接上。

#### 9.4.3 160K 头号指标

（本节数字由 `/tmp/tp_160k_three_20261028.sh` / `/tmp/tp_160k_hi_20261028.sh` 一单元一臂产出，每个臂一个全新
进程：g = 纯 TP、h = 分阶段（`--tp --layer-split 24`，无额外开关）、i = 分阶段 + 六个 Turing 开关。对比基线：
串行层切 160K ≈1450 tok/s prefill / 37.6–42.4 tok/s decode；TP 采纳配置 870.3–870.8 / 861.9 prefill、32.1
decode。注意基线是**长期运行的服务器（专家热）**，本轮各臂是**冷启动**，见 §7 冷热 2.6 倍。）

| 臂 | 配置 | prefill | decode | 备注 |
|---|---|---|---|---|
| g | 纯 TP，`auto` = 5,550 槽，8192 分块 | 845.5 tok/s | 27.5 tok/s | 今天的纯 TP 基线 |
| h | 分阶段，`auto` = 4,674/4,368 槽，3072 分块 | 854.1 tok/s | 27.8 tok/s（128 token，草稿 99/141） | 交接 64.3 ms；分块被 CUDA1 的整宽表压到 3072 |
| i | h + `STRATA_PA_V2T/SM75_IL/SM75_EXP/EXP_MODE=8/ATTN_DAILY/GR_DOWN_TG=2` | 838.6 tok/s | 10.3 tok/s，且第 28 个 token 就 stop（草稿 9/69） | 这组开关在 CUDA 上是错的：回复退化、草稿命中 13%，不可用 |
| **k** | **分阶段，两卡钉 3,488 槽，8192 分块，CPU share 按默认关** | **1,167.4 tok/s** | 26.3 tok/s（草稿 108/142，命中 10.5%） | 分块从 3072 回到 8192 = +37%；两卡驻留 2,084（KV 拿走 1,404） |
| m | k + `STRATA_SM75_PERF=1`（位级等价伞） | 1,171.4 tok/s | 26.8 tok/s | +0.3%，伞开关在 160K 无杠杆 |
| l | k + `STRATA_PREFILL_CPU_SHARE=1` | 1,172.9 tok/s | 26.5 tok/s | +0.5%，8192 分块下 CPU 分摊可有可无 |

四条结论（写给下一步）：

1. **h 的"分阶段只打平纯 TP"是假象**：h→k 的差别只有"分块 3072 → 8192 + 两卡同槽位"，prefill 就
   从 854.1 涨到 1,167.4（+37%），比纯 TP 的 845.5 快 38%。分阶段的层切角色本身是对的（160K 上
   attention 占大头，层切把 12 个 QSA 层一分为二，TP 却把 attention 整宽复制到两卡）。
2. **k/m/l 三臂在噪声内**（1,167.4 / 1,171.4 / 1,172.9）：图灵位级伞和 CPU 分摊都不是 160K 的杠杆，
   唯一有效的是分块。三臂 decode 也一致（26.3–26.8），说明 decode 的天花板在结构上，不在开关。
3. **交接本身没问题**：64.2–64.3 ms，2,170 MiB QSA + 114 MiB GDN，20 个分块总共 1.3 s，占 137 s 的 1%。
4. **decode 的结构账**：TP decode 每层都整宽复制（attention 全宽做两遍），层切只用一半卡做一半层。
   160K 一个 window（6.4 token）要 243 ms，纯 TP + 生产开关集只要 148.6 ms——差的就是 attention
   那部分被复制。所以要超过串行 decode，必须要么切上下文（flash-decode 分半 K/V），要么在 160K 上
   干脆用层切 decode（那时候分阶段就退化成串行，没有收益）。



#### 9.5 双切点层切（prompt 在 A 读，decode 在 B 跑）

**设计**（用户指定的第 3 条）：长 prompt 在 **A = 24** 读、token 在 **B = 20** 解码。两卡各有一套 session：
CUDA0 按 `stage0_hi() = max(A,B)` 开 carve（[0,24)），CUDA1 的 session 是 [B,48)（是它 prompt 角色 [A,48) 的超集），
所以整个过程只有一份 session、一份权重表，不需要复制第三份。带 [B,A) = layer 20（QSA）+ 21,22,23（GDN）的
序列状态在 prompt 读完时**一次**搬到 CUDA1（整宽 GDN plan、单目的地、纯拷贝），decode 之后不再搬。decode 的两个
verifier 范围都取自 B（`generate.cpp:9798-9804`、`9889-9893`），所以 [B,A) 在 decode 既不重复也不遗漏。

**第一版为什么错**（已修）：`host_res` 是**跨卡合并**驻留表 —— `generate.cpp:365` 的注释写明"条目属于拥有该层的
那张卡"，`7975-7978` 用 `stage_of(l)` 构造。单切点自洽（每个 stage 只跑自己拥有的层），双切点不自洽：stage 0 的
prompt 跑 [0,24)，它自己的缓存只覆盖 [0,20)，于是 layer 20–23 的条目里是 **CUDA1 的槽号**，被当成 CUDA0 的
槽号去寻址。两种表现、同一个缺陷：

- 缓存钉小（`--expert-cache 1000`）：越界 → `prefill mmq: gather_native: an illegal memory access`，0 token；
- 缓存大（160K，`auto` = 5543 槽）：槽号恰好落进自己的显存 → 读到错误专家字节 → 第 5 个 token 后 logits 非有限
  （position 160004）。

**修法**（`src/prefill/prefill.cpp`，一处入口）：prefill 不再无条件信任合并表。`init()` 逐层比较"合并行"与
"本缓存自己的 `slot_of` 行"，记进 `res_row_ok`；`res_slot(l,e)` 对**本缓存自己的层原样返回合并表的值**（所以
solo / TP / 单切点位级不变，包括 `STRATA_SPLIT_SMALL_OWN` 的驱逐语义），只有合并行属于别的卡的层才改问本缓存
自己的 `residency_`（同形 O(1) 数组读，不是哈希）。十处消费点
（`prefill.cpp:2843/3756/3977/4019/4526/4542/4946/4986-4987` 一带）全部改走它；Aurora 驻留排序钩子只在行归属
本缓存时才拿到行指针（`res_row`），否则传 null。启动打一行证据：
`strata prefill: residency table: N of 48 layers belong to another card in the merged map (layers x-y),
M of them inside this stage's [lb, le)`。

**600-token 三臂验证**（`/tmp/tp_dual_fixcheck_20261028.sh`，单元 `strata-tp-dualfix-20261028`，全部钉
`--expert-cache 1000`、`STRATA_PREFILL_CPU_SHARE=0`、16 token）：

| 臂 | 配置 | 结果 |
|---|---|---|
| c600 | `--layer-split 20` | 16 token，草稿 6/13，`residency table` 在本 stage 范围内 0 层属别人 |
| e600 | `--layer-split 20 --prefill-cut 20`（空带） | **与 c600 逐位一致**：token 16/16 相同、`STATE_HASH` 全同、`STATE_HASH_GDN` 15 行全同 |
| d600 | `--layer-split 20 --prefill-cut 24`（原崩溃臂） | **16 token 完成，EXIT=0，无 illegal access**；草稿 6/13；证据行 `4 of them inside this stage's [0, 24)`；前 15 行 GDN 哈希与 c600 相同（layer 0–19 的 prompt 不受双切点影响），第 16 行起按预期不同（那正是搬过去的层的行） |

**方法论结论（写给以后）**：层切下"钉槽位位级 oracle"结构性不可用 —— 改切点就改每卡的层片与空闲显存，驻留集
随之改变，而驻留集改 token（见 `tp-residency-changes-tokens`）。层切唯一成立的位级 oracle 是**空带恒等**
（`--prefill-cut` == `--layer-split`：carve、角色、驻留集全不变，必须与不加该开关逐位相同），其余只能靠
"有限性 + 草稿接受率正常"。

#### 9.5.1 第二个根因：prompt 的借出把另一张卡的槽号当成自己的槽（这才是 40K/160K 崩溃的原因）

上一条修好后，600 token（1 chunk）干净通过，但 **40800 token 仍然在第 1–11 个解码 token 后非有限**，
而**空带臂（`--prefill-cut` == `--layer-split`）在 40800 上逐位等于单切点**。两者唯一的差别就是 stage 0 的
prompt 真的去跑 [B,A) 这 4 层。先把"分块"证伪：把四个臂（c48/e48/d48/d48w0）的 prefill 时序逐行比对，
**都是 13824/13824/40799/13151 四次读** —— 40K 与 600 的分块结构一样，分块不是触发条件，
`stage 1 一次读 40799` 是层切本来的行为。

真正的缺陷在 `generate.cpp` 的 `lend`（prompt 借出专家缓存槽）。它遍历 `p.lb..p.le` 并用**跨卡合并的
`host_res`** 判断"槽号 ≥ `first` 就借"：

- 合并表里第 20–23 层的条目是 **CUDA1 的槽号**（`generate.cpp:365` 注释已言明条目属于拥有该层的那张卡）；
- 槽号只是数字，CUDA1 的槽号落在 CUDA0 的借用区 `[first, slots)` 里完全可能；
- 借出后 `refill_issue` 执行 `fill_slot_queued(slot, blob_of(layer20))` —— 把**第 20 层的专家灌进 CUDA0 自己的
  槽 <那个外来编号>**，而那正是本卡某一层某个专家的槽；`host_res` 却仍说那个槽是原来的专家 ⇒ 表自洽、
  权重已错。decode 从第一个 token 起就在读错权重，几个 token 后发散成非有限。

边界完全对得上：600 token 的请求级借用区按 1200 token 的 chunk 大小算，`first ≈ 5700`，而 CUDA1 的槽号
最大 5311 —— 一个都够不到，所以不炸；40800 token 的 chunk 是 13824，`first ≈ 3500`，CUDA1 带层里约三分之一
的 resident 槽号 ≥ `first`。`--expert-cache 1000` 的 600-token 复核之所以通过，同样是因为小缓存下
`first` 高于 CUDA1 的全部槽号。

**修法**（`src/program/generate.cpp`，两处，都是"只借本缓存自己的行"）：

- `lend` 的借出判定改成 `host_res[i] >= first && host_res[i] == p.cache->slot_of(l, e)`：合并行**就是**本缓存
  这一行时才借（`slot_of` 同 `prefill.cpp` 的 `res_row_ok`，O(1) 数组读）。单切点下每行都成立，
  路径逐字节不变；双切点下带层被跳过，CUDA1 的驻留一个不动，prompt 照旧把带层专家从主机流式取。
  非零跳过数打一行证据：`CUDA%d's loan for N tokens skips M experts whose slots are the other card's`。
- `STRATA_SPLIT_SMALL_OWN` 的驱逐扫描同样加上 `r == p.cache->slot_of(l, e)`（那条只写 `kNotResident`，
  不会写错槽号，但不加会把另一张卡带层的 resident 从合并表里永久抹掉：槽还占着、表却说没有）。

**40800 token 三臂复核**（`/tmp/tp_dual_fix2_20261029.sh`，单元 `strata-tp-dualfix2-20261029`，
`--prefill 16384`、`--pipeline-windows 2`、`--expert-cache auto`、16 token）：

| 臂 | 结果 |
|---|---|
| d48 `--layer-split 20 --prefill-cut 24` | **EXIT=0，16 token，无 non-finite**；读 1251.6 tok/s，草稿 6/23，命中 74.4%；证据行 `skips 301 experts whose slots are the other card's`；交接 [20,24) 1 QSA 46.1 MiB + 3 GDN 9.4 MiB / 22.1 ms |
| c48 `--layer-split 20`（控制） | EXIT=0，16 token；读 1197.7 tok/s，草稿 4/18，命中 74.9%；**skip 行一条都没有**（守卫对单切点惰性） |

**d48 与 c48 的输出 token 完全相同**：`1450 77 220 17 15 17 18 25 220 15 25 220 15 25 220 15`。
解码速度 11.8 vs 13.4 tok/s 不构成比较：token 相同而草稿接受数不同（6/23 vs 4/18），
decode tok/s 由接受率主导（见 `decode-tokps-dominated-by-acceptance`）。

**逐 ordinal 的 GDN 哈希对齐（比 token 相同更强）**：`STRATA_STATE_HASH_GDN=1` 每行一个 ordinal。

- d48 的 stage-0 行比 c48 多 3 个哈希，这 3 个正是带层 ords 15/16/17，**与 c48 在 stage 1 上算出的同一
  ordinal 逐字符相同**（`bde9 3e0a 2984`）；
- d48 的 CUDA1 行以 `8383 8383 8383` 开头（交接尚未发生，那 3 行还是零态），其后 18 个哈希与 c48 的
  同一 ordinal 一一相同（整体错位 3 行，因为 c48 的 stage 1 拥有 [20,48)、d48 的拥有 [24,48)）。

即**全模型 36 个 GDN ordinal 在 40800 token 上双切点与单切点逐位一致**：带层在 CUDA0 上算出的序列状态
与它在 CUDA1 上算出的完全一样，交接搬的正是这份状态。这也说明"带层 prompt 算术"本身没问题 ——
之前那个非有限纯粹是专家槽被写坏。

**方法论补充**：空带恒等只证明"双切点管线是惰性的"，**证明不了它与借出/驱逐机制的交互**。
这一轮真正定位问题的两个 oracle 是：①借出跳过数的证据行（301 vs 0，机制是否真的被触发）；
②跨切点按全局 ordinal 对齐 GDN 哈希（带层换卡算的结果是否逐位相同）。以后再动 prompt 的槽位管理，
这两个都比"能跑完"更早给出答案。

#### 9.5.2 160K 实测：prefill +8.6%，decode 无结构性代价

（`/tmp/tp_160k_dual_fix_20261029.sh`，单元 `strata-tp-dualat-fix-20261029`，一臂一进程；两臂共用一个
160000 token prompt、`GEN 128 spec_min_p=0.20`、`--spec 4 --pipeline-windows 2 --expert-cache auto
--prefill auto --max-context 204800 --kv int8 --vram-reserve-mib 700 --pcie-frac 0.55
--adapt-every 0 --adapt-swaps 0`、`STRATA_BF16_TC=1` + 那六个 Turing 开关。）

| 臂 | 结果 |
|---|---|
| **24on20**（`--layer-split 20 --prefill-cut 24`） | EXIT=0，**128 token，0 次 non-finite**；读 84947 ms = **1883.5 tok/s**；decode **22.0 tok/s**（34 窗，170.84 ms/窗，3.76 token/窗），草稿 98/113，命中 50.6%（5544 槽）；交接 [20,24) 于 159999 token：1 QSA **180.9 MiB** + 3 GDN 9.4 MiB / **25.2 ms**；借出跳过 516 个外卡槽；6 检查点 |
| **20**（控制，`--layer-split 20`） | EXIT=0，128 token，0 次 non-finite；读 92276 ms = **1733.9 tok/s**；decode **29.1 tok/s**（39 窗，112.87 ms/窗，3.28 token/窗），草稿 90/100，命中 53.2%（5643 槽） |

**prefill +8.6%**（1883.5 vs 1733.9），把阶段层数从 20/28 变成 24/24 —— 正是设计第 3 条要的东西。
交接代价 **25.2 ms / 84947 ms = 0.03%**（190.3 MiB 走 NVLink ≈ 7.9 GB/s），可以忽略。
两臂都高于串行层切 160K 的 ~1450 tok/s 热基线（**新进程**口径，但专家 blob 的页缓存在同一 unit 里
`dd` 预热过；+19.6% / +29.9%）。

**decode 的 22.0 vs 29.1 不是退步，是窗口配比。** 日志里的 `pipeline classes` 直接给出两条账：

| 类（`A.spec` 0/1） | 双切点 | 控制 cut-20 | 每 token |
|---|---|---|---|
| fresh（链重开） | 24 窗 × 169.94 ms，3.08 tok | 12 窗 × 91.68 ms，1.58 tok | 55.2 vs 58.0 ms（双切点 **−5%**） |
| speculative（链续） | 10 窗 × 155.58 ms，5.80 tok | 27 窗 × 116.45 ms，4.07 tok | 26.8 vs 28.6 ms（双切点 **−6%**） |
| 合计 | 24×169.94 + 10×155.58 = 5634 ms ✓ | 12×91.68 + 27×116.45 = 4244 ms ✓ | — |

即 **同一类窗口内，双切点每 token 都更快**；聚合变差只因为双切点把 70% 的窗口放进了昂贵的 fresh 类
（对照臂 69% 在便宜的 speculative 类）。触发它的全是 token 流性质：门控拒绝 `below the gate` 22 vs 7，
suffix drafter 出手 11 窗 49/49 vs 2 窗 7/7。逐 layer-window 的单位成本也印证同一结论：
CPU 专家 7.56 ms（16.55 entry）vs 5.75 ms（13.33 entry）→ **0.457 vs 0.431 ms/entry（+6%）**；
PCIe 8.03 vs 5.90 → **0.485 vs 0.443 ms/entry（+9%）**；entry 数 +24% 正好等于 T 比 4.32/3.56 = 1.21。

**为什么 token 流会不同**：两臂 decode 的层→卡映射完全相同（都是 [0,20)/[20,48)），唯一差别是驻留
——双切点把 CUDA0 的 carve 撑到 [0,24)，专家槽 5544 vs 5643（−1.8%，命中 50.6% vs 53.2%），
这正是 `tp-residency-changes-tokens` 记的那条（改驻留 token 就变）。对照臂自身可复现：
修前批次 39 窗 / 3.28 token/窗 / 113.65 ms/窗，本批 39 窗 / 3.28 / 112.87（−0.7%），说明测试台是确定性的，
配比差异不是跑动噪声。

**这条要写进方法论**：比较带投机解码的 decode 时，`ms/window` 也会被窗口配比污染（本批 170.84 vs 112.87
就是纯配比）。干净的 A/B 是 **class 内 ms/token**（本节的 55.2/26.8 vs 58.0/28.6）或固定宽度的非投机臂。

**仍未回答的**：两臂 decode 都低于串行层切的热基线 37.6–42.4 tok/s，但口径不同 —— 本轮是冷启动、
`--spec 4`、`--adapt-every 0`（关自适应），基线是长期运行、`--spec 8`、自适应开着的服务器。
按 §9.4.3 的冷热 2.6 倍和 §4.7 的 spec 8 杠杆，这个差距不能归给双切点，需要在生产配置（spec 8 + 自适应开）
下再取一次 160K decode 才算数。

#### 9.5.3 生产口径（`--spec 8` + 自适应开）160K：双切点 decode 打平，prefill +8.5%

同一脚本只改两处（`--spec 4`→`--spec 8`、去掉 `--adapt-every 0 --adapt-swaps 0`），
`/tmp/tp_160k_prod_20261029.sh`，单元 `strata-tp-dualprod-20261029`：

| 臂 | prefill | decode | 窗口 | 草稿 | 命中 | adapt |
|---|---|---|---|---|---|---|
| **24on20** | **1882.5 tok/s**（84992 ms） | **42.0 tok/s**（3046 ms） | 28 窗，4.57 tok/窗，108.78 ms/窗 | 102/119 | **88.1%** | 71.6 swaps/窗 |
| **20**（控制） | 1735.2 tok/s（92209 ms） | 41.5 tok/s（3081 ms） | 29 窗，4.41 tok/窗，106.25 ms/窗 | 104/128 | 86.1% | 68.9 swaps/窗 |

- **prefill +8.5%**，与冷口径的 +8.6% 一致 —— 这条是结构性的，与 spec/自适应无关。
- **decode 打平**（2827 vs 2822 ms / 128 token，差 0.2%，在噪声内）。这一轮两臂的窗口配比也几乎一样
  （fresh 20 窗 × 104.90 ms 3.50 tok vs 21 窗 × 100.72 ms 3.29 tok；speculative 8 窗 × 91.07 ms 7.50 tok
  vs 8 窗 × 88.30 ms 8.00 tok），所以这次是真正的同条件对照：**双切点在 decode 上没有代价**。
  冷口径那轮的 22.0 vs 29.1 由此可以判定为 `--adapt-every 0` 下驻留不同引起的 token 流配比差异。
- 自适应打开把命中率从 50.6% 抬到 **88.1%**，decode 从 22.0/29.1 抬到 42.0/41.5 —— **自适应不是可选项**，
  冷口径的 decode 数字不能拿去和生产基线比。
- 对串行层切 160K 基线（~1450 prefill / 37.6–42.4 decode）：**prefill +30%**；**decode 42.0 落在基线区间的
  上沿**（基线最好一次 42.4），是打平不是超过。想"完美超过"还差一个 decode 杠杆，见 §5 / §10 的
  GPU-reach wait（本臂 36.16 ms/窗 = 窗时的 36%）与每 expert 计算成本（VRAM hits 43.64 ms/窗）。

#### 9.5.4 decode 切点第三点（cut 22）与一个必须改的测量协议

`/tmp/tp_160k_cut22_20261029.sh`（生产口径、GEN 128），两臂：

| 臂 | prefill | decode | 窗口 | 草稿 | 命中 | class 拆分 |
|---|---|---|---|---|---|---|
| 22on24（prefill 24 / decode 22） | **1882.2** | 39.0 tok/s（3286 ms/128） | 22 窗，5.82 tok/窗，149.35 ms/窗 | 112/115 | 84.5% | fresh 22 窗 × 138.02 ms **6.09 tok**，speculative **0 窗** |
| 22（单切点 22） | 1819.1 | **43.8 tok/s**（2923 ms/128） | 31 窗，4.13 tok/窗，94.29 ms/窗 | 101/114 | 87.3% | fresh 24 × 90.43 ms 3.42 tok，spec 7 × 74.47 ms 7.14 tok |

**按 class 归一化（ms/token）才是可比量**：

| 臂 | fresh | speculative |
|---|---|---|
| 20（生产控制，§9.5.3） | 30.6 | 11.0 |
| 24on20（双切点，§9.5.3） | 30.0 | 12.1 |
| **22on24** | **22.7** | 无窗口 |
| **22（单切点）** | **26.4** | **10.4** |

两个结论：
1. **cut 22 的每 token 成本在两个 class 里都更低**（fresh −14%/−25%，spec −5%）——阶段 22/26 比 20/28
   均衡是真的有效；但 22on24 那一轮的 token 流一次 draft 链都没续上（0 个 speculative 窗、5 次门控拒绝），
   全部走昂贵的 fresh 类，聚合反而只有 39.0。
2. **128 token 的样本已经被配比噪声主导**：同一套结构，聚合在 39.0–43.8 tok/s 之间摆动，
   而每窗成本其实很稳。这解释了 §9.5.2 那次 22.0 vs 29.1 的全部差额。

**协议修正（§9.5.6 已再修正一次，见下）**：160K 的速度臂一律 `GEN 512`。~~decode 从 128 加到 512 只多约 10 s，
而 prompt 那 85–92 s 是固定的；窗口数 ×4，配比才平均得掉。比较顺序仍然是
**class 内 ms/token → ms/窗 → tok/s**。~~
**这条已作废**：`GEN 512` 太短，切点排序在 512 和 2048 之间会翻；最终口径是 `GEN 2048` + `--ignore-eos`
\+ `--suffix-draft 0`，见 §9.5.6。比较顺序不变。

#### 9.5.5 `GEN 512` 口径的五个臂：双切点 prefill 稳 +8%，decode 掉 6-11%

`GEN 512`、160K、生产口径（`--spec 8` + 自适应开）。三条臂在同一个 unit 里跑，24on20 跑两轮当重复性检查。

| 臂 | prefill | decode | 窗口 | token/窗 | ms/窗 | fresh ms/token | spec ms/token | 命中 | 接受 |
|---|---|---|---|---|---|---|---|---|---|
| 双切点 24on20 #1 | 1874.9 | 73.3 | 81 | 6.32 | 86.22 | 18.18 | 8.06 | 94.7% | 431/484 |
| 双切点 24on20 #2 | 1869.7 | 70.7 | 81 | 6.32 | 89.40 | 18.79 | 8.39 | 94.4% | 431/487 |
| 单切点 20（控制） | 1733.1 | **78.1** | 74 | 6.92 | 88.55 | 16.31 | 7.92 | 93.8% | 442/464 |
| 单切点 22 | 1818.9 | **78.6** | 79 | 6.48 | 82.43 | 16.33 | 8.14 | 94.4% | 433/456 |
| 双切点 22on24 | **1880.1** | 65.5 | 70 | 7.31 | 111.61 | **14.60** | —（0 个） | 94.0% | 448/451 |

**结论一：prefill 的 +8% 是稳的。** 两轮复现（1874.9 / 1869.7，差 0.3%），对控制臂 1733.1 是 +8.2%；
与 §9.5.2 的 128 口径（1882.5 vs 1735.2，+8.5%）一致 —— 三次独立测量同一个结论。

**结论二：decode 上双切点落后 6-11%，不是噪声。** 两轮 24on20 的窗口数、每窗 token、接受数（431）
完全一样，decode 73.3 / 70.7；控制臂 78.1、单切点 22 是 78.6。**最好的 decode 是不带双切点的单切点**
（20 与 22 打平）。

**结论三：落后的直接原因是草稿接受率，不是管线。** 24on20 两轮都是 431/484 = 89.1%，控制臂 442/464 =
95.3%。两臂 decode 都用切点 20，差别只在 prefill 用了切 24：带层 [20,24) 换了主人，驻留轨迹跟着变，
两条 token 流从某处起分叉，接受率和每窗 token 数就分开了（6.32 vs 6.92）。class 内每 token 也更贵
（fresh 18.18/18.79 vs 16.31），但这一项同样被"每窗几行"污染（5.31 行 vs 6.19 行），不能单独当作
"双切点管线更慢"的证据。

**结论四：22on24 prefill 最好（1880.1）、decode 最差（65.5），因为它一次链式窗都没拿到。** 70 个窗全是
fresh，但每窗 7.31 token、MTP 接受 448/451 = 99.3%，所以它的 **fresh ms/token 14.60 是五臂里最低的** ——
同样的窗口类型它最快，只是窗口类型全是贵的那种。

**一个窗口成本模型**（详见 `AUDIT_2026-10-29.md` §2.1）：fresh 窗 ≈ **65 ms 固定 + 5.8 ms/行**，
第三个点独立验证误差 0.7%。所以 decode 的主杠杆是"每窗塞几行"，专家管线只占窗口的 11-14%；
但链式窗有 7.6-8.0 行却只要 61-66 ms/窗，明显低于模型，因此**这个模型不能外推到 8 行以上**。

#### 9.5.6 `GEN 2048` 口径的八个臂（2026-10-30 01:58–02:36）：prefill 切点定在 24，双切点作废

**口径再改一次，这次是最终口径。** §9.5.4 末尾定的"速度臂一律 `GEN 512`"作废了：`GEN 512` 太短，
切点排序会翻（`GEN 512` 上 cut20≈cut22，`GEN 2048` 上 100.7 vs 115.6）。现在一律 `GEN 2048`。
更严重的是一条**潜伏的 35% 误差源**：跑到 EOS 就停的臂会停在更冷的专家缓存状态里，命中率和 tok/s 双双下坠。
本次 `w22b` 只生成 1501 token 就 EOS，decode 87.4；`w24` 跑满 2048，decode 110.4 —— 同一份代码、同样的
槽位数（5613 vs 5636），差出来的 35% 全在"解码了几个窗口"上。已给引擎加 `--ignore-eos`
（`generate.cpp` 三处 EOS 判定点全部 `!o.ignore_eos &&` 门控），测量时逐臂解码同样多。

| 臂 | 层切 | prefill 切点 | prefill | decode | 生成 | 窗口 | token/窗 | ms/窗 | 命中 | 备注 |
|---|---|---|---|---|---|---|---|---|---|---|
| w24 | 24 | — | 1859.9 | **110.4** | 2048 | 276 | 7.42 | 67.21 | 97.8% | |
| w24b | 24 | — | 1861.2 | **111.1** | 2048 | 270 | 7.59 | 68.29 | 97.8% | 复现臂 |
| w23 | 23 | — | **1859.0** | 73.2 | 548 | 80 | 6.85 | 93.61 | 93.0% | EOS 截断，decode 不可比 |
| w22b | 22 | — | 1817.6 | 87.4 | 1501 | 198 | 7.58 | 86.71 | 95.7% | EOS 截断，decode 不可比 |
| mp0 | 22 | — | 1815.5 | 106.0 | 2048 | 259 | 7.91 | 74.57 | 97.3% | `--spec-min-p 0`（不剪窗） |
| pw0 | 22 | — | 1818.7 | 97.1 | 2048 | 259 | 7.91 | 81.40 | 97.4% | `--pipeline-windows 0`，0 个 suffix 窗 |
| d22o24 | 22 | 24 | **1879.0** | 95.8 | 2048 | 259 | 7.91 | 82.52 | 97.4% | 双切点，0 个 suffix 窗 |
| mtp0 | 22 | — | 1763.3 | 28.6 | 87 | 29 | 3.00 | 104.76 | 80.3% | `STRATA_MTP_STAGE0=1`，崩 |

**结论一：prefill 切点是 24（或 23），+2.4% 可复现到 0.2%。** 切 22 三臂 1817.6 / 1815.5 / 1818.7
（均值 1817.3，极差 0.18%），切 24 两臂 1859.9 / 1861.2（均值 1860.6，极差 0.07%），切 23 是 1859.0 ——
**23 和 24 打平**。这与 §9.5.3 的两阶段均衡模型完全吻合：wall ∝ max(A, 48−A)，24/24 是最平衡的点，
23/25 的比值 1.087 只比 1.000 差一点，所以分不出来。这也说明 **§9.5.2 里双切点那 8% 的 prefill 收益
其实是"24/24 均衡"的收益，不是双切点的功劳** —— 单切点 24 就拿到了。

**结论二：双切点作废。** `d22o24`（prefill 24 / decode 22，每 prompt 搬一次 [22,24)）prefill 1879.0，
比单切点 24 的 1860.6 只多 1.0%（在噪声边缘），代价是 decode 95.8 对 110.4（−13.5%）、槽位 5521 对 5613、
而且它这一跑一个 suffix 窗都没拿到。算术是对的，但没有一项赚回来。**用 `--layer-split 24` 单切点。**

**结论三：把草稿器挪到首卡是灾难。** `STRATA_MTP_STAGE0=1` 让平均窗宽从 7.4 行塌到 **3.76 行**，
decode 28.6 tok/s，命中率 80.3%。推理上"尾卡最忙、把 drafter 挪走"完全站不住 —— 引擎默认
（`mtp_first_default_on()` 在 CUDA 上返回 false，草稿器留在 CUDA1）是对的，这个方向封死。

**结论四：mp0 / pw0 都变慢，但这一批还不足以定罪。** 不剪窗（mp0）decode 106.0，`--pipeline-windows 0`
（pw0）97.1，都低于 24 臂的 110.4，可是它们跑在切 22 上、而且我们**没有一条跑满 2048 的、配置相同的
切 22 基线臂**（`w22b` EOS 截断了）。mp0 的 fresh ms/token 反而更好（10.16 对 10.90，符合固定成本摊薄的
预期），只是它拿到的 suffix 窗更少、更贵，总量才下降。所以"要不要剪窗""两阶段 decode 管线值不值"
这两个问题留到 §9.5.7 用钉死口径重测。

**结论五：本轮起，160K 速度臂的固定口径是** `GEN 2048` + `--ignore-eos` + `--suffix-draft 0`
（把 suffix 命中这个彩票变量钉零），比较顺序 `class 内 ms/token → ms/窗 → tok/s`。
`--suffix-draft 0` 只用于**排名**；发布口径另跑一条 suffix 打开的臂。

#### 9.5.7 `GEN 2048` 钉死口径的七个臂（02:39–03:12）：切点定 24，两阶段管线保留

口径 = `GEN 2048` + `--ignore-eos` + `--suffix-draft 0`（§9.5.6），最后一条例外：`s24prod` 把 suffix 打开当发布口径。

| 臂 | 层切 | 管线 | prefill | decode | 窗口 | token/窗 | ms/窗 | **ms/token** | 命中 |
|---|---|---|---|---|---|---|---|---|---|
| s24a | 24 | 2 | 1861.9 | 100.8 | 271 | 7.56 | 75.00 | **9.92** | 98.0% |
| s24b | 24 | 2 | 1861.4 | 101.0 | 271 | 7.56 | 74.85 | **9.90** | 98.0% |
| s22a | 22 | 2 | 1821.7 | 79.1 | 278 | 7.37 | 93.08 | 12.63 | 96.7% |
| s22b | 22 | 2 | 1821.6 | 58.2 | 365 | 5.61 | 96.42 | 17.16 | 96.1% |
| s22p0 | 22 | 0 | 1818.7 | 96.9 | 259 | 7.91 | 81.63 | 10.32 | 97.4% |
| s24p0 | 24 | 0 | 1860.9 | 75.2 | 302 | 6.78 | 90.14 | 13.30 | 97.1% |
| s24prod | 24 | 2 | 1858.4 | **111.5** | 267 | 7.67 | 68.82 | 8.97 | 97.8% |

**结论一：切点定 24，prefill 和 decode 同时赢。** prefill 两臂 1861.9 / 1861.4 对 1821.7 / 1821.6（+2.2%，两格各自极差 <0.03%）。
decode 在**同为全 fresh 窗**的可比条件下 74.85 / 75.00 ms/窗 对 93.08 / 96.42（快 21%），而且切 24 两臂逐位可复现
（窗口数 271/271、token/窗 7.56/7.56）。**"decode 该用切 22"这个旧结论死了**，它是被 suffix 窗喂出来的假象。

**结论二：慢的原因不是槽数，是每卡缓存压力。** 切 22 的槽数反而**更多**（5613 对 5521），命中率却更低
（96.7/96.1 对 98.0）。22/26 把 26 层的专家堆到尾卡上，而尾卡本来就是 decode 的瓶颈。**24/24 拉平的是每卡
的缓存压力，命中率是它的函数** —— 这条把下一步直接指向"提高有效缓存容量"。

**结论三：`ms/token` 才是可比量，`ms/窗` 不是。** 同一配置的 s22a/s22b 差了 36%（12.63 对 17.16），
差别不在窗宽（avgT 7.66 对 7.70）而在**接受率**：278 窗×7.37 对 365 窗×5.61。自适应交换竞态让两次运行的
token 流分叉，接受率跟着变。所以报 `ms/窗` 必须同时报 `token/窗`，否则不可比。

**结论四：不确定性只出现在"切 22 + 开管线"这一格。** 其余格子都是确定性的：切 24 开管线两臂完全一致；
`s22p0` 与上一批的 `pw0` 结构完全一致（259 窗、7.91 token/窗、81.63 对 81.40 ms/窗）。所以精细 A/B
可以在切 24 + 开管线上做，不必先买一个"冻结自适应"的口径。

**结论五：两阶段管线保留。** 切 24 上开管线 74.9 对关管线 90.1 ms/窗（快 20%）；但切 22 上关管线（81.6）
反而快过开管线（93-96）。两格方向相反、都被接受率污染，**唯一能复现的最优格是"切 24 + 管线 2"**，按它定。

**发布口径最好记录（本次）**：`--layer-split 24` + `--pipeline-windows 2` + suffix 打开 + `GEN 2048` 跑满，
**prefill 1858.4 / decode 111.5 tok/s**，68.82 ms/窗，65 个 suffix 窗 435/435 全接受。

#### 9.5.8 预取槽（`STRATA_FS_SLOTS`）实测：不划算，封死

审计 §4.1 的 L1（预取/前瞻，把 CPU 未命中搬离关键路径）此前被判死路、又因单位错误撤回、再改判"必须实测定"。
现在有数了：**它不划算，而且不是"差一点"，是差一个数量级。**

**臂 f4**（4 槽/层/卡，2 卡，reserve 760 与 §9.5.7 的切 24 对照完全相同）：

| 项 | 值 |
|---|---|
| prefill | 1859.6（与对照 1861.4/1861.9 一致，说明 chunk 划分没变） |
| decode | 95.7 tok/s，343 窗，5.97 token/窗，62.38 ms/窗，**10.45 ms/token**（对照 9.90） |
| 命中率 | 98.4% |
| 预取报告 | **2474 个专家由交换槽服务**，12784 次拷贝（其中 12056 次来自路由器前瞻），597 次没落地，**67747 次无空槽**，4595 次超预算 |

**结论一：它只接住了约 3% 的未命中。** decode 全程约 374 万次查表、命中 98% ⇒ 未命中约 8.2 万次；
交换槽真正服务的只有 **2474 次（3%）**，剩下 `67747 次无空槽` 说明槽位永远不够。

**结论二：为这 3% 付了 33 GB 的 PCIe 流量。** 12784 次拷贝 × 2.58 MB ≈ **33 GB**，而只有 2474 次被用到
（**81% 的拷贝字节从没被用过**）。这条臂按 token 统一算反而慢 **5.5%**（10.45 对 9.90 ms/token）。

**结论三：加槽这条路被 reserve 堵死。** 8 槽（495 MiB/卡）在 reserve 760 下只剩 63 MiB 空闲，
prefill 的 `cudaMalloc` 直接 OOM（EXIT=134），引擎自己提示"Add --vram-reserve-mib 1216" —— 但 §9.5.7 已证明
**改 reserve 就会换掉整条 token 流**，再测就不是同一件事了。

**结论四（结构性）：预取槽和专家缓存抢的是同一块显存，而缓存槽更聪明。** 一个"最近未命中"槽的利用率先验就
低于自适应缓存槽，拿缓存容量去换它，是稳亏的买卖。**L1 结案：不做。** 未命中的成本要靠"让未命中变少"
（命中率）或"让未命中的计算变快"（CPU 池）来解决，不是靠加一块更笨的缓存。

#### 9.5.9 两个纯开关在切 24 上都是中性（2026-10-30 04:0x）：`--lookup-chain 4`、`--pipeline-windows 3`

口径同 §9.5.7 的发布层（`GEN 2048` + `--ignore-eos`，suffix 开，`--spec 8`，reserve 760），对照用 `s24prod`。

| 臂 | prefill | decode | 窗 | token/窗 | ms/窗 | ms/token | 命中 |
|---|---|---|---|---|---|---|---|
| `s24prod`（对照） | 1858.4 | **111.5** | 267 | 7.67 | 68.82 | 8.97 | 97.8% |
| `lc4`（`--lookup-chain 4`） | 1862.8 | 111.1 | 267 | 7.67 | 69.03 | 8.99 | 97.8% |
| `pw3`（`--pipeline-windows 3`） | 1860.4 | 110.2 | 271 | 7.56 | 68.56 | 9.07 | 97.7% |

**结论一：两个开关都是中性的（±0.4%，在重复运行的噪声内），谁也不是生产端慢的原因。**

**结论二：`--lookup-chain` 在这个口径下根本没触发。** 能接长链的窗口本来就是 8.00 满窗（无剩余行可补），
fresh 窗没有可接的提示匹配。生产日志里更是直接写着 `lookup chain: 0 of 1518 windows` —— 它是个**空转开关**。

**结论三：`--pipeline-windows 3` 也不坏。** 这意味着生产配置里的 3 和 4 都不是问题所在，可以继续用。
（真因见 §9.5.10：是**每窗接受的 token 数**，不是每窗成本。）

#### 9.5.10 换真实提示后，之前所有 decode 数字作废：草稿门槛才是真杠杆（2026-10-30 04:3x–05:0x）

**起因**：查生产日志 `strata-nvfp4.log`（10-26）发现真实流量下每窗成本和我们一样（69.5 对 73.5 ms），
但每窗只接受 **3.04** 个 token（我们 7.60）。于是量了那条 160K 测试提示本身：**只有 2651 个不同 token，
95.0% 的 8-gram 是重复的** —— 它是个模式续写题。换成用本仓库源码拼的 160K 提示（8469 个不同 token，
重复 8-gram 15.3%，`/tmp/real160k_ids.txt`），同一套配置立刻打回原形。

同一提示、切 24、管线 2、reserve 760：

| 臂 | prefill | decode | 窗 | token/窗 | ms/窗 | ms/token | 草稿接受 |
|---|---|---|---|---|---|---|---|
| spec 8 + 门槛 0.20（旧口径） | 1507.7 | 61.7 | 383 | 5.35 | 86.70 | 16.21 | 1668/2488 = 67% |
| spec 8 + 0.70 | 1522.4 | 73.4 | 664 | 3.08 | 42.04 | 13.63 | 1387/1842 = 75% |
| spec 6 + 0.70（生产口径） | 1522.2 | 89.3 | 449 | 4.56 | 51.09 | 11.20 | 1604/1799 = 89% |
| spec 6 + 0.70 + 管线 3（生产完整口径） | 1530.0 | 101.6 | 388 | 5.28 | 51.95 | 9.84 | 1662/1783 = 93% |
| spec 4 + 0.70 | 1528.2 | 114.2 | 496 | 4.13 | 36.17 | 8.76 | 1552/1559 = **99.6%** |
| **spec 6 + 0.90** | 1529.6 | **116.7** | 300 | 6.83 | 58.51 | **8.57** | 1748/1752 = **99.7%** |

**结论一：`111.5 tok/s` 不是成绩，是填充提示的性质。** 同配置同切点在真实文本上只有 61.7。以后 decode 的
排名与发布数字必须用 `/tmp/real160k_ids.txt` 这一口径重算，填充提示只能当"结构测试+prefill 测试"。
prefill 也降了 18%（1860 → 1522），真实文本本来就更难读。

**结论二：每行成本是本构常量，吞吐由"每行接受多少 token"决定。** 六臂的 ms/行 几乎不动（10–11.5 ms/行），
真正变化的是每行换成多少 token：0.71（0.20）→ 0.91（spec6/0.70）→ 1.00（spec6/0.90）。**所以杠杆不是
让窗口更宽，而是别去验证注定被拒的行。** 两个赢家的草稿接受率都是 99.6%+。

**结论三：生产配置的 `--spec 6` 方向是对的，`--spec 8` 是填充提示的产物。** 门槛从 0.70 提到 0.90 在
spec 6 上再快 15%（89.3 → 116.7）；spec 4 + 0.70 与 spec 6 + 0.90 基本打平（114.2 / 116.7）。

**结论四：生产日志里旧二进制的每行成本是 15.8 ms（70.23 / 4.44），现在是 10.2 ms** —— 光把线上二进制
升到当前版本就约 +50%，与门槛无关，这两件事可以叠加。

> ⚠ **上面的结论二、三已被 §9.5.11 推翻，转发前先读那一节。** 门槛不是杠杆，每行成本也不是常量。

#### 9.5.11 门槛是噪声放大器，不是杠杆；成本模型与真实上限（2026-10-30 07:0x–07:2x）

run-7 用**完全相同的配置**重复了 run-6 的冠军臂，得到 86.9 而不是 116.7。顺着这条线把账算穿了。

**(a) 先立一条恒等式，它让所有报告可验算。** 投机解码每窗贡献 `接受的草稿 + 1 个 bonus token`，所以

```
生成的 token 数 = 接受的草稿数 + 窗口数
```

`s6p90b`：1201 + 847 = 2048 ✓；`s6p90`：1748 + 300 = 2048 ✓。**任何一臂对不上这个式子就是测错了。**

**(b) 34% 的摆幅全部来自"草稿链有多长"，而链长由逐草稿门槛卡出来的。**

| 臂 | 门槛 | decode | 窗 | token/窗 | ms/窗 | 每窗提出草稿 | 接受率 |
|---|---|---|---|---|---|---|---|
| s6p90（run-6 冠军） | 0.90 | **116.7** | 300 | 6.83 | 58.51 | 5.84 | 99.7% |
| **s6p90b（同配置重复）** | 0.90 | **86.9** | 847 | 2.42 | 27.82 | **1.44** | 98.1% |
| s8p90 | 0.90 | 101.0 | 318 | 6.44 | 63.78 | 5.66 | 96.4% |
| s4p90 | 0.90 | 69.3 | 1058 | 1.94 | 27.92 | 1.12 | 83.3% |
| s6p95 | 0.95 | 114.6 | 562 | 3.64 | 31.79 | 2.67 | 99.3% |

`--spec-min-p` 是**逐草稿**判定（generate.cpp:16552，`while (T < S_mtp && pr[from+T-1] >= floor) ++T`）。
真实文本上草稿概率正好游走在 0.9 附近，于是链长在 1.1–5.8 之间跳 —— 门槛越贴近草稿概率分布的众数，
摆幅越大（0.70 两次差 14%，0.90 两次差 34%）。**这不是杠杆，是把噪声放大。**

**(c) 成本模型（10 个真实提示臂，最小二乘）。**

```
ms/窗 = 9.56 + 8.75 * 行         R² = 0.59
```

- **固定开销只有 ~9.6 ms/窗，成本几乎全在"每验证一行 8.75 ms"。** 所以窗口开大并不值钱：
  T→∞ 时 tok/s 的上限是 `1000/8.75 = 114`（100% 接受）。**116.7 已经在这条线上，甚至略超**
  （那一次 ms/窗 58.51 比拟合值 69.33 便宜 11 ms，说明它同时拿到了低行成本）。
- **R²=0.59 说明每行成本本身也在漂**（实测 8.57–16.21 ms/行）。残差有系统性符号：接受率最高的几次
  比预测便宜，`spec8/0.20`（67% 接受）比预测贵 30 ms —— **回滚是要花钱的**，模型缺一个"被拒行"项。
- 因此**单次 160K decode 数字不可排名**。要排名必须同一配置跑 ≥3 次取中位数，并同时报
  `ms/窗`、`token/窗`、`每窗提出草稿数`、`接受率` 四项。

**(d) 对下一步的意义。** 一卡的算力已经把这个引擎按在 ~114–117 tok/s 的天花板上，`--spec`/`--spec-min-p`
在这一层再调没有空间。**唯一还没动过的量是那 8.75 ms/行** —— 它才是 TP 该去减半的东西。而 TP 在真实提示
口径下**从未测过**（`62.8 vs 44.0` 那个记录是填充提示），所以 run-9 直接做这个 A/B，方法上也改成
**比 ms/行（斜率），不比 tok/s**，这样 token 流漂移不再污染比较。



### 9.6 这一轮改了哪些文件

| 文件 | 改动 |
|---|---|
| `src/program/generate.cpp` | 角色按段定（`ls_role && a0 == 0`）、一段只交接一次（`r == b0`）、`give_way` 对分阶段读不让位、§8.3 块（3d）与付费 relayout（`pf_ls0_borrowed`）、`PhStageWt` + 通用化 `load_stage_weights` + 整宽表装载顺序（3e）、`phased_arena` 失败走降档重试 + 计量行 |
| `include/strata/core/verify.hpp` | `drop_solo_graphs()` 提到 public |
| `src/core/verify.cpp` | `drop_solo_graphs()` 实现（进自己的 device、`capture_all` 的图不动）；删掉 `run_tp` 的调试打印 |
| `include/strata/core/kernel_switches.hpp`（新） | `select_simt()` / `set_select_simt()` / `select_simt_env()` |
| `src/kernels/cuda/qsa_select.cu` | sm<8 路径改成运行时读（`const int simt = strata::select_simt()`） |
| `include/strata/core/kv_redistribute.hpp` | 契约注释改成"弹性池可以，流式池不行" |
| `src/program/generate.cpp`（双切点） | `--prefill-cut` 及校验、`stage0_hi()`、CUDA0 的 session/dense carve 到 A、CUDA1 的 session 取 [B,48)（prompt 角色 [A,48) 的超集）、`band_handover`（每段一次、整宽 GDN plan、单目的地纯拷贝、空带即空操作） |
| `src/prefill/prefill.cpp`（双切点修复） | `res_row_ok` / `res_slot()` / `res_row()`：prompt 的驻留判定改问**本缓存自己的表**，合并行属于本缓存时原样走旧路径（solo/TP/单切点位级不变）；十处消费点全部改走它，Aurora 驻留排序钩子只在行归属本缓存时拿到行指针 |
| `src/program/generate.cpp`（双切点修复 2） | prompt 借出的两处只借**本缓存自己的行**（`host_res[i] == p.cache->slot_of(l,e)`）：`lend` 不再把另一张卡的槽号当本卡槽借出并 refill；`STRATA_SPLIT_SMALL_OWN` 的驱逐不再抹掉另一张卡的带层 resident。`lend` 打跳过数证据行 |
| `src/program/generate.cpp`（测速口径） | 新增 `--ignore-eos`：声明（729）、解析（2647）、三处 EOS 判定点（约 16935 / 17288 / 18677）各加 `!o.ignore_eos &&` 门控。**只影响何时停止，不进任何算术路径**，开与不开在同长度下位级同流 |

**上表里双切点（`--prefill-cut`）整条已在 §9.5.6 判定作废**，代码保留但不再使用，生产配置是 `--layer-split 24`；
其余各行仍在使用。`--ignore-eos` 不是优化，是**测量口径**：不钉住它，跑到 EOS 的臂会停在更冷的专家缓存里，
同配置 decode 能差 35%（见 9.5.6 的 `w22b` 87.4 对 `w24` 110.4）。

## 10. 遗留与下一步（按性价比排序）

1. **`--tp` 会话检查点不能续跑（agent 用法的头号缺口）。** TP 的 GDN 状态在 `gdn_state_tp_`（每 rank 一份
   切分 arena），检查点只存 `ss.gdn_state`（整宽）。所以 agent 每一轮都要把整段上下文重新 prefill。做法：
   存档时把每 rank 的切分 arena 原样写进镜像（按 `QsaState` 逐项），恢复时直接灌回 `gdn_state_tp_`，
   恢复路径**跳过** `kv_redistribute`。这是"长上下文 agent"最值钱的一项。
2. **分阶段模式要求 batch 空闲（`bs.empty()`）。** 一个长提示与任何在飞请求重叠时不会被授予层切角色，
   而是退回纯 TP prefill——正确但慢。若 agent 场景是"一问一答"，可接受；多路并发要另行设计。
3. **`qsa_kv_elastic_cells(0)` / `need(0)` 硬编码 owner 0。** `kv_redistribute` 接受弹性池，但取容量时
   写死 0 号 owner（`kv_redistribute.cu:321-324`）。分阶段里两个角色的池都属于 owner 0，今天成立；将来
   若给 rank 1 单独建池就会静默按错 owner 算容量。
4. **`--tp` 的 BYIELD 路径 GDN 洞（既有问题，非本轮引入）。** `copy_to_slot` 只带 QSA，让位给解码后
   GDN 状态对不上；分阶段读已经用 `!phased_ls` 挡住，纯 TP 的 BYIELD 还开着。
5. **`NativeDense::set_layer_range` 是进程级静态**（`native_dense.cpp:40`）。分阶段路径上已经"设-复位"
   成对出现，但任何**未来**的多阶段构造都必须记得复位；更稳的做法是把它改成 RAII 守卫按调用栈生效。
6. **CUDA1 的 2,245 MiB 整宽表挤掉了 8192 chunk**（见 §8.4 更正）。**已修一半**：`auto` 定槽时现在扣掉
   每卡整宽表（`phased_hold_mib`，`generate.cpp:6513` 附近，只作用于定槽、不动 `--vram-reserve-mib` 别的
   用途）；rank 1 的 cap 也扣掉它自己的整宽表（`generate.cpp:10076` 附近）。剩下的问题：`auto` 在 rank 0
   上先定槽，rank 1 的 cap 后算且可能更小，两卡驻留集合于是变成"前缀"关系（rank 0 多一批），而 GPU 命中/
   CPU 未命中的算术不同 → 不再是等槽位口径。**分阶段现在就钉 `--expert-cache <cap>`**（启动日志会打印
   cap 和建议的 pin 值）。三条进一步的路：(a) 把 rank 1 的定槽提到 rank 0 之前（要先把 `tps.ss` 的构造
   前移）；(b) 把 pack 的整宽表换成"只装 native 未覆盖的张量"（今天不行，`native_dense.cpp:372` 要求每个
   native 张量在 canonical 表里都有 ref，得先加容错）；(c) 让层切角色直接用 `--pack` 的 Q8_0 dense（省
   1,474 MiB，但数值换成 pack 口径，需重新钉定）。
