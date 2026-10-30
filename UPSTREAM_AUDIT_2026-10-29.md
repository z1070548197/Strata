# 上游审计（2026-10-29）：`477ac13c..upstream/main`，2252 提交

> 两个审计 agent 交叉扫（一个专攻正确性/状态，一个专攻性能），结论按"对我们这个工况"排序。
> 我们的工况定死：2×2080Ti sm_75、NVLink2、层切 24、NVFP4-GPTQ native pack（每个专家 blob 有 16 B 尾）、
> `--spec 4 --pipeline-windows 2 --vram-reserve-mib 960 --ple-io mmap`、长上下文 agent 多轮。

---

## 0. 一句话结论

**上游没有修我们的 NaN。** 但提交 `2a9a67c1` 正文里记的 #3638 症状和我们的几乎逐字相同，而且是一条**完全不同的机制**
（后卡显存将满 → 窗口图静默创建失败），HEAD 上这些路径是**响亮失败而非静默 NaN**，机制未证；作者自己写 "no model was run"。
我们的根因（两张驻留表 / 借贷按 `slot_of` 门控 → 漏借 117 个 → 被提示词缓冲区覆盖成 NaN）是读代码 + run-51 字节级证据直接证明的，
不依赖上游，**修复按我们自己的走**。

~~**上游最该抄的一件事和 NaN 无关，而且今天就能免费做**：PLE 表常驻 RAM（逐位一致，PLE 块 451 ms → 132 ms）。~~
**〔已实测判负，2026-10-29 run-56〕`--ple-io ram` 让 160K 冷 prefill 慢 7.5%、PLE 等待 +29%。**
原因是本机装不下：专家 arena `cudaHostRegister` 了 63.28 GiB + 表 28.8 GB = 91.8 GiB > 93.6 GiB 总量，
`ulimit -l` 硬上限 11.7 GiB 让 mlock 直接失败。**三档 direct / mmap / ram 至此全部有实测，direct 最好。**
详见 `RESULTS_2026-10-29.md` §12.46。

---

## 1. 和 NaN 直接相关的提交

| hash | 标题 | 实际改了什么 | 判定 |
|---|---|---|---|
| `2a9a67c1` | Later-card VRAM: audit fixes for failed allocations, the reserve arithmetic as a tested helper, a per-card OOM report (#3638 #3702 #3797) | 提交正文原话：**"#3638: 268 and 666 MiB [后卡空闲], then every batched prompt read of more than ~400 tokens ends non-finite after hours"**。修一串**静默失败**：① `cudaGraphInstantiate` 失败时用 `cudaGetErrorString(ce)` 报错 → 打印 "no error"（**我们 HEAD `src/core/verify.cpp:2500` 正是这段**）；② `instantiate_evicting` 直接写成员，失败留下未定义句柄，而 `capture_commit()` 用 `commit_exec_ != nullptr` 当"已捕获"（**HEAD `verify.cpp:2381`**）；③ `cudaGraphUpload` 失败不清 pending error；④ `Prefill::carve` 共享区 take 失败后各子分配器仍在将满的卡上 `cudaMalloc` 数百 MiB。新增 `stage_reserve.hpp` 把 `stage_room` 算法抽出并**量化了 `--spec 4` 的缺口：后卡窗口图组需要 174 MiB，计划只留 96**。 | **MAYBE（最佳匹配）**。症状（layer split + 后卡 + >400 token 的 batched prefill + 数小时后 + non-finite）几乎逐字对应我们。但作者声明 "no model was run"，根因未认定；HEAD 上这些路径多数是**响亮失败**而非静默 NaN。**仍值得在 NaN 修好后对照我们 HEAD 逐条查**——尤其 `capture_commit()` 用 `!= nullptr` 当已捕获这条。 |
| `894f133b` | Fix #3465: after non-finite logits in a prompt window the retry reads the whole prompt batched | 加 `nf_prompt_windows` / `short_read_off_once`：`nonfinite_hit` 回调里若 `what=="prompt"` 就置位，下一条请求强制 `req_short_read = 0`，让重试整条 prompt 走 batched 而不走那组 verify 窗口。新增 `STRATA_NAN_INJECT` 注入测试。 | **MAYBE（只是绕行）**。0.2.2 release notes 明写 **"the cause is still open"**。症状同形：尾部 prompt 窗口 NaN、"每次重试都同样失败"、`--short-read 0` 通过。**但我们不是这条**：resume 159999 后是 **513 token**，远超 `--short-read` 默认 64，NaN 未必落在窗口段；且我们已证实是缓存槽内容为 NaN。零成本，可留作对照。 |
| `8dd297d0` | PLE table: `STRATA_PLE_CHECK=1` names the first row that decodes to an implausible value | 诊断开关：对可疑 PLE 行用带缓冲的读重新读盘，打印 "the delivered bytes ARE the file's bytes" / "THE DELIVERED BYTES DIFFER FROM THE FILE from byte N"。 | **工具，该拿**。能把"PLE 交付了非文件字节"这条假设一次性证伪。注意：我们跑 `--ple-io mmap`。 |
| `ebe3383a` | PLE reader: a failed read no longer leaves the next ticket waiting for ever; stress test (#3602) | 唯一真 bug：错误路径上 I/O worker 永久 park 而 `issue()` 继续排队 → `collect()` 永久等待。明确声明**没有找到 #3602 那个 NaN 的读故障**。 | **NO**。#3602 前提是 `--ple-io direct`；我们 `--ple-io mmap`。 |
| `f8ff3420` | conversation cache: never park one conversation with another's retained K/V (#1503) | `retain()` 记录会话 token ids；`take_reuse_for(live)` 只复用共享前缀。 | **NO**。依赖 `--batch` + parking。 |
| `cea3ed40` / `01e404c1` | serve: slot checkpoint give-back / idle slot parking (#2526) | 都是 `--batch` 专属路径。 | **NO**。 |
| `789d8390` | `STRATA_QFUSE=1`: a one-token window did not self-commit, but `commit()` skipped the commit graph as if it had | `commit()` 用 `last_t_==1 && !always_publish_ && one_token_self_commit()` 跳过 commit graph，但 `record_window` 的条件还多一个 `!g_qfuse()`；于是 QFUSE 下每个被接受的单 token 窗口**把 GDN 循环状态和 indexer tail 落下一 token**。 | **NO**（`g_qfuse()` 在 `verify.cpp:182` 是纯 env，我们没开）。但"状态落后一 token"和我们"出事窗口 avg T 1.00"形状神似，**作为反证排除**。 |

---

## 2. 性能 / 稳定性：值得移植的（按性价比）

### 2.1 `--ple-io ram` —— 今天就能做，零代码，**逐位**

```
3b34b6c7  PLE 表常驻 RAM（RAM 够时自动锁，原来只有显式 --ple-io ram 才锁）
472de9cd  准入条件 = table + 专家 arena + 8 GiB（原来只算 table + 8 GiB，在 64 GB Windows 上把引擎搞死）
```

- **逐位**。上游原话：`"the answer md5 was identical in all three"`（direct / mmap / resident）。
- 上游 PLE 块 @8K：**845 ms(direct) / 451 ms(mmap) / 132 ms(resident)** —— mmap→resident **3.4×**。
- 我们的表 320001536 行 × 90 B = **28.8 GB**，RAM 93.6 GiB，满足准入。
- **⚠️ 更正（2026-10-29 15:19 实测）：我们的生产臂跑的是 `--ple-io direct`，不是 mmap。** 脚本里从来没传
  `--ple-io`，引擎每次启动都自己警告：
  `--ple-io direct measures slow here (11691 rows/s, about 48 MB/s of random 4 KiB reads) … kept direct:
  pass --ple-io mmap or --ple-io ram to change it`。**所以我们是三档里最慢的那一档。**
- `--ple-io ram` 这条开关在**我们本地树里就有**：`src/program/generate.cpp:4433` 已经是 `pio.lock = (ple_io == "ram")`。
- **⚠️ 本机 `ulimit -l` 硬上限 11.7 GiB**（非特权不可提），锁不住 28.8 GB → mlock 失败，引擎回落到
  "把整张表 touch 进 page cache"（不 pin）。专家包 68 GB 会把缓存挤出去，所以这是**必须实测**而不是相信上游的 3.4×。
- 我们现在的 PLE 成本：**热 160K 28.6 s**（见 PLE 结构化墙那份记录）——这是当前单笔最大可白拿的。
- **风险**：锁 28.8 GB 会把专家包挤出 page cache。按 `472de9cd` 口径：table + 专家 arena + 8 GiB 才算得下。
- **动作**：`--ple-io mmap` → `--ple-io ram` 单变量 A/B，比 tok/s 且**比 token md5**（应完全一致）。

### 2.2 `#3976` layer-major prefill —— 对我们**直接 decline**，除非自己补 NVFP4 尾

```
ffe40c16 + 6eff377a + 42094da9（18 提交）  一次读 8K~32K 超块，逐层跑完再交给下一层
```

- 每层专家**每超块只 staging 一次**而不是每 chunk 一次；作者实测专家流量降 ~S/512（12.9K prompt: 509 GB → 79 GB）。
- 作者实测（2×A4000，32K）：**首次出词 141.8 s → 27.7 s**；decode 不变（38.6 vs 38.4）。
- **⚠️ 我们用不了**：`prefill.cpp:5993` 明写
  ```cpp
  if (f.tail_off || mmq::fp4_activations(f.gu_type)) { why = "an NVFP4 layer"; return 0; }
  ```
  我们的 native 包每个专家 blob 末尾有 16 B 的 `{s_gate,s_up,s_down,0}` 尾
  （`src/kernels/cpu/native_expert.cpp:83-86`，`tail_off = bytes`）→ `tail_off != 0` → **整条路拒走**。
  合并后日志只会多一行 `layer-major off for this prompt: an NVFP4 layer`。
- **要自己补的活**：`moe_mmq` 的 `gather_native_group` / `gather_native` 早就有 `tail_off/tail_dst/zero_bytes` 形参
  （`include/strata/prefill/moe_mmq.hpp:109-117`），`run_impl` 全套用了（`prefill.cpp:5191-5198/5284-5285/5332/5349/5365-5367/5486/5510`）；
  **layer-major 的唯一 gather 调用在 `prefill.cpp:6519`，没传 tail**。补 `grp_tail` 缓冲区 + gu 缩放"边读边乘" + `s_down` 在 combine 里按行乘，
  估 **100–200 行**，删掉 5993 那一行 decline。改完与 chunk-major **逐位一致**（同一份 kernel、同一份 tail 字节）。
- **改算术**（原注释 `"another summation order of the routed experts, so KL-checked, not bitwise"`）：实测 KL 0.0007~0.009、
  top1 每次都相同、12/12 needle 全中、质量套件 18/19 三臂一致。
- ~~我们能拿到的：**冷 160K prefill 估 103–110 s → 75–85 s**（命中 30.0 s 的 combine）。~~
  **〔已推翻，2026-10-29〕那个 "30.0 s 的 combine" 是**记错桶**：聚合行把 PLE 停顿记进了 `combine`。
  分块行的真实 combine 各块只有约 250 ms，合计 **1.75 s / 105.6 s = 1.7%**（两次独立运行对上 0.2% 以内，
  见 `RESULTS_2026-10-29.md` §12.46）。**layer-major 在本工况最多省 1.7 s，不值得补 NVFP4 尾。**
- **⚠️ 升级注意**：`37e4832a` 让它 **CUDA 层切默认开启**（super-chunk 32768、VRAM margin 96→512）。升级后必须
  `STRATA_PREFILL_LAYER_MAJOR=0` 钉住，否则 chunk 几何全变、污染一切对比。另：与 `--prefill-pipe` **互斥**（`prefill.cpp:5963`）。

### 2.3 稳定性四连 —— 无算术影响，正是我们的工况

| hash | 内容 | 为什么是我们的 |
|---|---|---|
| `3478382e` | 会话拷贝 stream/event 池化（#3350 #3470） | `conversation_detail::private_stream()` 用 `thread_local` 存 stream/event **且从不销毁**；层切的后续 stage 每个 prompt chunk 跑一个新 `std::async` 线程 → **每个长冷 prompt 的每次 checkpoint 都在后卡留几 MiB 显存，永不归还**，直到 `cudaStreamCreate` OOM 打死引擎。我们正是层切 + 反复长冷 prompt + 多轮 agent。 |
| `a8c1d4b7` | 窗口图 OOM 时按 largest-first 释放别的尺寸图（#3702） | 我们 `--spec 4` 有 1..6 共 6 个尺寸 × `--pipeline-windows 2` 两套 verifier；HEAD 上这种情况是**直接终止引擎**。 |
| `f62589fd` | checkpoint 拷贝 OOM 改为跳过而非终结引擎（#3350） | 同上。 |
| `2a9a67c1` | 见 §1 的静默失败四连 | 见 §1。 |

### 2.7 `STRATA_SM75_PERF=1` —— **最大的一笔免费午餐，之前完全漏掉了**

`docs/DETAILS.md:2398` + `include/strata/core/sm75_series.hpp`：PR **#2069-#2072, #2081, #2091-#2093,
#2145-#2146, #2220-#2223, #1821** 送进来一整组 **compute capability 7.5 专用 kernel**，上游原话：
*"measured on one **RTX 2080 Ti** (and a CMP 50HX) and are **bitwise equal** to the kernels they replace"*，
**decode +5..6% @4K-32K，prompt 最深 +30%**。他们默认关掉的理由是 **"We have no sm_75 card to confirm them"**
——**我们有，而且就是 2080 Ti。**

- 我们的臂只按名字开了其中两个（`STRATA_SM75_IL`、`STRATA_SM75_EXP`）。总开关 `STRATA_SM75_PERF=1` 还会打开：
  `STRATA_GR_UP75`、`STRATA_SCORES_HALVE`、`STRATA_TOPK_SPLIT`、`STRATA_TOPK_SAMPLE`、`STRATA_ATTN_V3`、
  `STRATA_Q2_WARP`、`STRATA_FUSE_WAIT=2`、`STRATA_QIL`、`STRATA_TC_SM75`、`STRATA_PA_V1_PREFETCH`、
  `STRATA_SCORES_THREAD`。
- **全部已在我们树里**（逐个 grep 确认存在）。零代码、单变量、逐位。
- 其中 `STRATA_FUSE_WAIT=2` + `STRATA_QIL`（*fewer launches per decode layer*）正打我们 decode 预算的痛处：
  run-55 的 `per-layer host 3.56 ms/window` 与 `GPU-reach wait 20.17 ms / 32.54 ms 窗`。
  `STRATA_SCORES_THREAD` / `STRATA_TOPK_SAMPLE` 打 prefill 的 `qsa select`（111→817 ms/块，随位置线性增长）。
- 明确**不属于**这个开关的改算术项（各自单独 opt-in，我们已在用）：`STRATA_SELECT_SIMT`、`STRATA_PA_V2T`、
  `STRATA_PF_FUSED_SM75`、`STRATA_BF16_TC`。
- **动作**：run-58，A B A B 四次独立引擎启动，同一 unit、同一 160K 四轮工作负载，单变量。
- **〔2026-10-29 16:05 审计，见 RESULTS §12.47〕这 11 个里真正会改变我们工况的只有 7 个**
  （qsa_select 的 `SCORES_THREAD`/`SCORES_HALVE`/`TOPK_SPLIT`/`TOPK_SAMPLE`、`qsa_decode_attn` 的 `ATTN_V3`、
  `qsa_prompt_attn` 的 `PA_V1_PREFETCH`、`fused_gr` 的 `GR_UP75`）。另外四个是空的：
  `EXP` 布局（所有 `em0` 分支都要求 IQ 类型 id，我们是 NVFP4=40）、`Q2_WARP`（只在 `native_q2_0_mmvq` 内）、
  `TC_SM75`（dequant_bf16 只覆盖类型 2/6/7/8/11/12/13/14/20/23/42），`SM75_IL`（基线一直显式 =1）。
  而且 `FUSE_WAIT` / `QIL` 的默认是 `device_is_sm75()`，**本来就开着**，不算这个开关的功劳。
  **推论：生产臂里的 `STRATA_EXP_MODE=8` 是幽灵变量，可删；run-59 的 A→B 步骤作废。**

### 2.4 `STRATA_GRAPH_PRIO=1`（`0049fa73`）—— 逐位

窗图按节点优先级派发，上游称 `"bitwise by construction"`，对准我们 decode 窗里 **29.73 ms 的 GPU-reach wait**。

### 2.5 `STRATA_SPEC_COST`（`68781055` + 6 提交）—— **改 token，且方向和我们的数据相反**

`DraftPolicy::best_rows()` 用链式概率 × 实测"概率→接受率"表 × 实测各窗宽往返成本，选 E(提交 token)/ms 最大的窗宽。
**我们的实测是"行越多越快"**（8 行 → 153 tok/s，见窗成本模型那份），纯 E/cost 最大化器可能收敛到更窄的窗反而掉速。**风险高。**

### 2.6 `a3ab37fc` `STRATA_DEFAULTS=026`

一个开关把 v0.2.6 之后改过的默认值全部退回，显式 flag 仍优先。**升级时的安全带。**

---

## 3. 否定结论（省时间，别再碰）

- **`--kv-resident` 上游完全没动**：`src/kernels/cuda/kv_stream.cu` 在该区间**零提交**。升级修不了它。
- `GDN_TC` 是 sm_80+ / 122 KB 共享内存**硬墙**——读代码即判死。
- `PF_FUSED_F16` 在 Turing 编成 `__trap()` 桩。
- int8 prompt attention 要求 `cc >= 80`。

---

## 4. 动作清单

1. ~~**（今天，零代码，逐位）** `--ple-io ram` 单变量 A/B；比 tok/s 和 token md5。~~
   **〔已做，判负〕** run-56：−7.5%，PLE 等待 +29%。**PLE 三档全部测完，direct 最好，这条线关死。**
2. **（NaN 修好后）** 对照我们 HEAD 查 `2a9a67c1` 的静默失败四连，尤其 `capture_commit()` 用 `commit_exec_ != nullptr` 当"已捕获"（`verify.cpp:2381`）和 `cudaGraphInstantiate` 报 "no error"（`verify.cpp:2500`）。**我们 `--vram-reserve-mib 960` 是有意压紧的，这条路径真有机会踩到。**
3. **（升级时）** `STRATA_PREFILL_LAYER_MAJOR=0` + `STRATA_DEFAULTS=026` 双保险。
4. ~~**（可选，1.5–2 人日）** 自己给 layer-major 补 NVFP4 尾，拿冷 160K prefill −25 s。~~
   **〔作废〕** 真实 combine 只有 1.75 s，见 §2.2 的更正。**不要做。**
4b. **（下一个候选，191 行）** 移植 `0049fa73` `STRATA_GRAPH_PRIO`——见 §2.4，对准 decode 窗里
   20 ms 的 GPU-reach wait（占窗口 62%）。**已确认：我们树里 `instantiate_evicting` 在，缺的只是
   `graph_prio_on` / `cudaGraphInstantiateFlagUseNodePriority` / 优先级流那三处。**
5. **（可选）** 移植 `3478382e` / `a8c1d4b7` / `f62589fd` 三个 OOM 相关的生命周期修复，防长跑后卡显存单调下降。

---

## 5. 来源

- 区间 `477ac13c..upstream/main`，2252 提交，两次独立审计（正确性/状态 + 性能）。
- 引用行号均为**上游 HEAD** 的行号，不是我们分支的。
