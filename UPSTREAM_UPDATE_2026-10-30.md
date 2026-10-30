# 上游更新与可移植清单（2026-10-30）

基线：本树 `merge/v0.1.45`（merge-base `477ac13c`），上游 `Niko1221/Strata` main = `870aedd9`（v0.2.8，engine 版本串 0.1.54），
我们落后 2322 提交、领先 177。本文件只记**能落到我们这台机器**（2×RTX 2080 Ti sm_75 + NVFP4-GPTQ，生产 `--layer-split 24`）的条目。

## 0. 结论

上游 0.2.x 的提交九成与我们无关：SYCL/Arc/Metal、`src/dsv41/` 的 DeepSeek 新引擎、batch/多槽流水线、
sm_80/90/120 的默认表、Windows/WDDM。真正值得动的只有 6 项，按「改动大小 / 是否改 token」分三档：

| 档 | 项 | 改什么 | 改 token？ | 预期 |
|---|---|---|---|---|
| A | `tool_call_recovery: true` | 生产配置加一行 | 否 | 工具调用成功率（不是速度） |
| — | `--prefill-pipe` | **空操作**：基类 `pipeline_chunk()` 恒返回 cap、无 override | — | 已作废；真开关是 `STRATA_SPLIT_PF_CHUNKS`（layer split 默认开），见 §1.2 |
| A（提醒） | `--pcie-mode dma` | 生产未启用 | 否（逐位） | 我们自己实测 +6.2%，还没上线 |
| B | `fec7d1ca` | `fused_gr.cu` 3 行 | 否（逐位 + 启动自检） | 上游 V100 +0.5~0.6% decode |
| B | `83081577` | `verify_kernels.cu` 4 行 | 否（逐位） | 上游 V100 +0.7~0.8% decode |
| C | `cfa197e2` | 先当**诊断**：不改代码 | 会（改槽数） | 解释 reserve 曲线，指出 1205 实验 |
| C | `d7113273` + 前置两补丁 | ~200–300 行解析层 | 否 | 量化 agent 的 tool-call 成功率 |
| C | `70f42407` + 别名组 | 解析层 | 否 | 未声明工具名不再冒充调用 |

不做整体升级 0.2.8（理由见 §4）。

## 1. A 档：只改配置 / 开关，不改 token

### 1.1 serve 的 `tool_call_recovery`（最值得先做）

机制在我们树里**已经完整存在**，只是默认关、生产配置里没这个键：

- `serve/server.py:3381` `self.tool_call_recovery = False`（默认）
- `serve/server.py:7466` `recovery = cfg.get("tool_call_recovery", False)`；`:7468` 校验；`:7469` `svc.tool_call_recovery = recovery`
- `serve/server.py:4492` `parser = OutputParser(..., recover=self.tool_call_recovery, ...)`
- 解析侧 `serve/frontend.py:1012`；测试 `serve/test_server.py:1058 / 1167 / 1173`
- 生产配置 `/media/zc/软件1/strata-models/strata-nvfp4-gptq.json` 里**没有任何含 "tool" 的键** → 现在是关的

作用：工具调用的参数被模型写坏（漏引号、多逗号、缺 `</function>`、多余的 `</parameter>`）时，不再整轮退化成纯文本，
而是修复后照常发出。对量化 agent 是直接的成功率，**不动任何算术 / token**。
落地 = 配置加 `"tool_call_recovery": true` 后重启服务（需用户点头，生产配置不擅改）。

### 1.2 `--prefill-pipe`（更正 2026-10-30：在本树是空操作）

`generate.cpp:4941` 只把值交给 `Prefill::set_pipe()` 存起来；`Prefill::pipeline_chunk()`
（`src/prefill/prefill.cpp:2231`）基类恒 `return cap`，全仓库 grep 无任何 override，`plan_chunk()`
（:2232）恒返回 0 → **这个旗标改变不了任何分块，不用测**。

上游的效果在我们树里由另一套机制实现，且**默认已开**（`extras/mgpu-fixes`）：

- `generate.cpp:15291-15360` + `split_rules::pipeline_chunk`（`include/strata/program/split_rules.hpp:174`）：
  一个 prompt 段切成 k 个等长块（256 对齐），块下限 `STRATA_SPLIT_PF_MIN`（2048）；
- 默认 k 由「借方缓存持有其层专家对的比例」`pf_res_min` 与阈值 `STRATA_SPLIT_PF_RES`（0.5）决定：
  ≥0.5 → `k = max(fewest, 2×parts)`、下限 2048；<0.5 → `k = max(fewest, parts+1)`、下限 **8192**；
- `fewest = ceil(tokens / prefill_chunk)`，`--prefill auto` 下 `prefill_chunk = 32768`；
- 我们生产实测：CUDA1 缓存 6106 / 12288 对 = **0.497，刚好低于阈值** → 默认 k=3、下限 8192。
  日志实证：`17515 prompt tokens in chunks of 8960`（k=3）、`120078 → 30208`（fewest=4 → k=4）。

所以真正要 A/B 的是显式 `STRATA_SPLIT_PF_CHUNKS=k` / `STRATA_SPLIT_PF_MIN`（0/1 = 关）。我们从没在生产口径下扫过 k。
注意 150K 上 `k≤4` 与关等价（`ceil(150K/k)` 仍 >32768）。prefill 口径：热 2173 tok/s、冷 1430，跑间方差 9% → 必须多跑。

### 1.3 提醒项：`--pcie-mode dma` 已验证但未上生产

不是上游的，是我们自己的结论（记忆 `pcie-mode-dma-is-the-transport-win`）：mode 0 的 copy engine 比生产默认
（auto → mode 2，`fetch_blobs` 串行、约 2.9 GB/s）快 6.2%，token 逐位全同。生产 `args` 里没有 `--pcie-mode`。

## 2. B 档：3–5 行逐位补丁（不改 token）

共同前提：两处上游补丁的**改前版本与我们现状一致**，可直接抄。

### 2.1 `fec7d1ca` — 寄存器分裂 norm 在 CUDA 上也 opt-in

- 我们：`src/kernels/cuda/fused_gr.cu:1522-1536`，`norm_fast_on()` 的 `__HIPCC__` 分支有 gfx12 判定，`#else`（CUDA）直接 `return false;`
- 上游补丁 = CUDA 分支改成读 `STRATA_GR_NORM_FAST=1`（3 行）
- 为什么我们有效：decode 走 `launch_multi → fused_gr_variant()`；`env_variant()` 默认 `kHcStaged`，
  `fused_gr_check()`（`src/core/verify.cpp:747`，每卡一次）会**自检 staged/split 与 plain 逐位一致才启用**。
  所以这个新核上线后，启动自检会替我们验证它；不一致就不启用，token 不可能变。
- 上游实测 +0.5~0.6%（V100）；sm_75 未证实。
- **本机实测（2026-10-30，`build/fused_gr_bench 200 1 4`）：16 种 (T,apply,inject) 组合全部 `bitwise equal`，
  但 fast 列在本卡更慢** —— T1 43.0→46.5 µs、T2 50.7→54.3、T3 61.5→64.7、T4 72.9→76.1 µs（约 +8%）。
  结论：**补丁已打（上游原文，保持 opt-in），生产不设 `STRATA_GR_NORM_FAST`**。

### 2.2 `83081577` — sm_7x 的 `argmax_rows` opt-in

- 我们：`src/kernels/cuda/verify_kernels.cu:1217-1232`，`argmax_rows_wanted()` 只认 `major == 8`
- 上游补丁 = `major == 7` 时读 `STRATA_ARGMAX_ROWS_SM70=1`（4 行）
- **它对我们仍然有用，尽管生产是 `temperature 1.0`**（已核到行）：
  1. `src/core/verify.cpp:2463` 的贪心 pick 是**录进图里、每次 replay 都跑**的（采样请求回放后再由主机重采样，图里这一发照跑）；
  2. `src/core/mtp.cpp:1220` 的草稿头 pick 恒为贪心 argmax —— `--spec 6` 每窗 6 行 × 词表。
  两条在 sm_75 上现在都落到 `sample_tokens(greedy)` 的一块核。
- 同族已在跑：`multi_block_head_ops()` 默认 true（`STRATA_MULTI_BLOCK_ARGMAX` 未设），`row_top_prob_split` 早就在我们卡上用着。
- 逐位证据本地可出：`src/kernels/verify_parity.cpp:70-100` 就是 `argmax_rows` vs `sample_tokens(greedy)` 的逐位对比，补丁后直接跑。
- 上游实测 +0.7~0.8%。
- **本机实测（2026-10-30）：`verify_parity --selftest` PASS**（"bitwise equal to the one-block kernels (ties, NaN, -inf, 1-8 rows on one scratch)"）；
  整模逐位 A/B 也通过（见 §7）；**已加入生产 env**。

### 2.3 B 档验收协议

1. `git diff` 确认只有这两处；
2. 固定 `--expert-cache N`（不用 auto），跑固定种子的 `GEN 8`，比 8 个 token 逐位 + 六份 md5（口径见记忆 `tp-9341-is-the-deterministic-token`）；
3. 测速一臂一 unit（`multi-arm-units-are-swap-biased`）；<10% 差异必须多跑。

## 3. C 档：要移植 / 会改 token

### 3.1 `cfa197e2` — auto 缓存漏算 ~245 MiB 迟到分配（**不改代码也能用**）

上游事实：`--expert-cache auto` 从「权重后的空闲 − reserve − draft head − 已计价缓冲」定容，但加载后仍会分配
约 245 MiB（cuBLAS handle/workspace、懒加载模块、verifier 小缓冲），这些**从 reserve 里出** → 卡上实际余量比
`--vram-reserve-mib` 少约 245。

我们树的定容公式形状不同（`generate.cpp:6574-6790`），已经 price 了 `phased_hold_mib` / `prefill_mib` /
`mtp_bind` / `pipe_first`（`:6681`），所以**不能照搬**。但它给了我们一条读数解释：

- 我们调的 `--vram-reserve-mib 960`，**真实余量约 715**；760 那一档（≈515，贴着引擎自己 512 的告警线）崩掉正好对上。
- `--vram-reserve-later-mib` **只是「加载完成后余量低于它就告警」的阈值**（`generate.cpp:12603` 及其后的告警文案），
  **不参与定容** → 生产里的 1018 是告警线，不是余量。
- 可做的事（无需移植）：想做「真实 960」的实验 = **名义 1205**。这是我们 reserve 曲线（760 崩 / 896→111.1 / 960→115.3）
  唯一没扫过的方向，而余量是我们实测里唯一稳定推动 decode 的变量。
- 若真要移植：新增 `include/strata/core/auto_cache.hpp`(47) + `src/core/auto_cache_test.cpp`(43) + `generate.cpp`(20) + `CMakeLists`(5)；
  **会改槽数 → 会改 token**，必须重钉基线（`tp-residency-changes-tokens`）。

### 3.2 `d7113273`（前置 `77794e34` / `b72f56e0`）— 宽松参数层

让「参数被写坏」的 tool call 仍能解析出来：`_Lenient` / `lenient_json` / `coerce_value`、`</parameter></tool_call>` 终止、
缺 `</function>`、游离 think 标签、`tool_args_repaired_total` 计数；上游带 49 例语料，点名 Qwen3.8-Flash-Next 与 `todo_write`。
我们树里 `serve/frontend.py:692 param_end(text, final=False)` 是老签名，**两个前置补丁必须先落**（`param_end(..., name=)`、`_QUOTED_OPEN`）。
规模 ~200–300 行，不改 token（纯解析层）。对量化 agent 的价值可能比 §1.1 更大：开关只兜「已能识别的坏参数」，这层负责「根本没识别出来」。

### 3.3 `70f42407`（+ `8bf178fe` / `fd8d37e9` / `805ebe17`）— 未声明工具名不外发 + 别名表

我们 `_finish_call` 在 `repair_names` 关时会把**未声明**的工具名当成调用发出去，前端拿到一个不存在的工具名。
同一组还带 Codex 的 `namespace=member` 形式与 `strata_rejected_tool_calls` 计数。不改 token。

### 3.4 其余 C 档（备查）

`11237f83`、`7adced3d`（Anthropic 用量字段 `cache_creation_input_tokens`）、`07d5ecfa`（`STRATA_ARENA_RAM_WAIT_S`）、
`2cc6ea16`、PLE 自检。
`aec327ee`（`gr_read` 6→4 launch）：**搁置** —— 我们 `src/kernels/cuda/gr.cu` 里根本没有 `gr_read` 这一层
（整条 `gr_read` 重构都不在树里），不是 5 行，是连锁移植。

## 4. D 档：明确不做

- **整体升级 0.2.8**：会带 layer-major prefill 默认、batch/多槽 pipeline 默认、sm_80/120 默认表，以及 `STRATA_DEFAULTS=026`
  这个「只改默认值」的主开关。升级必须钉 `STRATA_DEFAULTS=026` 或 `STRATA_PREFILL_LAYER_MAJOR=0`，否则基线全废、token 全变。
  收益侧 0.2.x 对我们只剩两项：auto 定容（§3.1 已单独拿走）与 16 行窗（移植极大，且 CUDA 宽窗上游未验证、sm_75 被显式排除）。
- 16 行宽窗（`9d3c8c36`）：要 `-DSTRATA_WINDOW_MAX_T=16`、`kVerifyMaxT=8→16`，牵动整条 verify 图。
- `STRATA_EXPERIMENTAL_SM60`、sm_120 门、KVA、route prior、`--ple-io ram`（实测 −7.5%）、`src/dsv41/`、SYCL/Arc/Metal、Windows/WDDM。
- gated MTP（`STRATA_SPEC_PROB`）与 PIPED_MTP：改草稿律或需要多槽；我们单流，且草稿层三条路已实测封死
  （`decode-drafter-latency-beats-quality`）。

## 5. 建议顺序

1. §1.1 `tool_call_recovery`（一行配置，等用户点头）
2. §2.1 + §2.2 两个逐位补丁（一起编一次；跑 `verify_parity` + 固定槽 token 逐位）
3. §1.2 `--prefill-pipe` 生产旗标 A/B（长上下文优先）
4. §3.1 的「名义 1205」实验（只测速，token 必变）
5. §3.2 / §3.3 解析层移植（按需，不影响速度）

## 6. 核验记录
- 上游 commit 全部 `git show` 过（本仓库已 fetch upstream 对象，可直接读）：`fec7d1ca`、`83081577`、`cfa197e2`、
  `d7113273`、`70f42407`、`77794e34`、`b72f56e0`、`aec327ee` 均存在，标题与队友报告一致。
- 三处「我们现状」都读到行号：`fused_gr.cu:1522-1536`（`#else return false`）、
  `verify_kernels.cu:1217-1232`（`major == 8`）、`generate.cpp:2578/2616/6681/12603`。
- 生产配置只读白名单键（`sampling` / `args` / `env` / `model_name` / `gpu` / `layer_split`），未打印 `api_key`：
  `temperature 1.0, top_p 0.95, top_k 20`；`gpu [1,0]`；`layer_split 24`；args 与 env 见 §1.3 讨论。

## 7. 执行记录（2026-10-30）

已落地，生产服务已重启并端到端验证（真实请求 200 / `end_turn` / 366 token / 11.6 s）：

1. §1.1 `tool_call_recovery: true` 写入生产 config（备份 `strata-nvfp4-gptq.json.bak-20261030-toolcall`）。
   机制在本树自证（`/tmp/tcr_check.py`，直接用 `serve.frontend.OutputParser`）：
   - 裸 `<function=todo_write>`（没有 `<tool_call>` 包裹）：关 = 纯文本；开 = `todo_write{content:"hello"}`
   - `<tool_call>` 里写 JSON：关 = 纯文本；开 = `todo_write{content:"hi"}`
   - 正常模板形式：开关两种都解析成同一个调用
2. §2.1 `fec7d1ca` 补丁已打（`fused_gr.cu`），**不启用** —— 本卡实测 fast 版本逐位但更慢约 8%（见 §2.1）。
3. §2.2 `83081577` 补丁已打（`verify_kernels.cu`）+ env `STRATA_ARGMAX_ROWS_SM70=1` 写入生产 config
   （备份 `...bak-20261030-argmax`）；`verify_parity --selftest` PASS。
4. 整模逐位 A/B（`/tmp/parity_arm.sh`；除两个新 env 外生产参数不变：`--expert-cache 5362 --adapt-every 0
   --spec 4 --max-context 4096`；`GEN 8 9707 11 1246`；一臂一 systemd 单元、同单元 dd 预热）：
   - 旧二进制 `/tmp/strata-pre-patch`：`79790 316 52933 91 79751 198 262 9341`
   - 新二进制 + **两个 env 全开**：完全相同（`diff` 无输出）
   - 速度 103.95 vs 101.21 ms/窗（6 窗单跑）——在噪声内，不作结论。
5. 启动后核查：引擎子进程 environ 有 `STRATA_ARGMAX_ROWS_SM70=1`、无 `STRATA_GR_NORM_FAST`；
   引擎参数仍是 `--layer-split 24 --expert-cache auto --vram-reserve-mib 960`。

回滚：配置用上面两个 `.bak-*` 覆盖任意一档；二进制 `cp /tmp/strata-pre-patch build/strata` 后
`systemctl --user restart strata-model-200k.service`。

## 8. prefill 分块流水实测（§1.2 的真正开关，2026-10-30）

协议：一臂一 `systemd-run --user` 单元（`MemoryHigh/MemoryMax=infinity`、`ManagedOOMPreference=avoid`），
`dd` 预热与引擎同单元，`/tmp/benchpf.py` 从 `/tmp/real160k_ids.txt`（160000 token 真实仓库文本）按 seed 取切片，
`/v1/chat/completions`，服务端自报 `prompt_ms`。32K 漂移 1.7%，150K 漂移 5.6%。

**32K（32810 token 切片，seed 32768）**

| 臂 | 分块 | prefill tok/s | 相对默认 |
|---|---|---|---|
| 默认 k=3 | 3 × 11K | 1123.6（复跑 1104.4） | — |
| k=4 | 4 × 8K | 1108.3 | −1.4%（噪声内） |
| k=2 | 2 × 16K | 975.1 | −13.2% |
| k=8 + PF_MIN=2048 | 8 × 4K | 959.8 | −14.6% |
| 关（=1） | 1 × 32K | 702.3 | **−37.5%** |

**150K（150059 token 切片）**

| 臂 | 分块 | prefill tok/s | 相对默认 | 墙钟 |
|---|---|---|---|---|
| 默认（k=5） | 5 × 30K | 1392.7（复跑 1383.4） | — | 116.6 s |
| k=8 | 8 × 18.7K | 1483.5 | +6.5% | 110.2 s |
| **k=12** | **12 × 12.5K** | **1550.8（复跑 1555.9）** | **+11.4 / +12.5%** | **105.6 s** |
| k=16 | 16 × 9.5K | 1530.7 | +9.9% | 106.3 s |

**已叠加进生产**：`STRATA_SPLIT_PF_CHUNKS=12` + `STRATA_SPLIT_PF_MIN=8192`
（备份 `strata-nvfp4-gptq.json.bak-20261030-134129-stack`，env 20→22 键，args 与 api_key 未动）。
选这个组合而不是裸 k=12，是因为 `pipeline_chunk()`（`split_rules.hpp:174`）的 `n = min(k, tokens/floor)`：

- 32K → 4 × 8K（实测与默认打平），裸 k=12 会切 2.8K 小块（4K 块实测 −15%）；
- 48K → 6 × 8K、64K → 8 × 8K、96K → 12 × 8K；150K → 12 × 12.5K；160K → 12 × 13.3K；
- 8–16K 提示 `n ≤ 1`，行为与之前完全一样。

**上生产后复验中间长度与端点（2026-10-30，两臂都带 `--pcie-mode dma`，唯一变量 = 两个分块 env）**

| 长度 | 叠加前 | 叠加后 | 差 | 墙钟 |
|---|---|---|---|---|
| 32K（32810 tok） | 1116.3（3 × 11K） | 1137.7（4 × 8K） | 打平 | — |
| **48K（49211 tok）** | **1115.0（3 × 16K）** | **1257.5（6 × 8K）** | **+12.8%** | 49.0 → 43.9 s |
| 150K（150059 tok） | 1383~1393（5 × 30K） | 1551~1556（12 × 12.5K） | +12.5% | 116.6 → 105.6 s |
| 160K（160059 tok，端点） | 未测 | **1570.3（12 × 13.3K）** | — | 104.1 s（无 OOM） |

48K 的 +12.8% 含首位臂偏慢约 3.4%，扣掉仍 >9%。结论：**这个组合在 32K 打平、48K/150K 各 +12%+、160K 无异常**。

**上生产后的实测（in-situ，同一常驻进程）**：冷启动首个 150K 请求 1388.8 tok/s（= 旧冷态水平，说明冷态收益小），
随后**热态新切片 150K = 1745.8 tok/s**（墙钟 94.3 s）、32K = 1256.5。注意重复同一提示会命中会话缓存
（`cache_n ≈ prompt_n`、`prompt_n 7`），prefill 数字作废 —— 复测必须换 seed 取新切片。

**`--pcie-mode dma` 上生产（2026-10-30，第三次复验）**：在已叠加分块流水的生产形态下，32K 单变量两臂
（`base_nodd` / `dma_nodd`，无 dd、同 32810-token 切片、先跑 8K 热身）：

| 臂 | prefill tok/s | decode tok/s | 墙钟 |
|---|---|---|---|
| 默认（mode 2） | 1116.3 | 33.4（3.06 行/窗） | 39.0 s |
| **`--pcie-mode dma`（mode 0）** | **1137.7（+1.9%）** | **37.3（+11.7%，2.79 行/窗）** | **37.5 s（−3.8%）** |

墙钟差压在 3.6% 噪声底上，但 decode 优势是在**接受率更差**的臂上取得的，方向与 2026-10-29 的 −4.7%/−7.5% 一致，
且 token 逐位全同 → 已写入生产 args（备份 `...bak-20261030-140041-stack`，args 54→56，env 不变）。
注意 `--pcie-mode` 与 `dma` 必须是两个独立 argv 元素（`generate.cpp:2663` 精确匹配 `a == "--pcie-mode"`）。

## 9. 2026-10-30 速度轮（生产口径，一臂一 unit，无 dd 瘦身协议）

除注明外均为 32K 上下文（32810 token 切片）、每臂同进程连发 3 次（r2/r3 是会话缓存命中 = 纯 decode）、
`spec 4/p0.9` + `--pcie-mode dma` + 分块流水的生产形态。

| 候选 | 结果 | 处置 |
|---|---|---|
| `--spec 4 --spec-min-p 0.9`（原 6/0.7） | decode 中位 **48.8 vs 43.3（+12.7%）**，三次全胜，行/窗 3.6 vs 3.3 | **已上生产**（`...bak-20261030-142207-spec4`） |
| `--vram-reserve-mib 1205`（真实余量≈960） | 中位 48.8 vs 50.0（**−2.4%**），prefill 完全相同 | 不上 |
| `STRATA_IO_THREADS=64`（默认 16） | 150K prefill 1539.8 vs 1530.6（+0.6%，噪声内） | 不上 |

reserve 的结论把 §3.1 的路也堵死了：**「余量杠杆」只在 160K + GEN 2048 + ignore-eos 那个口径成立**，
短上下文/短生成下无效。生产 150K prefill 现在稳定在 **1530**（改造前 1383–1393）。

**合成 agent 画像 A/B（`/tmp/agentbench.py 8000 100 256 append`，与真实流量口径不同，只能自比）**：
两臂各一 unit、同脚本同日，唯一变量 = 今早的生产配置（旧：spec 6/p0.7 + mode 2 + 无分块 env）。

| 指标 | 今早配置 | 现在 | 差 |
|---|---|---|---|
| decode 中位 | 74.2 | **84.8** | +14.3% |
| 端到端 | 45.9 | **51.0** | **+11.1%** |
| 总墙钟 | 557.5 s | **502.4 s** | −9.9% |
| TTFT 中位 | 1940 ms | 1730 ms | −10.8% |

即今天三项（分块流水 + dma + spec4/p0.9）合计在 agent 式负载上 **+11.1% 端到端**。
注意合成提示是模板重复文本，草稿几乎全中，decode 绝对值偏高；**真实流量画像（46.9 tok/s，44K–160K）
不能拿这个数去比**。TTFT 在合成口径下占墙钟约 36%（真实流量首 token 中位 1.51 s），是下一个观察点。

下一步（未做）：§3.2/3.3 解析层移植（与本树差 271/221 行，需手工移植）。

## 10. 2026-10-30 下午：本地未设开关普查 + 会话快照开销（本轮最大发现）

### 10.1 普查口径

`src/` 里 `getenv` 到的 `STRATA_*` 共 800 个，生产 env 只设 22 个。按"注释里带量化实测"筛出 28 个
未设开关，逐条读实现后分成三类：

| 类别 | 开关 | 判据 |
|---|---|---|
| **死开关**（对我们无效） | `STRATA_ADAPT_LINK_BUDGET` | 只对 `link_gbps < 20` 且**探测过链路**的 stage 生效；生产给了 `--pcie-frac 0.55`，链路不探测 → 空操作 |
| | `STRATA_ADAPT_OVERLAP` | 启动日志 `not on a layer split`；且 `--pipeline-windows>0` 也禁用它 |
| | `STRATA_ADAPT_FETCH` | 启动日志 `is off (a layer split)`；fork 默认 2 在层切下不生效 |
| | `STRATA_ADAPT_NOWAIT` | fork 默认已开（wait 版慢 ~5%），无需设 |
| | `STRATA_GDN_KEYHEAD` | 需要 `major >= 8`，sm_75 自动关 |
| | `STRATA_BATCH_ATTN_ROWS` | 要求所有行 `kv_mode != 1`；我们 `--kv-resident 32768 < 200704`，主层恒为 mode 1（流式） |
| | `STRATA_EXPERT_V2K` / `STRATA_HC_Q8` / `STRATA_GFX12_*` | gfx12(RDNA4) 默认项，CUDA 上未测；`V2K` 还只对 Q4_K/Q5_K 流 |
| | `STRATA_IQ_GATHER` / `STRATA_SIMT2_TILE=2` / `STRATA_GR_DOWN_MAX4` / `STRATA_KQ256` | 注释自报更慢或持平 |
| | `STRATA_PREFILL_CPU_SHARE_MAX` | NVFP4 包默认已是 4096（fork） |
| **可疑但代价/收益不划算** | `STRATA_HC_REQ8=1` | 改输出（HC 投影重量化成 int8），换 ~0.56× 字节 → 更多槽位；质量有损，不做 |
| | `STRATA_B6_MMVQ_ROWS` | llama.cpp 多列布局，只到浮点舍入级相等；本卡未测 |
| **值得测（逐位不变）** | `STRATA_DF_PLE=1` | 窗口内 PLE 后投影一次过，注释 `bitwise the per-token loop` |
| | `STRATA_TSUM=1` | GR down 的 warp0 求和换成转置蝶形，注释 `bitwise the same` |
| | `STRATA_LFUSE=1` | 共享专家 gate 并入 router GEMV（少一次发射） |
| | `STRATA_MMVQ_V2=1` | attention k/v/q 三个 Q8_0 矩阵一次发射，`mmvq_v2_parity` 逐位；5090 上引擎内只省 0.05–0.1 ms/窗 |

另确认：`STRATA_PCIE_TIMING` 不在生产 config 里，但 **systemd 用户管理器环境里残留了它**
（`systemctl --user show-environment`），所有 unit 都会继承。实测 dma 模式下窗内 stamp 对为 0
（该路径没有图内 H2D），所以**零成本、不影响任何对照**；要清理就 `systemctl --user unset-environment STRATA_PCIE_TIMING`。

### 10.2 会话快照（conversation cache park）是每轮 300–370 ms 的纯开销

`nocpushare` 全程（100 轮）日志统计：**park 99 次、restore 0 次、skip 0 次**。
每轮序列固定为：

```
conversation cache: parked <N> tokens in 246–487 ms; bytes=0.4–1.2 GB; reused_kv_bytes=0
conversation cache: dropped 1 superseded copy of this conversation; parked=0
```

即：上一轮存的快照在下一轮开头被丢弃，**存了从来没被用过**（客户端 97% 的"缓存命中"来自会话内
retained K/V，不是快照）。而 park 发生在**新请求开头**（`ParkStage: "request: parking the previous
conversation"`），所以直接计入 TTFT：

| 量 | 值 |
|---|---|
| 客户端 TTFT 中位 | 1.73 s |
| 引擎自报增量 prefill（164 token） | 1.36 s |
| 差 | **≈0.37 s = park** |

代码里已有现成开关，默认关闭（`--conversation-cache-park-gain` 默认 0 = 每轮都存）：
`parked_prefix_tokens()` 会算出"这次 park 只会覆盖上次那份"，若本轮增长 < N token 就**跳过**，
旧快照保留（仍然是本对话的前缀，只是短 N 个 token；将来 restore 只需重算这 N 个）。
`drop_superseded` 只在同一对话、且其最深 checkpoint 被本轮链持有时才命中 → **子 agent 分叉不受影响**
（共享 root 的其它对话 `parked_prefix_tokens` 返回 0，照常 park）。

预期：合成口径（每轮 +420 token）N=1024 → 每 3 轮存一次，省 ~2/3 park ≈ 200 ms/轮 ≈ 4%；
真实口径（每轮 +~700 token）N=1024 → 每 2 轮一次，约省 0.2–0.3 s/请求。
**不改变任何算术，只改变快照频率。**

### 10.3 融合臂判负 + 上游异步 park 无法直接打补丁

**`winfuse` 臂（`STRATA_DF_PLE=1` + `STRATA_TSUM=1` + `STRATA_LFUSE=1`）100 轮完整跑完，三项全略差 → 判负，不进生产：**

| 指标 | 生产基线 | winfuse | 差 |
|---|---|---|---|
| decode 中位 | 84.8 | 83.7 | −1.3% |
| TTFT 中位 | 1730 ms | 1770 ms | +2.3% |
| 墙钟总 | 502.4 s | 509.6 s | +1.4% |
| 端到端 | 51.0 tok/s | 50.2 tok/s | −1.6% |

三者都被上游注释标为"bitwise"，所以**不是正确性问题，是零收益**；在本机口径下合并进
同一臂还轻微变慢（每个融合都多一层分支/量化路径选择）。结论：**这三个开关保持不设。**

**上游 `f73b3c86`（`STRATA_PARK_ASYNC=1`）与 `0f7224d1`（`STRATA_CKPT_ASYNC` / `STRATA_CKPT_VRAM_SLOTS`）
合计 562 行 / 11 文件，均不在本树，`git apply --check` 双双重放失败：**

- `src/core/conversation_state.cpp` 本树相对其父提交 **+10 / −161**（fork 重写过这段）；
- `src/program/generate.cpp` 的 park 段落（`park_source` / `park_worth`）本树已有自己的版本，hunk 对不上。

→ **只能手工重写**，而它是 K/V 生命周期与后台线程 join 语义（读方、cache 释放、`SavedConversation::pending`），
改错的后果是会话缓存里出现半拷贝的 K/V（静默错误输出），风险远高于收益。
**决策：先测 `--conversation-cache-park-gain`（纯配置、零代码），拿到大部分收益就不移植。**

### 10.4 `--conversation-cache-park-gain 1024` 未观察到效果（疑点未结）

| 指标 | 生产基线 | park-gain 1024 | 差 |
|---|---|---|---|
| decode 中位 | 84.8 | 84.8 | 0 |
| TTFT 中位 | 1730 ms | 1700 ms | −30 ms |
| 墙钟总 | 502.4 s | 488.7 s | −2.7%（噪声内） |
| 端到端 | 51.0 | 52.4 | +2.7%（噪声内） |

**判据不是均值而是分布**：若真跳过 87% 的停车，TTFT 应左移 ~380 ms 或出现双峰。
实测两臂 TTFT 分位数（p10/25/50/75/90）为 `1640/1700/1770/1830/1900` vs `1570/1620/1700/1770/1850`
——**单峰、整体平移 ~70 ms**，即停车没有被跳过。

关键佐证：winfuse 全程日志 `dropped 1 superseded copy` 打印在 `parked` **之前**，且 `parked=0`，
说明每轮开始时缓存里确实有上一轮那份、并在 `park_current` 里先被丢弃
（`generate.cpp:10772` 的 `drop_superseded` 在 park 之前）。所以 `parked_prefix_tokens()` 理应返回 >0，
`park_worth` 理应为假。**理论成立、实测不成立，机制尚未查清。**

⚠️ 测量坑：`parkgain1024.engine.log` 只写下了启动的 18 行（3.2 KB），mtime 停在 15:32:25，
**运行期日志在 SIGTERM 时整段丢失**（winfuse 那次没丢）。→ 该臂无法用日志数停车次数，
只能靠 TTFT 分布判断。下次带日志的实验要先确认日志能落盘。

**因此改测上限**：`--conversation-cache-min-tokens 1000000` 走 `wants()` 这条**必定生效**的跳过路径
（`conversation_cache.hpp:228`，只门控停车、不影响 retained K/V 复用），日志会打
`skip parking (N tokens, below the minimum; parked=...)`。若它的 TTFT 下降 ~380 ms / 墙钟 −7%，
说明停车确实值这么多，再去修 park-gain 的触发条件；若同样没动静，**"停车 0.37 s" 的归因就是错的，这条线关掉**。

### 10.5 零停车上限实测：TTFT 省 350 ms，但 decode 全部吐回去（净零）

`nopark` 臂 = 生产 + `--conversation-cache-min-tokens 1000000`，日志确认 **99 次 `skip parking`、0 次真停车**，
即这是一次干净的"完全没有停车"的整场测量。

| 指标 | 生产基线 | nopark | 差 |
|---|---|---|---|
| TTFT 中位 | 1730 ms | **1380 ms** | **−350 ms（−20%）** |
| decode 中位 | 84.8 | **70.2** | **−17%** |
| 墙钟总 | 502.4 s | 504.6 s | +0.4% |
| 端到端 | 51.0 | 50.7 | −0.6% |

**结论：停车那 0.35 s 是真的，但去掉它并不省时间，只是把时间从 TTFT 挪进了 decode 窗口**
（TTFT 总和少 35 s，decode 窗口总和多 40 s，总墙钟持平）。端到端"变快"是幻觉——`agentsum.py` 的
TTFT 和 decode 是分开统计的，只有总墙钟可信（对应 [[decode-total-wallclock-is-the-metric]]）。
本轮不值得为它移植上游 562 行的异步停车。**这条线按"净零"关闭，除非真实长上下文口径出现相反证据。**

⚠️ 附带发现：`parked_prefix_tokens()` 返回的是**被取代快照里最深 checkpoint 的 token 数**，不是快照长度。
本工况 `live` 每轮 5103→5346→5405→5663，而 checkpoint 链（`--lookup-chain 4`）最深那份老得多，
差值恒 > 1024 → `park_worth` 恒真。**park-gain 1024 定小了，不是开关无效**；但既然上限是净零，不必再调。

### 10.6 真实 160K 长上下文：零停车是 **+12~19%/轮**，且 decode 不变慢（本轮唯一真赢家）

上面 10.5 的"净零"只在 20K 合成口径成立。换成真实文本 16 万 token 的追加式多轮
（`/tmp/prodarm/realctx.py`，15 轮 × 128 token，前 14 轮为热态；首轮是 16 万 token 冷读，比较时剔除）：

| 后 14 轮 | 生产 #1 | 生产 #2 | 零停车 | park-gain 65536 | park-gain 1e8 |
|---|---|---|---|---|---|
| TTFT 中位 | **1.96 s** | **1.96 s** | **1.60 s** | 1.55 s | **1.40 s** |
| 只看生成 128 token 的轮：TTFT | 2.04 s | 1.96 s | 1.59 s | 1.55 s | — |
| 只看生成 128 token 的轮：墙钟 | 4.38 s | 4.19 s | **2.86 s** | 3.12 s | — |
| 后 14 轮总墙钟 | 50.7 s | 55.0 s | **41.4 s** | 52.1 s | — |

两次生产复测 TTFT 完全相同（1.96/1.96），所以 −0.36~0.56 s 不是噪声（生产跑间方差 9%，这里 0%）。
生产日志直接印证规模：**160K 时一次停车搬 3151 MiB、均值 539 ms、单次最大 3092 ms**
（20K 时只有 961 MiB/378 ms）。park-gain 1e8 实测每次 1882.6 / 3092.6 / 889.0 ms。

**park-gain 只跳过一部分，不能等价于关停**：160K 快照 3.2 GB，`--conversation-cache-mib 8192` 只装得下两份，
满了就 evict（日志 `parked=2 … evictions=1`），`park_held` 归零后又恢复停车。所以 65536 与 1e8 两臂
各仍然停了 3 次、总墙钟没跟上零停车。**要拿满收益只能用 `--conversation-cache-min-tokens` 彻底关停。**

**为什么 20K 净零、160K 是净赢**：20K 合成口径每轮生成 256 token、decode 占比大，停车省下的 0.35 s
会被 decode 窗口吃掉；160K 口径每轮只生成 128 token、TTFT 本来就占整轮 60%，停车省下的直接落袋。

**落地开关选 `--conversation-cache-park-gain 65536` 而不是 `--conversation-cache-min-tokens 1000000`**：
后者是全局关闭（还顺带关掉 live 磁盘同步）；前者只在"本对话已有快照且本轮推进不到 6.4 万 token"时跳过，
快照被清掉/淘汰后 `park_held==0` 会自动恢复停车，且不影响磁盘同步——语义更窄、更安全。

⚠️ 质量口径：`--expert-cache auto` 下同提示词重跑本来就不是逐位相同的（[[tp-residency-changes-tokens]]），
所以不能用"两次输出不同"判定 park 影响质量。park 是纯缓存，不参与任何算术。
若要逐位证书，必须钉死 `--expert-cache N` 后两臂对比（待做）。

### 10.7 但是：真实流量里停车**是被用到的**，不能一关了之

今天生产日志（93090 行，覆盖全天真实 agent 流量）统计：

| 量 | 值 |
|---|---|
| 停车次数 | **4815** |
| `restored … (checkpoint/live)` | **1168**（≈24% 的停车带来一次恢复） |
| 停车时增量复用 `reused_kv_bytes>0` | 1160 次，合计 **920.9 GiB**，单次最多 2.58 GiB |
| 典型恢复 | `restored 28353 tokens (checkpoint) in 145 ms`；`restored 38703 tokens (live) in 141 ms` |

一次 28353 token 的恢复用 145 ms，**重算这段 prefill 要 15 s 量级**（1500–2000 tok/s）。
所以关掉停车会在真实负载上把 24% 的复用变成重算，**净亏**。

**因此生产决策不是"关停车"，而是"把停车挪出关键路径"**——即上游 `f73b3c86` 的 `STRATA_PARK_ASYNC`
（D2D 进 VRAM 暂存 + 后台线程分片搬回主机）。它保留全部复用语义，只把 0.3–3 s 的拷贝从请求路径上拿掉。
**代价**：562 行手工移植，涉及 K/V 生命周期与 join 语义。这是下一步唯一值得做的代码工作。

### 10.8 移植做完了，但**这块卡上装不了**：异步停车要的显存我们没有

两个上游补丁（`0f7224d1` 检查点异步 222 行、`f73b3c86` 停车异步 340 行）已经在临时 worktree 里
做完三方合并，冲突只有 6 处（`generate.cpp` 4 处、`conversation_cache.hpp` 2 处），全部解完：

- 我们的 `park_current_body(size_t held)` 对应上游的 `park_source(const ParkSource&, size_t)`，
  我们这棵树上没有 `ParkSource` 重构（#2526 未落地），所以 `src.session → ss`、去掉 `src.with_reuse` 门；
- `conversation_snapshot_save` 已带上 `storage` / `moved` 两个可选参数（上游把 `STRATA_CKPT_POOL`
  与"本次请求要丢弃的检查点搬进停车镜像"合到一次调用里）；
- SAVE/RESTORE 段加 `conversation_park_join()`；
- 停车决策前用 `park_drops` 标记本提示词路径之外的检查点（规则与我们已有的
  `c->ids.size() > resume || !starts_with(c->ids, c->imgs)` 一致）。

补丁存档：`upstream-ports/0001-port-ckpt-async-0f7224d1.patch`、`0002-port-park-async-*.patch`
（`git format-patch` 两份，可直接 `git apply` 到 `c9ab4bfe`）。

**为什么不装**：提交说明里写得很清楚——异步停车要先把 K/V 的 D2D 副本放进**显存暂存**，
条件是"暂存之外还要留 1.5 GB 空闲"（`conversation_state.cpp` 里的 gate）。实测本机稳态：

| | 总量 | 已用 | 空闲 |
|---|---|---|---|
| CUDA0 | 22528 MiB | 21233 MiB | **769 MiB** |
| CUDA1 | 22528 MiB | 21229 MiB | **773 MiB** |

（`--expert-cache` 把 17.22 GiB 专家缓存填满之后的稳态，与 [[vram-reserve-nominal-is-optimistic]]
记的"真实余量约 715 MiB"一致。）

于是门槛是：

| 上下文 | 停车字节 | 暂存 + 1.5 GB 门槛 | 我们有的 |
|---|---|---|---|
| 20K（agent 常见） | 961 MiB | 2497 MiB | 769 MiB |
| 160K | 3151 MiB | 4687 MiB | 769 MiB |

**任何长度都进不了异步分支**，只会走原来的同步拷贝。移植过去等于空改，还要冒 K/V 生命周期出错的风险
（上游自己也写了"背景拷贝失败会留下未填充字节，恢复会读到它们，且没有测试"）。因此：不装、不改生产。

### 10.9 这一轮的结论

- 速度型 A/B 已经到噪声底：同配置重跑、不同臂之间，端到端差 1–3%，而噪声底就是 3.6%。
- **逐位质量证书在这台机器上做不出来**：钉死 `--expert-cache 5362 --adapt-every 0` 之后，
  parA（停车）/ parB（不停车）第 0 轮冷启动 md5 完全相同（`1a2b8d38b2bd`），第 1 轮起分叉。
  原因是专家驻留的 PCIe 上传时序不同 → 命中走 GPU、未命中走 CPU 池，两条算路不逐位相同
  （[[tp-hit-miss-arithmetic-proof]]），不是停车改了算术。
- 停车必须留着（真实流量 1168 次恢复/天），而唯一的提速方案受显存硬约束。
- 剩下的差异全部在 769 MiB 余量这条硬墙之外。
