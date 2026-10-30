# 上游更新与可移植清单（2026-10-30）

基线：本树 `merge/v0.1.45`（merge-base `477ac13c`），上游 `Niko1221/Strata` main = `870aedd9`（v0.2.8，engine 版本串 0.1.54），
我们落后 2322 提交、领先 177。本文件只记**能落到我们这台机器**（2×RTX 2080 Ti sm_75 + NVFP4-GPTQ，生产 `--layer-split 24`）的条目。

## 0. 结论

上游 0.2.x 的提交九成与我们无关：SYCL/Arc/Metal、`src/dsv41/` 的 DeepSeek 新引擎、batch/多槽流水线、
sm_80/90/120 的默认表、Windows/WDDM。真正值得动的只有 6 项，按「改动大小 / 是否改 token」分三档：

| 档 | 项 | 改什么 | 改 token？ | 预期 |
|---|---|---|---|---|
| A | `tool_call_recovery: true` | 生产配置加一行 | 否 | 工具调用成功率（不是速度） |
| A | `--prefill-pipe` | 已有的旗标，从未 A/B | 否（只改分块） | 上游 2×A4000：4K +15%、8K +23% prompt |
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

### 1.2 `--prefill-pipe`（本树已有，默认 off）

`generate.cpp:649/652/1036/2616/4941` 全在，`prefill_pipe` 默认 `-1.0`（off），`prefill.hpp:113-138` 是它的计划接口。
上游在 2×A4000 上 4K prompt +15%、8K +23%，**只改分块、不改算术**。我们从未在生产口径下 A/B 过。
注意我们的 prefill 口径：热 2173 tok/s、冷 1430，且 <10% 差异单跑不可比 → 至少 2 臂 × 2 跑。

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
下一步（未做）：§1.2 `--prefill-pipe` A/B、§3.1 名义 1205、§3.2/3.3 解析层移植。
