# TP 定位结论（2026-10-26 晚，第二次修订）

> 本文件此前把根因写成「`native_mmvq_il` 的 n_out 行数分档」。**该假设已被实验否证**：
> `STRATA_MMVQ_IL_ROWS` 把 TP 的 cls 2 强制成 r=4 后，TP 输出**逐位不变**
> （`79790 9286 91 29 ...`）。`mmvq_il_parity.cpp:2` 与 `native_mmvq.hpp:164` 也都明确写着
> IL 对**每个 r** 都与 `native_mmvq` 逐位相同。IL 与本案无关。

## 一句话结论

**两个 out-projection（GDN 的 `ssm_out`、QSA 的 `attn_output`）切错了轴。** 它们原本切 `n_in`
（归约轴，6144→3072），每卡只算一半的点积，再由 allreduce 把两个部分和相加。
`(a1+a2)+b ≠ a1+(a2+b)` 在 float32 下不成立，所以结果与单卡**不可能**逐位相同。
这个差异逐层累积，到 l13 触发 q8_1 量化边界翻转，随后每层放大约 20 倍。

**正确切法是切 `n_out`（输出行）**：每卡拿 1280 行、算完整的 6144 项归约，
kernel、block 布局、warp 分配都与单卡一致 → 每个元素**逐位相同**；两卡的输出行互不相交，
这个投影**根本不需要 allreduce**。

## 判决实验（`tests/core/tp_outproj_split_test.cu`，单卡，<1 秒）

真实 Q8_0 kernel、真实形状 2560×6144：

| T | n_in 切分（旧实现） | n_out 切分（新实现） |
|---|---|---|
| 1 | 1912/2560 不同，max rel 7.8e-5 | **0/2560** |
| 2 | 3904/5120 不同，max rel 6.7e-4 | **0/5120** |
| 3 | 5679/7680 不同，max rel 4.4e-4 | **0/7680** |
| 4 | 7624/10240 不同，max rel 1.5e-3 | **0/10240** |

`n_out` 切分在每一个 token 数下都**逐位一致**。这就是修复的依据。

> 第一次写这个测试时夹具是坏的：把 float 的低 2 字节当 fp16 scale 写进 Q8_0 block，
> 产生 NaN 位型，`NaN != NaN` 让所有元素都"不同"。修正为真正的 `__float2half` 后结论才成立。

## 为什么之前一直找不到

`src/core/tp_split_plan.cpp` 的旧注释**自己就写明了根因**（attn_output 段）：

> "A split out-projection adds the two halves' partials, and (a1+a2)+b is not a1+(a2+b) in float32
> - 2e-7..1e-6 per layer, enough to flip a greedy argmax by token 2."

但作者的对策是 **FULL**（rank0 算全宽，`STRATA_TP_ATTN_FULL` / `STRATA_TP_GDN_FULL`）——
正是用户明确禁止的「把重计算推回 rank0」。第三条路（切 n_out）没被考虑。

同时 `tests/core/tp_split_plan_test.cpp` 和 `tp_layout_test.cpp` **从未注册进 CMakeLists**，
所以从来没人构建过它们；它们断言的恰恰是错误设计（`TpMode::Row`），也就无人纠正。

## 已排除（每条都有实验）

- 专家 FFN 半宽切分：`STRATA_TP_MOE_FULL=1` 下 l0–l12 仍差 2–4e-7 → 切分无罪
- 竞态：两次独立 TP 运行逐位相同（MAX relL2 = 0.000e+00）
- IL 行数分档：强制 r=4 后 TP 输出逐位不变（见开头）
- `q8_mixed` 地址算术、`fused_gr_read_multi` v3、半宽量化块划分、allreduce 加法顺序
  （`x + 0.0f == x` 逐位成立）
- `tp_zero_persistent` 在 `resume == 0` 时对两卡都调用（`generate.cpp:14305`）

## 本次改动

1. `src/core/tp_split_plan.cpp` — `ssm_out` 与 `attn_output` 由 `TpMode::Row` 改为 `TpMode::Col`
   （n_in 全宽 6144，n_out 半宽 1280）。`FULL` 环境变量保留为诊断开关。
2. `src/core/verify.cpp` —
   - GDN：两卡先交换值半份（rank1→rank0 的 `y_full_[ZVt,ZV)`，rank0→rank1 的 `y_full_[0,ZVt)`），
     然后各算自己 N/2 行，另一半清零交给 allreduce 拼接（`x+0` 逐位不变）。
   - QSA：`attn32_` 两卡本来就是全份，**无需交换**，直接全宽量化 + 半宽输出行。
   - 删掉每层白做一次的 `attn_tp_` 2D 拷贝与死缓冲区 `attn_tp_`。
3. `src/program/generate.cpp` — `set_tp_y_peer` 改为**双向**指向对方的 `y_full_`（原来 rank0 指自己）。
4. `tests/core/tp_outproj_split_test.cu` — 新增判决测试（已注册 CMake）。
5. `tests/core/tp_split_plan_test.cpp` / `tp_layout_test.cpp` — 断言更新为 Col，并**注册进 CMakeLists**。
6. `CMakeLists.txt` — 注册上述两个从未构建过的 host 测试。

## 修 n_out 切分时踩到的两个坑（都已修，值得记住）

### 坑 1：`native_mmvq` 的输出 pitch 是 `n_out`，不是 `N`

kernel 的写回是 `y[j * n_out + row0 + i]`（`native_mmvq.cu:1251/1261`，small-K 路径是
`y[row0+i]`）。把 `n_out` 从 2560 降到 1280 之后，**第 t 个 token 的结果会写到 `t*1280`
而不是 `t*2560`**，T=1 完全看不出来，T=4 的预填窗口整片错位（实测第一 token 从 `79790`
变成 `12`）。

修法：投影写**紧凑** `[T][N/2]` 的 `half_out_`，再按 token 用 `cudaMemcpyAsync` 搬进
`bo_` 的 `[T][N]` 正确半区。单卡 fixture 用的是紧凑数组，所以 fixture 测不出这个，
只有端到端 T>1 才会暴露。

### 坑 2：两个方向共用一个 flag，计数序列互相覆盖 → 死锁

我最初让两卡都用 `tp_flag_g_` 做值半份握手，但写的是**两套不同的递增序列**：
rank1 写 `l+1`（1,2,3…），rank0 写 `l+1+n_layers`（49,50,51…）。

`raise_flag_kernel` 是**直接赋值**（`*flag = value`，`verify_kernels.cu:1462`），不是取最大值。
所以后写的一方会把对方的计数**回退**：layer 1 时 rank0 的 `wait g>=2` 被上一轮遗留的 49 直接放行，
它随即把 g 写成 50；rank1 再把 g 写成 2，rank1 的 `wait g>=50` 就永远不成立。

日志正是这个形态（不是"卡住"，是"计数被回退"）：
```
strata verify: tp stall at layer 1: seq=1 want=2 flag_a=1 flag_b=1 (peer flag_a=1 flag_b=1)
ERR verify: tp: timed out at layer 1
```

修法：**每个方向一个 flag**。rank1→rank0 用 `g`，rank0→rank1 用 `c`
（`tp_flag_c_` 原本只被赋值、无人使用，正好空闲；两卡指向同一块 mapped host 内存）。

### 坑 3：`run_tp` 没有复位 flag g

`run_tp` 每请求把 a/b/d/e/f 归零，**g 不在列表里**。以前 g 只服务 `STRATA_TP_GDN_FULL`
诊断路径所以没暴露；默认路径改用 g 做握手后，第 2 个请求的 `wait_flag_ge(g, l+1)` 会
因为上一轮留下的高水位立刻返回，读到尚未到达的 P2P 数据。已把 `tp_flag_g_h_` 加进复位列表。

## 尚未解决的独立问题

- `STRATA_TP_GDN_FULL=1` 路径输出 `11 90006 11 90006`（诊断用，不影响默认路径）
- `STRATA_TP_FUSED=0` 的 allreduce 有 flag bug
- `--tp` + `--batch` 未拒绝
- 短 prompt（≤ `--short-read` 64）绕过 TP 窗口，读满 `ne0/ne1` 越过半宽分配

## 第三轮：MoE down 的归约轴切分（2026-10-26 晚）

### 已确认的第四个 bug：rank 1 漏写自己的 y_full_

GDN 列并行分支里，rank 1 把值半份 P2P 送到 rank 0，却**没有写进自己的** `y_full_[ZVt, ZV)`，
随后 `quant_in(y_full_, ZV)` 读的是上一层残留。rank 1 的 1280 行全错，allreduce 每个元素都错。
修好后 `l0 site0` 从「2560/2560 全差」变成逐位。

### 第五个 bug：flag c 不在每窗口复位列表里（审计代理发现，已核实并修）

`tp_flag_c_h_` 是 rank0→rank1 那条值半份握手的 flag。复位列表只清 a/b/d/e/f/g。
`raise_flag_kernel` 是**直接赋值不是取最大值**，所以上一请求留下的 96 会让
rank 1 的 `wait_flag_ge(c, l+1)` 立即返回 → 它在 rank 0 的 P2P 落地前就量化了 `y_full_`。
第一个窗口对，第二个窗口起错。已加进 `verify.cpp:4678` 复位列表。

### 第六个 bug：TP commit 不恢复 PLE 历史（已修）

`capture_commit`（solo，`verify.cpp:3300`）在提交后把 `ss.ple.hist` 回滚到保留行；
`capture_commit_tp` 里这一行被换成注释删掉了。而 TP 窗口的 layer-1 块**每个 token 都推进一次**历史
（`verify.cpp:3773`），提交却只保留 `commit_ + 1` 行 → 下一个窗口的 4 个卷积抽头读超前
`T - (commit_+1)` 行。第一个窗口对只因为 prefill 自己的提交已把历史停在保留行。
原注释说「恢复会改变 token」，那是因为当时快照没有 solo 的 `if (!keep_rows)` 守卫；
TP 不走 batch（`batch_rec_` 只在 `capture_batch` 里置真），所以该守卫在 TP 下等价恒真。
已补上 `copy_indexed(ss.ple.hist, hist_snap_, HS, commit_ + 1, HS, cs_)`。

### 判决实验：STRATA_TP_MOE_FULL=1 的逐层残差

`tp_moe_split` 把 gate/up 按**行**切（`ptr_off = rank*half*gu_row`）——这是正确的 n_out 切分；
但 down 是 `d_col_off = rank * d_row_half`，按**归约轴**切 640 项。
`row_dot`（`iq_kernels.cu:667`）是 `for (k = lane; k < nb*ipb; k += 32) s += dot(...)` + `warp_sum(s)`，
把 nb 从 20 降到 10 会改变每个 lane 的累加分组和归约树形状 —— 与原来 out-projection 的错误同一类。

MOE_FULL 把 down 恢复全宽后（`Lfull`，rank 1 贡献零）：

| 层 | 默认 TP site1 | MOE_FULL site1 |
|---|---|---|
| l0 | 1.676e-08 | **0.000e+00** |
| l1 | 7.451e-09 | 1.863e-09 |
| l12 | 5.588e-09 | 0.000e+00 |
| **l13** | **9.382e-05** | **7.451e-09** |
| l14 | 1.758e-04 | 7.451e-09 |
| l20 | 5.279e-04 | 1.024e-08 |
| l30 | 6.280e-04 | 7.451e-09 |
| l47 | 1.425e-01 | 8.445e-03 |

**l13 那个放大 10^4 倍的跳变完全消失**，l0–l41 全部稳定在 1e-9..1e-6（低 4 个数量级）。
默认路径的放大是 down 的半宽归约噪声逐层累积、到 l13 翻转 q8_1 桶的结果。

### 一个必须记住的测量陷阱：dump 宽度 2048 只覆盖了 80% 的一个 token

`STRATA_TP_VEC` 的站点是 `[T][N]` 缓冲区，宽度 N 只覆盖 token 0 的前 N 列。
之前默认 2048 < N=2560，所以**第 2048..2559 列和 token 1 从来没被 dump 过**，
据此得出的「l0 逐位」只对 80% 的一列 token 成立。已把 `tp_vec_len` 上限提到 16384
（最大窗口 6×2560），cmp.py 默认改成 5120（两个 token 全宽）。

### 仍然存在的第二个源

MOE_FULL 下 l42 site1 仍有 3.155e-06 的跳变、l45 到 1.046e-03，最终 token 是
`731 248046 8 ...` 而 solo 是 `79790 316 52933 ...`。l42 的 site0 是干净的 1.55e-06，
所以偏差在 MoE 内部进入。l42/l45 都不是 QSA 层（`is_qsa_layer` = `l%4==3`，即 43/47）。

**但注意**：solo 自己也跨请求不可复现，且量级相同 —— 同一 prompt、同一位置，
solo 请求1 vs 请求2 在 l12 差 6.197e-05、l40 差 3.765e-01。
所以深层逐位比较在默认 `ADAPT_NOWAIT` 下没有判据力，只能用请求 1 的 token 判定。


## 验收基线的一个重要前提：solo 自己也跨请求不可复现

默认设置下，solo 两个 `GEN 8 9707 11 1246`：
```
请求1: 79790 316 52933 91 79751 198 262 1992   <- 与验收目标逐位一致
请求2: 79790 316 52933 91 79751 198 262 9341   <- 第 8 个 token 变了
```
前 7 个 token 相同，第 8 个变。原因在 fork 的默认值而非模型：`STRATA_ADAPT_NOWAIT` 默认**开**
（`generate.cpp:1566`），verify 窗口不等自适应专家层上一轮换入的拷贝落地。

但 `STRATA_ADAPT_WAIT=1` 并**不能**把它变成可复现，反而**改变了请求 1 的数值**：
```
请求1: 79790 316 52933 91 79751 198 285 638   <- 偏离验收目标
请求2: 79790 316 52933 91 79751 198 262 9341
```

**结论：验收条目 1 只能用「默认设置 + 第一个请求」判定**，那才是逐位对上
`79790 316 52933 91 79751 198 262 1992` 的配置。第二个请求的差异是专家放置时序，
属于 fork 既有的非确定性，不是 TP 缺陷，也不能拿它当 TP 正确性的判据。
（要判第二个请求，得先让 solo 自己在同配置下可复现；目前它不可复现。）

## 环境状态

- 8090 服务：**停着**
- 未提交任何改动
- `build/strata` 已用本次修复重建
- 测试脚本在 `/tmp/tp-repair-20261026/`：`cmp.py`（solo vs TP）
