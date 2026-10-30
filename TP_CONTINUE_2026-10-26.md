# 双卡 TP 交接（2026-10-26 13:23 CST）

这份文件是最新交接。`TP_HANDOFF.md` 是较早的架构说明；`TP_AUDIT_2026-10-26.md` 有大量尚未证实或错误的结论，不能当作验收证据。用户现在要求换 agent 接续，不是在宣布任务完成。

## 用户目标和硬要求

仓库：`/media/zc/软件1/strata-nvfp4`，不在最初 pcai workspace 内。先读根目录 `AGENTS.md`。

模型 Qwen3.8-Flash-Next 125B MoE，NVFP4 专家 + Q8_0 dense，在 2×RTX 2080 Ti 22 GiB + NVLink 上实现真正 TP。两张卡都必须分担重计算，包括 MoE，不能把工作退回 rank0 然后宣称完成。

验收：
1. `GEN 8 9707 11 1246` 的 TP 输出 token 与单卡一致：`79790 316 52933 91 79751 198 262 1992`。
2. 同机同配置、正确输出下 decode 有可重复的明显加速，不能拿输出不同的两个 8-token 请求直接算加速；初始要求 >14.3 tok/s，实际单卡曾在 13.6～15.2 波动。
3. tests/core 全部 TP 测试通过，单卡无回归。还要测多请求 reset、采样、惩罚和 MTP 状态。
4. 用户强调效率：每次加载约 100～130 秒，不要一个猜测重启一次。先小测试证伪，再一次加载连续发多条 GEN。

不要把 token 分叉未经测量就归因于浮点误差；之前连续多轮这么做，用户不接受。没有测 logits/top1 margin，不能声称两个输出语义等价、误差不可修或验收不可能。

## 当前机器和进程状态（必须重新检查）

2026-10-26 13:22:52 CST 实际检查：
- `strata-model-200k.service` 为 inactive/dead，MainPID=0。
- 两卡显存 10 MiB / 7 MiB，没有 strata、serve.server、run_pair.py 或 wait_and_verify 进程。
- 用户早先明确允许临时停/启自己的模型做测试，但要求快。测试结束应恢复服务；别改服务配置。
- 服务启动：`systemctl --user start strata-model-200k.service`；配置 `/media/zc/软件1/strata-models/strata-nvfp4-gptq.json`；端口 8090；通常用两卡、200k context，各占约21.3 GiB。
- 不要执行旧脚本的宽泛 `pkill -9 -f "build/strata"`：会杀服务，也会匹配 shell 自身。只清理自己保存 PID 的测试进程；服务活跃/有GPU计算进程时先检查，别盲杀。
- 无 GPU reset、驱动重载、卸载文件系统等操作授权。

## 工作树与运行纪律

HEAD `8bc869bc`。已有大量未提交 TP 实现和用户原有修改；不能 git reset/checkout 覆盖。`serve/server.py` 等并非本轮修改，勿动。

大文件（verify.cpp/generate.cpp/iq_kernels.cu）按用户要求只能用 Bash 的 `rg`/`sed -n`/`awk`/`head`/`tail` 限段阅读，避免编辑器加载大文件。Edit 必须时可 Read 十几行登记后 Edit；不要一次 Read 整文件。

编译：`cmake --build /media/zc/软件1/strata-nvfp4/build --target strata -j8`。长命令后台，记录退出状态，不要重复挂好几个轮询 waiter。CUDA graph 下 compute-sanitizer 曾无法给出有用定位，不要继续耗时间。

## 架构（先核代码，旧文档有过时部分）

几何：N=2560,HC=4,L=48,NE=512,K=10,FF=640；GDN S=128,HK=16,HV=48，localHV=24，ZV=6144/local3072；q/k各2048复制，v3072分片，local conv Ct=7168。QSA每4层一层（3,7,...47），NH=24,HD=256,NKV=4。

- GDN：q/k复制、v-head切分；qkv/attn_gate按输出行切；ssm_out按输入列切，partial经allreduce。
- QSA：完整q/k/v、attention/cache在两卡复制；attn_output输入列分半。`tp_split_plan.cpp` 的 kKvNout=512 注释/常量与实际1024不一致，但这些k/v处于Replicate，先核真实用途，别机械改。
- MoE：`tp_moe_split.cpp` 按每个专家 FF=640 切320+320（不是专家ID分半），gate/up行并行，down列并行；两卡均算GPU常驻/PCIe专家的半宽。CPU专家由rank0承接，shared只计rank0一次；combine后allreduce。
- 当前allreduce是单写者P2P，rank0写两卡buffer；融合握手与非融合可通过 `STRATA_TP_FUSED` A/B。
- `STRATA_TP_ATTN_FULL` / `STRATA_TP_GDN_FULL` / `STRATA_TP_MOE_FULL` 为诊断路径；前两个默认关。不得用FULL-MoE作为真正TP交付。

## 以前确实找到并保留的修复

TP QSA分支缺少 K/V 的 FWHT 旋转；现在 norm_rope 后、kv_append前已补 `st.kv_rot` 时旋K+V、hybrid时旋V；查询和attention反变换原本已有。不要撤回。

融合allreduce最初漏写完成flag，后改成单block1024线程，每线程写完P2P后threadfence_system+syncthreads再发布flag。旧单元测试过，但测试本身有覆盖缺陷（见下）。

## 最近一轮（约13:00～13:19）实际新增、已编译的修改

没有重构TP架构，没有声称完成端到端修复。

### 1. 修正严重不可靠的诊断

`src/core/verify.cpp:728`：
- 原只申请 `(L*8+200)*4` 字节，却写到 `L*8+560+3` float。更早版本只+160也错。
- 已改为 `(L*8+564)*sizeof(float)`，覆盖现有最高探针。
- QSA3打印多传了q3[20:21]两个double给 `%d` 位置，造成 varargs 类型不匹配，attention读数不可信；已去掉多余double。solo和tp/r1都改了。
- W3权重打印也多传了w3/w0[20:21]，r1字段因此错位；已去掉。

因此以前拿 `attn=0` 推导attention无贡献、拿错位字段下结论都不成立；旧短head dumps也不能证明完整张量正确。

### 2. 真实权重测试竞态修复，已实跑通过

`tests/core/tp_real_split_test.cu` 在GPU1上生产dy1后马上让GPU0 P2P读它，没有跨卡顺序。已在allreduce前加 `cudaStreamSynchronize(s1)`（当时current device=1）。

证据：`/tmp/tp-repair-20261026/real-split.log`
- Column halves rel_err=0。
- Row-parallel ssm_out两卡 rel_err=1.080e-07，ALL REAL-WEIGHT TP SPLIT TESTS PASSED。

`/tmp/tp-repair-20261026/allreduce.log`：plain各尺寸误差0、fused误差0，报告20轮通过（注意此20轮测试有缺陷，不代表覆盖充分）。

### 3. 修正TP采样生命周期（编译通过，端到端尚未验证）

`include/strata/core/verify.hpp` 新增 private `bool sample_window(int T,const int32_t*,int64_t,std::string&)`。

`src/core/verify.cpp`：
- `sample_window()` 约2601行，从原solo `run()` 的host-side sampling块原样抽取：当前pos0计数、history/bias、sample_tokens_spec概率拒绝采样、清spec_q、sync。
- solo `run()` 原采样块改为调用helper（约2922）。保持计算行为，用基线回归验证。
- TP `record_window_tp` 的head只捕获argmax + nonfinite scan，不再在capture中调用sample_tokens/cudaStreamSynchronize。
- TP `run_tp` 两图join后、读取h_out之前调用同一个sample_window（约4614）。只rank0做最终请求采样，peer不需history/bias。

独立审计确认的真实缺陷：
1. 原TP在graph capture里按head_sampling_选择采样；prompt捕获时关head_sampling会把后续同T decode永久固定为argmax；新T采样时在capture里cudaStreamSynchronize会使capture失败。
2. sampler的counter/请求参数会被捕获而不是每次使用当前pos0/设置。
3. TP漏掉STRATA_SPEC_PROB路径，忽略spec_q，概率MTP不正确。
4. peer没有history/bias，旧peer采样在惩罚请求会失败；输出本来只用rank0。

这些采样bug不一定解释纯默认贪心的循环，不要宣称已经解释了。

构建证据：`/tmp/tp-repair-20261026/build2.log`，Built target strata。当前binary时间13:18:13，已含上述修复。

## 最新端到端测试：被会话结束打断，没有结果

新驱动 `/tmp/tp-repair-20261026/run_pair.py`（已写好，可以复用）：
```
python3 -I /tmp/tp-repair-20261026/run_pair.py --label samplefix2
```
- 每模式只加载一次，等待READY后连续送5条命令，等待每个DONE，最后QUIT。
- 支持 `--mode solo|tp|both`（默认both）和 `--diagnostic`。
- 清除三个FULL及FUSED/DUMP环境变量，启用STRATA_DECODE_TIMING、STRATA_DUMP_FIRST_LOGITS。
- 命令为：两次GEN8基线；GEN4惩罚；两次相同seed=12345的温度采样。
- 先检查用户服务inactive、无GPU计算进程；只终止本次Popen，不使用宽泛pkill。
- 日志在 `/tmp/tp-repair-20261026/<label>-{solo,tp}.log`；首decode logits为 `<label>-{solo,tp}-logits.N`。

上次命令 `--label samplefix` 在13:19启动；会话被结束，任务bxejsbvsu停止。13:22核查：driver只写 `solo pid=3055120 loading`，solo log仅967字节、无READY/T/DONE；无TP log、无存活进程、两卡已空。不能把它当作卡死，更不能算通过。启动在打印GPU0后安静约一分钟在旧版本属常见现象。

旧 `/tmp/tp_fast.sh` 不是可靠常驻通道：先把全部命令按固定12秒写进FIFO，加载未完成时feeder就退出，无法交互、失败计数不严格，还会kill所有相对路径实例。不要再用它做主要验证。

## 仍需处理的明确问题/重点疑点

### A. GDN_FULL诊断路径布局确实错，最近尚未改

verify.cpp约3858：
```
cudaMemcpyAsync(y_full_,y_tp_, T*ZVt*sizeof(float),...)
broadcast_to_a(y_tp_, y_peer_+ZVt, T*ZVt,...)
```
local是 `[T][3072]`，full是 `[T][6144]`。一次连续拷贝不是沿head轴拼接：T>1时把token1的数据填进token0后半，rank1还会覆盖重叠区域。需逐行/2D stride（full pitch=6144*4, local pitch=3072*4, width=3072*4,height=T）或明确gather kernel。之前 y_peer_ 指向错误曾修过，但stride依旧错。

`tp_flag_g_h_` 新增后没有在run_tp每个window开始清0；因此第二个window可能提前越过wait读旧数据。别复用MoE的f flag。

FULL只为诊断，修好用无模型小测试测T=1/2/4，多轮有不同数据，然后才可用于隔离误差。不要默认开，更不要因此把所有重计算挪rank0。

### B. 状态提交/请求reset/PLE历史（尚未收到完整独立审计报告）

先前叫 `tp-state-audit` 的代理没有交回结论，交接时已不可达，不能假装通过。

需重点比对：
- `capture_commit_tp`（~4775）vs solo `capture_commit`（~3100）：solo末尾会 `copy_indexed(ss.ple.hist,hist_snap_,HS,commit_+1,HS,cs_)`，TP末尾没这步。TP PLE stage本身会推进hist并保存hist_snap，拒绝draft后是否正确回滚？
- SessionState reset/会话恢复操作是否同时处理两rank独立的 `gdn_state_tp_`，并非只处理ss.gdn_state。旧fast1第2请求输出又变，不能只说正常舍入。
- `capture_commit_tp` 尝试generic gdn_commit_layers，但该函数在localHV24 % HK16 !=0 时返回false，因此当前模型走fallback；不要误报generic rank1 head_offset错误为已触发bug。fallback是否与solo状态-only commit的算术一致还需小测试。
- `last_tokens_` 不缺：stage_inputs会写last_t/pos/tokens，run_tp在启动前对两rank都调用。先核调用路径再断言。
- rank0 MTP feature cache捕获/commit是否缺失；最后head HC read和solo是否一致。另一个代理被要求继续查，但没有交回这段结论。

### C. allreduce测试覆盖问题

旧fused测试先将flags置7，之后循环k=1..20却不清flags；k<=7的wait可立即通过旧值。循环也未逐轮比较输出，只sync就宣称通过。需要重置或严格递增、每轮不同输入、实际读回两卡全部元素；测graph replay、多window、rank启动延迟、ragged sizes。

生产fusion单block1024线程是否真快需要CUDA event/graph下测，不能把主机launch估值当实测；当前每层2次collective，原每collective rank0三launch/rank1两launch=5、融合各一次=2，两次合计10→4（先前文档8→2错误）。

### D. 完整测试注册/资源生命周期

CMake只注册9个.cu测试；tp_layout_test.cpp、tp_split_plan_test.cpp、tp_moe_split_test.cpp尚需纳入。真实权重fixture在tests/core/tp_data，CTest需working_directory=源码根，否则报缺文件。

诊断h_tpdump_、tp_arena_、TP flags/graphs等析构资源释放还没全面审计；共享flags归谁free必须明确。

## 旧端到端数据（仅历史，不算交付）

`/tmp/fin_solo.log`：正确8token、553ms、14.5tok/s。
`/tmp/fin_tp.log`：`79790 9286 91 91 79790 9286 91 91`，525ms、15.2tok/s。

这组输出不同，只能说明旧TP能运行且计时如此，不能证明语义等价或正确加速。第一个输出token来自T=1 pos0=2，第二个来自后续MTP窗口；先前把W1/W2和输出token对应弄错过。

旧FULL模式有布局/P2P/flag问题，不能用旧FULL结果证明数值误差不可避免。

## 模型位置、测试命令和测量建议

根 `/media/zc/软件1/strata-models/NVFP4-GPTQ`：
- pack/experts.bin约64GiB，dense.bin约1.5GiB，index.txt
- qwen-nvfp4-gptq-dense.gguf、ple-fp8.gguf、token-embd-bf16.gguf、mtp/
完整参数已在run_pair.py的BASE。

文件系统fuseblk (NTFS)，不能因为wchan=fuse_get_req瞬间观测就断言模型损坏/启动死锁；不要大规模复制124GB模型，根ext4空间不足。

建议第一步只跑新runner完成solo+TP基线/重复/采样对比，同时看具体首次logits偏差。之后修状态缺陷，用小fixture测后再整模。最终benchmark需预热、较长生成、多次中位数、同输出同workload；profiling通过STRATA_DECODE_TIMING已有CPU wait/pool/draft/commit分解。

## 保存点和代码审计报告

`/tmp/tp-repair-20261026/strata.before` 是本轮修复前12:47的binary；`before.patch` 是当时tracked diff（不含所有untracked）；`tp_allreduce.cu` / `tp_allreduce_test.cu` 备份也在该目录。勿全量回滚，保护用户修改。

`tp-output-audit` 已确认采样缺陷，详见上述；只读未改代码。状态代理没有完成报告。

社区资料可参考但不能代替调试：
- vLLM #2891 https://github.com/vllm-project/vllm/issues/2891 （TP数值差异可能发生，不证明本项目无bug）
- https://github.com/vllm-project/vllm/blob/main/csrc/custom_all_reduce.cuh
- https://arxiv.org/abs/2511.17826 （TBIK跨TP归约一致性，先前代理给的性能数字需重新核原文）

不要继续说“全修好了/只有浮点误差/两卡只能这样”。以修复后实际证据完成用户原验收。
