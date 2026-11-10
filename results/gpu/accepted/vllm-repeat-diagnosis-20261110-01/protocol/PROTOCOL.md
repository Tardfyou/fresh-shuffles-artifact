# 诊断协议：未保护 vLLM 重复生成不一致（运行前固定）

目标：定位 vllm_baseline_b16 93/100、b32 92/100 两次生成不一致的原因，给出有证据的结论。不改主包/vllm-live 任何文件；本目录独立运作，全部结果与证据落盘于此。

## 固定事实（已核对）

- 原始实现（runner/extended.py:194-226）：同一引擎实例，同一样本连续两次 `engine.generate`（temp=0, max_tokens=16, max_num_seqs=1, enforce_eager=True, enable_prefix_caching=True, BF16, TP=1, uni 后端, VLLM_USE_V1=0；b16: blocks=72/max_len=1024，b32: blocks=40）。
- 每块尺寸 100 条记录均出自单次成功 invocation（b16=dd781495…，b32=c6ed3139…）；首次尝试的 CUDA invalid argument 错误发生在任何记录写入之前（并发互踩所致），与分歧无关。
- 分歧样本：b16 7 条 / b32 8 条（交集 5 sid：0068/0119/0180/0307/0346），运行序位置分散（15–73），首分歧位置 0/1/5/6/8/9/10 不等，两块尺寸共有的分歧 sid 首分歧位置多相同（0068@1、0119@9、0307@8、0346@6）。
- 输入直接取自原 records 的 input_ids（100 条），不重新 tokenize；b16/b32 输入集合断言一致。

## 变体设计（每变体独立引擎进程，除研究项外与原设置逐项相同）

| 变体 | prefix_caching | 每次生成前动作 | 比较结构 | 对应问题 |
|---|---|---|---|---|
| A | False | 无 | 同引擎两次 | 缓存关掉后是否仍分歧 |
| B | True | 每次生成前 reset | 冷—冷 | 逐次全新 prefill 是否复现 |
| C | True | 每对开始前 reset | 冷—热 | 复刻原结构，是否复现 |
| C0 | True | 每对开始前 reset | 冷—热 | 同 C 但无 logprobs（插桩等价性对照） |
| D | True | 预热一次后不 reset | 热—热 | 全命中路径是否一致 |
| E(fp32) | True | 每对开始前 reset | 冷—热 | 仅 15 条分歧样本，dtype=float32，单列标示 |

- reset 使用 `scheduler.reset_prefix_cache()`（0.8.5 正式方法）；每次 reset 后记录返回值与 `get_num_free_gpu_blocks()`（应等于全部容量），未回满即标记 reset_failed 并中止该变体。
- 每次生成前后记录空闲块数；输出记录 RequestOutput.metrics 全字段（时序证据）。

## 观测与数值证据

- 全部生成 `SamplingParams(temperature=0, max_tokens=16, logprobs=20)`（C0 除外）。
- 每对生成记录：两次 token_ids、相等性、首个分歧位置；分歧步与前一一致步的 top-20 (token, logprob)、top1−top2 间隔、对侧 run 的同位 top-k；只比较同一生成历史下的同位分布。
- 插桩等价性：C0（无 logprobs）与 C 的 token 输出必须完全一致，否则视为观测影响结果，作废并改用离线副本插桩。

## 样本与顺序

- 阶段 1：15 条分歧样本 + 5 条一致样本（原运行序原样），变体顺序 C0→C→B→D→A→E。
- 阶段 2（定位后）：关键对照在原 100 条 × b16/b32 各 ≥3 轮完整测量。

## 判定框架（预先固定）

- A 分歧 + B 分歧 → 与缓存无关的内核/调度非确定性。
- A 一致 + B 分歧 → 冷 prefill 本身非确定（同一输入两次全量 prefill 出不同 logits）。
- A 一致 + B 一致 + C 分歧 → 冷—热路径数值差（命中缓存 vs 重算）为因。
- C 一致 + 原始仍分歧 → 原始运行中跨样本缓存状态（样本间共享前缀/驱逐）为因，需在阶段 2 复刻原逐样本连续结构验证。
- D 分歧 → 热路径自身非确定。
- E 消除分歧 → BF16 数值近平局参与（单列证据，不以此替代主判定）。

不预先归因 BF16 误差；一切以对照矩阵 + 数值记录为准。正常分歧如实保留，不换种子、不删样本、不放宽指标。

## 交付

`analysis/` 汇总脚本与结论、`records/` 全部逐样本记录、`scripts/` 实际源码、HANDOFF.md 回答四问：原因、支持对照、原基线是否需修复、对保护接入正确性比较的影响。
