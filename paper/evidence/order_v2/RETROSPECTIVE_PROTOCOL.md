# 顺序恢复实验：回溯整理的协议记录（RETROSPECTIVE — 非运行前独立存档）

**性质声明**：本文件于 2026-10-15 实验完成后回溯整理，内容来自 2026-10-08 的任务
指令、代码中的 CONFIG 常量与代码评审记录。运行前未以独立文件形式预注册——这是
流程缺陷，如实记录。各条目的"固化证据"指向运行前即存在的载体。

## 1. 冻结的实验条件（固化载体：exp_order_v2.py CONFIG，运行前写入代码）
- 模型 Qwen2.5-0.5B；b=16；beam_width=16；top-1 输出；上下文上限 96；迭代 2 轮
- 种子：校准 42；混淆 base 9000+样本序号；采样 dev=20260929 / main=+1 / boundary=+2
  （main/boundary 独立种子，与 dev 无样本重叠——2026-10-07 修复并验证首个样本不同）

## 2. 采样设计（固化载体：samples_main/boundary.jsonl，逐样本含原文与分组）
- main：3 类别 × 4 长度桶(b1/b2/b4/b8) × 50，长度由 generate_sample_of_length 保证
- boundary A：100 对同词袋不同方向（wrap 上下文两方向相同，多重集逐 token 一致、
  仅主体位置不同——2026-10-13 验证记录：36 token/2 块、Counter 相等、2 位置差异）
- boundary B：100 条随机 token 序列
- dev：60 条（3×4×5），已观察、不进正式结果

## 3. 对照与指标（固化载体：exp_order_v2.py run_experiment/evaluate，运行前实现）
- 条件：end2end / random_order / oracle_ms / no_cross_block
- 指标：position_accuracy（主）、exact_match、block_exact、trigram_overlap、
  multiset_conservation、pa_by_block、random_baseline_pa（Σc²/b² 公式）
- 禁止用真值选择 key/候选/seed；主结果只报 top-1

## 4. 已知数据局限（论文措辞需对应）
- 逐 token 预测序列与原始词袋 Counter 未落盘；再生依赖固定种子确定性重跑
  （全量约 35 小时），未执行
- multiset_conservation 是 pred-vs-recovered 输入；recovered-vs-true 的逐样本等价
  仅在 dev 期验证（60/60 oracle≡end2end），main 期为聚合一等价
- random_order 基线在 resume 分段后使用续跑时的 rng 状态（记录于 bugfix 过程）

## 5. 运行事故与处置（固化载体：resume_info_*.json、代码 git 无—以文件哈希为准）
- 2026-10-07 磁盘满丢全量 → 改增量写盘重跑；输出迁移机械盘
- 三进程并发同写 → 种子一致内容一致，按样本 ID 去重（600 唯一/0 坏行）
- pair 样本首跑全 skip（<16 token）→ 共享 wrap 修复 + --resume 补跑
