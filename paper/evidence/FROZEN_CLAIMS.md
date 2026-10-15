# 冻结清单（最终版）

2026-09-29。替代此前所有版本的冻结清单。修正后的数字从已保存 g/m 重算，未重跑模型。

## 冻结的主张

| # | 主张 | 状态 | 证据 | 措辞边界 |
|---|---|---|---|---|
| 1 | 10 模型×12 配置 token 多重集恢复率 99.67%（原始计数） | ✅ | `results/STATS.json` | 逐头×逐 token 评价，非独立请求泛化 |
| 2 | 官方入口端到端 100%（LMSYS 16 条对话） | ✅ | `results/official_repro.json` | — |
| 3 | 跨租户生命周期（校准→10 用户→换钥→复活） | ✅ | `results/lifecycle_05b.json` | — |
| 4 | 跨密钥稳定性（3 模型×5 周期，15/15 校准 100%） | ✅ | `results/independent_eval.json` | 非独有会话上的稳定性观察 |
| 5 | 隐私敏感值识别 | ✅ 有限观察 | `results/privacy_v3_corrected.json` | 见下方详细措辞 |
| 6 | S/a 按请求轮换阻断已测攻击 | ✅ 有限观察 | `results/sarotate_05b.json` | 见下方详细措辞 |
| 7 | 状态坍缩至 ≤b 类 | ✅ | 10 模型实验 | — |
| 8 | 秘密恢复精度（S 列 cos≥0.999996 等） | ✅ | 各模型 recovery 字段 | — |
| 9 | fused 模式下攻击仍 100%（权限隔离版） | ✅ | `results/fused_isolated.json` | — |
| 10 | need_ratio 缩放加固下攻击仍 100% | ✅ | `results/ratio_05b.json` | — |

## 撤回或降级的主张

| # | 原主张 | 处置 | 原因 |
|---|---|---|---|
| A | S/a 轮换"功能已验证" | ❌ 撤回 | 排序检查不充分 |
| B | "held-out sessions" | ❌ 撤回 | LMSYS 流前 10 条与历史实验重叠 |
| C | PBCM 作为独立防御贡献 | ❌ 降级 | S/a 轮换是更简单的修复 |
| D | Beacon 二分法定理 | ❌ 撤回 | 仅有 k=1 实验观察，k≥2 解码器验证未完成 |

## VeilKV-Plus v4 防御候选（补充证据，不改变攻击主线）

| 主张 | 状态 | 证据 | 措辞边界 |
|---|---|---|---|
| TEE profile 双模型 fp32 功能、TEE-BF16 零额外误差 | ✅ strong candidate evidence | `experiments/kvcloak/results/veilkv_plus_v4.json` | 真实 attestation 未实现；仍是模拟 TEE 边界 |
| 公共前缀与私有尾部链感知复用 | ✅ | 同上 | stored-artifact reuse 1/1、chain-aware cross-decode、错误连接点全检测；独立重算仍可能漂移 |
| 完整性、重放和跨租户替换 | ✅ | 同上 | 12/12 检测；nonce 重启后拒绝 |
| 私有字典与攻击电池 | ✅ 有限边界证据 | 同上及 run1 archive | 双模型 0/100 私有字典；arithmetic 谱泄露疑点仍排除 |
| TEE 性能 | ✅ | 同上 | protect/recover 中位数约0.116/0.100 s/1k；attestation 延迟未测 |
| arithmetic profile | ❌ falsified/limited | `arithmetic_optimized_summary.json`、`arithmetic_spectrum_blocks.jsonl`、`arithmetic_optimized_perf.jsonl` | 双模型各100块：Gram 23/100、38/100；whitening 18/100、6/100；随机对照0/100、2/100；永久退出机密性声明。最终464-token基准均值protect/recover 0.7325/0.5789 s/1k；旧约11.4值为短基准，不混比 |
| TEE 与 no-sharing AEAD 基线 | ✅ 有限成本证据 | `veilkv_plus_v4.json` 的继承性能记录 | protect p50 0.1159 vs 0.1091 s/1k（约多6.2%）；未保存no-sharing recover指标；优势是已测公开前缀工件复用与私有尾部完整性集成，不声称唯一或比认证加密更快 |

## 隐私任务冻结措辞（修正后）

**可用表述**（对应 `privacy_v3_corrected.json`）：

> 在一个模型（Qwen2.5-0.5B）、一个秘密配置、30 个合成闭集候选案例中，真实值始终达到最高分（允许并列；g=0 对全部 30 例成立）。按记录的同分组均匀破平，top-1 期望 77.36%、top-5 期望 90.95%、top-10 期望 95.24%（均匀随机猜 100 候选的基线为 1%/5%/10%）。

**不可宣称**：
- ~~"30/30 unique-first"~~ → 正确数字：**21/30 = 70% 唯一第一**（g=0 且 m=1）；30/30 是"并列或唯一第一"
- ~~"top-5 = 83%"~~ → 正确数字：**top-5 期望 90.95%**（原代码漏了 min(1,...) 上限）
- 各类实体的"稳定识别能力"（每类仅 5 个案例，且候选抽样为固定前 5 而非均匀随机）
- "匹配生成过程的先验基线"（实际为均匀随机猜测基线）
- "完整字段恢复"（实体完整性由 token 集交集近似判断，非验证 span）
- 完整候选分数、输入 token ID、恢复多重集等原始证据已保存（实际未保存——仅保存了真值分数及 g/m 摘要）

## S/a 缓解冻结措辞

**可用表述**：

> 在 Qwen2.5-0.5B 上，保存的 S/a 按请求轮换（M 静态）跨请求恢复率为 0%，同周期对照为 97.44%。该 JSON 仅保留舍入率，未保留这次运行的原始分子分母。S/a 全模型参数生成微基准约 2.9ms（不代表端到端请求时延）。

**不可宣称**：
- 功能正确性（K/V 对应关系或注意力输出等价未验证）
- 完整安全修复
- "全部权重必须重融合"（仅 M 融合进权重，S/a 不在权重中）

## 2026-10-15 顺序恢复补充（回溯验收，不替换主实验分母）

- 交付：main 600指标ID、boundary 300指标ID，四条件；逐token预测与恢复Counter未保存。物理prediction/cost行存在续跑重复与旧skip，全部保留，指标不重复计数。
- main：每记录位置准确率22.2933%，random_order 7.4017%，no_cross 18.8958%；日志整段精确18/600、块精确108/2360。
- 3×4×50为标称最低生成块数组，不是严格16/32/64/128-token桶；109条实际长度超出标签。实际评估span唯一数581，不宣称600独立/唯一文本。
- main与boundary所有逐记录指标向量end2end/oracle相同。oracle输出来自真值词袋，而conservation与合并后的恢复词袋比较，单位conservation给出整体词袋相符的间接证据；不等于独立逐块100%审计，也不替换主实验99.668%。
- 随机token100条：位置准确率5.4584%，random_order 7.1250%，支持语言先验作用，不证明所有位置相关信息都不存在。
- 方向对：a/b组位置准确率18.9684%/18.4683%，并非方向分类准确率。全局词袋相同100/100，但逐块词袋相同0/100，不能宣称方向零泄露。整段精确0/200也不证明事实无法识别。双方向random为7.1244%，交付派生文件6.69%仅选a组。
- 独立运行前协议不存在；RETROSPECTIVE_PROTOCOL明确回溯性质。按当前源码、交付文件和离线tokenizer做回溯验收；不重跑模型、不承诺确定性再生。
- 正式依据：evidence/order_v2/*与review/order_v2_audit.json。旧探索65–100%、9.1%/15.9%和dev调试最高值不用于正式顺序恢复结论。

## 性能对照补充

Figure中的KV-Cloak计时脚本设置fused=False；保护/恢复1.93×/4.20×是对该未融合路径，不能宣称超过NDSS论文的融合版或真实服务。
