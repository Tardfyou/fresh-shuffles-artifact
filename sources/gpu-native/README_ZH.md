# 单张80GiB GPU实验包：从上传到验收

本包用于复现 Repeat-to-Reveal / VeilKV 的论文实验。只需要一张80GiB NVIDIA GPU，默认同一时刻一个任务。所有样本、版本、算法、阈值与统计定义已经固定；Claude负责监督、资源调节和安装/路径适配。

新增模型为官方稠密Qwen3-14B（实际14.8B）和Qwen3-32B。32B完整模型用BF16权重驻留、FP32保护；FP32首层测量有独立的完整checkpoint等价性证据，并单独标注。14B完整FP32可在80GiB卡上运行。没有多卡或120GiB显存的假设。

## 1. 上传并解压

将ZIP上传到GPU服务器自己的家目录，例如用已有SSH客户端的文件传输功能。然后在GPU终端执行：

```bash
mkdir -p ~/veilkv_gpu
cd ~/veilkv_gpu
unzip ~/veilkv-gpu-80g-experiments-20261109.zip
cd veilkv-gpu-80g
ls
```

应看到runner、protocol、data、scripts、CLAUDE.md、SCIENTIFIC_LOCK.json及SCIENTIFIC_BASELINE.zip。包里没有模型权重、论文或服务器密码。模型和两个运行环境预计需要约180–220GB，加上结果建议预留至少250GB磁盘。

## 2. 安装原生环境（不需要空闲GPU）

```bash
bash scripts/bootstrap_native.sh
```

这会在包目录内创建独立的`.venv-native`，安装Torch2.6.0/CUDA12.4和Transformers4.51.3。上游官方固定Torch2.6.0、Transformers4.51.1、CUDA/FP32/eager；这里保留Torch版本，HF用与全部Qwen3代码一致的两补丁版本适配。A100及580系列驱动可运行这个CUDA构建；不需要改系统驱动。

若Conda不可见，脚本会使用现有Python创建venv。镜像、代理、缓存路径等安装问题由Claude在环境范围内处理，不能用升级科学依赖、改精度或改算法来跳过闸门。

## 3. 先下载权重，空闲卡来了即可开跑

```bash
.venv-native/bin/python -m runner.download_models --workers 2
```

全部仓库和版本在`protocol/models.json`。下载可断点续传；新14B和32B权重合计约95GB，所有主表模型及8B时延对照合计约160GB。可在`runtime/settings.json`修改`cache_dir`到空间充足的磁盘。不要改模型名称/版本或换成量化权重。

MMLU和SQuAD输入已经随包固定，分别完整保留14,042和10,570条。其prompt格式与原工作相同，SQuAD示例选取固定种子，原脚本的跳错样本行为已消除。`reference/accepted_cpu_artifact.zip`只读保留既有CPU证据，不要运行其中历史入口。

## 4. 安装独立vLLM环境

```bash
bash scripts/bootstrap_engine.sh
```

该环境只跑真实vLLM的未保护推理/重复前缀基线，不影响原生实验环境。原工作没有发布保护缓存接入vLLM的实现；它的“PagedAttention兼容”是原生HF中16/32/64块尺寸下的精度和时延检查。本包分别对齐这部分，并单列vLLM基线。不能把后者写成VeilKV或KV-Cloak已经集成vLLM。

## 5. 有空闲卡后检查运行时

```bash
nvidia-smi
.venv-native/bin/python -m runner.doctor --gpu 0
```

doctor只做256×256的FP32/bf16运算，保存GPU、驱动、依赖版本等信息。默认GPU编号0；若80GiB卡不是0，在`runtime/settings.json`中把`gpu_ids`改成它的编号，列表里只能放一张卡。

## 6. 先跑固定小规模闸门

```bash
.venv-native/bin/python -u -m runner.supervisor --phase smoke
```

只有完整小规模队列通过，才会产生`results/SMOKE_GATE.json`，正式队列才允许启动。主攻击/缩放/S_a的正常负结果不是失败闸门；独立plain基线、代码一致性、字节恢复和记录守恒等才是正确性检查。任何科学代码错误都交回Codex修复，Claude不得软化阈值或删除记录。

完整队列包含93个执行单元（含分片与精度对照），不是93个独立实验；启动测试有60个小任务，覆盖每个模型与主要精度/块尺寸路径。逐任务规模、显存要求见`audit/GPU_QUEUE.csv`。

## 7. 正式运行

```bash
nohup .venv-native/bin/python -u -m runner.supervisor --phase all > runtime/all-monitor.log 2>&1 &
```

也可按队列分阶段运行，上一阶段退出后再开下一阶段：

```bash
.venv-native/bin/python -u -m runner.supervisor --phase core
.venv-native/bin/python -u -m runner.supervisor --phase performance
.venv-native/bin/python -u -m runner.supervisor --phase official_utility
.venv-native/bin/python -u -m runner.supervisor --phase engine_baseline
```

默认一次只跑一个GPU任务，等待GPU空闲和足够显存再启动，不会停止别人的进程。保留名义2GiB余量，单进程分配上限78GiB，大模型及计时任务独占队列。若仅调小模型并发，可将`max_concurrent_jobs`由1改成2；大模型不能并驻，不允许多卡。

FP32官方主复现使用corrected校准器；历史CPU精度桥接单列legacy/bf16保护条件；完整大模型使用BF16权重＋FP32保护并单列。不能把不同profile混池挑最好结果。

## 8. 看进度、暂停、恢复

```bash
.venv-native/bin/python -m runner.status
```

查看每项已完成记录数、期望数、状态和基于近期记录速度的ETA（正式运行前不凭CPU速度估算A100总时长）。不要仅凭进程还在就判断实验正常。

暂停并在当前完整观察写盘后释放显存：

```bash
bash scripts/pause.sh
```

恢复同一队列，自动跳过已经完整测量的ID：

```bash
bash scripts/resume.sh all
```

若必须立即释放卡，只中止本包验证过的worker PID：

```bash
.venv-native/bin/python -m runner.stop_owned
```

保留原始结果。若中断恰好留下最后一行不完整，使用内置工具保存原始片段再恢复完整行，然后续跑同一ID：

```bash
.venv-native/bin/python -m runner.repair_transport --job 这里填队列中的job_id
```

错误、OOM、下载中断和未完成记录均保留。正常的错误预测、空恢复或功能负结果不会被重跑覆盖。

## 9. 完成后验数并打包给Codex

```bash
.venv-native/bin/python -m runner.verify_outputs --completed-only
.venv-native/bin/python -m runner.summarize
.venv-native/bin/python -m runner.export_results
```

最后一条生成`gpu-results-private-handoff.zip`，包含原始记录、错误、日志和恢复现场，供Codex验收；它不是匿名发布包。匿名材料由验收后统一整理。不能只交summary，也不能直接将未验收结果改进论文。

## 科学实现和数据说明

- `audit/EXPERIMENT_AUDIT_ZH.md`逐项解释全部21类研究的CPU/GPU边界、样本规模与新队列。
- 字节防御格式为版本2，修复了无公共前缀时私有尾部缺少模型/tokenizer/RoPE绑定的问题；旧CPU证据保留为版本1。测试密钥用于已知密钥软件验证。
- 保护前复制输入，防止官方原地操作污染下一种子/条件；每次运行环境单独保存，不能用CPU记录完成GPU任务。
- 误差日志保存实际使用的校准估计、有效保护参数、置乱及失败块矩阵，评估字段在预测完成后才使用，不给解码器真值或秘密参数。
- 原生计时同步CUDA两端，可信CPU密码学与D2H/H2D传输计入；当前缓存克隆/nonce和恢复输入准备在计时外，各方法同负载。
- 包内科学文件与随包只读基准直接逐字节比较；改动后队列拒绝执行。允许的环境适配只在`runtime/`和两个bootstrap脚本中。
- 本包已做本地逻辑验证；实际A100 CUDA执行由固定GPU闸门确认。没有把本地CPU测试冒充GPU测量。

精度策略见`protocol/NUMERICAL_PRECISION.json`：新主复现保护过程保留FP32；14B/32B的完整BF16模型也使用FP32缓存保护以避免已发现的标记舍入损失。TF32和BF16低精度归约关闭，历史BF16保护桥接结果独立报告。
