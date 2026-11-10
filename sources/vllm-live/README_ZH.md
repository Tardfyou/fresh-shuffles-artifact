# 真实 vLLM KV 页接入：单卡 GPU 补充实验（v2）

本包独立于原主 GPU 包，模型推理只在 CUDA/vLLM 上运行。复用已有 lifecycle100 输入，不另收集真实用户样本。不要覆盖服务器上正在运行的原包。

## 内容和规模

| 部分 | 固定规模 | 验证内容 |
|---|---|---|
| VeilKV 防御 | Qwen-0.5B、Llama-1B × 物理块16/32 | 每组100续写、20组公共前缀共享、10组相同私有文本隔离、40篡改案例 |
| 攻击兼容 | Qwen-0.5B，物理块16，100条原输入 | 真实KV页保护出口上的重复输入校准和内容恢复 |
| CUDA启动检查 | 四组防御＋一组攻击，共50案例 | 引擎接口、页布局、认证、实际消费、共享命中与记录守恒 |

正式队列共780案例。每个正常案例执行六步greedy生成；独立plain和duplicate-plain对照必须完全相同。公共前缀固定32 token，来自原清单，后接已有no_pad fixture正文，构建说明在protocol/INPUTS.json。保留全部输入、预测和正常负结果。

## 实际接入范围

固定vLLM0.8.5.post1的V0/UniProc/FlashAttention/eager路径。Worker读取引擎实际物理缓存页及请求块表，在每次模型执行后将有效KV保护成host对象，并清零对应GPU页；下一次执行前认证所有相关对象后恢复到原物理页，模型真实消费这些页。可信CPU密码学和传输边界明确记录，不补CPU模型实验。

请求进入调度器前加入缓存作用域：前32个策略公共token跨主体共享，后续私有块缓存身份加入主体和请求。记录真实num_cached_tokens及每步物理块表，核对共享页相同、私有页分离。VeilKV的16-token密码块组合成16/32-token物理页。

攻击观察来自同一真实页保护出口；秘密参数与排列不传给恢复程序。合法恢复端另持排列，并用实际矩阵逆恢复物理行顺序：这是新增接入适配器，不冒充上游原生deobfuscate。保护出口仍调用官方obfuscate；FP32保护结果按FP32保存，不再量化回BF16。合法续写数值差单列保留。

本包测试单GPU、greedy、host保护存储、认证恢复、真实PagedAttention消费、前缀复用与请求隔离。多GPU、beam/COW、引擎CPU swap、真实TEE、生产服务不在本小规模协议内。CPU合成单测不代替CUDA启动检查。

v1曾在源码守卫处误报；本版已修复，原现场保留。已有v1的恢复步骤见FIX_V2_ZH.md。

## 1. 上传、解压和环境

上传veilkv-vllm-live-20261110-v2.zip到GPU服务器，执行：

```bash
unzip ~/veilkv-vllm-live-20261110-v2.zip
cd veilkv-vllm-live-v2
```

在runtime/settings.json配置GPU编号、原主包模型缓存路径、输出路径、资源比例。复用原主包已有vLLM环境：

```bash
export VEILKV_VLLM_PYTHON=/你的原主包目录/.venv-vllm/bin/python
bash scripts/bootstrap.sh
```

也可以创建独立环境：

```bash
bash scripts/bootstrap.sh
export VEILKV_VLLM_PYTHON="$PWD/.venv-vllm/bin/python"
```

固定Torch2.6.0、HF4.51.3、vLLM0.8.5.post1。Claude只处理安装/路径和资源；科学接口代码不改。

可先做不加载模型的源码预检：

```bash
"$VEILKV_VLLM_PYTHON" -m integration.api_guard
```

## 2. GPU空闲后先启动检查

原主GPU队列优先。本包等待卡上无其它计算进程，不停止别人任务。这里仅0.5B/1B，不加载14B/32B；以vLLM资源比例调节，不设置原包70GiB硬上限。

```bash
"$VEILKV_VLLM_PYTHON" -u -m integration.driver --phase smoke
```

50案例完成且字节防御/认证门通过后产生SMOKE_GATE.json。攻击正常错误和KV-Cloak数值负结果照实保留，不要求恢复率100%才能完成账目验收。

## 3. 正式运行与查看进度

```bash
nohup "$VEILKV_VLLM_PYTHON" -u -m integration.driver --phase all > runtime/queue.log 2>&1 &
"$VEILKV_VLLM_PYTHON" -m integration.manage status
```

也可分--phase defense或--phase attack。单GPU串行。重复执行核对并跳过已完整测量的ID；正常错误不覆盖。异常保存在attempts.jsonl，缺失同ID继续。

前台Ctrl-C可以停止本包；保留所有文件后重复同一phase恢复。若写盘断在最后一行，只用以下工具保留原片段再修复：

```bash
"$VEILKV_VLLM_PYTHON" -m integration.manage repair --job 这里填job_id
```

科学断言或接口核对失败，保留目录、stdout和完整错误交Codex；不软化阈值绕过。环境适配写runtime/ENVIRONMENT_ADAPTATIONS.md。

## 4. 交付

```bash
"$VEILKV_VLLM_PYTHON" -m integration.verify
"$VEILKV_VLLM_PYTHON" -m integration.manage export
```

交回vllm-live-private-handoff.zip：包含原记录、每次实际环境、受保护探针、估计参数、物理页/作用域证据、异常和ledger。此为私有验收包，不能直接推匿名仓。GPU验收后作为主要数据更新对应paper，CPU作为辅助对照。

科学文件与随包基准逐字节核对，安装的关键vLLM接口也与固定上游源码比较。输入/精度/模型版本/算法/阈值固定。本包完成本地合成张量和结构测试；实际A100运行由CUDA smoke确认。
