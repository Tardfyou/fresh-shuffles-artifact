# PermLLM二维置换与原生三方协议的可见轨道验证

固定上游：zfscgy/PermLLM @ 1b120aeb65f18744a1cd73dd419dc15f39a73971，对应PermLLM: Private Inference of Large Language Models within 3 Seconds under WAN (arxiv2405.18744)。原代码不改；此目录只含我们的驱动、注册协议和复用输入。上游仓库目前无明确分发license，发行包通过固定revision取源码，不拷贝其模块。

复用第一轮120个合法文本fixture和公开Llama1B layer0输入表示，10词汇锚点、b4/8/16/32、三种重复度、FP32/bf16。不是新的用户集合，不跑模型前向。

## 测的是什么

实际LayerNorm/Softmax保护路径调用permute_2d_with_seed：每行独立坐标置换，再做行置换。因此原始输出仍有大量变化。分别记录完整输出的精确状态数，以及逐行排序这个公开可计算不变量的状态数；不能把可见签名只有b种写成完整密文只有b种。

同时实际执行原SS_ElementWise__RandPerm、SS_Perm及SimulatedCommunication三方offline/online流程。透明LayerNorm回调记录n1实际获得的permuted_x，随后原样调用F.layer_norm。公开已知自身探针的排序行作为原型，输出出现前按最小原型间隔1/4固定误差门限；逐例保留签名稳定性与LayerNorm恢复误差（阈值1e-3）。原模块通用API的输入宽2048、nativefloat32分享；不声称完整ChatGLM6B模型或真实多机部署。

## 复现

从仓库根目录准备固定上游，保持源码不变：

```bash
git clone https://github.com/zfscgy/PermLLM.git /tmp/permllm-upstream
git -C /tmp/permllm-upstream checkout 1b120aeb65f18744a1cd73dd419dc15f39a73971
```

实验使用Torch和numpy（已验收记录中的Torch2.14.0+cpu），不需要transformers4.27、ChatGLM权重或Pyfhel。两者只在完整模型/HE协议中需要，本算子及通用非线性协议不导入这些组件。

```bash
python run_check.py --upstream /tmp/permllm-upstream \
  --inputs /artifact/results/orbit_verification_20261111/public_embedding_rows.json.gz \
  --output /tmp/permllm-orbit-results
python audit_check.py --upstream /tmp/permllm-upstream \
  --inputs /artifact/results/orbit_verification_20261111/public_embedding_rows.json.gz \
  --results /tmp/permllm-orbit-results
```

使用新输出目录，不覆盖任何旧记录。审核器独立重建逐次二维排列及公开签名、重跑原生三方流程；所有正常功能负结果必须保留，不以达到100%为补跑目标。
