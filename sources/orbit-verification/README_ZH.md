# 公开置换实现的轨道验证

来源：Hidden No More作者公开仓库leopardracer/vma-external，固定revision见protocol/UPSTREAM.json。只调用原permute_states函数，不调用词表匹配攻击。GELO默认稠密混合不是纯置换；本实验不声称攻破它。

输入是已有lifecycle100内提取的10个单token英文词汇锚点（每类两个），构造合法重复文本并保留Llama默认BOS。不是新增100条自然语言恢复样本。b=4/8/16/32，三类token重复度，dtype FP32/bf16。保护输入是Llama3.2-1B原始layer0 embedding查表结果，未运行模型前向或后注意力层。S配置预计轨道由token行重数决定：b!/∏n_i!。BOS＋b−1相同行的轨道预计为b；它没有人为注入KV-Cloak marker。

D与SD每次刷新列置换，完整输出可能继续变化；不能把S配置的b状态扩展成SD完整随机性的b状态。即使观察到坍缩，也不等于已完成第二套参数识别或内容恢复。

## 已冻结数据直接执行

GPU服务器使用已有主GPU环境、已有Llama1B模型缓存，只需小量embedding行与置换张量；不加载完整模型，不启动服务，不用API。输出必须是新目录，原记录保留。

```bash
python run_orbit.py --cache /你的模型缓存/hub --device cuda:0 --out results/gpu_operator
python audit_orbit.py --results results/gpu_operator
```

CPU结构验证使用相同脚本的--device cpu，不等于完整CPU推理实验。准备阶段--prepare仅供构建输入，正式运行不重新选样本。

协议与fixtures在测量之前冻结。所有registered cells都保留，出现正常负结果不筛选。临时观测只记一次每个输入/精度/profile条件；每cell保存全部trial种子、完整置换、输出状态编号及精确索引对照。实际公开embedding行独立落盘；审核脚本按这些数据重建整个输出分区，不只检查summary。recording wrapper仅记录randperm返回值，正式采样前32个固定小例对照原函数确保不改变RNG/输出。

实验用于验证额外公开配置的输入稳定子效应，不改变原论文形式化内容或主实验。条件、来源、完整正负结果必须一起报告。
