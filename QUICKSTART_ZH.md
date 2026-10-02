# CSA-Adapter v0.2.1 快速开始

> v0.2.1 是针对长音频 memory inference 稳定性的补丁。旧 v0.2 的 encoder feature cache 可以继续使用，但 adapter checkpoint 需要重新训练。建议先阅读 `PATCH_NOTES_v0.2.1.md`。


本版本实现跨段持久声学记忆：冻结 Whisper，在最终 encoder 输出后加入一个
CSA-Adapter，读取历史片段，解码成功后才写入当前片段。默认约 52.9 万可训练参数。

## 推荐先跑：Earnings-22 自定义 6:2:2

```bash
unzip csa-adapter-v0.2.1.zip
cd csa-adapter-v0.2.1
bash scripts/00_setup.sh
bash scripts/01_prepare_earnings22.sh
GPUS=0,1,2,3,4,5,6,7 bash scripts/02_cache_features.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/07_smoke_train.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/03_train.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/04_eval_matrix.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/05_eval_earnings21.sh
```

- 按整场电话会分组，75 train / 25 validation / 25 test，默认 seed=42。
- 不是官方 split；不能把自定义 25 场测试结果称为官方 Earnings-22 全测试集成绩。
- 缓存支持八卡分片并行；训练脚本是单卡、微批量1、梯度累积8，不是 DDP。
- 先运行 smoke_train 检查实际数据和机器环境；完整训练尚未在本交付环境执行。
- 最终评测读取完整电话会，固定30秒分段，不使用参考文本指导测试切分。
- 开启文本上下文时，只使用之前的预测文本，绝不使用历史正确转录。
- 当前还没有接入原生 Whisper 的时间戳滑窗、温度回退和边界去重。

训练段由原录音和已有对齐时间戳合并为最多28秒，尽量靠近测试窗口长度。
超长单条对齐记录不能安全按词拆开，因此脚本显式跳过，并输出
excluded_segments.jsonl 和 preparation_report.json。完整测试音频不被丢弃。

## 另一套组合：E22 8:2，E21 外部测试

```bash
GPUS=0,1,2,3,4,5,6,7 CUDA_VISIBLE_DEVICES=0 \
  bash scripts/06_earnings22_80_20_to_earnings21.sh
```

100场 E22 训练、25场 E22 验证、44场 E21 外部测试。采用独立的数据、缓存和实验目录。
这不是严格的公司/说话人不重叠协议；如需该结论，需要进一步审核元数据。

## 输出和资源

- data/earnings22_622/split_calls.json：确切划分，preparation_report.json：小时数和排除项。
- runs/earnings22_csa/best_adapter/：按验证 NLL 选择的 adapter-only checkpoint。
- runs/earnings22_csa/eval_test/：有/无 CSA × 有/无历史预测文本的四组结果，以及清空记忆对照。
- metrics.json：标准化 WER（micro / macro）、原始 macro WER、RTF。
- calls.jsonl：逐录音预测、错误计数、耗时、显存与生成长度诊断。

默认最优模型依据固定的256个验证片段计算 NLL。正式论文应使用完整验证电话会的
WER 检查模型选择，然后固定模型再跑测试；不要依据 test 结果选超参数。

冻结 encoder 输出缓存约为每训练片段3.84 MB（large-v3）；一万片段约38.4 GB。
建议预留足够 NVMe 空间。训练历史默认为64个保留片段、最多4096条压缩记忆。
FIFO会淘汰旧记忆，所以不等于无限保留整场录音；无间断时约覆盖10.9分钟声学帧。
OOM时可以先减小历史，再增加容量：

```bash
RUN_DIR=runs/csa_small_history bash scripts/03_train.sh \
  --history-segments 8 --max-memory 1024 --grad-accum 8
```

目录已存在时训练不会覆盖，也不提供优化器恢复；新实验设置新的 RUN_DIR。
旧 v0.1 adapter 权重不能直接作为 v0.2 跨段记忆 checkpoint 使用，需要重新训练。
更多模块说明、消融命令和限制参见 README.md；已执行的检查参见 VALIDATION.md。
## H100 8卡推荐复现流程（v0.2.1）

如果 v0.2 已经生成了 `cache/earnings22_large_v3/{train,validation}`，可以直接复用，不需要重新 cache。

先用 8 张 H100 做稳定性筛查（default hard-sparse 与旧 warm100，各 4 个 seed）：

```bash
export PYTHONPATH=$PWD/src:$PYTHONPATH
GPUS=0,1,2,3,4,5,6,7 \
  bash scripts/run_h100_v021_stability_8gpu.sh
```

确认没有大面积 empty hypothesis 后，再跑完整 priority matrix：

```bash
GPUS=0,1,2,3,4,5,6,7 \
  bash scripts/run_h100_v021_priority_8gpu.sh
```

该脚本会：8卡并行训练 → 对每个保存的 checkpoint 跑 long-form validation → 按 validation WER-N 选 checkpoint/seed → 再跑 E22 test 与 E21 external test。主比较默认关闭 text history，避免把文本 prompt 的不稳定性混入 acoustic-memory 结论。

### 重要：确认当前仓库代码被 Python 使用

v0.2.1 的所有 `scripts/*.sh` 现在都会自动把当前仓库的 `src/` 放到
`PYTHONPATH` 最前面，并在 H100 脚本启动时运行 preflight。正常启动时应看到：

```text
[preflight] package   : .../csa-adapter-v0.2.1/src/csa_adapter
[preflight] v0.2.1 CLI/source checks: OK
```

如果你手工调用 Python，也建议先执行：

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python scripts/preflight_v021.py
```

这可以避免系统 site-packages 中残留的旧 `csa-adapter` 覆盖当前源码。
