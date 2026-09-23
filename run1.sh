#########################################################################
# File Name: run1.sh
# Author: Xianchao Wu
# mail: xianchaow@nvidia.com
# Created Time: Tue Sep 22 13:24:31 2026
#########################################################################
#!/bin/bash

bash scripts/00_setup.sh
bash scripts/01_prepare_earnings22.sh

# 八卡并行提取冻结 encoder 特征
GPUS=0,1,2,3,4,5,6,7 bash scripts/02_cache_features.sh

# 先检查真实数据上的训练流程
CUDA_VISIBLE_DEVICES=0 bash scripts/07_smoke_train.sh

# 完整训练：单卡，梯度累积
CUDA_VISIBLE_DEVICES=0 bash scripts/03_train.sh

# 基线/CSA × 历史文本提示关闭/开启
CUDA_VISIBLE_DEVICES=0 bash scripts/04_eval_matrix.sh

# Earnings-21 外部测试
CUDA_VISIBLE_DEVICES=0 bash scripts/05_eval_earnings21.sh
