#########################################################################
# File Name: run4.v3.debug.sh
# Author: Xianchao Wu
# mail: xianchaow@nvidia.com
# Created Time: Mon Sep 28 01:20:20 2026
#########################################################################
#!/bin/bash

RANKS=16 \
SEEDS=42 \
STAGES=eval,summary \
GPUS=1 \
bash run_h100_lora_csa_joint_v3.sh \
2>&1 | tee h100_lora_csa_rank16_s42_v3.debug.log
