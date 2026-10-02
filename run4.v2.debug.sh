#########################################################################
# File Name: run4.v2.debug.sh
# Author: Xianchao Wu
# mail: xianchaow@nvidia.com
# Created Time: Mon Sep 28 01:09:32 2026
#########################################################################
#!/bin/bash

RANKS=16 \
SEEDS=42 \
STAGES=csa \
GPUS=0,1,2,3,4,5,6,7 \
bash run_h100_lora_csa_joint_v2.sh \
2>&1 | tee h100_lora_csa_rank16_s42_v2_debug.log
