#########################################################################
# File Name: run4.v3.debug.sh
# Author: Xianchao Wu
# mail: xianchaow@nvidia.com
# Created Time: Mon Sep 28 01:20:20 2026
#########################################################################
#!/bin/bash


#SEEDS=42,43,44 \

RANKS=8,16,32,64 \
SEEDS=42 \
STAGES=csa,eval,summary \
GPUS=0,1,2,3,4,5,6,7 \
bash run_h100_lora_csa_joint_v3.sh \
2>&1 | tee h100_lora_csa_rank16_s42_v3_formal.log
