#########################################################################
# File Name: run4.joint.lora.csa.sh
# Author: Xianchao Wu
# mail: xianchaow@nvidia.com
# Created Time: Sun Sep 27 21:32:30 2026
#########################################################################
#!/bin/bash

#SEEDS=42 \

STAGES=cache,csa,eval,summary \
RANKS=8,16,32,64 \
SEEDS=43,44 \
GPUS=4,5,6,7 \
bash run_h100_lora_csa_joint_v2.sh \
2>&1 | tee h100_lora_csa_joint_v2.seed43.44.log
