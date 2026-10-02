#########################################################################
# File Name: run3.lora.whisper.sh
# Author: Xianchao Wu
# mail: xianchaow@nvidia.com
# Created Time: Sun Sep 27 11:30:42 2026
#########################################################################
#!/bin/bash

#SEEDS=42 \
#bash run_whisper_lora_earnings22_ranks.sh \
#2>&1 | tee lora_rank_s42.log


GPUS=0,1,2,3,4,5,6,7 \
RANKS=8,16,32,64 \
SEEDS=42,43,44 \
bash run_whisper_lora_earnings22_ranks_v2.sh \
2>&1 | tee run_whisper_lora_earnings22_ranks_v2.log
