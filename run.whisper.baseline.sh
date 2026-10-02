#########################################################################
# File Name: run.whisper.baseline.sh
# Author: Xianchao Wu
# mail: xianchaow@nvidia.com
# Created Time: Thu Sep 24 11:06:17 2026
#########################################################################
#!/bin/bash
GPU_LIST=0,1,2,3,4 DATASETS=test,e21 bash scripts/run_whisper_frozen_baselines.sh
