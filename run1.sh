#########################################################################
# File Name: run1.sh
# Author: Xianchao Wu
# mail: xianchaow@nvidia.com
# Created Time: Thu Sep 24 09:57:09 2026
#########################################################################
#!/bin/bash

GPUS=0,1,2,3,4,5,6,7 bash scripts/run_h100_v021_stability_8gpu.sh
