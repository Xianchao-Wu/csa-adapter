#########################################################################
# File Name: test.fastconformer.baseline.sh
# Author: Xianchao Wu
# mail: xianchaow@nvidia.com
# Created Time: Thu Sep 24 10:54:58 2026
#########################################################################
#!/bin/bash

CUDA_VISIBLE_DEVICES=0 \
python scripts/eval_fastconformer_frozen.py \
  --manifest data/earnings22_622/test.jsonl \
  --output runs/smoke_fastconformer \
  --batch-size 8 \
  --max-items 16
