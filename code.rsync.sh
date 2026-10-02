#########################################################################
# File Name: code.rsync.sh
# Author: Xianchao Wu
# mail: xianchaow@nvidia.com
# Created Time: Fri Oct  2 00:55:04 2026
#########################################################################
#!/bin/bash

rsync -av \
    --exclude='.git' \
    --exclude='runs/' \
    --exclude='checkpoints/' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='cache/' \
    --exclude='data/' \
    ../csa-adapter-v0.2.1/ ./
