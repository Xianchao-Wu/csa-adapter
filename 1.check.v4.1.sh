#########################################################################
# File Name: 1.check.v4.1.sh
# Author: Xianchao Wu
# mail: xianchaow@nvidia.com
# Created Time: Sat Sep 26 10:23:49 2026
#########################################################################
#!/bin/bash

echo "===="

echo "select_|select_checkpoint|checkpoint.*select|launch.*select|run_select|screen" 

echo "grep -n -E 'select_|select_checkpoint|checkpoint.*select|launch.*select|run_select|screen' run_h200_01_memory_compression_v4.sh"

grep -n -E \
'select_|select_checkpoint|checkpoint.*select|launch.*select|run_select|screen' \
run_h200_01_memory_compression_v4.sh

echo "===="

echo "launch_job\|launch_gpu\|run_job\|wait.*pid"

echo "grep -n -A35 -B10 'launch_job\|launch_gpu\|run_job\|wait.*pid' run_h200_01_memory_compression_v4.sh"

grep -n -A35 -B10 \
'launch_job\|launch_gpu\|run_job\|wait.*pid' \
run_h200_01_memory_compression_v4.sh

echo "===="
echo "bash -c\|nohup\|eval\|xargs\|setsid"
echo "grep -n 'bash -c\|nohup\|eval\|xargs\|setsid' run_h200_01_memory_compression_v4.sh"
grep -n \
'bash -c\|nohup\|eval\|xargs\|setsid' \
run_h200_01_memory_compression_v4.sh

echo "===="
echo "^[[:space:]]*[a-zA-Z_][a-zA-Z0-9_]*\(\)[[:space:]]*\{" 
echo "grep -nE '^[[:space:]]*[a-zA-Z_][a-zA-Z0-9_]*\(\)[[:space:]]*\{' run_h200_01_memory_compression_v4.sh"
grep -nE \
'^[[:space:]]*[a-zA-Z_][a-zA-Z0-9_]*\(\)[[:space:]]*\{' \
run_h200_01_memory_compression_v4.sh

