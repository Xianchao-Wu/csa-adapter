#########################################################################
# File Name: add.export.sh
# Author: Xianchao Wu
# mail: xianchaow@nvidia.com
# Created Time: Thu Sep 24 09:56:49 2026
#########################################################################
#!/bin/bash

#export PYTHONPATH=$PWD/src:$PYTHONPATH

export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
