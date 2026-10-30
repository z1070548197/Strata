#!/bin/bash
set -u
dd if=/media/zc/软件1/strata-models/NVFP4-GPTQ/pack/experts.bin of=/dev/null bs=8M 2>/dev/null
sync
echo "cache warmed: $(free -h | awk 'NR==2{print $6}')" > /tmp/warm_then_run.stage
/tmp/final_run.sh
