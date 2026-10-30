#!/bin/bash
set -u
cd /media/zc/软件1/strata-nvfp4 || exit 9
MODEL=/media/zc/软件1/strata-models/NVFP4-GPTQ
export STRATA_POOL_ORDER=1 STRATA_DECODE_TIMING=1
( printf 'GEN 8 9707 11 1246\n'; sleep 30 ) | timeout 540 ./build/strata --serve --tp --expert-cache-device1 6000 \
  --pack "$MODEL/pack" \
  --native "$MODEL/qwen-nvfp4-gptq-dense.gguf" \
  --native-dense-gguf "$MODEL/qwen-nvfp4-gptq-dense.gguf" \
  --ple-gguf "$MODEL/ple-fp8.gguf" \
  --embd-gguf "$MODEL/token-embd-bf16.gguf" \
  --mtp "$MODEL/mtp" --spec 4 --prefill auto \
  --expert-profile data/expert-profile.bin --expert-cache 5362 \
  --adapt-every 0 --adapt-swaps 0 \
  --max-context 4096 --kv int8 --vram-reserve-mib 960 > /tmp/final_tp.log 2>&1
echo "EXIT=$?" > /tmp/final_tp.status
