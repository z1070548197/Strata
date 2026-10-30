#!/bin/bash
set -euo pipefail

ROOT=/media/zc/软件1/strata-nvfp4
MODEL=/media/zc/软件1/strata-models/NVFP4-GPTQ
IDS=${IDS:-/tmp/long_ids.txt}
OUT=${1:-/tmp/strata-solo-prefill-8k}

cd "$ROOT"
rm -f "$OUT.log" "$OUT.status"
test "$(wc -w < "$IDS")" -eq 8000

dd if="$MODEL/pack/experts.bin" of=/dev/null bs=8M status=none

set +e
{
  printf 'GEN 32 '
  tr '\n' ' ' < "$IDS"
  printf '\nQUIT\n'
} | timeout 1800 env STRATA_POOL_ORDER=1 STRATA_DECODE_TIMING=1 \
  ./build/strata --serve \
    --pack "$MODEL/pack" \
    --native "$MODEL/qwen-nvfp4-gptq-dense.gguf" \
    --native-dense-gguf "$MODEL/qwen-nvfp4-gptq-dense.gguf" \
    --ple-gguf "$MODEL/ple-fp8.gguf" \
    --embd-gguf "$MODEL/token-embd-bf16.gguf" \
    --mtp "$MODEL/mtp" --spec 4 --prefill auto \
    --expert-profile data/expert-profile.bin --expert-cache 4000 \
    --adapt-every 0 --adapt-swaps 0 \
    --max-context 16384 --kv int8 --vram-reserve-mib 960 \
    >"$OUT.log" 2>&1
status=$?
set -e
printf 'EXIT=%d\n' "$status" >"$OUT.status"
exit "$status"
