#!/bin/bash
set -euo pipefail

ROOT=/media/zc/软件1/strata-nvfp4
MODEL=/media/zc/软件1/strata-models/NVFP4-GPTQ
OUT=${OUT:-/tmp/strata-tp-prefill-gatefix-20261028}
IDS65=${IDS65:-/tmp/short65_ids.txt}
IDS8K=${IDS8K:-/tmp/long_ids.txt}

cd "$ROOT"
test "$(wc -w < "$IDS65")" -eq 65
test "$(wc -w < "$IDS8K")" -eq 8000
rm -f "$OUT.log" "$OUT.status"

# Keep warmup and inference in this unit so their reclaim pressure is accounted together.
dd if="$MODEL/pack/experts.bin" of=/dev/null bs=8M status=none

# The prompt CPU share is measured per layer from wall time, so which experts the idle CPU takes changes run to
# run (and the CPU pool's arithmetic is not the GPU's).  Pin it off: solo/TP token identity is only meaningful
# under a deterministic expert path, like --adapt-every 0 is for the decode side.
pin_env=(STRATA_POOL_ORDER=1 "STRATA_PREFILL_CPU_SHARE=${STRATA_PREFILL_CPU_SHARE:-0}")

set +e
# One request per line, each line ending exactly once: the serve reader takes one request per line and a blank
# line is a protocol error.  SHORT: the accepted short generation (its 8th token is the deterministic 9341).
{
  if [[ ${SHORT:-1} == 1 ]]; then printf 'GEN 8 9707 11 1246\n'; fi
  if [[ ${ONLY8K:-0} != 1 ]]; then printf 'GEN 1 '; tr -s '\n' ' ' < "$IDS65"; printf '\n'; fi
  if [[ ${ONLY65:-0} != 1 ]]; then printf 'GEN 32 '; tr -s '\n' ' ' < "$IDS8K"; printf '\n'; fi
  printf 'QUIT\n'
} | timeout 3600 env "${pin_env[@]}" \
    ./build/strata --serve --tp --short-read 0 \
      --pack "$MODEL/pack" \
      --native "$MODEL/qwen-nvfp4-gptq-dense.gguf" \
      --native-dense-gguf "$MODEL/qwen-nvfp4-gptq-dense.gguf" \
      --ple-gguf "$MODEL/ple-fp8.gguf" \
      --embd-gguf "$MODEL/token-embd-bf16.gguf" \
      --mtp "$MODEL/mtp" --spec 4 --prefill 1536 --no-prefill-borrow \
      --expert-profile data/expert-profile.bin --expert-cache 4000 \
      --adapt-every 0 --adapt-swaps 0 \
      --max-context 16384 --kv int8 --vram-reserve-mib 960 \
      >"$OUT.log" 2>&1
status=$?
set -e
printf 'EXIT=%d\n' "$status" >"$OUT.status"
exit "$status"
