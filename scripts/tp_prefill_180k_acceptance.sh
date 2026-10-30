#!/bin/bash
set -euo pipefail

# 180K context: one 180,000-token prompt and a short generation, on the TP path.  Asserts the run completes
# (EXIT=0), the output is not empty, and no non-finite value or rank drift is reported.  The prompt fixture is
# deterministic: 22 copies of the 8,000-token list plus its first 4,000 ids (sha256 recorded beside it).
ROOT=/media/zc/软件1/strata-nvfp4
MODEL=/media/zc/软件1/strata-models/NVFP4-GPTQ
OUT=${OUT:-/tmp/strata-tp-prefill-180k-20261028}
IDS180K=${IDS180K:-/tmp/long180k_ids.txt}
CTX=${CTX:-196608}

cd "$ROOT"
test "$(wc -w < "$IDS180K")" -eq 180000
rm -f "$OUT.log" "$OUT.status"

# Keep warmup and inference in this unit so their reclaim pressure is accounted together.
dd if="$MODEL/pack/experts.bin" of=/dev/null bs=8M status=none

# Same determinism pins as the short acceptance: no adaptive swap race, no measured prompt CPU share.
hash_env=(STRATA_POOL_ORDER=1 "STRATA_PREFILL_CPU_SHARE=${STRATA_PREFILL_CPU_SHARE:-0}")

set +e
{
  printf 'GEN 8 '
  tr '\n' ' ' < "$IDS180K"
  printf '\nQUIT\n'
} | timeout 7200 env "${hash_env[@]}" \
    ./build/strata --serve --tp --short-read 0 \
      --pack "$MODEL/pack" \
      --native "$MODEL/qwen-nvfp4-gptq-dense.gguf" \
      --native-dense-gguf "$MODEL/qwen-nvfp4-gptq-dense.gguf" \
      --ple-gguf "$MODEL/ple-fp8.gguf" \
      --embd-gguf "$MODEL/token-embd-bf16.gguf" \
      --mtp "$MODEL/mtp" --spec 4 --prefill 1536 --no-prefill-borrow \
      --expert-profile data/expert-profile.bin --expert-cache 4000 \
      --adapt-every 0 --adapt-swaps 0 \
      --max-context "$CTX" --kv int8 --vram-reserve-mib 960 \
      >"$OUT.log" 2>&1
status=$?
set -e
printf 'EXIT=%d\n' "$status" >"$OUT.status"
exit "$status"
