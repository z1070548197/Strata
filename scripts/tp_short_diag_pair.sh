#!/bin/bash
set -euo pipefail

ROOT=/media/zc/软件1/strata-nvfp4
MODEL=/media/zc/软件1/strata-models/NVFP4-GPTQ
OUT=${1:-/tmp/strata-tp-short-diag}
COMMON=(
  --pack "$MODEL/pack"
  --native "$MODEL/qwen-nvfp4-gptq-dense.gguf"
  --native-dense-gguf "$MODEL/qwen-nvfp4-gptq-dense.gguf"
  --ple-gguf "$MODEL/ple-fp8.gguf"
  --embd-gguf "$MODEL/token-embd-bf16.gguf"
  --mtp "$MODEL/mtp" --spec 4 --prefill auto
  --expert-profile data/expert-profile.bin --expert-cache 5362
  --adapt-every 0 --adapt-swaps 0
  --max-context 4096 --kv int8 --vram-reserve-mib 960
)

cd "$ROOT"
rm -f "$OUT"* /tmp/strata-diag-vec.solo.* /tmp/strata-diag-vec.tp*.*
dd if="$MODEL/pack/experts.bin" of=/dev/null bs=8M status=none

run_one() {
  local mode=$1
  shift
  set +e
  printf 'GEN 8 9707 11 1246\nQUIT\n' | timeout 540 env \
    STRATA_POOL_ORDER=1 STRATA_DECODE_TIMING=1 \
    STRATA_TP_DUMP=1 STRATA_TP_VEC=15360 STRATA_TP_VEC_FILE=/tmp/strata-diag-vec \
    STRATA_PREFILL_DUMP_R="$OUT-$mode-prefill-r.bin" \
    STRATA_STATE_HASH=1 STRATA_STATE_HASH_GDN=1 \
    STRATA_DUMP_FIRST_LOGITS="$OUT-$mode-logits" \
    ./build/strata --serve "$@" "${COMMON[@]}" >"$OUT-$mode.log" 2>&1
  local status=$?
  set -e
  printf 'EXIT=%d\n' "$status" >"$OUT-$mode.status"
  return "$status"
}

run_one solo
run_one tp --tp --expert-cache-device1 6000
