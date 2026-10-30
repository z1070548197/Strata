#!/bin/bash
set -euo pipefail
ROOT=/media/zc/软件1/strata-nvfp4
TP_OUT=${TP_OUT:-/tmp/strata-tp-prefill-recurrencefix-20261028}
SOLO_OUT=${SOLO_OUT:-/tmp/strata-solo-prefill-recurrencefix-20261028}
OUT="$TP_OUT" HASH="${HASH:-0}" ONLY65="${ONLY65:-0}" bash "$ROOT/scripts/tp_prefill_gatefix_acceptance.sh"
OUT="$SOLO_OUT" HASH="${HASH:-0}" ONLY65="${ONLY65:-0}" bash "$ROOT/scripts/solo_prefill_gatefix_acceptance.sh"
