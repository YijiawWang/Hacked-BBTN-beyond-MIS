#!/usr/bin/env bash
#
# contract.sh — bash wrapper around `_lib/contract_mis_counting_slices.jl`.
#
# Loads an MIS slice dump (CountingAll / total-counting flavour),
# exactly re-contracts every slice through its **saved** OMEinsum code
# at `x = 1`, and **sums** the per-slice counts into the total IS count
# of the original graph. Per-slice + summary CSVs are written to
# `beyond_mis/results/mis_counting_slice_contract/<results-name>/`.
#
# Usage:
#   ./contract.sh <slice_dir> [--gpu=N|--no-cuda] [--quiet] \
#                 [--count-eltype=finitefield|Float64|Int128|BigInt] \
#                 [--max-crt-iter=<int>] \
#                 [--results-root=<path>] [--results-name=<name>]
#   ./contract.sh --help
#
# Environment overrides: JULIA_BIN, JULIA_PROJECT, JULIA_THREADS.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS_DIR="$(dirname "$SCRIPT_DIR")"
BEYOND_MIS_DIR="$(dirname "$SCRIPTS_DIR")"
JL_DRIVER="$SCRIPTS_DIR/_lib/contract_mis_counting_slices.jl"

JULIA_BIN="${JULIA_BIN:-julia}"
JULIA_PROJECT="${JULIA_PROJECT:-$BEYOND_MIS_DIR}"
JULIA_THREADS="${JULIA_THREADS:-32}"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<EOF
$(basename "$0") — exactly contract an MIS slice dump (CountingAll)

Usage: $(basename "$0") <slice_dir> [flags]

Forwards every flag verbatim to:
  $JL_DRIVER

Results land under:
  $BEYOND_MIS_DIR/results/mis_counting_slice_contract/<results-name>/
EOF
  exec "$JULIA_BIN" --project="$JULIA_PROJECT" -t "$JULIA_THREADS" \
    "$JL_DRIVER" --help
fi

if [[ $# -lt 1 ]]; then
  echo "ERROR: <slice_dir> is required (use --help for usage)" >&2
  exit 2
fi
if [[ ! -f "$JL_DRIVER" ]]; then
  echo "ERROR: missing Julia driver: $JL_DRIVER" >&2
  exit 1
fi

echo "[contract.sh] julia    : $JULIA_BIN"
echo "[contract.sh] project  : $JULIA_PROJECT"
echo "[contract.sh] threads  : $JULIA_THREADS"
echo "[contract.sh] driver   : $JL_DRIVER"
echo "[contract.sh] forward  : $*"

exec "$JULIA_BIN" --project="$JULIA_PROJECT" -t "$JULIA_THREADS" \
  "$JL_DRIVER" "$@"
