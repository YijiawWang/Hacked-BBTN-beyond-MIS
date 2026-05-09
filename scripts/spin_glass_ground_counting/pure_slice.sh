#!/usr/bin/env bash
#
# pure_slice.sh — bash wrapper around `scripts/pure_slice_spin_glass.jl`.
#
# Runs standalone TreeSA slicing for every `.model` in a directory and dumps
# one representative slice or all slices into `beyond_mis/branch_results/`.
# The root index is written to `branch_results/pure_slicing_summary.csv`.
#
# Usage:
#   ./pure_slice.sh --models-dir=<dir> --sc-target=<int> \
#                   [--save-mode=one|all] [--recursive] [--h=<float>] \
#                   [--subdir=<name>] [--subdir-template=<template>] \
#                   [--seed=<int>] [--ntrials=<int>] [--niters=<int>] \
#                   [--overwrite=false] [--quiet]
#   ./pure_slice.sh --help
#
# Environment overrides:
#   JULIA_BIN     — julia executable (default: 'julia')
#   JULIA_PROJECT — Julia project path (default: <repo>/beyond_mis)
#   Julia is run with `-t 32`.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS_DIR="$(dirname "$SCRIPT_DIR")"
BEYOND_MIS_DIR="$(dirname "$SCRIPTS_DIR")"
JL_DRIVER="$SCRIPTS_DIR/pure_slice_spin_glass.jl"

JULIA_BIN="${JULIA_BIN:-julia}"
JULIA_PROJECT="${JULIA_PROJECT:-$BEYOND_MIS_DIR}"
JULIA_THREADS=32

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<EOF
$(basename "$0") — pure TreeSA-slice spin-glass model directories

Usage: $(basename "$0") --models-dir=<dir> --sc-target=<int> [...]

Forwards every flag verbatim to:
  $JL_DRIVER

Slice dumps land under:
  $BEYOND_MIS_DIR/branch_results/
Root summary:
  $BEYOND_MIS_DIR/branch_results/pure_slicing_summary.csv
EOF
  exec "$JULIA_BIN" --project="$JULIA_PROJECT" -t "$JULIA_THREADS" \
    "$JL_DRIVER" --help
fi

if [[ ! -f "$JL_DRIVER" ]]; then
  echo "ERROR: missing Julia driver: $JL_DRIVER" >&2
  exit 1
fi

echo "[pure_slice.sh] julia    : $JULIA_BIN"
echo "[pure_slice.sh] project  : $JULIA_PROJECT"
echo "[pure_slice.sh] threads  : $JULIA_THREADS"
echo "[pure_slice.sh] driver   : $JL_DRIVER"
echo "[pure_slice.sh] forward  : $*"

LOG_TMP="$(mktemp -t pure-slice-sh-XXXXXX.log)"
trap 'rm -f "$LOG_TMP"' EXIT

started_at_iso="$(date '+%Y-%m-%dT%H:%M:%S%z')"
started_ns="$(date +%s%N)"

set +e
"$JULIA_BIN" --project="$JULIA_PROJECT" -t "$JULIA_THREADS" \
  "$JL_DRIVER" "$@" 2>&1 | tee "$LOG_TMP"
status=${PIPESTATUS[0]}
set -e

finished_at_iso="$(date '+%Y-%m-%dT%H:%M:%S%z')"
finished_ns="$(date +%s%N)"
wall_ns=$(( finished_ns - started_ns ))
wall_seconds="$(awk -v n="$wall_ns" 'BEGIN { printf "%.3f", n / 1.0e9 }')"

mapfile -t slice_dirs < <(grep -E '^[[:space:]]*saved to[[:space:]]+/' "$LOG_TMP" \
  | awk '{print $NF}' | sort -u)

if [[ "${#slice_dirs[@]}" -gt 0 ]]; then
  forwarded_q=$(printf '%q ' "$@")
  for slice_dir in "${slice_dirs[@]}"; do
    [[ -d "$slice_dir" ]] || continue
    cat > "$slice_dir/pure_slice_runtime.txt" <<EOF
wrapper        = $(basename "${BASH_SOURCE[0]}")
wall_seconds   = $wall_seconds
started_at     = $started_at_iso
finished_at    = $finished_at_iso
exit_status    = $status
threads        = $JULIA_THREADS
julia_bin      = $JULIA_BIN
julia_project  = $JULIA_PROJECT
driver         = $JL_DRIVER
forwarded_args = ${forwarded_q% }
EOF
    cp "$LOG_TMP" "$slice_dir/pure_slice.log"
    echo "[pure_slice.sh] runtime  : ${wall_seconds}s (wall) -> $slice_dir/pure_slice_runtime.txt"
  done
else
  echo "[pure_slice.sh] WARN: could not locate slice dump dirs; wall=${wall_seconds}s not persisted" >&2
fi

exit "$status"
