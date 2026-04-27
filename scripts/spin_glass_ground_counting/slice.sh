#!/usr/bin/env bash
#
# slice.sh — bash wrapper around `_lib/slice_spin_glass.jl`.
#
# Reads a pre-generated `.model` file (the format written by
# `hacked_funcs/benchmarks/models/spin_glass_model_generator.jl`) off
# disk and dumps the slicer's output into
# `beyond_mis/branch_results/<subdir>/`. The slice subdir, graph_type,
# default uniform field h, and meta are auto-inferred from the model
# filename; pass --subdir / --graph-type / --h to override.
#
# Usage:
#   ./slice.sh --model=<path/to/file.model> --sc-target=<int> \
#              [--h=<float>] [--subdir=<name>] [--graph-type=<name>] \
#              [--no-lp] [--ntrials=<int>] [--niters=<int>] \
#              [--code-seeds=<lo:hi>] [--lp-time-limit=<sec>] [--quiet]
#   ./slice.sh --help
#
# Environment overrides:
#   JULIA_BIN     — julia executable (default: 'julia')
#   JULIA_PROJECT — Julia project path (default: <repo>/beyond_mis)
#   JULIA_THREADS — passed via -t (default: '32')

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS_DIR="$(dirname "$SCRIPT_DIR")"
BEYOND_MIS_DIR="$(dirname "$SCRIPTS_DIR")"
JL_DRIVER="$SCRIPTS_DIR/_lib/slice_spin_glass.jl"

JULIA_BIN="${JULIA_BIN:-julia}"
JULIA_PROJECT="${JULIA_PROJECT:-$BEYOND_MIS_DIR}"
JULIA_THREADS="${JULIA_THREADS:-32}"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<EOF
$(basename "$0") — slice a spin-glass instance into branches

Usage: $(basename "$0") --model=<path/to/file.model> --sc-target=<int> [...]

Forwards every flag verbatim to:
  $JL_DRIVER

Run with --help to see the full driver-level usage; the most common
flags are listed in the script header. Slice dumps land under:
  $BEYOND_MIS_DIR/branch_results/
EOF
  exec "$JULIA_BIN" --project="$JULIA_PROJECT" -t "$JULIA_THREADS" \
    "$JL_DRIVER" --help
fi

if [[ ! -f "$JL_DRIVER" ]]; then
  echo "ERROR: missing Julia driver: $JL_DRIVER" >&2
  exit 1
fi

echo "[slice.sh] julia    : $JULIA_BIN"
echo "[slice.sh] project  : $JULIA_PROJECT"
echo "[slice.sh] threads  : $JULIA_THREADS"
echo "[slice.sh] driver   : $JL_DRIVER"
echo "[slice.sh] forward  : $*"

LOG_TMP="$(mktemp -t slice-sh-XXXXXX.log)"
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

slice_dir="$(grep -E '^[[:space:]]*saved .+ to /' "$LOG_TMP" \
             | tail -n 1 | awk '{print $NF}')"

if [[ -n "$slice_dir" && -d "$slice_dir" ]]; then
  forwarded_q=$(printf '%q ' "$@")
  cat > "$slice_dir/slice_runtime.txt" <<EOF
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
  cp "$LOG_TMP" "$slice_dir/slice.log"
  echo "[slice.sh] runtime    : ${wall_seconds}s (wall) -> $slice_dir/slice_runtime.txt"
else
  echo "[slice.sh] WARN: could not locate slice dump dir; wall=${wall_seconds}s not persisted" >&2
fi

exit "$status"
