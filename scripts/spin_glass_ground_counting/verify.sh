#!/usr/bin/env bash
#
# verify.sh — bash wrapper around `_lib/verify_spin_glass.jl`.
#
# End-to-end correctness driver:
#   1) call slice_spin_glass.main(...) to slice and dump,
#   2) call contract_spin_glass_slices.main(...) to exactly contract,
#   3) build a fresh GenericTensorNetwork(SpinGlass(...)) and solve
#      CountingMax() to obtain the strict baseline,
#   4) assert (energy match within --atol) AND (degeneracy exact match).
#
# Usage:
#   ./verify.sh --model=<path/to/file.model> --sc-target=<int> [...]
#   ./verify.sh --help
#
# All slicer flags are forwarded as-is. Verifier-only flags include:
#   --atol=<float>            (energy comparison tolerance, default 1e-4)
#   --gpu=<id> | --no-cuda    (CUDA policy for both slice contraction
#                              and strict baseline)
#   --energy-scale, --count-eltype, --max-crt-iter
#                             (forwarded to contract step)
#   --keep-existing           (skip slice step; reuse existing dump;
#                              requires --subdir=<name>)
#   --quiet                   (verbose=0 throughout)
#
# Environment overrides: JULIA_BIN, JULIA_PROJECT, JULIA_THREADS.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS_DIR="$(dirname "$SCRIPT_DIR")"
BEYOND_MIS_DIR="$(dirname "$SCRIPTS_DIR")"
JL_DRIVER="$SCRIPTS_DIR/_lib/verify_spin_glass.jl"

JULIA_BIN="${JULIA_BIN:-julia}"
JULIA_PROJECT="${JULIA_PROJECT:-$BEYOND_MIS_DIR}"
JULIA_THREADS="${JULIA_THREADS:-32}"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<EOF
$(basename "$0") — verify the spin-glass slice → contract pipeline

Usage: $(basename "$0") --model=<path/to/file.model> --sc-target=<int> [...]

Forwards every flag verbatim to:
  $JL_DRIVER

Exits non-zero on (energy or degeneracy) mismatch.
EOF
  exec "$JULIA_BIN" --project="$JULIA_PROJECT" -t "$JULIA_THREADS" \
    "$JL_DRIVER" --help
fi

if [[ ! -f "$JL_DRIVER" ]]; then
  echo "ERROR: missing Julia driver: $JL_DRIVER" >&2
  exit 1
fi

echo "[verify.sh] julia    : $JULIA_BIN"
echo "[verify.sh] project  : $JULIA_PROJECT"
echo "[verify.sh] threads  : $JULIA_THREADS"
echo "[verify.sh] driver   : $JL_DRIVER"
echo "[verify.sh] forward  : $*"

exec "$JULIA_BIN" --project="$JULIA_PROJECT" -t "$JULIA_THREADS" \
  "$JL_DRIVER" "$@"
