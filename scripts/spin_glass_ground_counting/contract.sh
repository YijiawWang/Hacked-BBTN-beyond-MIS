#!/usr/bin/env bash
#
# contract.sh — bash wrapper around `_lib/contract_spin_glass_slices.jl`.
#
# Loads a spin-glass slice dump (produced by slice.sh or by any of the
# three benchmarks under `beyond_mis/hacked_funcs/benchmarks/`) and
# exactly re-contracts every slice through its **saved** OMEinsum code,
# combining the per-slice `(E_i + r_i, c_i)` into the global ground
# state `(E_g, deg)`. Per-slice + summary CSVs are written to
# `beyond_mis/results/spin_glass_slice_contract/<results-name>/`.
#
# Usage:
#   ./contract.sh <slice_dir> [--gpu=N|--no-cuda] [--atol=1e-6] [--quiet] \
#                 [--energy-scale=<int>] [--count-eltype=<sym>] \
#                 [--max-crt-iter=<int>] \
#                 [--results-root=<path>] [--results-name=<name>]
#   ./contract.sh --help
#
# `<slice_dir>` may be either:
#   * the subdir name under `beyond_mis/branch_results/` (e.g.
#     `grid_Jpm1_n=20_seed=1_cheating`), or
#   * an absolute path / pre-existing directory.
#
# Environment overrides: JULIA_BIN, JULIA_PROJECT, JULIA_THREADS.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS_DIR="$(dirname "$SCRIPT_DIR")"
BEYOND_MIS_DIR="$(dirname "$SCRIPTS_DIR")"
JL_DRIVER="$SCRIPTS_DIR/_lib/contract_spin_glass_slices.jl"

JULIA_BIN="${JULIA_BIN:-julia}"
JULIA_PROJECT="${JULIA_PROJECT:-$BEYOND_MIS_DIR}"
JULIA_THREADS="${JULIA_THREADS:-32}"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<EOF
$(basename "$0") — exactly contract a spin-glass slice dump

Usage: $(basename "$0") <slice_dir> [flags]

Forwards every flag verbatim to:
  $JL_DRIVER

Results land under:
  $BEYOND_MIS_DIR/results/spin_glass_slice_contract/<results-name>/
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
