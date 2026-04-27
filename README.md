# Hacked BBTN for General Applications

BBTN slicing + exact contraction for three problems:

- `mis_counting` — total IS count
- `mis_ground_counting` — maximum (W)IS size + degeneracy
- `spin_glass_ground_counting` — ground-state energy + degeneracy

Workflow: run `slice.sh` to dump slices into `branch_results/`, then run
`contract.sh` to contract them and write CSVs into `results/`.

## Usage

Each flavour lives under `scripts/<flavour>/`:

```
scripts/mis_counting/{slice.sh, contract.sh, verify.sh}
scripts/mis_ground_counting/{slice.sh, contract.sh, verify.sh}
scripts/spin_glass_ground_counting/{slice.sh, contract.sh, verify.sh}
```

Run any script with `--help` for the full flag list. Env overrides:
`JULIA_BIN` (default `julia`), `JULIA_PROJECT` (default `<repo>/beyond_mis`),
`JULIA_THREADS` (default `32`).

### 1. Slice

```bash
scripts/<flavour>/slice.sh --model=<path> --sc-target=<int> [flags]

# e.g.
scripts/spin_glass_ground_counting/slice.sh \
    --model=hacked_funcs/benchmarks/models/spin_glass_models/spin_glass_J±1_grid_n=20_seed=1.model \
    --sc-target=15
```

Output: `beyond_mis/branch_results/<subdir>/` (subdir auto-inferred from the
input filename; override with `--subdir=<name>`). Also written:
`slice_runtime.txt` and `slice.log`.

Common flags (defaults in parentheses):

- `--sc-target=<int>` *(required)* — target space complexity per slice
- `--code-seed=<int>` *(= seed)* — TreeSA seed
- `--ntrials=<int>` *(50)*, `--niters=<int>` *(100)* — TreeSA budget
- `--quiet` — `verbose=0` (default `1`)

Flavour-specific:

- `mis_counting` / `mis_ground_counting`: legacy random-KSG mode
  `--n=<int> [--density=<float>] [--seed=<int>]` *(density 0.8, seed 1)*
- `mis_ground_counting`: `--weights=unit|random` *(unit)*,
  `--weights-seed=<int>`, `--no-lp`, `--use-cuda`, `--gpu=<id>`
- `spin_glass_ground_counting`: `--h=<float>`, `--no-lp`,
  `--code-seeds=<lo:hi>` *(`1:2`)*, `--lp-time-limit=<sec>` *(300)*

### 2. Contract

```bash
scripts/<flavour>/contract.sh <slice_dir> [flags]
```

`<slice_dir>` is either the subdir name under `branch_results/` or an
absolute path.

Output:

```
beyond_mis/results/<flavour>_slice_contract/<results-name>/per_slice.csv
beyond_mis/results/<flavour>_slice_contract/<results-name>/summary.csv
```

(`<flavour>` ∈ {`mis_counting`, `mis_slice`, `spin_glass_slice`}.) A
cross-instance row is also appended to `<flavour>_slice_contract/summary.csv`.

Common flags (defaults in parentheses):

- `--gpu=<N>` / `--no-cuda` — pick a CUDA device or force CPU
- `--count-eltype=finitefield|Float64|Int128|BigInt` *(`finitefield`)*
  (BigInt forces CPU)
- `--max-crt-iter=<int>` *(8)*
- `--results-root=<path>` *(`beyond_mis/results`)*
- `--results-name=<name>` *(`basename(<slice_dir>)`)*
- `--quiet`

Flavour-specific:

- `mis_ground_counting`: `--atol=<float>` *(1e-6)*, `--scale=<int>` *(1)*
- `spin_glass_ground_counting`: `--atol=<float>` *(1e-6)*,
  `--energy-scale=<int>` *(2)*

### Example

```bash
scripts/mis_counting/slice.sh \
    --model=path/to/random_ksg_n=20_seed=1.graph --sc-target=20 --quiet
# -> branch_results/mis_counting_random_ksg_n=20_seed=1/

scripts/mis_counting/contract.sh \
    mis_counting_random_ksg_n=20_seed=1 --no-cuda --quiet
# -> results/mis_counting_slice_contract/mis_counting_random_ksg_n=20_seed=1/
```

`verify.sh` runs slice + contract and compares against a strict full-graph
solve (small instances only).

## Caveats

### MWIS / spin-glass weights must be representable as integers

The contractors expect integer weights / couplings even when
`--count-eltype=Float64`. For non-integer floats, multiply by an integer
factor first (`--scale=<int>` for MIS, `--energy-scale=<int>` for spin glass)
so that `round.(Int, w * scale)` is lossless (max rounding error `< 1e-10`),
then divide the result back. Pure 0/1 / integer-valued instances need no
scaling. Otherwise you get:

```
MIS weights cannot be represented as Int after scaling by 1
(max rounding error = ...); pick a larger weight_scale.
```

### Positive-energy / non-negative-weight assumption

The branch-and-bound code initialises the primal bound to `0.0`. This is
safe **only** when the optimum is non-negative. Negative-optimum instances
can be incorrectly pruned away.

### GPU device selection

`hacked_funcs/src/` no longer hard-codes `CUDA.device!(...)`. Pass
`--gpu=N` to the scripts (or call `CUDA.device!(N)` yourself before invoking
the library entry points) to pick the device.
