"""
    test_spin_glass_slice_contract.jl

Read-only verifier for an existing slice dump.

Given the path / subdir of an instance under `beyond_mis/branch_results/`
(produced earlier by `save_spin_glass_slices`, e.g. by the three
benchmarks under `beyond_mis/hacked_funcs/benchmarks/`), this script:

  1. invokes `run_spin_glass_slice_contract.main([slice_dir, ...])` to
     reload the slices, contract each one through its **saved** code,
     and combine `(E_i + r_i, c_i)` into the slice-based ground-state
     `(E_g, deg)`;
  2. loads the **original** spin glass `(g, J, h)` saved alongside the
     slices (`original_graph.lg`, `original_J.txt`, `original_h.txt`)
     and runs `solve(..., CountingMax(); usecuda=...)` on the full
     network (same `usecuda` flag as the slice path when `--gpu=N` is
     set) to obtain the strict ground state;
  3. asserts that energy and degeneracy match.

It does *not* rerun any slicing or branching.

Usage:

    julia --project=beyond_mis beyond_mis/contractors/test_spin_glass_slice_contract.jl \\
        [<slice_dir>] [--gpu=N] [--no-cuda] [--atol=1e-4] [--quiet]

If `<slice_dir>` is omitted, it defaults to
`grid_Jpm1_n=20_d=3_seed=1` under `beyond_mis/branch_results/`.

`<slice_dir>` accepts the same forms as
`run_spin_glass_slice_contract.jl`:

  * subdir name under `beyond_mis/branch_results/` (e.g.
    `grid_Jpm1_n=20_d=3_seed=1`), or
  * absolute path / pre-existing directory.

If the dump does not contain the original `(g, J, h)` (see
`save_spin_glass_slices(...; original = (g, J, h))`), the script aborts
with an informative error.
"""

using Printf
using Graphs
using GenericTensorNetworks
using ProblemReductions: SpinGlass
using OMEinsumContractionOrders: TreeSA

include(joinpath(@__DIR__, "spin_glass_slice_contract.jl"))
include(joinpath(@__DIR__, "run_spin_glass_slice_contract.jl"))

# Default `branch_results/` subdir when no positional `<slice_dir>` is passed.
const DEFAULT_SLICE_CONTRACT_TEST_DIR = "grid_Jpm1_n=20_seed=1_cheating"

# ----------------------------------------------------------------------
# Argument parsing
# ----------------------------------------------------------------------

function _parse_test_args(args)
    slice_dir = ""
    gpu_id    = -1
    atol      = 1e-4
    verbose   = true
    for a in args
        if startswith(a, "--gpu=")
            gpu_id = parse(Int, split(a, "=", limit = 2)[2])
        elseif a == "--no-cuda" || a == "--cpu"
            gpu_id = -1
        elseif startswith(a, "--atol=")
            atol = parse(Float64, split(a, "=", limit = 2)[2])
        elseif a == "--quiet"
            verbose = false
        elseif !startswith(a, "--")
            slice_dir = String(a)
        end
    end
    if isempty(slice_dir)
        slice_dir = DEFAULT_SLICE_CONTRACT_TEST_DIR
    end
    return (; slice_dir, gpu_id, atol, verbose)
end

# ----------------------------------------------------------------------
# Strict baseline: full GenericTensorNetwork contraction on (g, J, h)
# ----------------------------------------------------------------------

# `CountingMax` with `usecuda=true` runs contractions on `CuArray` via
# GenericTensorNetworks → OMEinsum + CUDA.jl kernels (not a separate
# “CountingMax CUDA package”; the flag selects the CUDA array backend).

function _strict_counting_solve!(problem, usecuda::Bool)
    raw = solve(problem, CountingMax(); usecuda = usecuda)
    return Array(raw)[]
end

function strict_ground_state(g, J, h; optimizer = TreeSA(), usecuda::Bool = false)
    @info "[strict] building GenericTensorNetwork ($(nv(g)) vertices, $(ne(g)) edges) ..."
    problem = GenericTensorNetwork(SpinGlass(g, collect(J), collect(h));
                                   optimizer = optimizer)
    @info "[strict] solving CountingMax(; usecuda=$(usecuda)) ..."
    t = @elapsed begin
        result = if usecuda
            Base.invokelatest(_strict_counting_solve!, problem, true)
        else
            _strict_counting_solve!(problem, false)
        end
    end
    return (energy = Float64(result.n),
            count  = _to_bigint(result.c),
            runtime = t)
end

# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------

function run_test(args)
    cfg = _parse_test_args(args)

    resolved = slice_results_dir(cfg.slice_dir)
    isdir(resolved) || error("slice directory does not exist: $resolved")
    @info "[test] slice directory: $resolved"

    has_original_spin_glass(resolved; root = "") ||
        error("$(resolved) does not contain the pre-slicing problem " *
              "(`original_graph.lg`, `original_J.txt`, `original_h.txt`). " *
              "Re-dump with `save_spin_glass_slices(..., original = (g, J, h))`.")

    usecuda = (cfg.gpu_id >= 0) && setup_cuda!(cfg.gpu_id)

    # ------------------------------------------------------------------
    # 1) Slice + sum via the run script (CLI entry point)
    # ------------------------------------------------------------------
    run_args = String[resolved, "--atol=$(cfg.atol)"]
    cfg.gpu_id >= 0 && push!(run_args, "--gpu=$(cfg.gpu_id)")
    cfg.verbose || push!(run_args, "--quiet")
    @info "[contract] invoking run_spin_glass_slice_contract.main(...) with: $run_args"
    sliced = main(run_args)

    # ------------------------------------------------------------------
    # 2) Strict baseline on the saved original (g, J, h)
    # ------------------------------------------------------------------
    orig = load_original_spin_glass(resolved; root = "")
    strict = strict_ground_state(orig.g, orig.J, orig.h; usecuda = usecuda)

    # ------------------------------------------------------------------
    # 3) Compare
    # ------------------------------------------------------------------
    @printf("\n========== summary ==========\n")
    @printf("strict GTN     : energy = %s,   count = %s,   runtime = %.3fs\n",
            string(strict.energy), string(strict.count), strict.runtime)
    @printf("slice + sum    : energy = %s,   count = %s,   runtime = %.3fs\n",
            string(sliced.energy), string(sliced.count), sliced.total_runtime)
    @printf("slices used    : %d / %d\n",
            count(sliced.per_slice.used_in_total), nrow(sliced.per_slice))

    energy_ok = isapprox(sliced.energy, strict.energy; atol = cfg.atol)
    count_ok  = sliced.count == strict.count
    @printf("energy match   : %s\n", energy_ok ? "OK" : "MISMATCH")
    @printf("count  match   : %s\n", count_ok  ? "OK" : "MISMATCH")

    if !(energy_ok && count_ok)
        error("slice contraction disagrees with strict GTN: " *
              "Δenergy = $(sliced.energy - strict.energy), " *
              "strict count = $(strict.count), sliced count = $(sliced.count)")
    end
    @info "[test] PASSED"
    return (; strict, sliced, slice_dir = resolved)
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_test(ARGS)
end
