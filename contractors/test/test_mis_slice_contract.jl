"""
    test_mis_slice_contract.jl

Read-only verifier for an existing MIS / MWIS slice dump.

Given the path / subdir of an instance under `beyond_mis/branch_results/`
(produced earlier by `init_mis_slice_writer` / `save_mis_slices`, e.g.
by the two MIS benchmarks under `beyond_mis/hacked_funcs/benchmarks/`),
this script:

  1. invokes `run_mis_slice_contract.main([slice_dir, ...])` to reload
     the slices, contract each one through its **saved** code, and
     combine `(size_i + r_i, c_i)` into the slice-based maximum-IS
     `(S_g, deg)`;
  2. loads the **original** MIS problem `(g, weights)` saved alongside
     the slices (`original_graph.lg`, `original_weights*.txt`,
     `original_weights_kind.txt`) and runs
     `solve(..., CountingMax(); usecuda=...)` on the full network
     (same `usecuda` flag as the slice path when `--gpu=N` is set) to
     obtain the strict maximum-IS;
  3. asserts that size and degeneracy match.

It does *not* rerun any slicing or branching.

Usage:

    julia --project=beyond_mis beyond_mis/contractors/test/test_mis_slice_contract.jl \\
        [<slice_dir>] [--gpu=N] [--no-cuda] [--atol=1e-4] [--quiet]

If `<slice_dir>` is omitted, it defaults to the value of
`DEFAULT_MIS_SLICE_CONTRACT_TEST_DIR` below.

`<slice_dir>` accepts the same forms as `run_mis_slice_contract.jl`:

  * subdir name under `beyond_mis/branch_results/`, or
  * absolute path / pre-existing directory.

If the dump does not contain the original `(g, weights)` (see
`init_mis_slice_writer(...; original = (g, weights))`), the script
aborts with an informative error.
"""

using Printf
using Graphs
using GenericTensorNetworks
using ProblemReductions: IndependentSet
using OMEinsumContractionOrders: TreeSA

include(joinpath(@__DIR__, "..", "src", "run_mis_slice_contract.jl"))
# (`run_mis_slice_contract.jl` already pulls in `mis_slice_contract.jl`,
# which exports `slice_results_dir`, `list_mis_slices`, `load_mis_slice`,
# `load_original_mis`, `has_original_mis`, `_to_bigint`, etc.)

# Default `branch_results/` subdir when no positional `<slice_dir>` is passed.
# Pick something a fresh user is most likely to have generated; both
# `mis_counting.jl` and `mis_ground_counting.jl` use the prefix
# `mis_(ground_)counting_<basename>_seed=<seed>`.
const DEFAULT_MIS_SLICE_CONTRACT_TEST_DIR = "mis_ground_counting_random_ksg_n=20_seed=1"

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
        slice_dir = DEFAULT_MIS_SLICE_CONTRACT_TEST_DIR
    end
    return (; slice_dir, gpu_id, atol, verbose)
end

# ----------------------------------------------------------------------
# Strict baseline: full GenericTensorNetwork contraction on (g, weights)
# ----------------------------------------------------------------------

# `CountingMax` with `usecuda=true` runs contractions on `CuArray` via
# GenericTensorNetworks → OMEinsum + CUDA.jl kernels (not a separate
# “CountingMax CUDA package”; the flag selects the CUDA array backend).

function _strict_counting_solve!(problem, usecuda::Bool)
    raw = solve(problem, CountingMax(); usecuda = usecuda)
    return Array(raw)[]
end

function strict_max_independent_set(g, weights; optimizer = TreeSA(), usecuda::Bool = false)
    @info "[strict] building GenericTensorNetwork ($(nv(g)) vertices, $(ne(g)) edges) ..."
    problem = GenericTensorNetwork(IndependentSet(g, weights);
                                   optimizer = optimizer)
    @info "[strict] solving CountingMax(; usecuda=$(usecuda)) ..."
    t = @elapsed begin
        result = if usecuda
            Base.invokelatest(_strict_counting_solve!, problem, true)
        else
            _strict_counting_solve!(problem, false)
        end
    end
    return (size = Float64(result.n),
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

    has_original_mis(resolved; root = "") ||
        error("$(resolved) does not contain the pre-slicing problem " *
              "(`original_graph.lg`, `original_weights_kind.txt`, " *
              "and—when non-unit—`original_weights.txt`). Re-dump with " *
              "`init_mis_slice_writer(...; original = (g, weights))`.")

    usecuda = (cfg.gpu_id >= 0) && setup_cuda!(cfg.gpu_id)

    # ------------------------------------------------------------------
    # 1) Slice + sum via the run script (CLI entry point)
    # ------------------------------------------------------------------
    run_args = String[resolved, "--atol=$(cfg.atol)"]
    cfg.gpu_id >= 0 && push!(run_args, "--gpu=$(cfg.gpu_id)")
    cfg.verbose || push!(run_args, "--quiet")
    @info "[contract] invoking run_mis_slice_contract.main(...) with: $run_args"
    sliced = main(run_args)

    # ------------------------------------------------------------------
    # 2) Strict baseline on the saved original (g, weights)
    # ------------------------------------------------------------------
    orig = load_original_mis(resolved; root = "")
    strict = strict_max_independent_set(orig.g, orig.weights; usecuda = usecuda)

    # ------------------------------------------------------------------
    # 3) Compare
    # ------------------------------------------------------------------
    @printf("\n========== summary ==========\n")
    @printf("strict GTN     : size = %s,   count = %s,   runtime = %.3fs\n",
            string(strict.size), string(strict.count), strict.runtime)
    @printf("slice + sum    : size = %s,   count = %s,   runtime = %.3fs\n",
            string(sliced.size), string(sliced.count), sliced.total_runtime)
    @printf("slices used    : %d / %d\n",
            count(sliced.per_slice.used_in_total), nrow(sliced.per_slice))

    size_ok  = isapprox(sliced.size, strict.size; atol = cfg.atol)
    count_ok = sliced.count == strict.count
    @printf("size  match    : %s\n", size_ok  ? "OK" : "MISMATCH")
    @printf("count match    : %s\n", count_ok ? "OK" : "MISMATCH")

    if !(size_ok && count_ok)
        error("slice contraction disagrees with strict GTN: " *
              "Δsize = $(sliced.size - strict.size), " *
              "strict count = $(strict.count), sliced count = $(sliced.count)")
    end
    @info "[test] PASSED"
    return (; strict, sliced, slice_dir = resolved)
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_test(ARGS)
end
