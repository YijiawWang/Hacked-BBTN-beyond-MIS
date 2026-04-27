"""
    test_mis_counting_slice_contract.jl

Read-only verifier for an existing **MIS total-counting** slice dump
(slices produced by `mis_counting.jl`'s `slice_bfs`, which uses
`optimal_branches_counting` to keep the per-slice IS sets disjoint).

Given the path / subdir of an instance under `beyond_mis/branch_results/`,
this script:

  1. invokes `run_mis_counting_slice_contract.main([slice_dir, ...])`
     to reload the slices, contract each one with `CountingAll()`
     through its **saved** code, and **sum** the per-slice counts into
     the slice-based total IS count `Σ_i count_i`;
  2. loads the **original** MIS problem `(g, weights)` saved alongside
     the slices and runs `solve(..., CountingAll(); usecuda=...)` on
     the full network (same `usecuda` flag as the slice path when
     `--gpu=N` is set) to obtain the strict total IS count;
  3. asserts that the two counts are *exactly* equal (no `atol`
     because `CountingAll` returns an integer count).

Usage:

    julia --project=beyond_mis beyond_mis/contractors/test/test_mis_counting_slice_contract.jl \\
        [<slice_dir>] [--gpu=N] [--no-cuda] [--quiet]

If `<slice_dir>` is omitted, it defaults to
`DEFAULT_MIS_COUNTING_SLICE_CONTRACT_TEST_DIR` below.

If the dump does not contain the original `(g, weights)` (see
`init_mis_slice_writer(...; original = (g, weights))`), the script
aborts with an informative error.
"""

using Printf
using Graphs
using GenericTensorNetworks
using ProblemReductions: IndependentSet
using OMEinsumContractionOrders: TreeSA

include(joinpath(@__DIR__, "..", "src", "run_mis_counting_slice_contract.jl"))
# (`run_mis_counting_slice_contract.jl` already pulls in
# `mis_slice_contract.jl`, which exports `slice_results_dir`,
# `list_mis_slices`, `load_mis_slice`, `load_original_mis`,
# `has_original_mis`, `_to_bigint`, etc.)

# Default `branch_results/` subdir when no positional `<slice_dir>` is passed.
# `mis_counting.jl` writes to `mis_counting_<basename>_seed=<seed>`.
const DEFAULT_MIS_COUNTING_SLICE_CONTRACT_TEST_DIR = "mis_counting_random_ksg_n=20_seed=1"

# ----------------------------------------------------------------------
# Argument parsing
# ----------------------------------------------------------------------

function _parse_test_args(args)
    slice_dir = ""
    gpu_id    = -1
    verbose   = true
    for a in args
        if startswith(a, "--gpu=")
            gpu_id = parse(Int, split(a, "=", limit = 2)[2])
        elseif a == "--no-cuda" || a == "--cpu"
            gpu_id = -1
        elseif a == "--quiet"
            verbose = false
        elseif !startswith(a, "--")
            slice_dir = String(a)
        end
    end
    if isempty(slice_dir)
        slice_dir = DEFAULT_MIS_COUNTING_SLICE_CONTRACT_TEST_DIR
    end
    return (; slice_dir, gpu_id, verbose)
end

# ----------------------------------------------------------------------
# Strict baseline: full GenericTensorNetwork CountingAll on (g, weights)
# ----------------------------------------------------------------------

function _strict_counting_all_solve!(problem, usecuda::Bool)
    raw = solve(problem, CountingAll(); usecuda = usecuda)
    return Array(raw)[]
end

function strict_total_is_count(g, weights; optimizer = TreeSA(), usecuda::Bool = false)
    @info "[strict] building GenericTensorNetwork ($(nv(g)) vertices, $(ne(g)) edges) ..."
    problem = GenericTensorNetwork(IndependentSet(g, weights);
                                   optimizer = optimizer)
    @info "[strict] solving CountingAll(; usecuda=$(usecuda)) ..."
    t = @elapsed begin
        result = if usecuda
            Base.invokelatest(_strict_counting_all_solve!, problem, true)
        else
            _strict_counting_all_solve!(problem, false)
        end
    end
    return (count = _to_bigint(result),
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
    run_args = String[resolved]
    cfg.gpu_id >= 0 && push!(run_args, "--gpu=$(cfg.gpu_id)")
    cfg.verbose || push!(run_args, "--quiet")
    @info "[contract] invoking run_mis_counting_slice_contract.main(...) with: $run_args"
    sliced = main(run_args)

    # ------------------------------------------------------------------
    # 2) Strict baseline on the saved original (g, weights)
    # ------------------------------------------------------------------
    orig = load_original_mis(resolved; root = "")
    strict = strict_total_is_count(orig.g, orig.weights; usecuda = usecuda)

    # ------------------------------------------------------------------
    # 3) Compare
    # ------------------------------------------------------------------
    @printf("\n========== summary ==========\n")
    @printf("strict GTN     : count = %s,   runtime = %.3fs\n",
            string(strict.count), strict.runtime)
    @printf("slice + sum    : count = %s,   runtime = %.3fs\n",
            string(sliced.count), sliced.total_runtime)
    @printf("slices summed  : %d\n", nrow(sliced.per_slice))

    count_ok = sliced.count == strict.count
    @printf("count match    : %s\n", count_ok ? "OK" : "MISMATCH")

    if !count_ok
        error("slice contraction disagrees with strict GTN: " *
              "strict count = $(strict.count), sliced count = $(sliced.count)")
    end
    @info "[test] PASSED"
    return (; strict, sliced, slice_dir = resolved)
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_test(ARGS)
end
