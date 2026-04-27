"""
    verify_mis.jl

End-to-end correctness check for the MIS ground-counting slicing →
exact-contraction pipeline implemented by the other two MIS drivers in
this folder:

  1. drive `_lib/slice_mis.jl::main` to **build & slice** a random KSG
     instance and dump every slice to
     `beyond_mis/branch_results/<subdir>/`;
  2. drive `_lib/contract_mis_slices.jl::main` to load that dump back,
     contract every slice through its **saved** OMEinsum code, and
     combine the per-slice results into the slice-based maximum-IS
     `(S_g_slice, deg_slice)`;
  3. independently re-build the same `(g, weights)` (also reloaded from
     `original_*` files in the dump for a sanity check), wrap it in a
     plain `GenericTensorNetwork(IndependentSet(g, w))` and solve
     `CountingMax()` on it to obtain the **strict** baseline
     `(S_g_strict, deg_strict)`;
  4. assert `S_g_slice ≈ S_g_strict` (within `--atol`) and
     `deg_slice == deg_strict`. Exits non-zero on mismatch.

Usage mirrors `_lib/slice_mis.jl` plus a few verifier-only flags:

    julia --project=beyond_mis beyond_mis/scripts/_lib/verify_mis.jl \\
        --n=<int> --sc-target=<int> \\
        [--density=<float>] [--seed=<int>] [--code-seed=<int>] \\
        [--weights=unit|random] [--weights-seed=<int>] \\
        [--subdir=<name>] [--no-lp] \\
        [--ntrials=<int>] [--niters=<int>] \\
        [--gpu=<id>] [--no-cuda] \\
        [--use-cuda]                       # forwarded to slicer (LP-bound CUDA)
        [--atol=<float>]                   # default 1e-4
        [--count-eltype=<symbol>]          # forwarded to contractor (default :finitefield)
        [--scale=<int>]                    # forwarded to contractor (default 1)
        [--max-crt-iter=<int>]             # forwarded to contractor (default 8)
        [--results-root=<path>] [--results-name=<name>] \\
        [--keep-existing]                  # do NOT re-slice; reuse existing dump
        [--quiet]
"""

using Printf
using Graphs
using GenericTensorNetworks
using ProblemReductions: IndependentSet
using OMEinsumContractionOrders: TreeSA
using DataFrames: nrow

module _Slicer
end
Base.include(_Slicer, joinpath(@__DIR__, "slice_mis.jl"))

module _Contractor
end
Base.include(_Contractor, joinpath(@__DIR__, "contract_mis_slices.jl"))

const _SETUP_CUDA = _Contractor._setup_cuda!

# ---------------------------------------------------------------------------
# Argument parsing (verifier-side)
# ---------------------------------------------------------------------------

struct VerifyConfig
    slicer_args::Vector{String}
    extra_contractor_args::Vector{String}
    gpu_id::Int
    atol::Float64
    keep_existing::Bool
    verbose::Bool
    results_root::String
    results_name::String
    subdir_hint::String
end

function _parse_args(args)
    slicer_args           = String[]
    extra_contractor_args = String[]
    gpu_id                = -1
    atol                  = 1e-4
    keep_existing         = false
    verbose               = true
    results_root          = ""
    results_name          = ""
    subdir_hint           = ""

    for a in args
        if startswith(a, "--gpu=")
            gpu_id = parse(Int, split(a, "="; limit = 2)[2])
        elseif a == "--no-cuda" || a == "--cpu"
            gpu_id = -1
        elseif startswith(a, "--atol=")
            atol = parse(Float64, split(a, "="; limit = 2)[2])
        elseif a == "--keep-existing"
            keep_existing = true
        elseif a == "--quiet"
            verbose = false
            push!(slicer_args, "--quiet")
        elseif startswith(a, "--results-root=")
            results_root = String(split(a, "="; limit = 2)[2])
        elseif startswith(a, "--results-name=")
            results_name = String(split(a, "="; limit = 2)[2])
        elseif startswith(a, "--subdir=")
            subdir_hint = String(split(a, "="; limit = 2)[2])
            push!(slicer_args, a)
        elseif startswith(a, "--count-eltype=") ||
               startswith(a, "--scale=") ||
               startswith(a, "--weight-scale=") ||
               startswith(a, "--max-crt-iter=")
            push!(extra_contractor_args, a)
        elseif a in ("-h", "--help")
            println(stderr, replace(string(@doc Main), "    " => "  "))
            exit(0)
        else
            push!(slicer_args, a)
        end
    end

    return VerifyConfig(slicer_args, extra_contractor_args, gpu_id, atol,
                        keep_existing, verbose, results_root, results_name,
                        subdir_hint)
end

# ---------------------------------------------------------------------------
# Strict baseline: full GenericTensorNetwork(IndependentSet) CountingMax
# ---------------------------------------------------------------------------

function _strict_solve!(problem, usecuda::Bool)
    raw = solve(problem, CountingMax(); usecuda = usecuda)
    return Array(raw)[]
end

"""
    strict_max_is(g, weights; usecuda=false, optimizer=TreeSA())

Build a fresh `GenericTensorNetwork(IndependentSet(g, weights))` (with a
freshly-optimized contraction order, *not* the slicer's saved code),
solve it under `CountingMax()`, and return `(size, count, runtime)`.
"""
function strict_max_is(g::SimpleGraph, weights;
                        usecuda::Bool = false, optimizer = TreeSA())
    @info "[strict] building GenericTensorNetwork (nv=$(nv(g)), ne=$(ne(g)))"
    problem = GenericTensorNetwork(IndependentSet(g, weights);
                                   optimizer = optimizer)
    @info "[strict] solving CountingMax(; usecuda=$(usecuda)) ..."
    t = @elapsed begin
        result = if usecuda
            Base.invokelatest(_strict_solve!, problem, true)
        else
            _strict_solve!(problem, false)
        end
    end
    return (size = Float64(result.n),
            count = _Contractor._to_bigint(result.c),
            runtime = t)
end

# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

function _step1_slice(cfg::VerifyConfig)
    println("\n", "="^72)
    println("[verify_mis] STEP 1 — slice  (slice_mis.main(...))")
    println("="^72)
    println("        args = ", cfg.slicer_args)
    return _Slicer.main(cfg.slicer_args)
end

function _step2_contract(cfg::VerifyConfig, slice_dir::AbstractString)
    println("\n", "="^72)
    println("[verify_mis] STEP 2 — contract  (contract_mis_slices.main(...))")
    println("="^72)
    args = String[slice_dir, "--atol=$(cfg.atol)"]
    cfg.gpu_id ≥ 0 && push!(args, "--gpu=$(cfg.gpu_id)")
    cfg.verbose || push!(args, "--quiet")
    isempty(cfg.results_root) || push!(args, "--results-root=$(cfg.results_root)")
    isempty(cfg.results_name) || push!(args, "--results-name=$(cfg.results_name)")
    append!(args, cfg.extra_contractor_args)
    println("        args = ", args)
    return _Contractor.main(args)
end

function _step3_strict(slice_dir::AbstractString, usecuda::Bool)
    println("\n", "="^72)
    println("[verify_mis] STEP 3 — strict GTN baseline")
    println("="^72)
    _Contractor.has_original_mis(slice_dir; root = "") ||
        error("$(slice_dir) does not contain the pre-slicing problem " *
              "(original_graph.lg + original_weights_kind.txt). " *
              "The slicer should have written these; rerun without " *
              "--keep-existing.")
    orig = _Contractor.load_original_mis(slice_dir; root = "")
    return strict_max_is(orig.g, orig.weights; usecuda = usecuda)
end

function _resolve_existing_subdir(cfg::VerifyConfig)
    isempty(cfg.subdir_hint) &&
        error("--keep-existing requires --subdir=<name> so we know which " *
              "dump to verify against")
    return _Contractor.slice_results_dir(cfg.subdir_hint)
end

function main(args)
    cfg = _parse_args(args)

    slice_info = if cfg.keep_existing
        slice_dir = _resolve_existing_subdir(cfg)
        @info "[verify_mis] reusing existing slice dump at $(slice_dir) (--keep-existing)"
        (slice_dir = slice_dir, slice_count = -1)
    else
        _step1_slice(cfg)
    end

    sliced = _step2_contract(cfg, slice_info.slice_dir)

    usecuda = (cfg.gpu_id ≥ 0) && _SETUP_CUDA(cfg.gpu_id)
    strict  = _step3_strict(slice_info.slice_dir, usecuda)

    println("\n", "="^72)
    println("[verify_mis] FINAL COMPARISON")
    println("="^72)
    @printf("strict GTN     : size = %s,   count = %s,   runtime = %.3fs\n",
            string(strict.size), string(strict.count), strict.runtime)
    @printf("slice + sum    : size = %s,   count = %s,   runtime = %.3fs\n",
            string(sliced.size), string(sliced.count), sliced.total_runtime)
    @printf("slices used    : %d / %d\n",
            count(sliced.per_slice.used_in_total), nrow(sliced.per_slice))

    size_match  = isapprox(sliced.size, strict.size; atol = cfg.atol)
    count_match = sliced.count == strict.count
    @printf("size  match    : %s\n", size_match  ? "OK" : "MISMATCH")
    @printf("count match    : %s\n", count_match ? "OK" : "MISMATCH")

    if !(size_match && count_match)
        error("verification FAILED: " *
              "Δsize = $(sliced.size - strict.size), " *
              "strict count = $(strict.count), sliced count = $(sliced.count)")
    end

    @info "[verify_mis] PASSED"

    return (
        slice_dir   = slice_info.slice_dir,
        sliced      = sliced,
        strict      = strict,
        size_match  = size_match,
        count_match = count_match,
        atol        = cfg.atol,
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
