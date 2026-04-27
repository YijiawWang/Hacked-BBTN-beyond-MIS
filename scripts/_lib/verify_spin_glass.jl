"""
    verify_spin_glass.jl

End-to-end correctness check for the slicing → exact-contraction
pipeline implemented by the other two scripts in this folder:

  1. drive `scripts/slice_spin_glass.jl::main`  to **build & slice** a
     spin glass instance and dump every slice to
     `beyond_mis/branch_results/<subdir>/`;
  2. drive `scripts/contract_spin_glass_slices.jl::main`  to load that
     dump back, contract every slice through its **saved** OMEinsum
     code and combine the per-slice `(E_i + r_i, c_i)` into the
     slice-based ground-state `(E_g_slice, deg_slice)`;
  3. independently re-build the same `(g, J, h)` (also reloaded from
     `original_*` files in the dump for a sanity check), wrap it in a
     plain `GenericTensorNetwork(SpinGlass(g, J, h))` and solve
     `CountingMax()` on it to obtain the **strict** ground state
     `(E_g_strict, deg_strict)`;
  4. assert `E_g_slice ≈ E_g_strict` (within `--atol`) and
     `deg_slice == deg_strict`. Exits non-zero on mismatch.

Usage mirrors `scripts/_lib/slice_spin_glass.jl` plus a few
verifier-only flags:

    julia --project=beyond_mis beyond_mis/scripts/_lib/verify_spin_glass.jl \\
        --model=<path/to/file.model> \\
        --sc-target=<int> \\
        [--h=<float>] [--subdir=<name>] [--graph-type=<name>] [--no-lp] \\
        [--ntrials=<int>] [--niters=<int>] [--code-seeds=<lo:hi>] \\
        [--lp-time-limit=<sec>] \\
        [--gpu=<id>] [--no-cuda] \\
        [--atol=<float>]                  # default 1e-4
        [--energy-scale=<int>]            # forwarded to script 2 (default 2)
        [--count-eltype=<symbol>]         # forwarded to script 2 (default :finitefield)
        [--max-crt-iter=<int>]            # forwarded to script 2 (default 8)
        [--results-root=<path>]           # forwarded to script 2
        [--results-name=<name>]           # forwarded to script 2
        [--keep-existing]                 # do NOT re-slice; reuse existing dump
        [--quiet]

The "calls the first two scripts" requirement is satisfied by
`include`-ing them into private modules (so their `main`/helper names
do not collide) and invoking each module's `main(args::Vector{String})`
in turn. The same Julia session is reused for all three steps, so all
the heavy packages (`OptimalBranching`, `TensorBranching`,
`GenericTensorNetworks`, `OMEinsum`, …) are precompiled / loaded once.
"""

using Printf
using Graphs
using GenericTensorNetworks
using ProblemReductions: SpinGlass
using OMEinsumContractionOrders: TreeSA
using DataFrames: nrow

# ---------------------------------------------------------------------------
# Load the two sibling scripts into private modules so their `main`s and
# helper-name spaces don't collide in a single Julia session.
# ---------------------------------------------------------------------------

module _Slicer
end
Base.include(_Slicer, joinpath(@__DIR__, "slice_spin_glass.jl"))

module _Contractor
end
Base.include(_Contractor, joinpath(@__DIR__, "contract_spin_glass_slices.jl"))

# Reach into `_Contractor` to reuse its CUDA helper for the strict
# baseline, so the strict and sliced contractions share the same
# `usecuda` policy when `--gpu=N` is passed.
const _SETUP_CUDA = _Contractor._setup_cuda!

# ---------------------------------------------------------------------------
# Argument parsing (verifier-side)
# ---------------------------------------------------------------------------

struct VerifyConfig
    # forwarded to script 1
    slicer_args::Vector{String}
    # extra args forwarded to script 2 (in addition to the ones derived
    # from gpu_id / atol / results_*).
    extra_contractor_args::Vector{String}
    # verifier-only
    gpu_id::Int
    atol::Float64
    keep_existing::Bool
    verbose::Bool
    results_root::String
    results_name::String
    subdir_hint::String      # for --keep-existing reuse
end

# Args that the verify script needs to interpret itself (and possibly
# strip / forward). Everything else is forwarded verbatim to the
# slicer.
const _VERIFY_ONLY_FLAGS_PREFIX = (
    "--gpu=", "--atol=", "--results-root=", "--results-name=",
)
const _VERIFY_ONLY_FLAGS = (
    "--no-cuda", "--cpu", "--keep-existing", "--quiet",
)

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
        elseif startswith(a, "--energy-scale=") ||
               startswith(a, "--count-eltype=") ||
               startswith(a, "--max-crt-iter=")
            # contractor-only flags: forward verbatim
            push!(extra_contractor_args, a)
        elseif a in ("-h", "--help")
            println(stderr, replace(string(@doc Main), "    " => "  "))
            exit(0)
        else
            # Forward to the slicer. Anything it doesn't recognise will
            # blow up over there, which is what we want.
            push!(slicer_args, a)
        end
    end

    return VerifyConfig(slicer_args, extra_contractor_args, gpu_id, atol,
                        keep_existing, verbose, results_root, results_name,
                        subdir_hint)
end

# ---------------------------------------------------------------------------
# Strict baseline (full GenericTensorNetwork CountingMax, no slicing)
# ---------------------------------------------------------------------------

function _strict_solve!(problem, usecuda::Bool)
    raw = solve(problem, CountingMax(); usecuda = usecuda)
    return Array(raw)[]
end

"""
    strict_ground_state(g, J, h; usecuda=false, optimizer=TreeSA())

Build a fresh `GenericTensorNetwork(SpinGlass(g, J, h))` (with a
freshly-optimized contraction order, *not* the slicer's saved code),
solve it under `CountingMax()`, and return `(energy, count, runtime)`.
This is the "exact contraction function" the verifier compares against.
"""
function strict_ground_state(g::SimpleGraph, J::AbstractVector, h::AbstractVector;
                              usecuda::Bool = false, optimizer = TreeSA())
    @info "[strict] building GenericTensorNetwork (nv=$(nv(g)), ne=$(ne(g)))"
    problem = GenericTensorNetwork(SpinGlass(g, collect(J), collect(h));
                                   optimizer = optimizer)
    @info "[strict] solving CountingMax(; usecuda=$(usecuda)) ..."
    t = @elapsed begin
        result = if usecuda
            Base.invokelatest(_strict_solve!, problem, true)
        else
            _strict_solve!(problem, false)
        end
    end
    return (energy = Float64(result.n),
            count  = _Contractor._to_bigint(result.c),
            runtime = t)
end

# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

function _step1_slice(cfg::VerifyConfig)
    println("\n", "="^72)
    println("[verify] STEP 1 — slice  (slice_spin_glass.main(...))")
    println("="^72)
    println("        args = ", cfg.slicer_args)
    return _Slicer.main(cfg.slicer_args)
end

function _step2_contract(cfg::VerifyConfig, slice_dir::AbstractString)
    println("\n", "="^72)
    println("[verify] STEP 2 — contract  (contract_spin_glass_slices.main(...))")
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
    println("[verify] STEP 3 — strict GTN baseline")
    println("="^72)
    _Contractor.has_original_spin_glass(slice_dir; root = "") ||
        error("$(slice_dir) does not contain the pre-slicing problem " *
              "(original_graph.lg, original_J.txt, original_h.txt). " *
              "The slicer should have written these; rerun without " *
              "--keep-existing.")
    orig = _Contractor.load_original_spin_glass(slice_dir; root = "")
    return strict_ground_state(orig.g, orig.J, orig.h; usecuda = usecuda)
end

function _resolve_existing_subdir(cfg::VerifyConfig)
    isempty(cfg.subdir_hint) &&
        error("--keep-existing requires --subdir=<name> so we know which " *
              "dump to verify against")
    return _Contractor.slice_results_dir(cfg.subdir_hint)
end

"""
    main(args::AbstractVector{<:AbstractString}) -> NamedTuple

Run all three steps of the end-to-end check on a fresh spin glass
instance and return:

    (slice_dir, sliced, strict, energy_match, count_match, atol)

Errors out on a mismatch (so the script's exit code reflects the test).
"""
function main(args)
    cfg = _parse_args(args)

    slice_info = if cfg.keep_existing
        slice_dir = _resolve_existing_subdir(cfg)
        @info "[verify] reusing existing slice dump at $(slice_dir) (--keep-existing)"
        (slice_dir = slice_dir, slice_count = -1)
    else
        _step1_slice(cfg)
    end

    sliced = _step2_contract(cfg, slice_info.slice_dir)

    # Strict baseline. We reuse the same `usecuda` policy as the slice
    # contraction, but we have to call `_setup_cuda!` again here in case
    # this is the keep-existing path (which never goes through script 2's
    # `main`, but does need GPU set up to honor `--gpu=N`).
    usecuda = (cfg.gpu_id ≥ 0) && _SETUP_CUDA(cfg.gpu_id)
    strict  = _step3_strict(slice_info.slice_dir, usecuda)

    # ------------------------------------------------------------------
    # Compare
    # ------------------------------------------------------------------
    println("\n", "="^72)
    println("[verify] FINAL COMPARISON")
    println("="^72)
    @printf("strict GTN     : energy = %s,   count = %s,   runtime = %.3fs\n",
            string(strict.energy), string(strict.count), strict.runtime)
    @printf("slice + sum    : energy = %s,   count = %s,   runtime = %.3fs\n",
            string(sliced.energy), string(sliced.count), sliced.total_runtime)
    @printf("slices used    : %d / %d\n",
            count(sliced.per_slice.used_in_total), nrow(sliced.per_slice))

    energy_match = isapprox(sliced.energy, strict.energy; atol = cfg.atol)
    count_match  = sliced.count == strict.count
    @printf("energy match   : %s\n", energy_match ? "OK" : "MISMATCH")
    @printf("count  match   : %s\n", count_match  ? "OK" : "MISMATCH")

    if !(energy_match && count_match)
        error("verification FAILED: " *
              "Δenergy = $(sliced.energy - strict.energy), " *
              "strict count = $(strict.count), sliced count = $(sliced.count)")
    end

    @info "[verify] PASSED"

    return (
        slice_dir    = slice_info.slice_dir,
        sliced       = sliced,
        strict       = strict,
        energy_match = energy_match,
        count_match  = count_match,
        atol         = cfg.atol,
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
