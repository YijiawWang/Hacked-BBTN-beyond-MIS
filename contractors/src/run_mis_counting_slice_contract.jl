"""
    run_mis_counting_slice_contract.jl

Driver script that loads a directory of saved MIS / MWIS slices (as
produced by `init_mis_slice_writer` / `save_mis_slices` from
`mis_slice_contract.jl`) and exactly contracts each one through
`GenericTensorNetwork(IndependentSet(g, w), code, Dict{Int,Int}())`
with the **`CountingAll()`** property — i.e. **total** independent-set
counting (every IS of every size), as opposed to the
ground-state-only `CountingMax` flavour handled by
`run_mis_slice_contract.jl`.

This is the contractor side of the slice dumps produced by
`beyond_mis/hacked_funcs/benchmarks/mis_counting.jl`. The combination
rule is a simple sum across slices: each slice carries a disjoint
sub-tree of the original IS enumeration, so the total IS count of the
original graph is `Σ_i count_i`. `r_i` is irrelevant here.

Usage:

    julia --project=beyond_mis beyond_mis/contractors/src/run_mis_counting_slice_contract.jl \\
        <slice_dir> [--gpu=N] [--no-cuda] [--csv=out.csv] \\
        [--count-eltype=finitefield|Float64|Int128|BigInt] \\
        [--max-crt-iter=8] [--quiet]

`<slice_dir>` may be either:
  * the *subdir name* under `beyond_mis/branch_results/` (e.g.
    `mis_counting_random_ksg_n=600_seed=1`), or
  * an absolute path / pre-existing directory.

The default `--count-eltype=finitefield` is the GPU-safe exact mode:
each slice is contracted with eltype `Mods.Mod{p, Int32}` (just `Mod`,
no `CountingTropical`, since `CountingAll` evaluates the IS polynomial
at `x = 1`) for several primes near `typemax(Int32)`, and the per-prime
counts are CRT-combined into an exact `BigInt`. Iteration stops once
the recovered count is identical for two consecutive primes (or after
`--max-crt-iter`).

The legacy `Float64` / `Int128` / `BigInt` choices defer to
`solve(problem, CountingAll(); usecuda = ...)`, whose internal
`big_integer_solve(Int32, 100)` already does CRT — so they all behave
identically and are kept only for API parity with the ground-state
runner. `--count-eltype=BigInt` always implies CPU.

GPU setup follows the same `setup_cuda!` contract as the ground-state
runner.
"""

using Printf
using CSV, DataFrames
using CUDA

include(joinpath(@__DIR__, "..", "mis_slice_contract.jl"))

function _parse_count_eltype(tag::AbstractString)
    s = lowercase(String(tag))
    if s in ("finitefield", "finite_field", "ff", "crt", "mod")
        return :finitefield
    elseif s in ("int128", "i128")
        return Int128
    elseif s in ("float64", "f64")
        return Float64
    elseif s in ("bigint",)
        return BigInt
    else
        error("unknown --count-eltype=$(tag); use finitefield (default) / " *
              "Float64 / Int128 / BigInt")
    end
end

function _parse_args(args)
    slice_dir    = ""
    gpu_id       = -1
    csv_path     = ""
    verbose      = true
    count_eltype = :finitefield
    max_crt_iter = 8

    for a in args
        if startswith(a, "--gpu=")
            gpu_id = parse(Int, split(a, "=", limit = 2)[2])
        elseif a == "--no-cuda" || a == "--cpu"
            gpu_id = -1
        elseif startswith(a, "--csv=")
            csv_path = String(split(a, "=", limit = 2)[2])
        elseif a == "--quiet"
            verbose = false
        elseif startswith(a, "--count-eltype=")
            count_eltype = _parse_count_eltype(split(a, "=", limit = 2)[2])
        elseif startswith(a, "--max-crt-iter=") ||
               startswith(a, "--max-iter=")
            max_crt_iter = parse(Int, split(a, "=", limit = 2)[2])
        elseif !startswith(a, "--")
            slice_dir = String(a)
        end
    end
    isempty(slice_dir) &&
        error("usage: julia run_mis_counting_slice_contract.jl <slice_dir> " *
              "[--gpu=N] [--no-cuda] [--csv=out.csv] " *
              "[--count-eltype=finitefield|Float64|Int128|BigInt] " *
              "[--max-crt-iter=8] [--quiet]")
    return (; slice_dir, gpu_id, csv_path, verbose,
              count_eltype, max_crt_iter)
end

"""
    setup_cuda!(gpu_id) -> Bool

Same contract as the ground-state driver's `setup_cuda!`. See
[`run_mis_slice_contract.jl`] for full notes.
"""
function setup_cuda!(gpu_id::Int)
    try
        if !CUDA.functional()
            @warn "CUDA.jl loaded but not functional; falling back to CPU."
            return false
        end
        ndev = length(CUDA.devices())
        if gpu_id < 0 || gpu_id >= ndev
            @warn "Requested GPU id=$gpu_id but only $ndev device(s) visible; falling back to CPU."
            return false
        end
        CUDA.device!(gpu_id)
        dev = CUDA.device()
        drv = CUDA.driver_version()
        if v"11.4" <= drv <= v"12.3"
            try
                @eval Main using CuTropicalGEMM
                @info "CUDA backend ready: device #$(gpu_id) ($(CUDA.name(dev))); CuTropicalGEMM loaded for Tropical GEMM."
            catch err
                @warn "CuTropicalGEMM failed to load ($(err)); using bare CUDA.jl on device #$(gpu_id) ($(CUDA.name(dev)))."
            end
        else
            @info "CUDA backend ready: device #$(gpu_id) ($(CUDA.name(dev))); driver $(drv) outside CuTropicalGEMM-supported range [11.4, 12.3], using generic OMEinsum + CUDA kernels (no hand-tuned Tropical GEMM)."
        end
        return true
    catch err
        @warn "Failed to initialise CUDA ($(err)); falling back to CPU."
        return false
    end
end

# Wrapper isolated so callers can use Base.invokelatest after setup_cuda!
# loaded `CuTropicalGEMM` at runtime (same shim as the ground runner).
function _contract_mis_counting_slices_with_cuda!(resolved, verbose,
                                                   count_eltype, max_crt_iter)
    return contract_mis_counting_slices(resolved;
                                         usecuda      = true,
                                         verbose      = verbose,
                                         count_eltype = count_eltype,
                                         max_crt_iter = max_crt_iter)
end

function main(args)
    cfg = _parse_args(args)

    if cfg.count_eltype === BigInt && cfg.gpu_id >= 0
        @info "BigInt is not isbits ⇒ forcing CPU even though --gpu=$(cfg.gpu_id) was given."
    end
    usecuda = (cfg.gpu_id >= 0 && cfg.count_eltype !== BigInt) &&
              setup_cuda!(cfg.gpu_id)

    resolved = slice_results_dir(cfg.slice_dir)
    @info "loading slices from $resolved"
    if cfg.count_eltype === :finitefield
        @info "count algebra: Mods.Mod{p, Int32} (CRT, max $(cfg.max_crt_iter) prime(s))"
    else
        @info "count algebra: solve(... CountingAll(); usecuda=$(usecuda)) " *
              "[built-in finite-field CRT, T=$(cfg.count_eltype) ignored]"
    end
    summary = list_mis_slices(resolved)
    @info "found $(nrow(summary)) slice(s) (max sc = $(maximum(summary.sc)), max tc = $(maximum(summary.tc)))"

    res = if usecuda
        Base.invokelatest(_contract_mis_counting_slices_with_cuda!,
                          resolved, cfg.verbose,
                          cfg.count_eltype, cfg.max_crt_iter)
    else
        contract_mis_counting_slices(resolved;
                                      usecuda      = false,
                                      verbose      = cfg.verbose,
                                      count_eltype = cfg.count_eltype,
                                      max_crt_iter = cfg.max_crt_iter)
    end

    @printf("\n=== Total independent-set count ===\n")
    @printf("  total IS count:           %s\n",   string(res.count))
    @printf("  slice count:              %d\n",   nrow(res.per_slice))
    @printf("  total contraction time:   %.3fs\n", res.total_runtime)

    if cfg.count_eltype === :finitefield && !isempty(res.per_slice.n_primes)
        @printf("  CRT primes per slice:     min=%d max=%d mean=%.2f\n",
                minimum(res.per_slice.n_primes),
                maximum(res.per_slice.n_primes),
                sum(res.per_slice.n_primes) / nrow(res.per_slice))
    end
    flush(stdout)

    if !isempty(cfg.csv_path)
        isdir(dirname(cfg.csv_path)) || mkpath(dirname(cfg.csv_path))
        CSV.write(cfg.csv_path, res.per_slice)
        @info "per-slice contribution table written to $(cfg.csv_path)"
    end

    return res
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
