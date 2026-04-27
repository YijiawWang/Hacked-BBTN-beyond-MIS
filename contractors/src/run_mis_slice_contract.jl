"""
    run_mis_slice_contract.jl

Driver script that loads a directory of saved MIS / MWIS slices (as
produced by `init_mis_slice_writer` / `save_mis_slices` from
`mis_slice_contract.jl`), exactly contracts each one through
`GenericTensorNetwork(IndependentSet(g, w), code, Dict{Int,Int}())`
(i.e. **reusing the saved contraction order**), shifts every slice's
size by its `r`, and prints / saves the global maximum-IS size +
degeneracy.

It works for any slice dump whose `slice.p` is a `MISProblem`, so it
handles the slices produced by both
`beyond_mis/hacked_funcs/benchmarks/mis_counting.jl` and
`mis_ground_counting.jl`.

Usage:

    julia --project=beyond_mis beyond_mis/contractors/src/run_mis_slice_contract.jl \\
        <slice_dir> [--gpu=N] [--no-cuda] [--csv=out.csv] [--atol=1e-6] \\
        [--count-eltype=finitefield|Float64|Int128|BigInt] [--scale=1] \\
        [--max-crt-iter=8] [--quiet]

`<slice_dir>` may be either:
  * the *subdir name* under `beyond_mis/branch_results/` (e.g.
    `mis_ground_counting_random_ksg_n=600_seed=1`), or
  * an absolute path / pre-existing directory.

The default `--count-eltype=finitefield` is the GPU-safe exact mode:
each slice is contracted with eltype
`CountingTropical{Float64, Mods.Mod{p, Int32}}` for several primes
near `typemax(Int32)`, and the per-prime counts are CRT-combined into
an exact `BigInt`. Iteration stops once the recovered count is
identical for two consecutive primes (or after `--max-crt-iter`).
This mirrors the spin-glass driver and sidesteps the LLVM NVPTX
≥ 256-bit codegen problems hit by `CountingTropical{Int128,Int128}`.

`--scale=N` multiplies vertex weights by `N` and rounds to `Int` before
contracting (only used when `count_eltype` requires integer weights;
the default `1` covers `UnitWeight` and integer-weight MIS).
`--count-eltype=BigInt` always implies CPU.

GPU setup follows `response_bbtn/counting_contract/bench_spin_glass.jl`
(`setup_cuda!`): top-level `using CUDA`, device bounds check, optional
`CuTropicalGEMM` when the driver is in ``[11.4, 12.3]``. Use `--no-cuda`
or `--cpu` to force CPU even if `--gpu=N` appears earlier on the
command line.
"""

using Printf
using CSV, DataFrames
using CUDA

include(joinpath(@__DIR__, "..", "mis_slice_contract.jl"))

# `count_eltype` accepts either a `DataType` (the legacy direct path) or
# the symbol `:finitefield` (the new CRT path). Same parser as the
# spin-glass driver so the CLI feels uniform across families.
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
    atol         = 1e-6
    verbose      = true
    count_eltype = :finitefield
    weight_scale = 1
    max_crt_iter = 8

    for a in args
        if startswith(a, "--gpu=")
            gpu_id = parse(Int, split(a, "=", limit = 2)[2])
        elseif a == "--no-cuda" || a == "--cpu"
            gpu_id = -1
        elseif startswith(a, "--csv=")
            csv_path = String(split(a, "=", limit = 2)[2])
        elseif startswith(a, "--atol=")
            atol = parse(Float64, split(a, "=", limit = 2)[2])
        elseif a == "--quiet"
            verbose = false
        elseif startswith(a, "--count-eltype=")
            count_eltype = _parse_count_eltype(split(a, "=", limit = 2)[2])
        elseif startswith(a, "--scale=") || startswith(a, "--weight-scale=")
            weight_scale = parse(Int, split(a, "=", limit = 2)[2])
        elseif startswith(a, "--max-crt-iter=") ||
               startswith(a, "--max-iter=")
            max_crt_iter = parse(Int, split(a, "=", limit = 2)[2])
        elseif !startswith(a, "--")
            slice_dir = String(a)
        end
    end
    isempty(slice_dir) &&
        error("usage: julia run_mis_slice_contract.jl <slice_dir> " *
              "[--gpu=N] [--no-cuda] [--csv=out.csv] [--atol=1e-6] " *
              "[--count-eltype=finitefield|Float64|Int128|BigInt] " *
              "[--scale=1] [--max-crt-iter=8] [--quiet]")
    return (; slice_dir, gpu_id, csv_path, atol, verbose,
              count_eltype, weight_scale, max_crt_iter)
end

"""
    setup_cuda!(gpu_id) -> Bool

Same contract as the spin-glass driver's `setup_cuda!`: select the CUDA
device for `solve(...; usecuda=true)`, optionally load `CuTropicalGEMM`
when the driver is in ``[11.4, 12.3]``, and return whether GPU
contraction should be used.
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

# If `setup_cuda!` used `@eval Main using CuTropicalGEMM`, callers still run at
# `main`'s entry world age; GPU contraction must see the latest methods.
function _contract_mis_slices_with_cuda!(resolved, atol, verbose,
                                         count_eltype, weight_scale, max_crt_iter)
    return contract_mis_slices(resolved;
                               usecuda      = true,
                               atol         = atol,
                               verbose      = verbose,
                               count_eltype = count_eltype,
                               weight_scale = weight_scale,
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
        @info "count algebra: CountingTropical{Float64, Mods.Mod{p, Int32}} " *
              "(CRT, max $(cfg.max_crt_iter) prime(s)); weight_scale = $(cfg.weight_scale)"
    else
        @info "count algebra: CountingTropical{$(cfg.count_eltype), $(cfg.count_eltype)}; " *
              "weight_scale = $(cfg.weight_scale)"
    end
    summary = list_mis_slices(resolved)
    @info "found $(nrow(summary)) slice(s) (max sc = $(maximum(summary.sc)), max tc = $(maximum(summary.tc)))"

    res = if usecuda
        Base.invokelatest(_contract_mis_slices_with_cuda!,
                          resolved, cfg.atol, cfg.verbose,
                          cfg.count_eltype, cfg.weight_scale, cfg.max_crt_iter)
    else
        contract_mis_slices(resolved;
                            usecuda      = false,
                            atol         = cfg.atol,
                            verbose      = cfg.verbose,
                            count_eltype = cfg.count_eltype,
                            weight_scale = cfg.weight_scale,
                            max_crt_iter = cfg.max_crt_iter)
    end

    @printf("\n=== Exact maximum independent set ===\n")
    @printf("  max IS size:              %.10g\n", res.size)
    @printf("  max IS degeneracy:        %s\n",   string(res.count))
    @printf("  contributing slices:      %d / %d\n",
            count(res.per_slice.used_in_total), nrow(res.per_slice))
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
