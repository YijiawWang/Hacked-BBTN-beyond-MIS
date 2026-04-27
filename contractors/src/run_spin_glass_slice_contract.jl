"""
    run_spin_glass_slice_contract.jl

Driver script that loads a directory of saved spin-glass slices (as
produced by `save_spin_glass_slices` from
`spin_glass_slice_contract.jl`), exactly contracts each one through
`GenericTensorNetwork(SpinGlass(g, J, h), code, Dict{Int,Int}())` (i.e.
**reusing the saved contraction order**), shifts every slice's energy
by its `r`, and prints / saves the global ground-state energy +
degeneracy.

Usage:

    julia --project=beyond_mis beyond_mis/contractors/run_spin_glass_slice_contract.jl \\
        <slice_dir> [--gpu=N] [--no-cuda] [--csv=out.csv] [--atol=1e-6] \\
        [--count-eltype=finitefield|Float64|Int128|BigInt] [--scale=2] \\
        [--max-crt-iter=8] [--quiet]

`<slice_dir>` may be either:
  * the *subdir name* under `beyond_mis/branch_results/` (e.g.
    `rrg_Jpm1_n=600_d=3_seed=1`), or
  * an absolute path / pre-existing directory.

The default `--count-eltype=finitefield` is the GPU-safe exact mode:
each slice is contracted with eltype
`CountingTropical{Float64, Mods.Mod{p, Int32}}` for several primes
near `typemax(Int32)`, and the per-prime counts are CRT-combined into
an exact `BigInt`.  Iteration stops once the recovered count is
identical for two consecutive primes (or after `--max-crt-iter`).

GPU setup is the same as `counting_contract/bench_spin_glass.jl`'s
`setup_cuda!`. Both `CUDA` and `CuTropicalGEMM` are loaded **at the
top of the file** so that all extension methods (`togpu`,
`Array(::CuArray)`, etc.) live at the same world-age as `main()` —
this is what the standalone `_bench_31x31_finitefield.jl` script does
too. Use `--no-cuda` (or `--cpu`) to force CPU even when `--gpu=N`
appears earlier on the command line. `--count-eltype=BigInt` always
implies CPU.
"""

using Printf
using CSV, DataFrames
using Logging
using CUDA

# When stdout is redirected to a file (e.g. via `nohup ... > log 2>&1 &`),
# Julia switches to block-buffered output and the default `ConsoleLogger`
# does not flush.  Wrap it so every log record is followed by an explicit
# `flush(stdout)` — otherwise long-running slice contractions look frozen
# even though the GPU is actively churning.
struct _FlushingLogger <: Logging.AbstractLogger
    inner::Logging.AbstractLogger
end
Logging.shouldlog(l::_FlushingLogger, args...) = Logging.shouldlog(l.inner, args...)
Logging.min_enabled_level(l::_FlushingLogger) = Logging.min_enabled_level(l.inner)
Logging.catch_exceptions(l::_FlushingLogger) = Logging.catch_exceptions(l.inner)
function Logging.handle_message(l::_FlushingLogger, args...; kwargs...)
    Logging.handle_message(l.inner, args...; kwargs...)
    flush(stdout); flush(stderr)
end
global_logger(_FlushingLogger(ConsoleLogger(stdout, Logging.Info)))
# CuTropicalGEMM is loaded eagerly so that its `mul!` overloads on
# Tropical{Float64} are visible at `main`'s world-age. On unsupported
# CUDA drivers (>12.3) the package emits a warning at import but does
# not error, and we simply fall back to GPUArrays' generic kernels.
try
    @eval using CuTropicalGEMM
catch err
    @warn "CuTropicalGEMM failed to load eagerly ($(err)); generic CUDA " *
          "kernels will be used instead."
end

include(joinpath(@__DIR__, "spin_glass_slice_contract.jl"))

# `count_eltype` accepts either a `DataType` (the legacy direct path) or
# the symbol `:finitefield` (the new CRT path). This helper turns a CLI
# tag into one of those two cases.
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
    energy_scale = 2
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
        elseif startswith(a, "--scale=") || startswith(a, "--energy-scale=")
            energy_scale = parse(Int, split(a, "=", limit = 2)[2])
        elseif startswith(a, "--max-crt-iter=") ||
               startswith(a, "--max-iter=")
            max_crt_iter = parse(Int, split(a, "=", limit = 2)[2])
        elseif !startswith(a, "--")
            slice_dir = String(a)
        end
    end
    isempty(slice_dir) &&
        error("usage: julia run_spin_glass_slice_contract.jl <slice_dir> " *
              "[--gpu=N] [--no-cuda] [--csv=out.csv] [--atol=1e-6] " *
              "[--count-eltype=finitefield|Float64|Int128|BigInt] " *
              "[--scale=2] [--max-crt-iter=8] [--quiet]")
    return (; slice_dir, gpu_id, csv_path, atol, verbose,
              count_eltype, energy_scale, max_crt_iter)
end

"""
    setup_cuda!(gpu_id) -> Bool

Select the CUDA device for `solve(...; usecuda=true)` and return whether
GPU contraction should be used. `CuTropicalGEMM` is already loaded at the
top of the file (so all extension methods are visible at `main`'s
world-age), so this function only picks the device and validates it.
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
            @info "CUDA backend ready: device #$(gpu_id) ($(CUDA.name(dev))); CuTropicalGEMM available for Tropical GEMM."
        else
            @info "CUDA backend ready: device #$(gpu_id) ($(CUDA.name(dev))); driver $(drv) outside CuTropicalGEMM-supported range [11.4, 12.3], using generic OMEinsum + CUDA kernels (no hand-tuned Tropical GEMM)."
        end
        return true
    catch err
        @warn "Failed to initialise CUDA ($(err)); falling back to CPU."
        return false
    end
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
              "(CRT, max $(cfg.max_crt_iter) prime(s)); energy_scale = $(cfg.energy_scale)"
    else
        @info "count algebra: CountingTropical{$(cfg.count_eltype), $(cfg.count_eltype)}; " *
              "energy_scale = $(cfg.energy_scale)"
    end
    summary = list_spin_glass_slices(resolved)
    @info "found $(nrow(summary)) slice(s) (max sc = $(maximum(summary.sc)), max tc = $(maximum(summary.tc)))"

    res = contract_spin_glass_slices(resolved;
                                     usecuda      = usecuda,
                                     atol         = cfg.atol,
                                     verbose      = cfg.verbose,
                                     count_eltype = cfg.count_eltype,
                                     energy_scale = cfg.energy_scale,
                                     max_crt_iter = cfg.max_crt_iter)

    @printf("\n=== Exact ground state ===\n")
    @printf("  ground-state energy:      %.10g\n", res.energy)
    @printf("  ground-state degeneracy:  %s\n",   string(res.count))
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
