"""
    contract_mis_counting_slices.jl

Companion to `_lib/slice_mis_counting.jl`. Loads an MIS slice dump
produced by the slicer (or by
`beyond_mis/hacked_funcs/benchmarks/mis_counting.jl`) and exactly
re-contracts every slice through its **saved** OMEinsum code under the
`CountingAll()` property — i.e. counts **every** independent set
(not just the maximum-weight ones). Per-slice counts are summed into
the global total and persisted under
`beyond_mis/results/mis_counting_slice_contract/<results-name>/`.

The actual contraction kernel
(`contract_mis_counting_slices`, `load_mis_slice`, …) lives in
`beyond_mis/contractors/mis_slice_contract.jl`. This script just wires
it to the CLI and writes a tidy `results/` layout.

Usage:

    julia --project=beyond_mis beyond_mis/scripts/_lib/contract_mis_counting_slices.jl \\
        <slice_dir> [--gpu=N] [--no-cuda] [--quiet] \\
                    [--count-eltype=finitefield|Float64|Int128|BigInt] \\
                    [--max-crt-iter=<int>]    # default 8
                    [--results-root=<path>]   # default beyond_mis/results
                    [--results-name=<name>]   # default basename(<slice_dir>)
"""

using Printf
using CSV, DataFrames
using CUDA

include(joinpath(@__DIR__, "..", "..", "contractors",
                 "mis_slice_contract.jl"))

const RESULTS_ROOT_DEFAULT =
    abspath(joinpath(@__DIR__, "..", "..", "results"))
const RESULTS_SUBFOLDER = "mis_counting_slice_contract"

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

struct MISCountingContractConfig
    slice_dir::String
    gpu_id::Int
    verbose::Bool
    count_eltype::Any
    max_crt_iter::Int
    results_root::String
    results_name::String
end

function _parse_count_eltype(s::AbstractString)
    s2 = strip(String(s))
    s2 = startswith(s2, ":") ? s2[2:end] : s2
    sl = lowercase(s2)
    if sl in ("finitefield", "finite_field", "ff", "crt", "mod")
        return :finitefield
    elseif sl in ("float64", "f64")
        return Float64
    elseif sl in ("int128", "i128")
        return Int128
    elseif sl in ("bigint",)
        return BigInt
    end
    error("--count-eltype must be one of finitefield / Float64 / Int128 / BigInt; got '$s'")
end

function _derive_results_name(slice_dir::AbstractString)
    resolved = slice_results_dir(String(slice_dir))
    name = basename(rstrip(resolved, '/'))
    isempty(name) && (name = "unnamed_mis_counting_slice_run")
    return name
end

function _parse_args(args)
    slice_dir    = ""
    gpu_id       = -1
    verbose      = true
    count_eltype = :finitefield
    max_crt_iter = 8
    results_root = RESULTS_ROOT_DEFAULT
    results_name = ""

    for a in args
        if startswith(a, "--gpu=")
            gpu_id = parse(Int, split(a, "="; limit = 2)[2])
        elseif a == "--no-cuda" || a == "--cpu"
            gpu_id = -1
        elseif a == "--quiet"
            verbose = false
        elseif startswith(a, "--count-eltype=")
            count_eltype = _parse_count_eltype(split(a, "="; limit = 2)[2])
        elseif startswith(a, "--max-crt-iter=") || startswith(a, "--max-iter=")
            max_crt_iter = parse(Int, split(a, "="; limit = 2)[2])
        elseif startswith(a, "--results-root=")
            results_root = abspath(String(split(a, "="; limit = 2)[2]))
        elseif startswith(a, "--results-name=")
            results_name = String(split(a, "="; limit = 2)[2])
        elseif a in ("-h", "--help")
            println(stderr, replace(string(@doc Main), "    " => "  "))
            exit(0)
        elseif startswith(a, "--")
            error("unknown / unsupported argument: $a")
        else
            slice_dir = String(a)
        end
    end
    isempty(slice_dir) &&
        error("usage: julia contract_mis_counting_slices.jl <slice_dir> " *
              "[--gpu=N] [--no-cuda] [--quiet] " *
              "[--count-eltype=finitefield|Float64|Int128|BigInt] " *
              "[--max-crt-iter=8] [--results-root=…] [--results-name=…]")

    isempty(results_name) && (results_name = _derive_results_name(slice_dir))
    return MISCountingContractConfig(slice_dir, gpu_id, verbose,
                                       count_eltype, max_crt_iter,
                                       results_root, results_name)
end

# ---------------------------------------------------------------------------
# CUDA setup
# ---------------------------------------------------------------------------

function _setup_cuda!(gpu_id::Int)
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
                @info "CUDA backend ready: device #$(gpu_id) ($(CUDA.name(dev))); CuTropicalGEMM loaded."
            catch err
                @warn "CuTropicalGEMM failed to load ($(err)); using bare CUDA.jl."
            end
        else
            @info "CUDA backend ready: device #$(gpu_id) ($(CUDA.name(dev))); driver $(drv) outside CuTropicalGEMM range."
        end
        return true
    catch err
        @warn "Failed to initialise CUDA ($(err)); falling back to CPU."
        return false
    end
end

function _contract_with_cuda(resolved, cfg::MISCountingContractConfig)
    return contract_mis_counting_slices(resolved;
                                         usecuda      = true,
                                         verbose      = cfg.verbose,
                                         count_eltype = cfg.count_eltype,
                                         max_crt_iter = cfg.max_crt_iter)
end

# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

function _write_per_slice_csv(out_dir::AbstractString, df::DataFrame)
    mkpath(out_dir)
    path = joinpath(out_dir, "per_slice.csv")
    df_out = copy(df)
    df_out.count = map(string, df_out.count)
    CSV.write(path, df_out)
    return path
end

function _summary_row(cfg::MISCountingContractConfig, slice_path::AbstractString,
                       res::NamedTuple, total::Int)
    return DataFrame(
        slice_dir            = [slice_path],
        results_name         = [cfg.results_name],
        count                = [string(res.count)],
        total_slices         = [total],
        total_runtime        = [res.total_runtime],
        used_gpu             = [cfg.gpu_id ≥ 0],
    )
end

function _write_run_summary(out_dir::AbstractString, row::DataFrame)
    mkpath(out_dir)
    path = joinpath(out_dir, "summary.csv")
    CSV.write(path, row)
    return path
end

function _append_global_summary(global_dir::AbstractString, row::DataFrame)
    mkpath(global_dir)
    path = joinpath(global_dir, "summary.csv")
    df = if isfile(path)
        try
            existing = CSV.read(path, DataFrame)
            existing = filter(r -> r.results_name != row.results_name[1], existing)
            vcat(existing, row; cols = :union)
        catch err
            @warn "results summary at $path is unreadable, starting fresh ($(err))"
            row
        end
    else
        row
    end
    CSV.write(path, df)
    return path
end

# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

function main(args)
    cfg = _parse_args(args)

    resolved = slice_results_dir(cfg.slice_dir)
    isdir(resolved) || error("slice directory does not exist: $resolved")
    @info "[contract_mis_counting] slice directory: $resolved"

    summary = list_mis_slices(resolved)
    @info "[contract_mis_counting] found $(nrow(summary)) slice(s); max sc=$(maximum(summary.sc)), max tc=$(maximum(summary.tc))"

    usecuda = (cfg.gpu_id ≥ 0) && _setup_cuda!(cfg.gpu_id)
    if cfg.count_eltype === BigInt && usecuda
        @info "BigInt is not isbits ⇒ forcing CPU even though --gpu=$(cfg.gpu_id) was given."
        usecuda = false
    end

    res = if usecuda
        Base.invokelatest(_contract_with_cuda, resolved, cfg)
    else
        contract_mis_counting_slices(resolved;
                                      usecuda      = false,
                                      verbose      = cfg.verbose,
                                      count_eltype = cfg.count_eltype,
                                      max_crt_iter = cfg.max_crt_iter)
    end

    @printf("\n=== Total IS count (slice + sum) ===\n")
    @printf("  total IS count:           %s\n",   string(res.count))
    @printf("  slice count:              %d\n",   nrow(res.per_slice))
    @printf("  total contraction time:   %.3fs\n", res.total_runtime)

    out_dir   = joinpath(cfg.results_root, RESULTS_SUBFOLDER, cfg.results_name)
    per_path  = _write_per_slice_csv(out_dir, res.per_slice)
    total     = nrow(res.per_slice)
    row       = _summary_row(cfg, resolved, res, total)
    sum_path  = _write_run_summary(out_dir, row)
    glob_path = _append_global_summary(joinpath(cfg.results_root,
                                                RESULTS_SUBFOLDER), row)

    @printf("\n[results] per-slice CSV : %s\n", per_path)
    @printf("[results] run summary   : %s\n", sum_path)
    @printf("[results] global index  : %s\n", glob_path)

    return (
        count         = res.count,
        per_slice     = res.per_slice,
        total_runtime = res.total_runtime,
        slice_dir     = resolved,
        results_dir   = out_dir,
        per_slice_csv = per_path,
        summary_csv   = sum_path,
        global_csv    = glob_path,
        used_gpu      = usecuda,
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
