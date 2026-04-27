"""
    contract_spin_glass_slices.jl

Companion to `scripts/slice_spin_glass.jl`. Loads a slice dump produced
by the slicer (or by any of the three benchmarks under
`beyond_mis/hacked_funcs/benchmarks/`) and exactly re-contracts each
slice through its **saved** OMEinsum code. Per-slice results are
combined into the global ground-state `(E_g, deg)` and persisted under
`beyond_mis/results/spin_glass_slice_contract/<results-name>/`.

The actual contraction code (`contract_spin_glass_slices`,
`load_spin_glass_slice`, …) lives in
`beyond_mis/contractors/spin_glass_slice_contract.jl`. This script just
wires it to the CLI and writes a tidy `results/` layout.

Usage:

    julia --project=beyond_mis beyond_mis/scripts/contract_spin_glass_slices.jl \\
        <slice_dir> [--gpu=N] [--no-cuda] \\
                    [--atol=1e-6] [--quiet] \\
                    [--energy-scale=<int>]    # default: 2 (covers J,h ∈ ±0.5,±1)
                    [--count-eltype=<symbol>] # default: finitefield
                    [--max-crt-iter=<int>]    # default: 8
                    [--results-root=<path>]   # default: beyond_mis/results
                    [--results-name=<name>]   # default: basename(<slice_dir>)

`<slice_dir>` accepts either:
  * a subdir name under `beyond_mis/branch_results/` (e.g.
    `grid_Jpm1_n=20_seed=1_cheating`), or
  * an absolute path / pre-existing directory.

Two files are written per run:

  * `<results-root>/spin_glass_slice_contract/<results-name>/per_slice.csv`
    – one row per slice with `(id, r, energy, count, runtime, used_in_total)`.
  * `<results-root>/spin_glass_slice_contract/<results-name>/summary.csv`
    – global `(slice_dir, energy, count, contributing_slices, total_slices,
       total_runtime, …)` row.

A single rolling index is also kept at
`<results-root>/spin_glass_slice_contract/summary.csv` (one row per
`results-name`, replaced on re-runs).
"""

using Printf
using CSV, DataFrames
using CUDA

include(joinpath(@__DIR__, "..", "..", "contractors",
                 "spin_glass_slice_contract.jl"))

const RESULTS_ROOT_DEFAULT =
    abspath(joinpath(@__DIR__, "..", "..", "results"))
const RESULTS_SUBFOLDER = "spin_glass_slice_contract"

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

struct ContractConfig
    slice_dir::String                 # raw, as passed on the CLI
    gpu_id::Int
    atol::Float64
    verbose::Bool
    energy_scale::Int
    count_eltype::Any                 # Symbol or DataType
    max_crt_iter::Int
    results_root::String              # absolute
    results_name::String              # subfolder under <results_root>/<RESULTS_SUBFOLDER>
end

# Map a CLI string to a value `contract_spin_glass_slices` understands
# for `count_eltype`. Accepts the symbolic `:finitefield` form as well as
# the legacy direct paths via the corresponding type names.
function _parse_count_eltype(s::AbstractString)
    s2 = strip(String(s))
    s2 = startswith(s2, ":") ? s2[2:end] : s2
    if s2 == "finitefield"
        return :finitefield
    elseif s2 == "Float64"
        return Float64
    elseif s2 == "Int128"
        return Int128
    elseif s2 == "BigInt"
        return BigInt
    end
    error("--count-eltype must be one of :finitefield, Float64, Int128, BigInt; got '$s'")
end

function _derive_results_name(slice_dir::AbstractString)
    resolved = slice_results_dir(String(slice_dir))
    name = basename(rstrip(resolved, '/'))
    isempty(name) && (name = "unnamed_slice_run")
    return name
end

function _parse_args(args)
    slice_dir    = ""
    gpu_id       = -1
    atol         = 1e-6
    verbose      = true
    energy_scale = 2
    count_eltype = :finitefield
    max_crt_iter = 8
    results_root = RESULTS_ROOT_DEFAULT
    results_name = ""

    for a in args
        if startswith(a, "--gpu=")
            gpu_id = parse(Int, split(a, "="; limit = 2)[2])
        elseif a == "--no-cuda" || a == "--cpu"
            gpu_id = -1
        elseif startswith(a, "--atol=")
            atol = parse(Float64, split(a, "="; limit = 2)[2])
        elseif a == "--quiet"
            verbose = false
        elseif startswith(a, "--energy-scale=")
            energy_scale = parse(Int, split(a, "="; limit = 2)[2])
        elseif startswith(a, "--count-eltype=")
            count_eltype = _parse_count_eltype(split(a, "="; limit = 2)[2])
        elseif startswith(a, "--max-crt-iter=")
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
        error("usage: julia contract_spin_glass_slices.jl <slice_dir> " *
              "[--gpu=N] [--no-cuda] [--atol=…] [--quiet] " *
              "[--energy-scale=…] [--count-eltype=…] [--max-crt-iter=…] " *
              "[--results-root=…] [--results-name=…]")

    isempty(results_name) && (results_name = _derive_results_name(slice_dir))
    return ContractConfig(slice_dir, gpu_id, atol, verbose,
                          energy_scale, count_eltype, max_crt_iter,
                          results_root, results_name)
end

# ---------------------------------------------------------------------------
# CUDA setup (mirrors `…/contractors/src/run_spin_glass_slice_contract.setup_cuda!`)
# ---------------------------------------------------------------------------

"""
    _setup_cuda!(gpu_id) -> Bool

Pin the requested CUDA device for `solve(...; usecuda=true)`, optionally
load `CuTropicalGEMM` when the driver is in the supported `[11.4, 12.3]`
range (matching the existing benchmark + driver scripts), and return
whether GPU contraction should actually be used. Falls back to CPU on
any failure.
"""
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

# When `_setup_cuda!` `@eval`'d `using CuTropicalGEMM`, the new methods
# live in a later world-age than `main`. Wrap the contraction call so we
# can `Base.invokelatest` it.
function _contract_with_cuda(resolved, cfg::ContractConfig)
    return contract_spin_glass_slices(resolved;
                                      usecuda      = true,
                                      atol         = cfg.atol,
                                      verbose      = cfg.verbose,
                                      energy_scale = cfg.energy_scale,
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
    # `count` is a Vector{BigInt} → render via `string` so CSV.jl is happy.
    df_out.count = map(string, df_out.count)
    CSV.write(path, df_out)
    return path
end

function _summary_row(cfg::ContractConfig, slice_path::AbstractString,
                       res::NamedTuple, used::Int, total::Int)
    return DataFrame(
        slice_dir            = [slice_path],
        results_name         = [cfg.results_name],
        energy               = [res.energy],
        count                = [string(res.count)],
        contributing_slices  = [used],
        total_slices         = [total],
        total_runtime        = [res.total_runtime],
        atol                 = [cfg.atol],
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

"""
    main(args::AbstractVector{<:AbstractString}) -> NamedTuple

Resolve `<slice_dir>`, contract every slice (optionally on GPU), persist
per-slice and aggregate results under
`<results-root>/spin_glass_slice_contract/<results-name>/`, and return:

    (energy, count, per_slice, total_runtime,
     slice_dir, results_dir, per_slice_csv, summary_csv, global_csv,
     used_gpu)
"""
function main(args)
    cfg = _parse_args(args)

    resolved = slice_results_dir(cfg.slice_dir)
    isdir(resolved) || error("slice directory does not exist: $resolved")
    @info "[contract] slice directory: $resolved"

    summary = list_spin_glass_slices(resolved)
    @info "[contract] found $(nrow(summary)) slice(s); max sc=$(maximum(summary.sc)), max tc=$(maximum(summary.tc))"

    usecuda = (cfg.gpu_id ≥ 0) && _setup_cuda!(cfg.gpu_id)

    res = if usecuda
        Base.invokelatest(_contract_with_cuda, resolved, cfg)
    else
        contract_spin_glass_slices(resolved;
                                   usecuda      = false,
                                   atol         = cfg.atol,
                                   verbose      = cfg.verbose,
                                   energy_scale = cfg.energy_scale,
                                   count_eltype = cfg.count_eltype,
                                   max_crt_iter = cfg.max_crt_iter)
    end

    @printf("\n=== Exact ground state (slice + sum) ===\n")
    @printf("  ground-state energy:      %.10g\n", res.energy)
    @printf("  ground-state degeneracy:  %s\n",   string(res.count))
    @printf("  contributing slices:      %d / %d\n",
            count(res.per_slice.used_in_total), nrow(res.per_slice))
    @printf("  total contraction time:   %.3fs\n", res.total_runtime)

    out_dir   = joinpath(cfg.results_root, RESULTS_SUBFOLDER, cfg.results_name)
    per_path  = _write_per_slice_csv(out_dir, res.per_slice)
    used      = count(res.per_slice.used_in_total)
    total     = nrow(res.per_slice)
    row       = _summary_row(cfg, resolved, res, used, total)
    sum_path  = _write_run_summary(out_dir, row)
    glob_path = _append_global_summary(joinpath(cfg.results_root,
                                                RESULTS_SUBFOLDER), row)

    @printf("\n[results] per-slice CSV : %s\n", per_path)
    @printf("[results] run summary   : %s\n", sum_path)
    @printf("[results] global index  : %s\n", glob_path)

    return (
        energy        = res.energy,
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
