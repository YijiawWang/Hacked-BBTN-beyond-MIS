"""
    slice_spin_glass.jl

Parameterized driver that slices a **pre-generated** spin-glass `.model`
file (the same format `models/spin_glass_model_generator.jl` writes and
`hacked_funcs/benchmarks/spin_glass_ground_counting.jl` reads) into
finished slices, and dumps every slice into
`beyond_mis/branch_results/<subdir>/` exactly the way the benchmarks do
it. Subsequent scripts in this folder
(`contract_spin_glass_slices.jl`, `verify_spin_glass.jl`,
`beyond_mis/contractors/spin_glass_slice_contract.jl`) can then pick the
dump up unchanged.

Three known filename patterns are auto-recognised so that `subdir`,
`model_name`, `graph_type`, the default uniform field `h`, and the
`meta` dict written into `summary.csv` exactly match those produced by
the benchmark script:

  * `spin_glass_J±1_grid_n=<n>_seed=<seed>.model`
        -> graph_type = `grid_Jpm1`,            default h = 0.5
  * `spin_glass_J1J2_grid_L=<L>_J1pm1_J2pm1_seed=<seed>.model`
        -> graph_type = `j1j2_grid_open`,       default h = 0.5
  * `spin_glass_J1J2_pbc_afm_grid_L=<L>_g=<gf>.model`
        -> graph_type = `j1j2_grid_pbc_afm`,    default h = 0.0

Files that don't match any of these patterns are sliced with a generic
graph_type / `h = 0.0` / `subdir = <basename>_cheating`, which is fine
for ad-hoc debugging.

Usage:

    julia --project=beyond_mis beyond_mis/scripts/_lib/slice_spin_glass.jl \\
        --model=<path/to/file.model> \\
        --sc-target=<int> \\
        [--h=<float>]            # override default uniform field
        [--subdir=<name>]        # override the auto-inferred slice subdir
        [--graph-type=<name>]    # override auto-inferred graph_type tag
        [--no-lp]                # use slice_dfs (no Gurobi LP) instead of
                                 # slice_dfs_lp (the benchmarks' default)
        [--ntrials=<int>]        # TreeSA ntrials for code search (default 50)
        [--niters=<int>]         # TreeSA niters    for code search (default 100)
        [--code-seeds=<lo:hi>]   # range of RNG seeds for the code search
                                 # (default 1:2, lowest-tc kept)
        [--lp-time-limit=<sec>]  # Gurobi initial LP time limit (default 300)
        [--quiet]                # set verbose=0 (default verbose=1)

The script prints (and returns from `main`) the absolute slice directory
path on success, which is what `verify_spin_glass.jl` consumes.
"""

using Random
using Graphs, OMEinsum, OMEinsumContractionOrders
using OptimalBranching
using OptimalBranching.OptimalBranchingCore, OptimalBranching.OptimalBranchingMIS
using OptimalBranchingMIS: TensorNetworkSolver
using OMEinsumContractionOrders: TreeSA, uniformsize
using TensorBranching: ContractionTreeSlicer, GreedyBrancher, ScoreRS,
                       SlicedBranch, complexity, initialize_code

# Pull in the benchmark layer (which itself includes
# `hacked_funcs/src/spin_glass_ground_counting.jl` and the
# `contractors/spin_glass_slice_contract.jl` writer/reader). The
# benchmark file's `main(args)` is guarded by
# `if abspath(PROGRAM_FILE) == @__FILE__`, so the include only defines
# functions; the per-family `run_*` loops are not executed here.
# Driver-side `main(args)` (defined further below) intentionally
# overrides the benchmark's `main(args)` because of include order.
include(joinpath(@__DIR__, "..", "..", "hacked_funcs", "benchmarks",
                 "spin_glass_ground_counting.jl"))

# ---------------------------------------------------------------------------
# `.model` file reader (same format produced by
# `hacked_funcs/benchmarks/models/spin_glass_model_generator.jl`).
# ---------------------------------------------------------------------------

"""
    _read_spin_glass_model(path; T=Float32)
        -> (graph, J::Vector{T}, header_meta::Dict{String,Any})

Parse the on-disk `.model` format. `J` is ordered to match
`edges(graph)`. `header_meta` collects every `key = value` line from
the header (lines preceding `vertices:`), which lets us pull `L`,
`seed`, `n`, `g`, `J1`, `J2` out without re-deriving them from the
filename.
"""
function _read_spin_glass_model(path::AbstractString; T::Type = Float32)
    graph = SimpleGraph()
    edge_weights = Dict{Tuple{Int,Int}, Float64}()
    header_meta = Dict{String,Any}()
    open(path, "r") do io
        in_edges = false
        while !eof(io)
            line = readline(io)
            if !in_edges
                stripped = strip(line)
                isempty(stripped) && continue
                if startswith(stripped, "vertices:")
                    n_vertices = parse(Int, split(stripped)[2])
                    graph = SimpleGraph(n_vertices)
                elseif startswith(stripped, "edges:")
                    # written by the generator but not needed: ne(graph)
                    # falls out of edge insertion below
                    continue
                elseif stripped == "edges_with_weights:"
                    in_edges = true
                elseif startswith(stripped, "#")
                    continue
                elseif occursin('=', stripped)
                    k, v = split(stripped, '='; limit = 2)
                    key = String(strip(k))
                    val = String(strip(v))
                    # strip inline comments on the value (e.g. "# |J2/J1|")
                    if occursin('#', val)
                        val = String(strip(split(val, '#'; limit = 2)[1]))
                    end
                    parsed = tryparse(Int, val)
                    if parsed === nothing
                        parsed = tryparse(Float64, val)
                    end
                    header_meta[key] = parsed === nothing ? val : parsed
                end
            else
                isempty(strip(line)) && continue
                parts = split(line)
                length(parts) < 3 && continue
                u = parse(Int, parts[1])
                v = parse(Int, parts[2])
                w = parse(Float64, parts[3])
                add_edge!(graph, u, v)
                edge_weights[(min(u, v), max(u, v))] = w
            end
        end
    end
    nv(graph) > 0 || error("`.model` file $path is missing a `vertices:` line")
    J = Vector{T}(undef, ne(graph))
    for (k, e) in enumerate(edges(graph))
        J[k] = T(edge_weights[(min(src(e), dst(e)), max(src(e), dst(e)))])
    end
    return graph, J, header_meta
end

# ---------------------------------------------------------------------------
# Filename → (graph_type, h_default, subdir, model_name, meta) inference
# ---------------------------------------------------------------------------

# Pull a numeric value for `key` out of either the `.model` header or
# from a `key=<value>` chunk in the filename. Falls back to `default`.
function _infer_value(header_meta, basename_str, key, default; integer = false)
    if haskey(header_meta, key)
        v = header_meta[key]
        return integer ? Int(v) : Float64(v)
    end
    m = match(Regex("(?:^|_)$(key)=([^_]+)"), basename_str)
    if m !== nothing
        s = m.captures[1]
        if integer
            p = tryparse(Int, s)
            p !== nothing && return p
        else
            p = tryparse(Float64, s)
            p !== nothing && return p
        end
    end
    return default
end

"""
    _classify_model(path, header_meta)
        -> NamedTuple{(:family,:graph_type,:h_default,:subdir,
                       :model_name,:meta)}

Map a `.model` filename to the same `subdir` / `model_name` /
`graph_type` / `h_default` / `meta` the benchmark used to write its
slice dump. The recognised patterns are exhaustive over what
`spin_glass_model_generator.jl` produces; an unrecognised name falls
back to a generic dump under `<basename>_cheating` with `h=0`.
"""
function _classify_model(path::AbstractString, header_meta::AbstractDict)
    basename_str = splitext(basename(path))[1]

    # Family 1: spin_glass_J±1_grid_n=<n>_seed=<seed>
    m = match(r"^spin_glass_J±1_grid_n=(\d+)_seed=(\d+)$", basename_str)
    if m !== nothing
        n    = parse(Int, m.captures[1])
        seed = parse(Int, m.captures[2])
        return (
            family     = "j1pm1",
            graph_type = "grid_Jpm1",
            h_default  = 0.5,
            subdir     = "grid_Jpm1_n=$(n)_seed=$(seed)_cheating",
            model_name = "$(basename_str)_cheating",
            meta       = Dict{String,Any}("n" => n, "seed" => seed),
        )
    end

    # Family 2: spin_glass_J1J2_grid_L=<L>_J1pm1_J2pm1_seed=<seed>
    m = match(r"^spin_glass_J1J2_grid_L=(\d+)_J1pm1_J2pm1_seed=(\d+)$",
              basename_str)
    if m !== nothing
        L    = parse(Int, m.captures[1])
        seed = parse(Int, m.captures[2])
        return (
            family     = "j1j2",
            graph_type = "j1j2_grid_open",
            h_default  = 0.5,
            subdir     = "j1j2_grid_J1pm1_J2pm1_L=$(L)_seed=$(seed)_cheating",
            model_name = "$(basename_str)_cheating",
            meta       = Dict{String,Any}("L" => L, "g" => 1.0, "seed" => seed),
        )
    end

    # Family 3: spin_glass_J1J2_pbc_afm_grid_L=<L>_g=<gf>
    m = match(r"^spin_glass_J1J2_pbc_afm_grid_L=(\d+)_g=([\-0-9eE\.]+)$",
              basename_str)
    if m !== nothing
        L  = parse(Int, m.captures[1])
        gf = parse(Float64, m.captures[2])
        J1 = _infer_value(header_meta, basename_str, "J1", -1.0)
        J2 = _infer_value(header_meta, basename_str, "J2", J1 * gf)
        return (
            family     = "j1j2_pbc",
            graph_type = "j1j2_grid_pbc_afm",
            h_default  = 0.0,
            subdir     = "j1j2_pbc_afm_grid_L=$(L)_g=$(gf)_cheating",
            model_name = "$(basename_str)_cheating",
            meta       = Dict{String,Any}(
                "L"  => L,
                "g"  => gf,
                "J1" => J1,
                "J2" => J2,
            ),
        )
    end

    # Generic fallback: just slice it, no family-specific metadata.
    @warn "unrecognised `.model` filename pattern; using generic " *
          "(graph_type=`generic`, h=0). Pass --subdir / --graph-type " *
          "/ --h to override." path
    meta = Dict{String,Any}()
    for (k, v) in header_meta
        meta[k] = v
    end
    return (
        family     = "generic",
        graph_type = "generic",
        h_default  = 0.0,
        subdir     = "$(basename_str)_cheating",
        model_name = "$(basename_str)_cheating",
        meta       = meta,
    )
end

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

struct SliceConfig
    model_path::String
    sc_target::Int
    h_override::Union{Nothing,Float64}
    subdir_override::String
    graph_type_override::String
    use_lp::Bool
    ntrials::Int
    niters::Int
    code_seeds::UnitRange{Int}
    lp_time_limit::Float64
    verbose::Int
end

_parse_range(s::AbstractString) = begin
    if occursin(':', s)
        a, b = split(s, ':'; limit = 2)
        return parse(Int, a):parse(Int, b)
    else
        v = parse(Int, s)
        return v:v
    end
end

function _parse_args(args)
    model_path          = ""
    sc_target           = nothing
    h_override          = nothing
    subdir_override     = ""
    graph_type_override = ""
    use_lp              = true
    ntrials             = 50
    niters              = 100
    code_seeds          = 1:2
    lp_time_limit       = 300.0
    verbose             = 1

    for a in args
        if startswith(a, "--model=")
            model_path = String(split(a, "="; limit = 2)[2])
        elseif startswith(a, "--sc-target=")
            sc_target = parse(Int, split(a, "="; limit = 2)[2])
        elseif startswith(a, "--h=")
            h_override = parse(Float64, split(a, "="; limit = 2)[2])
        elseif startswith(a, "--subdir=")
            subdir_override = String(split(a, "="; limit = 2)[2])
        elseif startswith(a, "--graph-type=")
            graph_type_override = String(split(a, "="; limit = 2)[2])
        elseif a == "--no-lp"
            use_lp = false
        elseif startswith(a, "--ntrials=")
            ntrials = parse(Int, split(a, "="; limit = 2)[2])
        elseif startswith(a, "--niters=")
            niters = parse(Int, split(a, "="; limit = 2)[2])
        elseif startswith(a, "--code-seeds=")
            code_seeds = _parse_range(split(a, "="; limit = 2)[2])
        elseif startswith(a, "--lp-time-limit=")
            lp_time_limit = parse(Float64, split(a, "="; limit = 2)[2])
        elseif a == "--quiet"
            verbose = 0
        elseif a in ("-h", "--help")
            println(stderr, replace(string(@doc Main), "    " => "  "))
            exit(0)
        else
            error("unknown / unsupported argument: $a\n" *
                  "(tip: pass --help for usage)")
        end
    end

    isempty(model_path) &&
        error("--model=<path> is required (point at a `.model` file " *
              "produced by `models/spin_glass_model_generator.jl`)")
    sc_target === nothing &&
        error("--sc-target=<int> is required")
    isfile(model_path) ||
        error("--model file does not exist: $model_path")

    return SliceConfig(model_path, sc_target, h_override, subdir_override,
                       graph_type_override, use_lp, ntrials, niters,
                       code_seeds, lp_time_limit, verbose)
end

# ---------------------------------------------------------------------------
# Slicer driver
# ---------------------------------------------------------------------------

"""
    _best_initial_code(graph, ntrials, niters, code_seeds, verbose)

Run `initialize_code(graph, TreeSA(...))` for several RNG seeds and
return the candidate with the lowest `tc`. Mirrors the “best-of-K
TreeSA” loop in the benchmarks, so the same seeds (defaults `1:2`)
will pick the same code as the benchmarks do for the same inputs.
"""
function _best_initial_code(graph::SimpleGraph, ntrials::Int, niters::Int,
                            code_seeds::AbstractRange, verbose::Int)
    optimizer = TreeSA(ntrials = ntrials, niters = niters,
                       βs = 0.01:0.02:20.0)
    code     = nothing
    cc       = nothing
    best_tc  = Inf
    for code_seed in code_seeds
        Random.seed!(code_seed)
        cand_code = initialize_code(graph, optimizer)
        cand_cc   = contraction_complexity(cand_code, uniformsize(cand_code, 2))
        if verbose ≥ 1
            println("    code seed=$code_seed -> tc=$(cand_cc.tc), " *
                    "sc=$(cand_cc.sc), rw=$(cand_cc.rwc)")
        end
        if cand_cc.tc < best_tc
            best_tc = cand_cc.tc
            code = cand_code
            cc = cand_cc
        end
    end
    verbose ≥ 1 && println("  best code complexity over seeds " *
                           "$(collect(code_seeds)): $cc")
    return code, cc
end

"""
    main(args::AbstractVector{<:AbstractString}) -> NamedTuple

Parse `args`, load `(graph, J)` from the `.model` file in
`--model=<path>`, run `slice_dfs_lp` (or `slice_dfs` when `--no-lp` is
passed) on it, stream every finished slice to disk, and return a
`NamedTuple`:

    (slice_dir, subdir, root, slice_count, model_name, graph_type,
     branching_time, code_complexity)

`slice_dir` is the absolute path written under
`beyond_mis/branch_results/`; pass it (or `subdir`) to
`contract_spin_glass_slices.jl` next.
"""
function main(args)
    cfg = _parse_args(args)

    println("="^60)
    println("[slice_spin_glass] model=$(cfg.model_path)  sc_target=$(cfg.sc_target)")
    println("="^60)

    graph, J, header_meta = _read_spin_glass_model(cfg.model_path)
    info = _classify_model(cfg.model_path, header_meta)

    h_default  = cfg.h_override === nothing ? info.h_default : cfg.h_override
    subdir     = isempty(cfg.subdir_override)     ? info.subdir     : cfg.subdir_override
    graph_type = isempty(cfg.graph_type_override) ? info.graph_type : cfg.graph_type_override
    model_name = info.model_name

    h = fill(Float32(h_default), nv(graph))

    meta = Dict{String,Any}(
        "sc_target"  => cfg.sc_target,
        "h"          => Float64(h_default),
        "model_path" => abspath(cfg.model_path),
        "family"     => info.family,
    )
    for (k, v) in info.meta
        meta[k] = v
    end

    p = SpinGlassProblem(graph, J, h)

    println("  model_name : $model_name")
    println("  family     : $(info.family)")
    println("  graph_type : $graph_type")
    println("  vertices   : $(nv(graph))")
    println("  edges      : $(ne(graph))")
    println("  h_default  : $h_default")
    println("  meta       : $meta")

    code, cc = _best_initial_code(graph, cfg.ntrials, cfg.niters,
                                  cfg.code_seeds, cfg.verbose)

    slicer = ContractionTreeSlicer(
        sc_target       = cfg.sc_target,
        table_solver    = TensorNetworkSolver(),
        region_selector = ScoreRS(n_max = 10),
        brancher        = GreedyBrancher(),
    )

    mkpath(SLICE_RESULTS_ROOT)
    full_meta = merge(meta, Dict{String,Any}(
        "vertices" => nv(graph),
        "edges"    => ne(graph),
    ))
    println("  branch results root : $SLICE_RESULTS_ROOT")
    println("  slice subdir        : $subdir")

    # NB: the on-disk API in `spin_glass_slice_contract.jl` is the
    # batch-style `save_spin_glass_slices(subdir, slices; ...)`. We
    # collect the slices in memory first and dump them in one shot. The
    # `on_finished_slice` callback is still used for live progress
    # output but no longer touches the filesystem.
    t0 = time()
    finished_slices = if cfg.use_lp
        slice_dfs_lp(p, slicer, code, true, cfg.verbose;
            on_finished_slice = slice -> begin
                cc_s = complexity(slice)
                println("  [slice produced] sc=$(cc_s.sc) tc=$(cc_s.tc) " *
                        "nv=$(nv(slice.p.g)) ne=$(ne(slice.p.g)) r=$(slice.r)")
                flush(stdout)
            end)
    else
        slice_dfs(p, slicer, code, cfg.verbose)
    end
    branching_time = time() - t0

    println("  finished $(length(finished_slices)) slice(s) in " *
            "$(round(branching_time, digits = 3))s; persisting ...")
    slice_dir = save_spin_glass_slices(subdir, finished_slices;
        original       = (graph, J, h),
        model_name     = model_name,
        graph_type     = graph_type,
        overwrite      = true,
        meta           = full_meta,
        update_summary = true,
    )
    println("  saved $(length(finished_slices)) slice(s) to $slice_dir")
    println("  branching wall time (s): ", branching_time)

    return (
        slice_dir       = slice_dir,
        subdir          = subdir,
        root            = SLICE_RESULTS_ROOT,
        slice_count     = length(finished_slices),
        model_name      = model_name,
        graph_type      = graph_type,
        branching_time  = branching_time,
        code_complexity = cc,
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
