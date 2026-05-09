"""
    pure_slice_spin_glass.jl

Run standalone `TreeSASlicer` slicing for every spin-glass `.model` in a
directory and persist the resulting slices in the existing
`beyond_mis/branch_results/<subdir>/` format.

Unlike `scripts/_lib/slice_spin_glass.jl`, this script does not use the
branching slicer. It mirrors the standalone slicing block in
`hacked_funcs/benchmarks/spin_glass_ground_counting.jl`: build the raw
spin-glass EinCode, run `optimize_code(...; slicer=TreeSASlicer(...))`,
then translate the sliced index assignments into `(graph, J, h, r, code)`
records that the existing spin-glass contractor can load.

Usage:

    julia --project=beyond_mis beyond_mis/scripts/pure_slice_spin_glass.jl \\
        --models-dir=<dir> --sc-target=<int> [--save-mode=one|all]

Common options:

    --models-dir=<dir>          directory containing `.model` files
    --sc-target=<int>           target slicing space complexity
    --save-mode=one|all         save one representative slice or all slices
                                (default: one)
    --representative-assignment=<int>
                                assignment used when --save-mode=one
                                (default: 0)
    --subdir=<name>             force subdir; only valid for one model
    --subdir-template=<tmpl>    placeholders: {basename}, {L}, {g}, {seed},
                                {n}, {family}; default is pure_slicing_{basename}
    --h=<float>                 override inferred uniform field
    --seed=<int>                fallback RNG seed when the model filename/header
                                has no seed (default fallback: 2)
    --ntrials=<int>             TreeSA ntrials (default: package default)
    --niters=<int>              TreeSA niters (default: package default)
    --recursive                 search models recursively
    --overwrite=false           do not clear existing output directories
    --quiet                     reduce per-model logging

The default subdir is `pure_slicing_{basename}`.
"""

using Random
using Graphs, OMEinsum, OMEinsumContractionOrders
using OptimalBranching
using OptimalBranching.OptimalBranchingMIS
using OMEinsumContractionOrders: TreeSA, uniformsize
using TensorBranching: SlicedBranch
import TensorBranching

include(joinpath(@__DIR__, "_lib", "slice_spin_glass.jl"))

struct PureSliceConfig
    models_dir::String
    sc_target::Int
    save_mode::Symbol
    representative_assignment::UInt64
    subdir_override::String
    subdir_template::String
    h_override::Union{Nothing,Float64}
    seed::Union{Nothing,Int}
    ntrials::Union{Nothing,Int}
    niters::Union{Nothing,Int}
    recursive::Bool
    overwrite::Bool
    verbose::Int
end

function _parse_bool(s::AbstractString)
    v = lowercase(strip(s))
    v in ("1", "true", "yes", "y") && return true
    v in ("0", "false", "no", "n") && return false
    error("invalid boolean value: $s")
end

function parse_pure_slice_args(args)
    models_dir = ""
    sc_target = nothing
    save_mode = :one
    representative_assignment = UInt64(0)
    subdir_override = ""
    subdir_template = ""
    h_override = nothing
    seed = nothing
    ntrials = nothing
    niters = nothing
    recursive = false
    overwrite = true
    verbose = 1

    positional = String[]
    for a in args
        if startswith(a, "--models-dir=")
            models_dir = String(split(a, "="; limit = 2)[2])
        elseif startswith(a, "--sc-target=")
            sc_target = parse(Int, split(a, "="; limit = 2)[2])
        elseif startswith(a, "--save-mode=")
            mode = Symbol(lowercase(split(a, "="; limit = 2)[2]))
            mode in (:one, :all) || error("--save-mode must be `one` or `all`")
            save_mode = mode
        elseif startswith(a, "--representative-assignment=")
            representative_assignment = parse(UInt64, split(a, "="; limit = 2)[2])
        elseif startswith(a, "--subdir=")
            subdir_override = String(split(a, "="; limit = 2)[2])
        elseif startswith(a, "--subdir-template=")
            subdir_template = String(split(a, "="; limit = 2)[2])
        elseif startswith(a, "--h=")
            h_override = parse(Float64, split(a, "="; limit = 2)[2])
        elseif startswith(a, "--seed=")
            seed = parse(Int, split(a, "="; limit = 2)[2])
        elseif startswith(a, "--ntrials=")
            ntrials = parse(Int, split(a, "="; limit = 2)[2])
        elseif startswith(a, "--niters=")
            niters = parse(Int, split(a, "="; limit = 2)[2])
        elseif a == "--recursive"
            recursive = true
        elseif startswith(a, "--overwrite=")
            overwrite = _parse_bool(split(a, "="; limit = 2)[2])
        elseif a == "--quiet"
            verbose = 0
        elseif a in ("-h", "--help")
            println(stderr, replace(string(@doc Main), "    " => "  "))
            exit(0)
        elseif startswith(a, "--")
            error("unknown / unsupported argument: $a")
        else
            push!(positional, String(a))
        end
    end

    if isempty(models_dir) && length(positional) == 1
        models_dir = positional[1]
    elseif !isempty(positional)
        error("unexpected positional arguments: $(join(positional, ", "))")
    end

    isempty(models_dir) && error("--models-dir=<dir> is required")
    isdir(models_dir) || error("--models-dir does not exist or is not a directory: $models_dir")
    sc_target === nothing && error("--sc-target=<int> is required")

    return PureSliceConfig(models_dir, sc_target, save_mode,
        representative_assignment, subdir_override, subdir_template,
        h_override, seed, ntrials, niters, recursive, overwrite, verbose)
end

function _model_paths(dir::AbstractString; recursive::Bool = false)
    paths = String[]
    if recursive
        for (root, _, files) in walkdir(dir)
            for f in files
                endswith(f, ".model") && push!(paths, joinpath(root, f))
            end
        end
    else
        for f in readdir(dir)
            p = joinpath(dir, f)
            isfile(p) && endswith(f, ".model") && push!(paths, p)
        end
    end
    return sort!(paths)
end

function _spin_glass_eincode(g::SimpleGraph)
    edge_ixs = [[minmax(src(e), dst(e))...] for e in Graphs.edges(g)]
    vertex_ixs = [[v] for v in 1:nv(g)]
    return OMEinsumContractionOrders.EinCode([vcat(edge_ixs, vertex_ixs)...], Int[])
end

function _tree_sa(cfg::PureSliceConfig)
    kwargs = Pair{Symbol,Any}[]
    cfg.ntrials !== nothing && push!(kwargs, :ntrials => cfg.ntrials)
    cfg.niters !== nothing && push!(kwargs, :niters => cfg.niters)
    return TreeSA(; kwargs...)
end

function pure_sliced_code(g::SimpleGraph, sc_target::Int;
                          seed::Int = 2,
                          optimizer = TreeSA())
    code0 = _spin_glass_eincode(g)
    Random.seed!(seed)
    optcode_sliced = optimize_code(code0, uniformsize(code0, 2), optimizer;
        slicer = TreeSASlicer(score = ScoreFunction(sc_target = sc_target)))
    total_tc, sc = OMEinsum.timespace_complexity(optcode_sliced, uniformsize(code0, 2))
    return (code0 = code0,
            optcode = optcode_sliced,
            sliced_labels = sort!(collect(optcode_sliced.slicing)),
            total_tc = total_tc,
            sc = sc)
end

function _check_all_mode_size(k::Int)
    k <= 62 || error("--save-mode=all would require 2^$k slices, which does not fit UInt64")
    return UInt64(1) << k
end

function _assignment_values(k::Int, save_mode::Symbol, representative_assignment::UInt64)
    if save_mode === :all
        return UInt64(0):(_check_all_mode_size(k) - UInt64(1))
    end
    (k >= 64 || representative_assignment < (UInt64(1) << k)) ||
        error("--representative-assignment=$representative_assignment is outside 0:$(2^k - 1)")
    return representative_assignment:representative_assignment
end

function pure_tree_sa_slices(g::SimpleGraph, J::AbstractVector, h::AbstractVector,
                             sc_target::Int;
                             seed::Int = 2,
                             optimizer = TreeSA(),
                             save_mode::Symbol = :all,
                             representative_assignment::UInt64 = UInt64(0))
    sliced = pure_sliced_code(g, sc_target; seed = seed, optimizer = optimizer)
    removed_vertices = sliced.sliced_labels
    k = length(removed_vertices)
    assignments = _assignment_values(k, save_mode, representative_assignment)

    g_i, vmap_i = induced_subgraph(g, setdiff(1:nv(g), removed_vertices))
    # `TreeSASlicer` returns an OMEinsumContractionOrders tree. Convert it
    # to OMEinsum's DynamicNestedEinsum before reindexing through TensorBranching.
    base_code = OMEinsum.decorate(sliced.optcode.eins)
    code_i = nv(g_i) == 0 ? nothing : TensorBranching.update_code(g_i, base_code, vmap_i)

    branches = SlicedBranch[]
    for val in assignments
        J_i, h_i, r_i = TensorBranching.induced_spin_glass_subproblem(
            g, g_i, vmap_i, Vector(J), Vector(h), val, removed_vertices, removed_vertices)
        p_i = SpinGlassProblem(g_i, J_i, h_i)
        push!(branches, SlicedBranch(p_i, code_i, r_i))
    end

    return (slices = branches,
            sliced_labels = removed_vertices,
            all_slice_count = BigInt(1) << k,
            total_tc = sliced.total_tc,
            sc = sliced.sc,
            optcode = sliced.optcode)
end

function _compact_value(v)
    v isa Integer && return string(v)
    x = Float64(v)
    isinteger(x) && return string(Int(round(x)))
    s = string(x)
    return replace(s, "." => "p", "-" => "m")
end

function _template_value(info, model_path::AbstractString, key::AbstractString)
    key == "basename" && return splitext(basename(model_path))[1]
    key == "family" && return info.family
    return haskey(info.meta, key) ? _compact_value(info.meta[key]) : "NA"
end

function _apply_subdir_template(template::AbstractString, info, model_path::AbstractString)
    out = String(template)
    for key in ("basename", "family", "L", "g", "seed", "n")
        out = replace(out, "{$key}" => _template_value(info, model_path, key))
    end
    return out
end

function _default_pure_subdir(info, model_path::AbstractString)
    return "pure_slicing_$(splitext(basename(model_path))[1])"
end

function _resolve_subdir(cfg::PureSliceConfig, info, model_path::AbstractString, nmodels::Int)
    if !isempty(cfg.subdir_override)
        nmodels == 1 || error("--subdir can only be used when exactly one model is processed")
        return cfg.subdir_override
    elseif !isempty(cfg.subdir_template)
        return _apply_subdir_template(cfg.subdir_template, info, model_path)
    end
    return _default_pure_subdir(info, model_path)
end

function _resolve_slicing_seed(cfg::PureSliceConfig, info, model_path::AbstractString)
    if haskey(info.meta, "seed")
        return Int(info.meta["seed"])
    end
    cfg.seed !== nothing && return cfg.seed
    @warn "model filename/header does not expose `seed`; falling back to 2" model_path
    return 2
end

function _write_pure_slice_timing(slice_dir::AbstractString; model_path::AbstractString,
                                  slicing_time::Real, slicing_seed::Int,
                                  sc_target::Int, save_mode::Symbol,
                                  saved_slice_count::Int, all_slice_count)
    open(joinpath(slice_dir, "pure_slice_timing.txt"), "w") do io
        println(io, "model_path        = ", abspath(model_path))
        println(io, "slicing_time_s    = ", Float64(slicing_time))
        println(io, "slicing_seed      = ", slicing_seed)
        println(io, "sc_target         = ", sc_target)
        println(io, "save_mode         = ", save_mode)
        println(io, "saved_slice_count = ", saved_slice_count)
        println(io, "all_slice_count   = ", all_slice_count)
    end
    return nothing
end

function process_model(model_path::AbstractString, cfg::PureSliceConfig; nmodels::Int = 1)
    graph, J, header_meta = _read_spin_glass_model(model_path)
    info = _classify_model(model_path, header_meta)
    h_default = cfg.h_override === nothing ? info.h_default : cfg.h_override
    h = fill(Float32(h_default), nv(graph))
    subdir = _resolve_subdir(cfg, info, model_path, nmodels)
    slicing_seed = _resolve_slicing_seed(cfg, info, model_path)

    cfg.verbose >= 1 && println("\n[pure_slice] model=$model_path")
    cfg.verbose >= 1 && println("  subdir      : $subdir")
    cfg.verbose >= 1 && println("  vertices    : $(nv(graph))")
    cfg.verbose >= 1 && println("  edges       : $(ne(graph))")
    cfg.verbose >= 1 && println("  save_mode   : $(cfg.save_mode)")
    cfg.verbose >= 1 && println("  seed        : $slicing_seed")

    t0 = time()
    result = pure_tree_sa_slices(graph, J, h, cfg.sc_target;
        seed = slicing_seed,
        optimizer = _tree_sa(cfg),
        save_mode = cfg.save_mode,
        representative_assignment = cfg.representative_assignment)
    slicing_time = time() - t0

    meta = Dict{String,Any}(
        "family"                    => info.family,
        "model_path"                => abspath(model_path),
        "h"                         => Float64(h_default),
        "sc_target"                 => cfg.sc_target,
        "pure_tree_sa_slicing"       => true,
        "sliced_labels"             => join(result.sliced_labels, ","),
        "all_slice_count"           => string(result.all_slice_count),
        "saved_slice_count"         => length(result.slices),
        "save_mode"                 => string(cfg.save_mode),
        "slicing_seed"              => slicing_seed,
        "representative_assignment" => string(cfg.representative_assignment),
        "total_tc_slicing"          => Float64(result.total_tc),
        "sc_slicing"                => Float64(result.sc),
        "slicing_time"              => slicing_time,
        "vertices"                  => nv(graph),
        "edges"                     => ne(graph),
    )
    for (k, v) in info.meta
        meta[k] = v
    end

    slice_dir = save_spin_glass_slices(subdir, result.slices;
        original = (graph, J, h),
        model_name = info.model_name,
        graph_type = info.graph_type,
        overwrite = cfg.overwrite,
        meta = meta,
        update_summary = true,
        summary_filename = "pure_slicing_summary.csv")
    _write_pure_slice_timing(slice_dir;
        model_path = model_path,
        slicing_time = slicing_time,
        slicing_seed = slicing_seed,
        sc_target = cfg.sc_target,
        save_mode = cfg.save_mode,
        saved_slice_count = length(result.slices),
        all_slice_count = result.all_slice_count)

    cfg.verbose >= 1 && println("  sliced labels : $(result.sliced_labels)")
    cfg.verbose >= 1 && println("  all slices    : $(result.all_slice_count)")
    cfg.verbose >= 1 && println("  saved slices  : $(length(result.slices))")
    cfg.verbose >= 1 && println("  total tc      : $(result.total_tc)")
    cfg.verbose >= 1 && println("  sc            : $(result.sc)")
    cfg.verbose >= 1 && println("  time (s)      : $slicing_time")
    cfg.verbose >= 1 && println("  timing file   : $(joinpath(slice_dir, "pure_slice_timing.txt"))")
    cfg.verbose >= 1 && println("  saved to      : $slice_dir")

    return (model_path = model_path,
            slice_dir = slice_dir,
            subdir = subdir,
            saved_slice_count = length(result.slices),
            all_slice_count = result.all_slice_count,
            sliced_labels = result.sliced_labels,
            total_tc = result.total_tc,
            sc = result.sc,
            slicing_time = slicing_time,
            slicing_seed = slicing_seed)
end

function main(args)
    cfg = parse_pure_slice_args(args)
    paths = _model_paths(cfg.models_dir; recursive = cfg.recursive)
    isempty(paths) && error("no `.model` files found in $(cfg.models_dir)")

    println("[pure_slice] models_dir=$(cfg.models_dir)")
    println("[pure_slice] models=$(length(paths)) sc_target=$(cfg.sc_target) save_mode=$(cfg.save_mode)")

    results = map(p -> process_model(p, cfg; nmodels = length(paths)), paths)
    println("\n[pure_slice] completed $(length(results)) model(s)")
    for r in results
        println("  $(basename(r.model_path)) -> $(r.slice_dir)")
    end
    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
