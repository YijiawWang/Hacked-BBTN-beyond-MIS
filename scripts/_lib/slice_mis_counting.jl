"""
    slice_mis_counting.jl

Parameterized driver for the **MIS total-counting** branching benchmark
(`beyond_mis/hacked_funcs/benchmarks/mis_counting.jl`). It can either

  * **read** a pre-generated graph from disk via `--model=<path>`
    (the format produced by
    `hacked_funcs/benchmarks/models/mis_model_generator.jl`), or

  * **regenerate** a random KSG instance from `(n, density, seed)` —
    the legacy behaviour, kept for backward compatibility.

After the graph is in hand the script builds the initial contraction
code via `TreeSA` and runs `slice_bfs` (from
`hacked_funcs/src/mis_counting.jl`) to decompose the contraction tree
into slices that fit a target space-complexity budget. The slices form
a **disjoint** sub-tree of the original IS enumeration, so total IS
count = sum of per-slice CountingAll contributions (no `r`-shift, no
max-filtering).

Every finished slice is written into
`beyond_mis/branch_results/<subdir>/` using the streaming writer from
`beyond_mis/contractors/mis_slice_contract.jl`, so the dump is
immediately consumable by `contract_mis_counting_slices.jl`.

Vertex weights are irrelevant to total IS counting (the constraint is
purely topological), so this driver always uses unit weights.

The recognised `.graph` filename pattern is

    random_ksg_n=<n>_seed=<seed>.graph
        -> graph_type = `random_ksg`,
           subdir     = `mis_counting_random_ksg_n=<n>_seed=<seed>`

Files that don't match are sliced with a generic graph_type and a
`<basename>` subdir suffix; pass `--subdir` / `--graph-type` to
override.

Usage:

    julia --project=beyond_mis beyond_mis/scripts/_lib/slice_mis_counting.jl \\
        --sc-target=<int>        # required
        (--model=<path> | --n=<int>)
        [--density=<float>]      # KSG diagonal-coupled density (default 0.8,
                                 #   only used when --model is not given)
        [--seed=<int>]           # graph generation seed (default 1, only
                                 #   used when --model is not given)
        [--code-seed=<int>]      # TreeSA seed (default = --seed)
        [--subdir=<name>]        # override the auto-generated slice subdir
        [--graph-type=<name>]    # override the auto-inferred graph_type tag
        [--ntrials=<int>]        # TreeSA ntrials (default 50)
        [--niters=<int>]         # TreeSA niters  (default 100)
        [--quiet]                # set verbose=0 (default verbose=1)

Prints (and `main` returns) the absolute slice directory written under
`beyond_mis/branch_results/`; `contract_mis_counting_slices.jl`
consumes that path next.
"""

using Random
using Graphs, OMEinsum, OMEinsumContractionOrders
using OptimalBranching
using OptimalBranching.OptimalBranchingCore, OptimalBranching.OptimalBranchingMIS
using OptimalBranchingMIS: TensorNetworkSolver
using OMEinsumContractionOrders: TreeSA, uniformsize
using TensorBranching: ContractionTreeSlicer, GreedyBrancher, ScoreRS,
                       SlicedBranch, complexity, initialize_code
using GenericTensorNetworks: random_diagonal_coupled_graph

# Pull in the benchmark layer (which itself includes
# `hacked_funcs/src/mis_counting.jl` and the
# `contractors/mis_slice_contract.jl` writer/reader). The benchmark
# file's `main()` is guarded by
# `if abspath(PROGRAM_FILE) == @__FILE__`, so this include only
# defines functions; the benchmark's `for graph_file in graph_files`
# loop is not executed here. Driver-side `main(args)` (defined
# further below) takes a different signature, so the two coexist
# cleanly.
include(joinpath(@__DIR__, "..", "..", "hacked_funcs", "benchmarks",
                 "mis_counting.jl"))
include(joinpath(@__DIR__, "mis_graph_io.jl"))

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

struct MISCountSliceConfig
    model_path::String              # "" if no model (regenerate path)
    n::Int                          # 0 when --model is set and --n omitted
    sc_target::Int
    density::Float64
    seed::Int                       # only used if model_path == ""
    code_seed::Int
    subdir_override::String
    graph_type_override::String
    ntrials::Int
    niters::Int
    verbose::Int
end

function _parse_args(args)
    model_path          = ""
    n                   = nothing
    sc_target           = nothing
    density             = 0.8
    seed                = 1
    code_seed           = nothing
    subdir_override     = ""
    graph_type_override = ""
    ntrials             = 50
    niters              = 100
    verbose             = 1

    for a in args
        if startswith(a, "--model=")
            model_path = String(split(a, "="; limit = 2)[2])
        elseif startswith(a, "--n=")
            n = parse(Int, split(a, "="; limit = 2)[2])
        elseif startswith(a, "--sc-target=")
            sc_target = parse(Int, split(a, "="; limit = 2)[2])
        elseif startswith(a, "--density=")
            density = parse(Float64, split(a, "="; limit = 2)[2])
        elseif startswith(a, "--seed=")
            seed = parse(Int, split(a, "="; limit = 2)[2])
        elseif startswith(a, "--code-seed=")
            code_seed = parse(Int, split(a, "="; limit = 2)[2])
        elseif startswith(a, "--subdir=")
            subdir_override = String(split(a, "="; limit = 2)[2])
        elseif startswith(a, "--graph-type=")
            graph_type_override = String(split(a, "="; limit = 2)[2])
        elseif startswith(a, "--ntrials=")
            ntrials = parse(Int, split(a, "="; limit = 2)[2])
        elseif startswith(a, "--niters=")
            niters = parse(Int, split(a, "="; limit = 2)[2])
        elseif a == "--quiet"
            verbose = 0
        elseif a in ("-h", "--help")
            println(stderr, replace(string(@doc Main), "    " => "  "))
            exit(0)
        else
            error("unknown / unsupported argument: $a (pass --help for usage)")
        end
    end

    sc_target === nothing && error("--sc-target=<int> is required")

    if isempty(model_path)
        n === nothing &&
            error("--n=<int> is required when --model is not provided")
    else
        isfile(model_path) ||
            error("--model file does not exist: $model_path")
    end
    code_seed === nothing && (code_seed = seed)

    return MISCountSliceConfig(model_path, something(n, 0), sc_target,
                                density, seed, code_seed, subdir_override,
                                graph_type_override, ntrials, niters,
                                verbose)
end

# ---------------------------------------------------------------------------
# Problem builder (read `.graph` file or regenerate random KSG)
# ---------------------------------------------------------------------------

function _build_problem(cfg::MISCountSliceConfig)
    if !isempty(cfg.model_path)
        graph, header_meta = _read_mis_graph(cfg.model_path)
        info = _classify_mis_model(cfg.model_path, header_meta)
        graph_type = isempty(cfg.graph_type_override) ?
                     info.graph_type : cfg.graph_type_override
        graph_name = info.graph_name
        base_id    = info.base_id
        family     = info.family
        meta = Dict{String,Any}(
            "model_path" => abspath(cfg.model_path),
            "family"     => family,
        )
        for (k, v) in info.meta
            meta[k] = v
        end
        for (k, v) in header_meta
            haskey(meta, k) || (meta[k] = v)
        end
    else
        Random.seed!(cfg.seed)
        g_raw = random_diagonal_coupled_graph(cfg.n, cfg.n, cfg.density)
        graph = SimpleGraph(g_raw)
        graph_type = isempty(cfg.graph_type_override) ?
                     "random_ksg" : cfg.graph_type_override
        graph_name = "random_ksg_n=$(cfg.n)_seed=$(cfg.seed).graph"
        base_id    = "random_ksg_n=$(cfg.n)_seed=$(cfg.seed)"
        family     = "random_ksg"
        meta = Dict{String,Any}(
            "n"       => cfg.n,
            "density" => cfg.density,
            "seed"    => cfg.seed,
            "family"  => family,
        )
    end

    weights = ones(Float32, nv(graph))

    meta["weights_kind"] = "unit"
    meta["sc_target"]    = cfg.sc_target
    meta["vertices"]     = nv(graph)
    meta["edges"]        = ne(graph)

    p = MISProblem(graph, weights)
    return p, graph, weights, meta, graph_type, graph_name, base_id, family
end

# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

function main(args)
    cfg = _parse_args(args)

    println("="^60)
    if !isempty(cfg.model_path)
        println("[slice_mis_counting] model=$(cfg.model_path)  " *
                "sc_target=$(cfg.sc_target)")
    else
        println("[slice_mis_counting] regenerate random KSG n=$(cfg.n) " *
                "seed=$(cfg.seed) density=$(cfg.density)  " *
                "sc_target=$(cfg.sc_target)")
    end
    println("="^60)

    p, graph, weights, meta, graph_type, graph_name, base_id, family =
        _build_problem(cfg)

    subdir = isempty(cfg.subdir_override) ?
             "mis_counting_$(base_id)" :
             cfg.subdir_override

    println("  graph_name : $graph_name")
    println("  graph_type : $graph_type")
    println("  family     : $family")
    println("  vertices   : $(nv(graph))")
    println("  edges      : $(ne(graph))")
    println("  meta       : $meta")

    Random.seed!(cfg.code_seed)
    optimizer = TreeSA(ntrials = cfg.ntrials, niters = cfg.niters,
                       βs = 0.01:0.02:20.0)
    code = initialize_code(graph, optimizer)
    cc = contraction_complexity(code, uniformsize(code, 2))
    println("  initial code complexity: ", cc)

    slicer = ContractionTreeSlicer(
        sc_target       = cfg.sc_target,
        table_solver    = TensorNetworkSolver(),
        region_selector = ScoreRS(n_max = 10),
        brancher        = GreedyBrancher(),
    )

    mkpath(SLICE_RESULTS_ROOT)
    println("  branch results root : $SLICE_RESULTS_ROOT")
    println("  slice subdir        : $subdir")

    writer = init_mis_slice_writer(subdir;
        original   = (graph, weights),
        graph_name = graph_name,
        graph_type = graph_type,
        overwrite  = true,
        meta       = meta)
    println("  streaming slices to $(writer.dirname)")

    t0 = time()
    finished_slices = slice_bfs(p, slicer, code, cfg.verbose;
        on_finished_slice = slice -> begin
            sid = append_mis_slice!(writer, slice; flush_summary = true)
            cc_s = complexity(slice)
            println("  [slice $sid saved] sc=$(cc_s.sc) tc=$(cc_s.tc) " *
                    "nv=$(nv(slice.p.g)) ne=$(ne(slice.p.g)) r=$(slice.r) " *
                    "(total saved: $(length(writer.ids)))")
            flush(stdout)
        end)
    branching_time = time() - t0
    slice_dir = finalize_mis_slice_writer!(writer)

    println("  finished $(length(finished_slices)) slice(s) in " *
            "$(round(branching_time, digits = 3))s")
    println("  saved slice dump to $slice_dir")

    return (
        slice_dir       = slice_dir,
        subdir          = subdir,
        root            = SLICE_RESULTS_ROOT,
        slice_count     = length(finished_slices),
        graph_name      = graph_name,
        graph_type      = graph_type,
        branching_time  = branching_time,
        code_complexity = cc,
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
