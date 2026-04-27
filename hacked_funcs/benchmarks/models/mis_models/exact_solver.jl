"""
    exact_solver.jl  (mis_models)

Stand-alone exact solver for a single `*.graph` file passed on the
command line (the format produced by `../mis_model_generator.jl`).

Wraps the graph in a `GenericTensorNetwork(IndependentSet(g, w))` with
`w = ones(Int, nv(g))` and solves three properties exactly via
`GenericTensorNetworks.solve`:

  * `SizeMax()`     — the maximum-IS size (cardinality, since w = 1).
  * `CountingMax()` — the **ground counting**: number of maximum IS.
  * `CountingAll()` — the **total counting**: number of *every* IS.

`CountingMax` / `CountingAll` results are converted to exact `BigInt`
(matching `beyond_mis/scripts/_lib/verify_mis*.jl`); `solve(...,
CountingAll())` already runs an internal `big_integer_solve` CRT loop,
so the count is exact even for very large graphs.

Results are appended to `exact_solver_results.csv` next to this script
(or to the file given by `--out=`). Counts are stored as strings to
preserve `BigInt` precision.

Usage (from this directory):

    julia --project=../../../.. exact_solver.jl \\
        --graph=<path/to/file.graph> \\
        [--out=<path/to/results.csv>] \\
        [--seed=<int>]                # default 1, controls TreeSA path search

`--graph=` accepts either an absolute path or a name relative to this
directory.
"""

using Graphs
using GenericTensorNetworks
using ProblemReductions: IndependentSet
using OMEinsumContractionOrders: TreeSA
using Random
using CSV, DataFrames
using Printf


# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------

const _USAGE = """
Usage:

    julia exact_solver.jl --graph=<path/to/file.graph>
        [--out=<path/to/results.csv>]   # default: ./exact_solver_results.csv
        [--seed=<int>]                  # default: 1
"""

function _parse_args(args)
    graph_path = ""
    out_path   = ""
    seed       = 1
    for a in args
        if startswith(a, "--graph=")
            graph_path = String(split(a, "="; limit = 2)[2])
        elseif startswith(a, "--out=")
            out_path = String(split(a, "="; limit = 2)[2])
        elseif startswith(a, "--seed=")
            seed = parse(Int, split(a, "="; limit = 2)[2])
        elseif a in ("-h", "--help")
            println(_USAGE)
            exit(0)
        else
            error("unknown / unsupported argument: $a\n$_USAGE")
        end
    end
    isempty(graph_path) && error("--graph=<path> is required\n$_USAGE")
    return (; graph_path, out_path, seed)
end


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

"""
    read_mis_graph(path) -> SimpleGraph

Read a `.graph` file written by `../mis_model_generator.jl`. The file
format is

    # <free-form comments / metadata>
    n = ...
    seed = ...
    type = ...

    vertices: <nv>
    edges:
    u1 v1
    u2 v2
    ...
"""
function read_mis_graph(path::AbstractString)
    graph = SimpleGraph()
    open(path, "r") do io
        while !eof(io)
            line = readline(io)
            if startswith(line, "vertices: ")
                n_vertices = parse(Int, split(line)[2])
                graph = SimpleGraph(n_vertices)
            elseif strip(line) == "edges:"
                while !eof(io)
                    line = readline(io)
                    isempty(strip(line)) && continue
                    parts = split(line)
                    if length(parts) >= 2
                        u = parse(Int, parts[1])
                        v = parse(Int, parts[2])
                        add_edge!(graph, u, v)
                    end
                end
                break
            end
        end
    end
    return graph
end

# Coerce whatever element type `solve(... CountingMax/All ...)` happened to
# return into an exact `BigInt`. Mirrors `_to_bigint` in
# `beyond_mis/contractors/spin_glass_slice_contract.jl`.
_to_bigint(x::Integer)        = BigInt(x)
_to_bigint(x::AbstractFloat)  = isfinite(x) ? BigInt(round(x)) :
    error("non-finite count $(x); use a wider count eltype")


# ---------------------------------------------------------------------------
# Exact solver (single graph)
# ---------------------------------------------------------------------------

"""
    solve_exact(graph; optimizer=TreeSA()) -> NamedTuple

Solve `SizeMax`, `CountingMax`, `CountingAll` on the unweighted IS
problem defined by `graph` and return their numeric values together
with their wall-clock runtimes.
"""
function solve_exact(graph::SimpleGraph; optimizer = TreeSA())
    weights = ones(Int, nv(graph))

    # CountingMax / SizeMax need the weights (we use w = 1 → cardinality).
    problem_w = GenericTensorNetwork(IndependentSet(graph, weights);
                                     optimizer = optimizer)

    # CountingAll is weight-independent; using `IndependentSet(g)` here
    # avoids `Mod{Int32}^Float` issues inside the internal CRT loop
    # (see `beyond_mis/scripts/_lib/verify_mis_counting.jl`).
    problem_uw = GenericTensorNetwork(IndependentSet(graph);
                                      optimizer = optimizer)

    t_size = @elapsed begin
        sm = Array(solve(problem_w, SizeMax()))[]
    end
    mis_size = Float64(sm.n)

    t_gc = @elapsed begin
        cm = Array(solve(problem_w, CountingMax()))[]
    end
    ground_count = _to_bigint(cm.c)

    t_all = @elapsed begin
        ca = Array(solve(problem_uw, CountingAll()))[]
    end
    all_count = _to_bigint(ca)

    return (
        mis_size      = mis_size,
        ground_count  = ground_count,
        all_count     = all_count,
        size_runtime  = t_size,
        gc_runtime    = t_gc,
        all_runtime   = t_all,
    )
end


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

const _SCHEMA = DataFrame(
    model_name        = String[],
    vertices          = Int[],
    edges             = Int[],
    mis_size          = Float64[],
    ground_count      = String[],
    all_count         = String[],
    size_runtime_s    = Float64[],
    counting_max_s    = Float64[],
    counting_all_s    = Float64[],
    total_runtime_s   = Float64[],
)

function _resolve_graph_path(p::AbstractString)
    isabspath(p) && return p
    isfile(p)    && return abspath(p)
    here = abspath(joinpath(@__DIR__, p))
    isfile(here) && return here
    error("graph file not found: $p (also tried $here)")
end

function main(args = ARGS)
    cfg         = _parse_args(args)
    graph_path  = _resolve_graph_path(cfg.graph_path)
    output_file = isempty(cfg.out_path) ?
        joinpath(@__DIR__, "exact_solver_results.csv") :
        cfg.out_path
    isfile(output_file) || CSV.write(output_file, _SCHEMA)

    println("="^72)
    println("[exact_solver mis]")
    println("  graph  : $graph_path")
    println("  output : $output_file")
    println("  seed   : $(cfg.seed)")
    println("="^72)

    graph = read_mis_graph(graph_path)
    println("  vertices: $(nv(graph))")
    println("  edges:    $(ne(graph))")

    Random.seed!(cfg.seed)        # deterministic TreeSA path search

    t_total = @elapsed r = solve_exact(graph)

    @printf("  MIS size                  : %g\n",     r.mis_size)
    println("  ground count (CountingMax): ", r.ground_count)
    println("  all count    (CountingAll): ", r.all_count)
    @printf("  SizeMax runtime           : %.4f s\n", r.size_runtime)
    @printf("  CountingMax runtime       : %.4f s\n", r.gc_runtime)
    @printf("  CountingAll runtime       : %.4f s\n", r.all_runtime)
    @printf("  total runtime             : %.4f s\n", t_total)

    row = DataFrame(
        model_name      = [basename(graph_path)],
        vertices        = [nv(graph)],
        edges           = [ne(graph)],
        mis_size        = [r.mis_size],
        ground_count    = [string(r.ground_count)],
        all_count       = [string(r.all_count)],
        size_runtime_s  = [r.size_runtime],
        counting_max_s  = [r.gc_runtime],
        counting_all_s  = [r.all_runtime],
        total_runtime_s = [t_total],
    )
    CSV.write(output_file, row, append = true)

    println("\n[exact_solver mis] done. Results -> $output_file")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
