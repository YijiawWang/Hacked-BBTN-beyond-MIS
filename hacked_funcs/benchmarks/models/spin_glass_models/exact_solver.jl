"""
    exact_solver.jl  (spin_glass_models)

Stand-alone exact ground-state + ground-counting solver for a single
`*.model` file passed on the command line (the format produced by
`../spin_glass_model_generator.jl`).

Loads `(graph, J)` from the model, takes a uniform external field `h`
from `--h=<float>` (the field convention is the same as in
`spin_glass_ground_counting.jl`: the optimizer maximises
`Σ J_ij s_i s_j + Σ h_i s_i`), wraps the problem in
`GenericTensorNetwork(SpinGlass(g, J, h))` and solves two properties
exactly via `GenericTensorNetworks.solve`:

  * `SizeMax()`     — the ground-state energy.
  * `CountingMax()` — the **ground counting**: number of optimal spin
                       configurations.

Counts are converted to exact `BigInt` (matching
`beyond_mis/scripts/_lib/verify_spin_glass.jl`).

Results are appended to `exact_solver_results.csv` next to this script
(or to the file given by `--out=`). Counts are stored as strings to
preserve `BigInt` precision.

Usage (from this directory):

    julia --project=../../../.. exact_solver.jl \\
        --model=<path/to/file.model> \\
        --h=<float>                        # uniform external field
        [--out=<path/to/results.csv>] \\
        [--seed=<int>]                     # default 1, controls TreeSA path search

`--model=` accepts either an absolute path or a name relative to this
directory.
"""

using Graphs
using GenericTensorNetworks
using ProblemReductions: SpinGlass
using OMEinsumContractionOrders: TreeSA
using Random
using CSV, DataFrames
using Printf


# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------

const _USAGE = """
Usage:

    julia exact_solver.jl --model=<path/to/file.model> --h=<float>
        [--out=<path/to/results.csv>]   # default: ./exact_solver_results.csv
        [--seed=<int>]                  # default: 1
"""

function _parse_args(args)
    model_path = ""
    out_path   = ""
    h_val      = nothing
    seed       = 1
    for a in args
        if startswith(a, "--model=")
            model_path = String(split(a, "="; limit = 2)[2])
        elseif startswith(a, "--out=")
            out_path = String(split(a, "="; limit = 2)[2])
        elseif startswith(a, "--h=")
            h_val = parse(Float64, split(a, "="; limit = 2)[2])
        elseif startswith(a, "--seed=")
            seed = parse(Int, split(a, "="; limit = 2)[2])
        elseif a in ("-h", "--help")
            println(_USAGE)
            exit(0)
        else
            error("unknown / unsupported argument: $a\n$_USAGE")
        end
    end
    isempty(model_path) && error("--model=<path> is required\n$_USAGE")
    h_val === nothing   && error("--h=<float> is required\n$_USAGE")
    return (; model_path, out_path, h_val, seed)
end


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

"""
    read_spin_glass_model(path) -> (graph, edge_weights_vec)

Read a `.model` file written by `../spin_glass_model_generator.jl`. The
returned `edge_weights_vec` is `Float32`-typed and ordered to match
`edges(graph)`. Same parser as
`beyond_mis/hacked_funcs/benchmarks/spin_glass_ground_counting.jl`.
"""
function read_spin_glass_model(path::AbstractString)
    graph = SimpleGraph()
    edge_weights = Dict{Tuple{Int,Int}, Float64}()
    open(path, "r") do io
        while !eof(io)
            line = readline(io)
            if startswith(line, "vertices: ")
                n_vertices = parse(Int, split(line)[2])
                graph = SimpleGraph(n_vertices)
            elseif strip(line) == "edges_with_weights:"
                while !eof(io)
                    line = readline(io)
                    isempty(strip(line)) && continue
                    parts = split(line)
                    if length(parts) >= 3
                        u = parse(Int, parts[1])
                        v = parse(Int, parts[2])
                        w = parse(Float64, parts[3])
                        add_edge!(graph, u, v)
                        edge_weights[(min(u, v), max(u, v))] = w
                    end
                end
                break
            end
        end
    end
    edge_weights_vec = Vector{Float32}(undef, ne(graph))
    for (k, e) in enumerate(edges(graph))
        edge_weights_vec[k] = Float32(
            edge_weights[(min(src(e), dst(e)), max(src(e), dst(e)))])
    end
    return graph, edge_weights_vec
end

_to_bigint(x::Integer)       = BigInt(x)
_to_bigint(x::AbstractFloat) = isfinite(x) ? BigInt(round(x)) :
    error("non-finite count $(x); use a wider count eltype")


# ---------------------------------------------------------------------------
# Exact solver (single model)
# ---------------------------------------------------------------------------

"""
    solve_exact(graph, J, h; optimizer=TreeSA()) -> NamedTuple

Solve `SizeMax` and `CountingMax` on `SpinGlass(g, J, h)` and return
their numeric values together with their wall-clock runtimes.
"""
function solve_exact(graph::SimpleGraph,
                     J::AbstractVector,
                     h::AbstractVector;
                     optimizer = TreeSA())
    problem = GenericTensorNetwork(
        SpinGlass(graph, collect(J), collect(h));
        optimizer = optimizer)

    t_size = @elapsed begin
        sm = Array(solve(problem, SizeMax()))[]
    end
    energy = Float64(sm.n)

    t_gc = @elapsed begin
        cm = Array(solve(problem, CountingMax()))[]
    end
    ground_count = _to_bigint(cm.c)

    return (
        energy        = energy,
        ground_count  = ground_count,
        size_runtime  = t_size,
        gc_runtime    = t_gc,
    )
end


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

const _SCHEMA = DataFrame(
    model_name        = String[],
    vertices          = Int[],
    edges             = Int[],
    h_uniform         = Float64[],
    ground_energy     = Float64[],
    ground_count      = String[],
    size_runtime_s    = Float64[],
    counting_max_s    = Float64[],
    total_runtime_s   = Float64[],
)

function _resolve_model_path(p::AbstractString)
    isabspath(p) && return p
    isfile(p)    && return abspath(p)
    here = abspath(joinpath(@__DIR__, p))
    isfile(here) && return here
    error("model file not found: $p (also tried $here)")
end

function main(args = ARGS)
    cfg         = _parse_args(args)
    model_path  = _resolve_model_path(cfg.model_path)
    output_file = isempty(cfg.out_path) ?
        joinpath(@__DIR__, "exact_solver_results.csv") :
        cfg.out_path
    isfile(output_file) || CSV.write(output_file, _SCHEMA)

    println("="^72)
    println("[exact_solver spin_glass]")
    println("  model  : $model_path")
    println("  output : $output_file")
    @printf("  h      : %g (uniform)\n", cfg.h_val)
    println("  seed   : $(cfg.seed)")
    println("="^72)

    graph, edge_weights_vec = read_spin_glass_model(model_path)
    h = fill(Float32(cfg.h_val), nv(graph))
    println("  vertices: $(nv(graph))")
    println("  edges:    $(ne(graph))")

    Random.seed!(cfg.seed)        # deterministic TreeSA path search

    t_total = @elapsed r = solve_exact(graph, edge_weights_vec, h)

    @printf("  ground energy             : %g\n",     r.energy)
    println("  ground count (CountingMax): ", r.ground_count)
    @printf("  SizeMax runtime           : %.4f s\n", r.size_runtime)
    @printf("  CountingMax runtime       : %.4f s\n", r.gc_runtime)
    @printf("  total runtime             : %.4f s\n", t_total)

    row = DataFrame(
        model_name      = [basename(model_path)],
        vertices        = [nv(graph)],
        edges           = [ne(graph)],
        h_uniform       = [Float64(cfg.h_val)],
        ground_energy   = [r.energy],
        ground_count    = [string(r.ground_count)],
        size_runtime_s  = [r.size_runtime],
        counting_max_s  = [r.gc_runtime],
        total_runtime_s = [t_total],
    )
    CSV.write(output_file, row, append = true)

    println("\n[exact_solver spin_glass] done. Results -> $output_file")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
