"""
    mis_graph_io.jl

I/O helpers for MIS graph files (the on-disk format produced by
`hacked_funcs/benchmarks/models/mis_model_generator.jl`). Used by
`slice_mis.jl` / `slice_mis_counting.jl` so the slicer can be pointed
at a *pre-generated* graph instead of regenerating it from
`(n, density, seed)`.

The reader is intentionally permissive: it accepts both the lean
`.graph` shape that `mis_model_generator.jl` writes today

    # Random KSG graph
    n = 20
    seed = 1
    type = random_ksg

    vertices: 320
    edges:
    1 2
    1 17
    ...

and the richer header / `edges_with_weights:` shape used by the
spin-glass `.model` files (the third weight column is ignored — MIS
problems don't carry edge weights). This way an ad-hoc `.model` /
`.graph` file from any of the generators in
`hacked_funcs/benchmarks/models/` can be sliced as MIS.
"""

using Graphs

"""
    _read_mis_graph(path)
        -> (graph::SimpleGraph, header_meta::Dict{String,Any})

Parse a MIS graph file. Header is everything above the `edges:` (or
`edges_with_weights:`) section start; `key = value` lines are folded
into `header_meta`. The edge block accepts either `u v` lines or
`u v w` lines (the third column is ignored).
"""
function _read_mis_graph(path::AbstractString)
    graph = SimpleGraph()
    header_meta = Dict{String,Any}()
    open(path, "r") do io
        in_edges = false
        while !eof(io)
            line = readline(io)
            if !in_edges
                stripped = strip(line)
                isempty(stripped) && continue
                if startswith(stripped, "#")
                    continue
                elseif startswith(stripped, "vertices:")
                    parts = split(stripped)
                    if length(parts) >= 2
                        n_vertices = parse(Int, parts[2])
                        graph = SimpleGraph(n_vertices)
                    end
                elseif stripped == "edges:" || stripped == "edges_with_weights:"
                    in_edges = true
                elseif startswith(stripped, "edges:") ||
                       startswith(stripped, "edges_with_weights:")
                    # informational "edges: <count>" line written by some
                    # generators — skip; the real section header comes later.
                    continue
                elseif occursin('=', stripped)
                    k, v = split(stripped, '='; limit = 2)
                    key = String(strip(k))
                    val = String(strip(v))
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
                length(parts) < 2 && continue
                u = parse(Int, parts[1])
                v = parse(Int, parts[2])
                add_edge!(graph, u, v)
            end
        end
    end
    nv(graph) > 0 ||
        error("graph file $path is missing a `vertices:` line")
    return graph, header_meta
end

"""
    _classify_mis_model(path, header_meta)
        -> NamedTuple{(:family, :graph_type, :base_id,
                       :graph_name, :meta)}

Map a graph filename to the same `graph_type` / slice-subdir tail that
the legacy regenerate path produces. The default
`mis_model_generator.jl` only emits one pattern,
`random_ksg_n=<n>_seed=<seed>.graph`, but ad-hoc files fall back to a
`generic` family (`graph_type=`generic``) so they still slice.

`base_id` is the trailing chunk that `slice_mis.jl` /
`slice_mis_counting.jl` prepend their own `mis_ground_counting_` /
`mis_counting_` prefix to when auto-naming the slice subdir.
"""
function _classify_mis_model(path::AbstractString,
                             header_meta::AbstractDict)
    basename_str = splitext(basename(path))[1]

    # Family 1: random_ksg_n=<n>_seed=<seed>
    m = match(r"^random_ksg_n=(\d+)_seed=(\d+)$", basename_str)
    if m !== nothing
        n    = parse(Int, m.captures[1])
        seed = parse(Int, m.captures[2])
        return (
            family     = "random_ksg",
            graph_type = "random_ksg",
            base_id    = "random_ksg_n=$(n)_seed=$(seed)",
            graph_name = basename(path),
            meta       = Dict{String,Any}("n" => n, "seed" => seed),
        )
    end

    # Generic fallback: just slice it, no family-specific metadata.
    @warn "unrecognised MIS graph filename pattern; using generic " *
          "(graph_type=`generic`). Pass --subdir / --graph-type to " *
          "override." path
    meta = Dict{String,Any}()
    for (k, v) in header_meta
        meta[k] = v
    end
    return (
        family     = "generic",
        graph_type = "generic",
        base_id    = basename_str,
        graph_name = basename(path),
        meta       = meta,
    )
end
