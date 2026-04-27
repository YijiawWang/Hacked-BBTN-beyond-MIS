using Graphs, Random

# ============================================================================
# Spin-glass model generator (three families). Family selection and
# parameter ranges are passed via CLI flags so the same script can
# produce any subset of models without editing the file. See `--help`
# (or the `_USAGE` string below) for the full grammar.
#
# Run from `beyond_mis/hacked_funcs/benchmarks/models/`; the readers in
# `beyond_mis/hacked_funcs/benchmarks/spin_glass_ground_counting.jl`
# pick the resulting `.model` files up via `models/spin_glass_models/`.
# ============================================================================

const OUTPUT_DIR = "spin_glass_models"

const _USAGE = """
Usage:

    julia spin_glass_model_generator.jl --family={j1pm1|j1j2|j1j2_pbc|all}
        [--ns=lo[:step]:hi]        # j1pm1 only        (default 75:5:80)
        [--seeds=lo:hi]            # j1pm1 / j1j2      (default 1:10 / 1:1)
        [--Ls=lo[:step]:hi]        # j1j2 / j1j2_pbc   (default 50:1:50 / 19:1:19)
        [--gs=g1,g2,...]           # j1j2_pbc only     (default 1//2,1//10)
        [--J1=<float>]             # j1j2_pbc only     (default -1.0)

`--gs` accepts either rationals (`1//2`) or decimals (`0.5`).
`--family=all` runs all three families; overrides apply to whichever
family takes that parameter.
"""


# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------

function _parse_int_range(s::AbstractString)
    parts = split(s, ':')
    if length(parts) == 1
        v = parse(Int, parts[1])
        return v:v
    elseif length(parts) == 2
        return parse(Int, parts[1]):parse(Int, parts[2])
    elseif length(parts) == 3
        return parse(Int, parts[1]):parse(Int, parts[2]):parse(Int, parts[3])
    end
    error("invalid range spec: $s (expected lo, lo:hi, or lo:step:hi)")
end

function _parse_g_list(s::AbstractString)
    out = Float64[]
    for tok in split(s, ',')
        tok = strip(tok)
        isempty(tok) && continue
        if occursin("//", tok)
            a, b = split(tok, "//"; limit = 2)
            push!(out, parse(Int, a) / parse(Int, b))
        elseif occursin('/', tok)
            a, b = split(tok, '/'; limit = 2)
            push!(out, parse(Int, a) / parse(Int, b))
        else
            push!(out, parse(Float64, tok))
        end
    end
    isempty(out) && error("--gs: empty list")
    return Tuple(out)
end

function _parse_args(args)
    family = ""
    ns     = nothing
    seeds  = nothing
    Ls     = nothing
    gs     = nothing
    J1     = -1.0
    J1_set = false

    for a in args
        if startswith(a, "--family=")
            family = String(split(a, "="; limit = 2)[2])
        elseif startswith(a, "--ns=")
            ns = _parse_int_range(split(a, "="; limit = 2)[2])
        elseif startswith(a, "--seeds=")
            seeds = _parse_int_range(split(a, "="; limit = 2)[2])
        elseif startswith(a, "--Ls=")
            Ls = _parse_int_range(split(a, "="; limit = 2)[2])
        elseif startswith(a, "--gs=")
            gs = _parse_g_list(split(a, "="; limit = 2)[2])
        elseif startswith(a, "--J1=")
            J1 = parse(Float64, split(a, "="; limit = 2)[2])
            J1_set = true
        elseif a in ("-h", "--help")
            println(_USAGE)
            exit(0)
        else
            error("unknown / unsupported argument: $a\n$_USAGE")
        end
    end

    isempty(family) &&
        error("--family={j1pm1|j1j2|j1j2_pbc|all} is required\n$_USAGE")
    family in ("j1pm1", "j1j2", "j1j2_pbc", "all") ||
        error("unknown family: $family\n$_USAGE")

    return (; family, ns, seeds, Ls, gs, J1, J1_set)
end


# ---------------------------------------------------------------------------
# Common writer
# ---------------------------------------------------------------------------

"""
    write_spin_glass_model(filepath, header_lines, graph, edge_weights)

Write `graph` plus `edge_weights` (keyed by `(min(u,v), max(u,v))`) to
`filepath` in the format expected by `read_spin_glass_model` over in
`spin_glass_ground_counting.jl`:

    # <free-form header>
    <header_lines...>
    vertices: <nv>
    edges: <ne>

    edges_with_weights:
    u v w
    ...
"""
function write_spin_glass_model(filepath::AbstractString,
                                header_lines::Vector{<:AbstractString},
                                graph::SimpleGraph,
                                edge_weights::AbstractDict)
    open(filepath, "w") do io
        for line in header_lines
            println(io, line)
        end
        println(io, "vertices: ", nv(graph))
        println(io, "edges: ", ne(graph))
        println(io)
        println(io, "edges_with_weights:")
        for e in edges(graph)
            s, d = src(e), dst(e)
            w = edge_weights[(min(s, d), max(s, d))]
            println(io, "$s $d $w")
        end
    end
    println("Generated: $filepath")
end


# ---------------------------------------------------------------------------
# Family 1: J = ±1 grid spin glass (NN only, open boundary)
# ---------------------------------------------------------------------------

function generate_j1pm1(; ns = 75:5:80, seeds = 1:10)
    println("\n[generate j1pm1] ns=$(collect(ns))  seeds=$(collect(seeds))")
    for n in ns, seed in seeds
        Random.seed!(seed)
        graph = SimpleGraph(Graphs.grid([n, n]))

        edge_weights = Dict{Tuple{Int,Int}, Int}()
        for e in edges(graph)
            edge_weights[(min(src(e), dst(e)), max(src(e), dst(e)))] = rand([-1, 1])
        end

        filename = "spin_glass_J±1_grid_n=$(n)_seed=$(seed).model"
        filepath = joinpath(OUTPUT_DIR, filename)
        write_spin_glass_model(filepath,
            ["# Grid Spin Glass Model with ±1 couplings",
             "n = $n",
             "seed = $seed",
             "J_type = ±1"],
            graph, edge_weights)
    end
end


# ---------------------------------------------------------------------------
# Family 2: J1-J2 spin glass on open square lattice (NN ±1, NNN ±0.5)
# ---------------------------------------------------------------------------

"""
    build_j1j2_square(L; rng) -> (graph, edge_weights)

`L x L` open-boundary square lattice. NN bonds i.i.d. ∈ {±1}, NNN
diagonal bonds i.i.d. ∈ {±0.5}. `edge_weights` is keyed by
`(min(u,v), max(u,v))`.
"""
function build_j1j2_square(L::Int; rng::AbstractRNG)
    graph = SimpleGraph(L * L)
    idx(i, j) = (i - 1) * L + j
    bond_is_nn = Dict{Tuple{Int,Int}, Bool}()

    for i in 1:L, j in 1:L
        u = idx(i, j)
        if i < L
            v = idx(i + 1, j); add_edge!(graph, u, v); bond_is_nn[minmax(u, v)] = true
        end
        if j < L
            v = idx(i, j + 1); add_edge!(graph, u, v); bond_is_nn[minmax(u, v)] = true
        end
        if i < L && j < L
            v = idx(i + 1, j + 1); add_edge!(graph, u, v); bond_is_nn[minmax(u, v)] = false
        end
        if i < L && j > 1
            v = idx(i + 1, j - 1); add_edge!(graph, u, v); bond_is_nn[minmax(u, v)] = false
        end
    end

    edge_weights = Dict{Tuple{Int,Int}, Float64}()
    for e in edges(graph)
        key = minmax(src(e), dst(e))
        if bond_is_nn[key]
            edge_weights[key] = rand(rng, Bool) ? 1.0 : -1.0
        else
            edge_weights[key] = rand(rng, Bool) ? 0.5 : -0.5
        end
    end
    return graph, edge_weights
end

function generate_j1j2_open(; Ls = 50:1:50, seeds = 1:1)
    println("\n[generate j1j2 open] Ls=$(collect(Ls))  seeds=$(collect(seeds))")
    for L in Ls, seed in seeds
        rng = MersenneTwister(seed)
        graph, edge_weights = build_j1j2_square(L; rng = rng)

        filename = "spin_glass_J1J2_grid_L=$(L)_J1pm1_J2pm1_seed=$(seed).model"
        filepath = joinpath(OUTPUT_DIR, filename)
        write_spin_glass_model(filepath,
            ["# J1-J2 spin glass, open boundary square lattice",
             "L = $L",
             "seed = $seed",
             "J1_type = ±1   (NN,  i.i.d.)",
             "J2_type = ±0.5 (NNN, i.i.d.)"],
            graph, edge_weights)
    end
end


# ---------------------------------------------------------------------------
# Family 3: J1-J2 AFM spin glass on PBC square lattice (torus)
#
# Sign convention in this codebase: the optimizer maximizes
#     sum_<i,j> J_ij s_i s_j  +  sum_i h_i s_i
# so  J < 0  is antiferromagnetic. Hence the standard frustrated AFM
# J1-J2 model is `J1 < 0` and `J2 = g · J1` (also < 0).
# ---------------------------------------------------------------------------

"""
    build_j1j2_square_pbc(L; J1, J2) -> (graph, edge_weights)

`L x L` PBC square lattice (torus) with NN coupling `J1` and NNN
coupling `J2`. Wrap-around duplicates are dropped, so each pair
appears at most once in `edge_weights` (keyed by
`(min(u,v), max(u,v))`).
"""
function build_j1j2_square_pbc(L::Int; J1::Float64, J2::Float64)
    graph = SimpleGraph(L * L)
    idx(i, j) = (i - 1) * L + j
    wrap(k) = mod1(k, L)

    edge_weights = Dict{Tuple{Int,Int}, Float64}()

    function add_bond!(u, v, mag)
        u == v && return                                # avoid self-loops at L=1
        key = minmax(u, v)
        haskey(edge_weights, key) && return             # avoid wrap-around dupes
        add_edge!(graph, u, v)
        edge_weights[key] = mag
    end

    for i in 1:L, j in 1:L
        u = idx(i, j)
        add_bond!(u, idx(wrap(i + 1), j),           J1)
        add_bond!(u, idx(i,           wrap(j + 1)), J1)
        add_bond!(u, idx(wrap(i + 1), wrap(j + 1)), J2)
        add_bond!(u, idx(wrap(i + 1), wrap(j - 1)), J2)
    end
    return graph, edge_weights
end

function generate_j1j2_pbc(; Ls = 19:1:19,
                            gs = (1/2, 1/10),
                            J1::Float64 = -1.0)
    println("\n[generate j1j2 pbc] Ls=$(collect(Ls))  gs=$(collect(gs))  J1=$J1")
    for L in Ls, g in gs
        gf = Float64(g)
        J2 = J1 * gf
        graph, edge_weights = build_j1j2_square_pbc(L; J1 = J1, J2 = J2)

        filename = "spin_glass_J1J2_pbc_afm_grid_L=$(L)_g=$(gf).model"
        filepath = joinpath(OUTPUT_DIR, filename)
        write_spin_glass_model(filepath,
            ["# J1-J2 AFM spin glass, PBC square lattice (torus)",
             "L = $L",
             "g = $gf   # |J2 / J1|",
             "J1 = $J1",
             "J2 = $J2"],
            graph, edge_weights)
    end
end


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

function main(args)
    cfg = _parse_args(args)
    mkpath(OUTPUT_DIR)

    println("="^60)
    println("[spin_glass_model_generator] family=$(cfg.family)")
    println("="^60)

    if cfg.family == "j1pm1" || cfg.family == "all"
        kw = Dict{Symbol,Any}()
        cfg.ns    !== nothing && (kw[:ns]    = cfg.ns)
        cfg.seeds !== nothing && (kw[:seeds] = cfg.seeds)
        generate_j1pm1(; kw...)
    end

    if cfg.family == "j1j2" || cfg.family == "all"
        kw = Dict{Symbol,Any}()
        cfg.Ls    !== nothing && (kw[:Ls]    = cfg.Ls)
        cfg.seeds !== nothing && (kw[:seeds] = cfg.seeds)
        generate_j1j2_open(; kw...)
    end

    if cfg.family == "j1j2_pbc" || cfg.family == "all"
        kw = Dict{Symbol,Any}()
        cfg.Ls !== nothing && (kw[:Ls] = cfg.Ls)
        cfg.gs !== nothing && (kw[:gs] = cfg.gs)
        cfg.J1_set        && (kw[:J1] = cfg.J1)
        generate_j1j2_pbc(; kw...)
    end

    println("\nAll requested models generated and saved to $OUTPUT_DIR directory")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
