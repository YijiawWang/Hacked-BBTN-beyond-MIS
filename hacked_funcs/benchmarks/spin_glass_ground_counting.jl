using OptimalBranching
using OptimalBranching.OptimalBranchingCore, OptimalBranching.OptimalBranchingMIS
using GenericTensorNetworks, ProblemReductions
using Graphs, TropicalNumbers, OMEinsum, OMEinsumContractionOrders, Random
using Base.Threads
using TensorBranching: ob_region, optimal_branches, ContractionTreeSlicer, uncompress, SlicedBranch, complexity
using TensorBranching: optimal_branches_ground_counting, initialize_code, GreedyBrancher, ScoreRS
using OptimalBranchingMIS: TensorNetworkSolver
using OMEinsumContractionOrders: TreeSA
using OMEinsum: DynamicNestedEinsum
using OMEinsumContractionOrders: uniformsize
using CSV, DataFrames
using OrderedCollections
include("../src/spin_glass_ground_counting.jl")
include(joinpath(@__DIR__, "..", "..", "contractors", "spin_glass_slice_contract.jl"))

# ============================================================================
# Unified ground-counting benchmark for three spin-glass families:
#
#   * Family 1 (`j1pm1`)   : J = ±1 NN-only grid (open boundary)
#   * Family 2 (`j1j2`)    : J1-J2 square lattice, open boundary,
#                            NN i.i.d. ±|J1|, NNN i.i.d. ±|g·J1|
#                            (defaults |J1|=1, g=0.5 reproduce the old
#                             ±1 / ±0.5 behaviour)
#   * Family 3 (`j1j2_pbc`): J1-J2 AFM square lattice with PBC (torus),
#                            J2 = g · J1
#
# The family to run (and its parameter ranges) is selected via CLI flags;
# see `--help` (or the `_USAGE` string below) for the full grammar. All
# graphs are loaded from `models/spin_glass_models/*.model`, which are
# generated up-front by `models/spin_glass_model_generator.jl`. Each
# family writes its own CSV under `results/<family>/...` (paths and
# schemas are identical to the previous per-family scripts so downstream
# analysis is unchanged), and streams every produced slice into
# `beyond_mis/branch_results/<subdir>/` for later contraction.
# ============================================================================

const INPUT_DIR   = "models/spin_glass_models"
const TIMING_COLS = (:slicing_time, :branching_time, :total_time)

const _USAGE = """
Usage:

    julia spin_glass_ground_counting.jl --family={j1pm1|j1j2|j1j2_pbc|all}
        [--sc-target=<int>]        # default 31
        [--ns=lo:hi]               # j1pm1 only        (default 70:1:70)
        [--seeds=lo:hi]            # j1pm1 / j1j2      (default 2:2 / 1:1)
        [--Ls=lo:hi]               # j1j2 / j1j2_pbc   (default 50:1:50 / 19:1:19)
        [--gs=g1,g2,...]           # j1j2 / j1j2_pbc   (default 0.5    / 1//2,1//10)
        [--J1=<float>]             # j1j2 / j1j2_pbc   (default 1.0    / -1.0)
        [--h=<float>]              # uniform field     (default 0.5/0.5/0.0)
        [--J2-scale=<float>]       # j1j2 legacy alias for --gs=<g>
        [--ntrials=<int>]          # TreeSA ntrials    (default 50)
        [--niters=<int>]           # TreeSA niters     (default 100)
        [--code-seeds=<lo:hi>]     # TreeSA RNG seeds  (default 1:2)

`--gs` accepts either rationals (`1//2`) or decimals (`0.5`).
`--family=all` runs all three families; overrides apply to whichever
family takes that parameter. Lower `--ntrials --niters --code-seeds`
for quick sanity checks on small instances; benchmark defaults are
sized for L≈50.
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
    family   = ""
    sc_target = 31
    ns       = nothing
    seeds    = nothing
    Ls       = nothing
    gs       = nothing
    J1       = -1.0
    J1_set   = false
    h        = nothing                  # `nothing` → use family default
    J2_scale = 1.0
    J2_scale_set = false
    ntrials    = 50
    niters     = 100
    code_seeds = 1:2

    for a in args
        if startswith(a, "--family=")
            family = String(split(a, "="; limit = 2)[2])
        elseif startswith(a, "--sc-target=")
            sc_target = parse(Int, split(a, "="; limit = 2)[2])
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
        elseif startswith(a, "--h=")
            h = parse(Float64, split(a, "="; limit = 2)[2])
        elseif startswith(a, "--J2-scale=")
            J2_scale = parse(Float64, split(a, "="; limit = 2)[2])
            J2_scale_set = true
        elseif startswith(a, "--ntrials=")
            ntrials = parse(Int, split(a, "="; limit = 2)[2])
        elseif startswith(a, "--niters=")
            niters = parse(Int, split(a, "="; limit = 2)[2])
        elseif startswith(a, "--code-seeds=")
            code_seeds = _parse_int_range(split(a, "="; limit = 2)[2])
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

    return (; family, sc_target, ns, seeds, Ls, gs,
              J1, J1_set, h, J2_scale, J2_scale_set,
              ntrials, niters, code_seeds)
end


# ---------------------------------------------------------------------------
# Common helpers
# ---------------------------------------------------------------------------

"""
    read_spin_glass_model(path) -> (graph, edge_weights_vec)

Read a `.model` file written by `models/spin_glass_model_generator.jl`.
The returned `edge_weights_vec` is `Float32`-typed and ordered to match
`edges(graph)`.
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
            elseif line == "edges_with_weights:"
                while !eof(io)
                    line = readline(io)
                    isempty(line) && continue
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
        edge_weights_vec[k] =
            Float32(edge_weights[(min(src(e), dst(e)), max(src(e), dst(e)))])
    end
    return graph, edge_weights_vec
end


"""
    ensure_results_csv!(results_file, schema_df)

If `results_file` does not exist, create it with the columns from
`schema_df`. If it already exists but is missing any of `TIMING_COLS`,
back-fill those columns with `missing` so the schema stays stable across
appends.
"""
function ensure_results_csv!(results_file::AbstractString, schema_df::DataFrame)
    if isfile(results_file)
        df = CSV.read(results_file, DataFrame)
        added_any = false
        for col in TIMING_COLS
            if !(string(col) in names(df))
                df[!, col] = Vector{Union{Missing,Float64}}(missing, nrow(df))
                added_any = true
            end
        end
        added_any && CSV.write(results_file, df)
    else
        CSV.write(results_file, schema_df)
    end
end


"""
    run_case!(; graph, edge_weights_vec, h, model_name, slice_subdir,
              graph_type, meta_extra, sc_target) -> Union{NamedTuple, Nothing}

Shared per-instance pipeline: best-of-K TreeSA code search → standalone
slicing pass → `slice_dfs_lp` branching → batch-persist finished slices
to disk via `save_spin_glass_slices` → accumulate total `tc`. Returns a
`NamedTuple` of metrics, or `nothing` when no slice was produced.
"""
function run_case!(; graph::SimpleGraph,
                    edge_weights_vec::AbstractVector,
                    h::AbstractVector,
                    model_name::AbstractString,
                    slice_subdir::AbstractString,
                    graph_type::AbstractString,
                    meta_extra::AbstractDict,
                    sc_target::Int,
                    ntrials::Int = 50,
                    niters::Int  = 100,
                    code_seeds::AbstractRange = 1:2)
    t_total_start = time()
    p = SpinGlassProblem(graph, edge_weights_vec, h)

    println("\nProcessing model: $model_name")
    println("  Vertices: ", nv(graph))
    println("  Edges:    ", ne(graph))

    # Best-of-K TreeSA code search: spend more time looking for a good
    # contraction path; keep the candidate with the lowest tc.
    bbtn_optimizer = TreeSA(
        ntrials = ntrials,    # default 10 in OMEinsumContractionOrders
        niters  = niters,     # default 50
        βs      = 0.01:0.02:20.0,
    )
    code = nothing
    cc   = nothing
    best_tc = Inf
    for code_seed in code_seeds
        Random.seed!(code_seed)
        cand_code = initialize_code(graph, bbtn_optimizer)
        cand_cc   = contraction_complexity(cand_code, uniformsize(cand_code, 2))
        println("    seed=$code_seed -> tc=$(cand_cc.tc), sc=$(cand_cc.sc), rw=$(cand_cc.rwc)")
        if cand_cc.tc < best_tc
            best_tc = cand_cc.tc
            code = cand_code
            cc = cand_cc
        end
    end
    println("  Best code complexity over seeds $(collect(code_seeds)): ", cc)

    println("  " * "="^40)
    println("  sc_target: $sc_target")

    slicer = ContractionTreeSlicer(
        sc_target       = sc_target,
        table_solver    = TensorNetworkSolver(),
        region_selector = ScoreRS(n_max = 10),
        brancher        = GreedyBrancher(),
    )

    # Standalone slicing complexity (for the entire graph)
    edge_ixs   = [[minmax(e.src, e.dst)...] for e in Graphs.edges(graph)]
    vertex_ixs = [[v] for v in 1:nv(graph)]
    ixs        = vcat(edge_ixs, vertex_ixs)
    code0      = OMEinsumContractionOrders.EinCode([ixs...], Int[])
    Random.seed!(2)
    t_slicing_start = time()
    optcode_sliced = optimize_code(code0, uniformsize(code0, 2), TreeSA(),
        slicer = TreeSASlicer(score = ScoreFunction(sc_target = sc_target)))
    slicing_time = time() - t_slicing_start
    total_tc_slicing, sc_slicing =
        OMEinsum.timespace_complexity(optcode_sliced, uniformsize(code0, 2))
    slice_num_slicing = length(optcode_sliced.slicing)
    println("  Slicing results:")
    println("    Total tc (slicing):           ", total_tc_slicing)
    println("    Space complexity (slicing):   ", sc_slicing)
    println("    Number of slices (slicing):   ", slice_num_slicing)
    println("    Slicing wall time (s):        ", slicing_time)

    # Persist every produced slice (graph + J + h + r + saved code)
    # under `beyond_mis/branch_results/<subdir>`. Re-contraction is
    # later done by `beyond_mis/contractors/spin_glass_slice_contract.jl`.
    # NB: the contractor's on-disk API is the batch-style
    # `save_spin_glass_slices(subdir, slices; ...)`. We collect the
    # slices in memory first via `slice_dfs_lp` and dump them in one
    # shot. The `on_finished_slice` callback is still used for live
    # progress logging but no longer touches the filesystem.
    full_meta = merge(Dict{String,Any}(
        "sc_target" => sc_target,
        "vertices"  => nv(graph),
        "edges"     => ne(graph),
    ), Dict{String,Any}(string(k) => v for (k, v) in pairs(meta_extra)))

    t_branching_start = time()
    finished_slices = slice_dfs_lp(p, slicer, code, true, 1;
        on_finished_slice = slice -> begin
            cc_s = complexity(slice)
            println("  [slice produced] sc=$(cc_s.sc) tc=$(cc_s.tc) " *
                    "nv=$(nv(slice.p.g)) ne=$(ne(slice.p.g)) r=$(slice.r)")
            flush(stdout)
        end)
    branching_time = time() - t_branching_start

    println("  finished $(length(finished_slices)) slice(s) in " *
            "$(round(branching_time, digits = 3))s; persisting ...")
    slice_dir = save_spin_glass_slices(slice_subdir, finished_slices;
        original       = (graph, edge_weights_vec, h),
        model_name     = model_name,
        graph_type     = graph_type,
        overwrite      = true,
        meta           = full_meta,
        update_summary = true,
    )
    println("  Saved $(length(finished_slices)) slice(s) to $slice_dir")
    println("  Branching (slice_dfs_lp) wall time (s): ", branching_time)

    isempty(finished_slices) && return nothing

    # Accumulate total tc in Float32 (log-sum-exp over slice tc's)
    total_tc = -Inf32
    for slice in finished_slices
        cc_val = complexity(slice).tc
        total_tc = total_tc == -Inf32 ?
                   Float32(cc_val) :
                   log2(2^total_tc + 2^cc_val)
    end
    println("  Total tc (branching): ", total_tc)
    println("  Slice num (branching): ", length(finished_slices))

    total_time = time() - t_total_start
    println("  Total wall time (s): ", total_time)

    return (
        model_name        = model_name,
        vertices          = nv(graph),
        edges             = ne(graph),
        total_tc          = total_tc,
        slice_num         = length(finished_slices),
        total_tc_slicing  = total_tc_slicing,
        sc_slicing        = sc_slicing,
        slice_num_slicing = slice_num_slicing,
        slicing_time      = slicing_time,
        branching_time    = branching_time,
        total_time        = total_time,
    )
end


# ---------------------------------------------------------------------------
# Family 1: J = ±1 grid spin glass (NN only, open boundary)
# ---------------------------------------------------------------------------

function run_family_j1pm1(; n_values    = 70:1:70,
                            seed_values = 2:2,
                            h_default   = 0.5f0,
                            sc_target::Int,
                            ntrials::Int = 50,
                            niters::Int  = 100,
                            code_seeds::AbstractRange = 1:2)
    output_dir = "results/spin_glass_ground_counting"
    mkpath(output_dir)
    results_file = joinpath(output_dir, "spin_glass_ground_counting_results.csv")
    schema = DataFrame(
        model_name        = String[],
        vertices          = Int[],
        edges             = Int[],
        sc_target         = Int[],
        total_tc          = Float32[],
        slice_num         = Int[],
        total_tc_slicing  = Float32[],
        sc_slicing        = Float32[],
        slice_num_slicing = Int[],
        slicing_time      = Float64[],
        branching_time    = Float64[],
        total_time        = Float64[],
    )
    ensure_results_csv!(results_file, schema)

    println("\n" * "="^60)
    println("[family] J=±1 grid spin glass (open boundary, NN only)")
    println("  n values:    $(collect(n_values))")
    println("  seed values: $(collect(seed_values))")
    println("  sc_target:   $sc_target")

    for seed in seed_values, n in n_values
        model_file = joinpath(INPUT_DIR,
            "spin_glass_J±1_grid_n=$(n)_seed=$(seed).model")
        if !isfile(model_file)
            println("  Warning: File not found: $(basename(model_file))")
            continue
        end

        graph, edge_weights_vec = read_spin_glass_model(model_file)
        h = fill(Float32(h_default), nv(graph))

        r = run_case!(
            graph            = graph,
            edge_weights_vec = edge_weights_vec,
            h                = h,
            model_name       = "spin_glass_J±1_grid_n=$(n)_seed=$(seed)_cheating",
            slice_subdir     = "grid_Jpm1_n=$(n)_seed=$(seed)_cheating",
            graph_type       = "grid_Jpm1",
            meta_extra       = Dict("n" => n, "seed" => seed),
            sc_target        = sc_target,
            ntrials          = ntrials,
            niters           = niters,
            code_seeds       = code_seeds,
        )
        r === nothing && continue

        row = DataFrame(
            model_name        = [r.model_name],
            vertices          = [r.vertices],
            edges             = [r.edges],
            sc_target         = [sc_target],
            total_tc          = [r.total_tc],
            slice_num         = [r.slice_num],
            total_tc_slicing  = [Float32(r.total_tc_slicing)],
            sc_slicing        = [Float32(r.sc_slicing)],
            slice_num_slicing = [r.slice_num_slicing],
            slicing_time      = [r.slicing_time],
            branching_time    = [r.branching_time],
            total_time        = [r.total_time],
        )
        CSV.write(results_file, row, append = true)
        println("  Results saved to CSV: $(basename(results_file))")
    end
    println("[family j1pm1] results -> $results_file")
end


# ---------------------------------------------------------------------------
# Family 2: J1-J2 spin glass on open square lattice (NN ±1, NNN ±0.5)
# ---------------------------------------------------------------------------

"""
    _j1j2_open_model_paths(L, g, J1mag, seed) -> Vector{String}

Return candidate `.model` paths for the j1j2 open-boundary family in the
order they should be tried. The first entry is the new filename emitted
by the current generator
(`spin_glass_J1J2_grid_L=<L>_J1=<|J1|>_g=<g>_seed=<seed>.model`); the
second is the legacy `J1pm1_J2pm1` name, kept so existing dumps still
get picked up when `(|J1|, g) == (1.0, 0.5)`.
"""
function _j1j2_open_model_paths(L::Int, g::Float64, J1mag::Float64, seed::Int)
    paths = String[]
    push!(paths, joinpath(INPUT_DIR,
        "spin_glass_J1J2_grid_L=$(L)_J1=$(J1mag)_g=$(g)_seed=$(seed).model"))
    if J1mag == 1.0 && g == 0.5
        push!(paths, joinpath(INPUT_DIR,
            "spin_glass_J1J2_grid_L=$(L)_J1pm1_J2pm1_seed=$(seed).model"))
    end
    return paths
end

function run_family_j1j2_open(; L_values    = 50:1:50,
                                seed_values = 1:1,
                                gs          = (0.5,),
                                J1          = 1.0,
                                h_default   = 0.5f0,
                                sc_target::Int,
                                ntrials::Int = 50,
                                niters::Int  = 100,
                                code_seeds::AbstractRange = 1:2)
    J1mag = abs(Float64(J1))
    g_values = Tuple(Float64.(gs))

    output_dir = "results/spin_glass_j1j2_ground_counting"
    mkpath(output_dir)
    results_file = joinpath(output_dir,
        "spin_glass_j1j2_ground_counting_results.csv")
    schema = DataFrame(
        model_name        = String[],
        L                 = Int[],
        g                 = Float64[],
        J1                = Float64[],
        seed              = Int[],
        vertices          = Int[],
        edges             = Int[],
        sc_target         = Int[],
        total_tc          = Float32[],
        slice_num         = Int[],
        total_tc_slicing  = Float32[],
        sc_slicing        = Float32[],
        slice_num_slicing = Int[],
        slicing_time      = Float64[],
        branching_time    = Float64[],
        total_time        = Float64[],
    )
    ensure_results_csv!(results_file, schema)

    println("\n" * "="^60)
    println("[family] J1-J2 spin glass (open boundary, random ± signs, " *
            "NN magnitude |J1|, NNN magnitude |g·J1|)")
    println("  L values:    $(collect(L_values))")
    println("  g values:    $(collect(g_values))")
    println("  |J1|:        $J1mag")
    println("  seed values: $(collect(seed_values))")
    println("  sc_target:   $sc_target")

    for seed in seed_values, L in L_values, g in g_values
        gf  = Float64(g)
        candidates = _j1j2_open_model_paths(L, gf, J1mag, seed)
        model_file = ""
        for p in candidates
            if isfile(p)
                model_file = p
                break
            end
        end
        if isempty(model_file)
            println("  Warning: model file not found for " *
                    "L=$L g=$gf |J1|=$J1mag seed=$seed " *
                    "(tried: $(join(map(basename, candidates), ", ")))")
            continue
        end

        graph, edge_weights_vec = read_spin_glass_model(model_file)
        h = fill(Float32(h_default), nv(graph))

        tag = "L=$(L)_J1=$(J1mag)_g=$(gf)_seed=$(seed)"
        r = run_case!(
            graph            = graph,
            edge_weights_vec = edge_weights_vec,
            h                = h,
            model_name       = "spin_glass_J1J2_grid_$(tag)_cheating",
            slice_subdir     = "j1j2_grid_$(tag)_cheating",
            graph_type       = "j1j2_grid_open",
            meta_extra       = Dict("L"  => L,
                                    "g"  => gf,
                                    "J1" => J1mag,
                                    "J2" => J1mag * gf,
                                    "seed" => seed),
            sc_target        = sc_target,
            ntrials          = ntrials,
            niters           = niters,
            code_seeds       = code_seeds,
        )
        r === nothing && continue

        row = DataFrame(
            model_name        = [r.model_name],
            L                 = [L],
            g                 = [gf],
            J1                = [J1mag],
            seed              = [seed],
            vertices          = [r.vertices],
            edges             = [r.edges],
            sc_target         = [sc_target],
            total_tc          = [r.total_tc],
            slice_num         = [r.slice_num],
            total_tc_slicing  = [Float32(r.total_tc_slicing)],
            sc_slicing        = [Float32(r.sc_slicing)],
            slice_num_slicing = [r.slice_num_slicing],
            slicing_time      = [r.slicing_time],
            branching_time    = [r.branching_time],
            total_time        = [r.total_time],
        )
        CSV.write(results_file, row, append = true)
        println("  Results saved to CSV: $(basename(results_file))")
    end
    println("[family j1j2 open] results -> $results_file")
end


# ---------------------------------------------------------------------------
# Family 3: J1-J2 AFM spin glass on PBC square lattice (torus)
# ---------------------------------------------------------------------------

function run_family_j1j2_pbc(; L_values  = 19:1:19,
                               g_values  = (1/2, 1/10),
                               J1::Float64 = -1.0,
                               h_default = 0.0f0,
                               sc_target::Int,
                               ntrials::Int = 50,
                               niters::Int  = 100,
                               code_seeds::AbstractRange = 1:2)
    output_dir = "results/spin_glass_j1j2_pbc_ground_counting"
    mkpath(output_dir)
    results_file = joinpath(output_dir,
        "spin_glass_j1j2_pbc_ground_counting_results.csv")
    schema = DataFrame(
        model_name        = String[],
        L                 = Int[],
        g                 = Float64[],
        vertices          = Int[],
        edges             = Int[],
        sc_target         = Int[],
        total_tc          = Float32[],
        slice_num         = Int[],
        total_tc_slicing  = Float32[],
        sc_slicing        = Float32[],
        slice_num_slicing = Int[],
        slicing_time      = Float64[],
        branching_time    = Float64[],
        total_time        = Float64[],
    )
    ensure_results_csv!(results_file, schema)

    println("\n" * "="^60)
    println("[family] J1-J2 AFM spin glass (PBC torus)")
    println("  L values:  $(collect(L_values))")
    println("  g values:  $(collect(g_values))   (= |J2/J1|)")
    println("  J1 = $J1   (AFM, codebase convention: J<0)")
    println("  sc_target: $sc_target")

    for g_ratio in g_values, L in L_values
        gf = float(g_ratio)
        J2 = J1 * gf

        model_file = joinpath(INPUT_DIR,
            "spin_glass_J1J2_pbc_afm_grid_L=$(L)_g=$(gf).model")
        if !isfile(model_file)
            println("  Warning: File not found: $(basename(model_file))")
            continue
        end

        graph, edge_weights_vec = read_spin_glass_model(model_file)
        h = fill(Float32(h_default), nv(graph))

        r = run_case!(
            graph            = graph,
            edge_weights_vec = edge_weights_vec,
            h                = h,
            model_name       = "spin_glass_J1J2_pbc_afm_grid_L=$(L)_g=$(gf)_cheating",
            slice_subdir     = "j1j2_pbc_afm_grid_L=$(L)_g=$(gf)_cheating",
            graph_type       = "j1j2_grid_pbc_afm",
            meta_extra       = Dict("L" => L, "g" => gf,
                                    "J1" => J1, "J2" => J2),
            sc_target        = sc_target,
            ntrials          = ntrials,
            niters           = niters,
            code_seeds       = code_seeds,
        )
        r === nothing && continue

        row = DataFrame(
            model_name        = [r.model_name],
            L                 = [L],
            g                 = [gf],
            vertices          = [r.vertices],
            edges             = [r.edges],
            sc_target         = [sc_target],
            total_tc          = [r.total_tc],
            slice_num         = [r.slice_num],
            total_tc_slicing  = [Float32(r.total_tc_slicing)],
            sc_slicing        = [Float32(r.sc_slicing)],
            slice_num_slicing = [r.slice_num_slicing],
            slicing_time      = [r.slicing_time],
            branching_time    = [r.branching_time],
            total_time        = [r.total_time],
        )
        CSV.write(results_file, row, append = true)
        println("  Results saved to CSV: $(basename(results_file))")
    end
    println("[family j1j2 pbc] results -> $results_file")
end


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

function main(args)
    cfg = _parse_args(args)
    Random.seed!(12345)
    mkpath(SLICE_RESULTS_ROOT)

    println("="^60)
    println("[spin_glass_ground_counting] family=$(cfg.family)  sc_target=$(cfg.sc_target)")
    println("="^60)

    if cfg.family == "j1pm1" || cfg.family == "all"
        kw = Dict{Symbol,Any}(
            :sc_target  => cfg.sc_target,
            :ntrials    => cfg.ntrials,
            :niters     => cfg.niters,
            :code_seeds => cfg.code_seeds,
        )
        cfg.ns    !== nothing && (kw[:n_values]    = cfg.ns)
        cfg.seeds !== nothing && (kw[:seed_values] = cfg.seeds)
        cfg.h     !== nothing && (kw[:h_default]   = Float32(cfg.h))
        run_family_j1pm1(; kw...)
    end

    if cfg.family == "j1j2" || cfg.family == "all"
        kw = Dict{Symbol,Any}(
            :sc_target  => cfg.sc_target,
            :ntrials    => cfg.ntrials,
            :niters     => cfg.niters,
            :code_seeds => cfg.code_seeds,
        )
        cfg.Ls           !== nothing && (kw[:L_values]    = cfg.Ls)
        cfg.seeds        !== nothing && (kw[:seed_values] = cfg.seeds)
        cfg.h            !== nothing && (kw[:h_default]   = Float32(cfg.h))
        if cfg.gs !== nothing
            kw[:gs] = cfg.gs
        elseif cfg.J2_scale_set
            # legacy alias: --J2-scale=<g> ≡ --gs=<g>
            kw[:gs] = (cfg.J2_scale,)
        end
        cfg.J1_set                     && (kw[:J1] = cfg.J1)
        run_family_j1j2_open(; kw...)
    end

    if cfg.family == "j1j2_pbc" || cfg.family == "all"
        kw = Dict{Symbol,Any}(
            :sc_target  => cfg.sc_target,
            :ntrials    => cfg.ntrials,
            :niters     => cfg.niters,
            :code_seeds => cfg.code_seeds,
        )
        cfg.Ls !== nothing && (kw[:L_values] = cfg.Ls)
        cfg.gs !== nothing && (kw[:g_values] = cfg.gs)
        cfg.J1_set        && (kw[:J1]        = cfg.J1)
        cfg.h  !== nothing && (kw[:h_default] = Float32(cfg.h))
        run_family_j1j2_pbc(; kw...)
    end

    println("\nAll requested spin-glass ground-counting families finished.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
