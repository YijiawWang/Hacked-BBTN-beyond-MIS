"""
    estimate_sc.jl

Quick `(tc, sc, rwc)` probe for one or more `.model` files.

Two optimizers are supported:

* `--optimizer=greedy`   (default) — `GreedyMethod()`. A min-degree-style
  greedy contraction ordering. Cheap (sub-second on graphs up to ~3 k
  vertices, even hub-rich), always produces a finite upper bound on
  `sc`, and serves as a fast first-pass screen.
* `--optimizer=treesa`   — `TreeSA(ntrials, niters)`. The same SA
  optimiser `slice_spin_glass.jl` uses; gives a tighter `sc` bound but
  is much slower on dense / hub-rich graphs.

`sc` is the log2 of the largest tensor in the contraction tree, so for
the spin-glass / IndependentSet tensor network `sc ≈ tw + 1`.

Usage:

    julia --project=beyond_mis beyond_mis/scripts/_lib/estimate_sc.jl \\
        <path/to/file.model> [<path2.model> ...] \\
        [--optimizer=greedy|treesa]   # default: greedy
        [--ntrials=<int>]             # treesa only; default: 10
        [--niters=<int>]              # treesa only; default: 50
        [--code-seeds=<lo:hi>]        # treesa only; default: 1:1
        [--quiet]                     # suppress per-seed output

Prints a summary table sorted by `sc`, plus a `degeneracy` column
(= max k-core number, which is a lower bound on `tw`).
"""

using Random, Printf
using Graphs, OMEinsum, OMEinsumContractionOrders
using OMEinsumContractionOrders: TreeSA, GreedyMethod, uniformsize
using TensorBranching: initialize_code

include(joinpath(@__DIR__, "..", "..", "hacked_funcs", "benchmarks",
                 "spin_glass_ground_counting.jl"))

const _DEFAULT_TREESA = TreeSA()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

function _parse_args(args)
    paths      = String[]
    optimizer  = "greedy"
    ntrials    = _DEFAULT_TREESA.ntrials
    niters     = _DEFAULT_TREESA.niters
    code_seeds = 1:1
    quiet      = false

    for a in args
        if startswith(a, "--optimizer=")
            optimizer = String(split(a, "="; limit = 2)[2])
            optimizer in ("greedy", "treesa") ||
                error("--optimizer must be `greedy` or `treesa`, got: $optimizer")
        elseif startswith(a, "--ntrials=")
            ntrials = parse(Int, split(a, "="; limit = 2)[2])
        elseif startswith(a, "--niters=")
            niters = parse(Int, split(a, "="; limit = 2)[2])
        elseif startswith(a, "--code-seeds=")
            spec = split(a, "="; limit = 2)[2]
            parts = split(spec, ':')
            if length(parts) == 1
                v = parse(Int, parts[1])
                code_seeds = v:v
            elseif length(parts) == 2
                code_seeds = parse(Int, parts[1]):parse(Int, parts[2])
            else
                error("invalid --code-seeds: $spec")
            end
        elseif a == "--quiet"
            quiet = true
        elseif a in ("-h", "--help")
            println(@doc(@__MODULE__))
            exit(0)
        elseif startswith(a, "--")
            error("unknown flag: $a")
        else
            push!(paths, a)
        end
    end
    isempty(paths) &&
        error("give at least one path to a .model file (or pass --help)")
    return (; paths, optimizer, ntrials, niters, code_seeds, quiet)
end


# ---------------------------------------------------------------------------
# sc probe for one .model file
# ---------------------------------------------------------------------------

"""
    probe_sc(path; optimizer, ntrials, niters, code_seeds, quiet) ->
        (graph, best_cc, best_seed)

Run the requested optimizer (`greedy` or `treesa`) and return the best
`(tc, sc, rwc)` together with the seed that produced it. For `greedy`
the result is deterministic and `code_seeds` is ignored.
"""
function probe_sc(path::AbstractString;
                  optimizer::String,
                  ntrials::Int, niters::Int,
                  code_seeds::AbstractRange,
                  quiet::Bool = false)
    graph, _ = read_spin_glass_model(path)

    if optimizer == "greedy"
        opt = GreedyMethod()
        cand_code = initialize_code(graph, opt)
        cand_cc   = contraction_complexity(cand_code, uniformsize(cand_code, 2))
        if !quiet
            @printf("    greedy: tc=%7.2f  sc=%5.2f  rwc=%7.2f\n",
                    cand_cc.tc, cand_cc.sc, cand_cc.rwc)
        end
        return (; graph, best_cc = cand_cc, best_seed = 0)
    end

    # TreeSA path
    best_cc   = nothing
    best_seed = -1
    for code_seed in code_seeds
        Random.seed!(code_seed)
        opt = TreeSA(ntrials = ntrials, niters = niters)
        cand_code = initialize_code(graph, opt)
        cand_cc   = contraction_complexity(cand_code, uniformsize(cand_code, 2))
        if !quiet
            @printf("    treesa seed=%2d  tc=%7.2f  sc=%5.2f  rwc=%7.2f\n",
                    code_seed, cand_cc.tc, cand_cc.sc, cand_cc.rwc)
        end
        if best_cc === nothing || cand_cc.tc < best_cc.tc
            best_cc   = cand_cc
            best_seed = code_seed
        end
    end
    return (; graph, best_cc, best_seed)
end


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

function main(args)
    cfg = _parse_args(args)

    println("="^80)
    if cfg.optimizer == "greedy"
        println("[estimate_sc] optimizer=greedy")
    else
        println("[estimate_sc] optimizer=treesa  ntrials=$(cfg.ntrials)" *
                "  niters=$(cfg.niters)  code_seeds=$(collect(cfg.code_seeds))")
    end
    println("="^80)

    summary = NamedTuple{(:name, :nv, :ne, :max_deg, :degeneracy, :tc, :sc, :rwc),
                         Tuple{String,Int,Int,Int,Int,Float64,Float64,Float64}}[]

    for path in cfg.paths
        name = splitext(basename(path))[1]
        println("\n--- $name ---")
        println("  $path")

        result = probe_sc(path;
                          optimizer  = cfg.optimizer,
                          ntrials    = cfg.ntrials,
                          niters     = cfg.niters,
                          code_seeds = cfg.code_seeds,
                          quiet      = cfg.quiet)

        max_deg    = isempty(vertices(result.graph)) ? 0 : maximum(degree(result.graph))
        degeneracy = isempty(vertices(result.graph)) ? 0 : maximum(Graphs.core_number(result.graph))
        @printf("  best: tc=%.2f  sc=%.2f  rwc=%.2f\n",
                result.best_cc.tc, result.best_cc.sc, result.best_cc.rwc)
        @printf("  graph: nv=%d  ne=%d  max_deg=%d  degeneracy=%d\n",
                nv(result.graph), ne(result.graph), max_deg, degeneracy)

        push!(summary, (name = name,
                        nv = nv(result.graph),
                        ne = ne(result.graph),
                        max_deg = max_deg,
                        degeneracy = degeneracy,
                        tc = Float64(result.best_cc.tc),
                        sc = Float64(result.best_cc.sc),
                        rwc = Float64(result.best_cc.rwc)))
    end

    # Sorted summary
    sort!(summary, by = r -> r.sc)
    println("\n" * "="^80)
    println("Summary (sorted by sc; sc ≈ tw+1; degeneracy is a lower bound on tw):")
    println("="^80)
    @printf("  %-50s %6s %7s %5s %5s %7s %7s %7s\n",
            "name", "nv", "ne", "Δ", "deg.", "tc", "sc", "rwc")
    for r in summary
        @printf("  %-50s %6d %7d %5d %5d %7.2f %7.2f %7.2f\n",
                first(r.name, 50), r.nv, r.ne, r.max_deg, r.degeneracy,
                r.tc, r.sc, r.rwc)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
