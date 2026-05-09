using Printf
using CUDA
using GenericTensorNetworks, ProblemReductions
using OMEinsumContractionOrders: TreeSA

include(joinpath(@__DIR__, "pure_slice_spin_glass.jl"))

const CASES = [
    (
        seed = 1,
        slice_dir = joinpath(@__DIR__, "..", "branch_results",
            "pure_slicing_spin_glass_J±1_grid_n=10_seed=1"),
        model_path = joinpath(@__DIR__, "..", "hacked_funcs", "benchmarks", "models",
            "spin_glass_models", "test", "spin_glass_J±1_grid_n=10_seed=1.model"),
    ),
    (
        seed = 2,
        slice_dir = joinpath(@__DIR__, "..", "branch_results",
            "pure_slicing_spin_glass_J±1_grid_n=10_seed=2"),
        model_path = joinpath(@__DIR__, "..", "hacked_funcs", "benchmarks", "models",
            "spin_glass_models", "test", "spin_glass_J±1_grid_n=10_seed=2.model"),
    ),
    (
        seed = 3,
        slice_dir = joinpath(@__DIR__, "..", "branch_results",
            "pure_slicing_spin_glass_J±1_grid_n=10_seed=3"),
        model_path = joinpath(@__DIR__, "..", "hacked_funcs", "benchmarks", "models",
            "spin_glass_models", "test", "spin_glass_J±1_grid_n=10_seed=3.model"),
    ),
]

function _parse_args(args)
    usecuda = CUDA.functional()
    gpu_id = 0
    for arg in args
        if arg == "--cpu" || arg == "--no-cuda"
            usecuda = false
        elseif startswith(arg, "--gpu=")
            gpu_id = parse(Int, split(arg, "="; limit = 2)[2])
            usecuda = true
        else
            error("unknown argument: $arg")
        end
    end
    return (; usecuda, gpu_id)
end

function _setup_cuda!(usecuda::Bool, gpu_id::Int)
    usecuda || return false
    if !CUDA.functional()
        @warn "CUDA is not functional; falling back to CPU"
        return false
    end
    ndev = length(CUDA.devices())
    if gpu_id < 0 || gpu_id >= ndev
        @warn "requested GPU id=$gpu_id but only $ndev CUDA device(s) are visible; falling back to CPU"
        return false
    end
    CUDA.device!(gpu_id)
    try
        @eval Main using CuTropicalGEMM
    catch err
        @warn "CuTropicalGEMM failed to load; continuing with CUDA.jl" exception = err
    end
    return true
end

_to_bigint_count(x::Integer) = BigInt(x)
_to_bigint_count(x::AbstractFloat) = isfinite(x) ? BigInt(round(x)) :
    error("non-finite count $x")

function direct_gtn(model_path::AbstractString; usecuda::Bool)
    graph, J, header_meta = _read_spin_glass_model(model_path)
    info = _classify_model(model_path, header_meta)
    h = fill(Float32(info.h_default), nv(graph))
    problem = GenericTensorNetwork(SpinGlass(graph, J, h); optimizer = TreeSA())

    elapsed = @elapsed raw = solve(problem, CountingMax(); T = Float64, usecuda = usecuda)
    result = Array(raw)[]
    return (;
        energy = Float64(result.n[1]),
        count = _to_bigint_count(result.c[1]),
        runtime = elapsed,
        vertices = nv(graph),
        edges = ne(graph),
    )
end

function verify_case(case; usecuda::Bool)
    sliced = contract_spin_glass_slices(case.slice_dir;
        root = "",
        usecuda = usecuda,
        count_eltype = :finitefield)
    direct = direct_gtn(case.model_path; usecuda = usecuda)

    energy_match = isapprox(sliced.energy, direct.energy; atol = 1e-6)
    count_match = sliced.count == direct.count
    return (;
        case.seed,
        direct.vertices,
        direct.edges,
        sliced_energy = sliced.energy,
        direct_energy = direct.energy,
        sliced_count = sliced.count,
        direct_count = direct.count,
        sliced_runtime = sliced.total_runtime,
        direct_runtime = direct.runtime,
        energy_match,
        count_match,
        ok = energy_match && count_match,
    )
end

function main(args = ARGS)
    cfg = _parse_args(args)
    usecuda = _setup_cuda!(cfg.usecuda, cfg.gpu_id)
    println("[verify] backend = ", usecuda ? "CUDA gpu=$(cfg.gpu_id)" : "CPU")

    results = [verify_case(case; usecuda = usecuda) for case in CASES]

    println()
    println("seed,nv,ne,sliced_energy,direct_energy,sliced_count,direct_count,energy_match,count_match,sliced_time_s,direct_time_s")
    for r in results
        @printf("%d,%d,%d,%.12g,%.12g,%s,%s,%s,%s,%.4f,%.4f\n",
            r.seed, r.vertices, r.edges,
            r.sliced_energy, r.direct_energy,
            string(r.sliced_count), string(r.direct_count),
            string(r.energy_match), string(r.count_match),
            r.sliced_runtime, r.direct_runtime)
    end

    all(r -> r.ok, results) || error("at least one sliced result differs from direct GTN")
    println("\n[verify] all 3 pure-slicing results match direct GTN.")
    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
