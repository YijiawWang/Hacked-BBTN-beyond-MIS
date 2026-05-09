using Test
using Graphs
using GenericTensorNetworks, ProblemReductions
using OMEinsumContractionOrders: TreeSA
using CUDA

include(joinpath(@__DIR__, "..", "..", "scripts", "pure_slice_spin_glass.jl"))

function _setup_cuda_for_test!(gpu_id::Int = 0)
    if !CUDA.functional()
        @warn "CUDA is not functional; skipping GPU contraction test"
        return false
    end
    ndev = length(CUDA.devices())
    if gpu_id < 0 || gpu_id >= ndev
        @warn "requested GPU id=$gpu_id but only $ndev CUDA device(s) are visible; skipping GPU contraction test"
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

function test_pure_tree_sa_slicing_matches_strict(; sc_target::Int = 18,
                                                  gpu_id::Int = 0)
    _setup_cuda_for_test!(gpu_id) || return nothing

    g = Graphs.grid([20, 20])
    J = Float32.(ones(ne(g)))
    h = fill(Float32(0.5), nv(g))

    pure = pure_tree_sa_slices(g, J, h, sc_target;
        optimizer = TreeSA(ntrials = 1, niters = 1),
        save_mode = :all)

    @test !isempty(pure.sliced_labels)
    @test length(pure.slices) == Int(pure.all_slice_count)

    root = mktempdir(; cleanup = false)
    slice_dir = save_spin_glass_slices("pure_tree_sa_slicing_test", pure.slices;
        root = root,
        original = (g, J, h),
        overwrite = true,
        update_summary = false)

    sliced = contract_spin_glass_slices(slice_dir;
        root = "",
        usecuda = true,
        count_eltype = Float64)
    strict_raw = solve(GenericTensorNetwork(SpinGlass(g, J, h); optimizer = TreeSA()),
        CountingMax(); T = Float64, usecuda = true)
    strict = Array(strict_raw)[]

    @test isapprox(sliced.energy, Float64(strict.n[1]); atol = 1e-6)
    @test sliced.count == BigInt(round(strict.c[1]))

    one = pure_tree_sa_slices(g, J, h, sc_target;
        optimizer = TreeSA(ntrials = 1, niters = 1),
        save_mode = :one,
        representative_assignment = UInt64(0))

    @test length(one.slices) == 1
    @test one.sliced_labels == pure.sliced_labels
    @test one.all_slice_count == pure.all_slice_count

    return (; pure, sliced, strict, slice_dir)
end

function test_pure_tree_sa_single_slice_matches_strict_gpu(; sc_target::Int = 1000,
                                                           gpu_id::Int = 0)
    _setup_cuda_for_test!(gpu_id) || return nothing

    g = Graphs.grid([20, 20])
    J = Float32.(ones(ne(g)))
    h = fill(Float32(0.5), nv(g))

    pure = pure_tree_sa_slices(g, J, h, sc_target;
        optimizer = TreeSA(ntrials = 1, niters = 1),
        save_mode = :all)

    @test isempty(pure.sliced_labels)
    @test pure.all_slice_count == BigInt(1)
    @test length(pure.slices) == 1

    root = mktempdir(; cleanup = false)
    slice_dir = save_spin_glass_slices("pure_tree_sa_single_slice_test", pure.slices;
        root = root,
        original = (g, J, h),
        overwrite = true,
        update_summary = false)

    sliced = contract_spin_glass_slices(slice_dir;
        root = "",
        usecuda = true,
        count_eltype = Float64)
    strict_raw = solve(GenericTensorNetwork(SpinGlass(g, J, h); optimizer = TreeSA()),
        CountingMax(); T = Float64, usecuda = true)
    strict = Array(strict_raw)[]

    @test isapprox(sliced.energy, Float64(strict.n[1]); atol = 1e-6)
    @test sliced.count == BigInt(round(strict.c[1]))

    return (; pure, sliced, strict, slice_dir)
end

if abspath(PROGRAM_FILE) == @__FILE__
    @testset "pure spin-glass TreeSA slicing" begin
        test_pure_tree_sa_slicing_matches_strict()
        test_pure_tree_sa_single_slice_matches_strict_gpu()
    end
end
