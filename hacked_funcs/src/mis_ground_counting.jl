using OptimalBranching
using OptimalBranching.OptimalBranchingCore, OptimalBranching.OptimalBranchingMIS
using Graphs, TropicalNumbers, OMEinsum, OMEinsumContractionOrders
using Base.Threads
using TensorBranching: ob_region, optimal_branches, optimal_branches_ground_counting_induced_sparsity, ContractionTreeSlicer, uncompress, SlicedBranch, complexity
using OMEinsum: DynamicNestedEinsum
using OMEinsumContractionOrders: uniformsize
using TensorBranching: show_status, LP_MWIS, solve_slice
using CUDA, CuTropicalGEMM, BenchmarkTools
CUDA.device!(2)

function slice_bfs(p::MISProblem{INT, VT}, slicer::ContractionTreeSlicer, code::DynamicNestedEinsum{Int}, verbose::Int = 0) where {INT, VT}
    # Initialize with the first branch
    initial_branch = SlicedBranch(p, code, zero(eltype(p.weights)))
    # Use Vector{SlicedBranch} to allow different INT types from branching
    unfinished_slices = SlicedBranch[initial_branch]
    finished_slices = SlicedBranch[]
    size_dict = uniformsize(code, 2)

    while !isempty(unfinished_slices)
        new_slices, new_scs = _slice_bfs(unfinished_slices, slicer,size_dict, verbose)
        verbose ≥ 1 && show_status(new_scs, slicer.sc_target, length(unfinished_slices), length(finished_slices))
        empty!(unfinished_slices)
        for (slice, sc) in zip(new_slices, new_scs)
            if sc ≤ slicer.sc_target
                push!(finished_slices, slice)
            else
                push!(unfinished_slices, slice)
            end
        end
    end

    return finished_slices
end

function _slice_bfs(unfinished_slices::Vector{SlicedBranch}, slicer::ContractionTreeSlicer, size_dict::Dict{Int, Int}, verbose::Int)
    n = length(unfinished_slices)
    # Use Vector{SlicedBranch} instead of Vector{SlicedBranch{INT, VT, RT}} to allow different INT types
    temp_slices = Vector{Vector{SlicedBranch}}(undef, n)
    nt = Threads.nthreads()
    chunks = collect(Iterators.partition(1:n, ceil(Int, n/nt)))

    Threads.@threads for chunk in chunks
        for i in chunk
            branch = unfinished_slices[i]
            uncompressed_code = uncompress(branch.code)
            region, loss = ob_region(branch.p.g, uncompressed_code, slicer, slicer.region_selector, size_dict, verbose)
            branches = optimal_branches_ground_counting(branch.p, uncompressed_code, branch.r, slicer, region, size_dict, verbose)
            temp_slices[i] = branches
        end
    end
    
    new_slices = vcat(temp_slices...)
    new_scs = [complexity(slice).sc for slice in new_slices]

    return new_slices, new_scs
end

function slice_dfs_lp(p::MISProblem{INT, VT}, slicer::ContractionTreeSlicer, code::DynamicNestedEinsum{Int}, usecuda::Bool, verbose::Int) where {INT, VT, RT}
    initial_branch = SlicedBranch(p, code, zero(eltype(p.weights)))
    unfinished_slices = SlicedBranch[initial_branch]
    finished_slices = SlicedBranch[]
    scs = [complexity(initial_branch).sc]
    size_dict = uniformsize(uncompress(initial_branch.code), 2)

    lp_bound = LP_MWIS(initial_branch.p.g, initial_branch.p.weights)
    primal_bound = 0.0
    finished_count = 0

    verbose ≥ 1 && @info "initial lp_bound: $lp_bound"

    while !isempty(unfinished_slices)
        branch_to_slice = popfirst!(unfinished_slices)
        sc_to_slice = popfirst!(scs)

        verbose ≥ 1 && @info "slicing branch with sc: $sc_to_slice"

        cc = complexity(branch_to_slice)
        if cc.sc ≤ slicer.sc_target 
            if primal_bound <= 0.01
                if iszero(cc.sc)
                    feasible_solution = branch_to_slice.r
                else
                    feasible_solution = solve_slice(branch_to_slice,Float32, usecuda) + branch_to_slice.r
                end
                primal_bound = max(primal_bound, feasible_solution)
                verbose ≥ 1 && @info "feasible solution: $feasible_solution, primal bound: $primal_bound"
            end
            

            push!(finished_slices, branch_to_slice)
            finished_count += 1
            
            
        else
            new_slices, new_scs, lp_scores = _slice_single(branch_to_slice, primal_bound, slicer, size_dict, verbose)
            if !isempty(new_slices)
                for i in length(new_slices):-1:1
                    pushfirst!(unfinished_slices, new_slices[i])
                    pushfirst!(scs, new_scs[i])
                end
            end
        end

        if !isempty(unfinished_slices)
            verbose ≥ 1 && show_status(scs, slicer.sc_target, length(unfinished_slices), finished_count)
        end
    end

    return finished_slices
end


function _slice_single(slice::ST, primal_bound::Float64, slicer::ContractionTreeSlicer, size_dict::Dict{Int, Int}, verbose::Int) where ST

    uncompressed_code = uncompress(slice.code)
    region, loss = ob_region(slice.p.g, uncompressed_code, slicer, slicer.region_selector, size_dict, verbose)
    branches = optimal_branches_ground_counting_induced_sparsity(slice.p, uncompressed_code, slice.r, primal_bound, slicer, region, size_dict, verbose)
    temp_slices = branches
    
    new_slices = SlicedBranch[]
    new_scs = Int[]
    lp_scores = Float64[]
    for (i, slice) in enumerate(temp_slices)
        lp_score = LP_MWIS(slice.p.g, slice.p.weights) + slice.r
        verbose ≥ 1 && @info "slice $i, lp_score: $lp_score, primal_bound: $primal_bound"
        if lp_score > primal_bound - 0.0001
            push!(lp_scores, lp_score)
            push!(new_scs, complexity(slice).sc)
            push!(new_slices, slice)
        end
    end
    
    if isempty(lp_scores)
        return new_slices, new_scs, lp_scores
    end

    graph_sizes = [nv(slice.p.g) for slice in new_slices]
    sorted_indices = sortperm(1:length(lp_scores), by = i -> (- lp_scores[i], graph_sizes[i]))

    return new_slices[sorted_indices], new_scs[sorted_indices], lp_scores[sorted_indices]
end