using OptimalBranching
using OptimalBranching.OptimalBranchingCore, OptimalBranching.OptimalBranchingMIS
using Graphs, TropicalNumbers, OMEinsum, OMEinsumContractionOrders
using Base.Threads
using TensorBranching: ob_region, optimal_branches, ContractionTreeSlicer, uncompress, SlicedBranch, complexity
using OMEinsum: DynamicNestedEinsum
using OMEinsumContractionOrders: uniformsize
using TensorBranching: show_status

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
