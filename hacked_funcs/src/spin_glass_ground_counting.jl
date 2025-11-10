using OptimalBranching
using OptimalBranching.OptimalBranchingCore, OptimalBranching.OptimalBranchingMIS
using Graphs, TropicalNumbers, OMEinsum, OMEinsumContractionOrders
using Base.Threads
using TensorBranching: ob_region, optimal_branches, ContractionTreeSlicer, uncompress, SlicedBranch, complexity
using OMEinsum: DynamicNestedEinsum
using OMEinsumContractionOrders: uniformsize
using TensorBranching: show_status, optimal_branches_ground_counting_induced_sparsity


function slice_dfs(p::SpinGlassProblem{INT, VT}, slicer::ContractionTreeSlicer, code::DynamicNestedEinsum{Int}, verbose::Int) where {INT, VT, RT}
    initial_branch = SlicedBranch(p, code, zero(eltype(p.J)))
    unfinished_slices = SlicedBranch[initial_branch]
    finished_slices = SlicedBranch[]
    scs = [complexity(initial_branch).sc]
    size_dict = uniformsize(uncompress(initial_branch.code), 2)

    primal_bound = 0.0
    finished_count = 0

    while !isempty(unfinished_slices)
        branch_to_slice = pop!(unfinished_slices)
        sc_to_slice = pop!(scs)

        verbose ≥ 1 && @info "slicing branch with sc: $sc_to_slice"

        cc = complexity(branch_to_slice)
        if cc.sc ≤ slicer.sc_target
            push!(finished_slices, branch_to_slice)
            finished_count += 1
            
        else
            new_slices, new_scs = _slice_single(branch_to_slice, primal_bound, slicer, size_dict, verbose)
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
    
    new_slices = temp_slices
    
    new_scs = [complexity(slice).sc for slice in temp_slices]
    
    return new_slices, new_scs
end