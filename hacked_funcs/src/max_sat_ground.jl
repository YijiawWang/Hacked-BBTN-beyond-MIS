using OptimalBranching
using OptimalBranching.OptimalBranchingCore, OptimalBranching.OptimalBranchingMIS
using Graphs, TropicalNumbers, OMEinsum, OMEinsumContractionOrders
using Base.Threads
using TensorBranching: ob_region, ob_region_factorg, optimal_branches, ContractionTreeSlicer, uncompress, SlicedBranch, complexity
using OMEinsum: DynamicNestedEinsum
using OMEinsumContractionOrders: uniformsize
using TensorBranching: optimal_branches_ground_induced_sparsity
using TensorBranching: show_status
using CUDA, CuTropicalGEMM, BenchmarkTools
CUDA.device!(4)


function slice_dfs(p::MaxSatProblem{INT, VT}, slicer::ContractionTreeSlicer, code::DynamicNestedEinsum{Int}, verbose::Int) where {INT, VT, RT}
    initial_branch = SlicedBranch(p, code, 0.0)
    unfinished_slices = SlicedBranch[initial_branch]
    finished_slices = SlicedBranch[]
    scs = [complexity(initial_branch).sc]
    size_dict = uniformsize(uncompress(initial_branch.code), 2)

    finished_count = 0

    while !isempty(unfinished_slices)
        branch_to_slice = popfirst!(unfinished_slices)
        sc_to_slice = popfirst!(scs)

        verbose ≥ 1 && @info "slicing branch with sc: $sc_to_slice"

        cc = complexity(branch_to_slice)
        if cc.sc ≤ slicer.sc_target
            push!(finished_slices, branch_to_slice)
            finished_count += 1
            
        else
            new_slices, new_scs = _slice_single(branch_to_slice, slicer, size_dict, verbose)
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

function _slice_single(slice::ST, slicer::ContractionTreeSlicer, size_dict::Dict{Int, Int}, verbose::Int) where ST

    uncompressed_code = uncompress(slice.code)
    n_clauses = length(slice.p.clauses)
    region, loss = ob_region_factorg(slice.p.g, n_clauses, uncompressed_code, slicer, slicer.region_selector, size_dict, verbose)
    println("region: $region")
    branches = optimal_branches_ground_induced_sparsity(slice.p, uncompressed_code, slice.r, slicer, region, size_dict, verbose)
    temp_slices = branches
    
    new_slices = temp_slices
    
    new_scs = [complexity(slice).sc for slice in temp_slices]
    
    return new_slices, new_scs
end
