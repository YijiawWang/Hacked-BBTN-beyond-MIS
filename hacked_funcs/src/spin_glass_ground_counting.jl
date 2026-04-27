using OptimalBranching
using OptimalBranching.OptimalBranchingCore, OptimalBranching.OptimalBranchingMIS
using Graphs, TropicalNumbers, OMEinsum, OMEinsumContractionOrders
using Base.Threads
using TensorBranching: ob_region, optimal_branches, ContractionTreeSlicer, uncompress, SlicedBranch, complexity
using OMEinsum: DynamicNestedEinsum
using OMEinsumContractionOrders: uniformsize
using TensorBranching: optimal_branches_ground_counting_induced_sparsity
using TensorBranching: show_status, spinglass_linear_LP_bound, solve_slice, QP_bound, QIP_bound, QIP_gurobi_bound, QIP_value, QIP_value_gurobi
# NOTE: top-level GPU initialisation is intentionally not performed here.
# Callers that want GPU acceleration should call `CUDA.device!(...)` themselves
# *before* invoking `slice_dfs_lp(..., usecuda=true, ...)`.


function slice_dfs(p::SpinGlassProblem{INT, VT}, slicer::ContractionTreeSlicer, code::DynamicNestedEinsum{Int}, verbose::Int) where {INT, VT, RT}
    initial_branch = SlicedBranch(p, code, zero(eltype(p.J)))
    unfinished_slices = SlicedBranch[initial_branch]
    finished_slices = SlicedBranch[]
    scs = [complexity(initial_branch).sc]
    size_dict = uniformsize(uncompress(initial_branch.code), 2)

    primal_bound = 0.0
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

function slice_dfs_lp(p::SpinGlassProblem{INT, VT}, slicer::ContractionTreeSlicer, code::DynamicNestedEinsum{Int}, usecuda::Bool, verbose::Int;
                       on_finished_slice = nothing) where {INT, VT, RT}
    initial_branch = SlicedBranch(p, code, zero(eltype(p.J)))
    unfinished_slices = SlicedBranch[initial_branch]
    finished_slices = SlicedBranch[]
    scs = [complexity(initial_branch).sc]
    size_dict = uniformsize(uncompress(initial_branch.code), 2)

    lp_bound = QIP_gurobi_bound(initial_branch.p.g, initial_branch.p.J, initial_branch.p.h, time_limit=300.0)
    primal_bound = 0.0
    finished_count = 0

    verbose ≥ 1 && @info "initial lp_bound: $lp_bound"

    while !isempty(unfinished_slices)
        branch_to_slice = popfirst!(unfinished_slices)
        sc_to_slice = popfirst!(scs)

        verbose ≥ 1 && @info "slicing branch with sc: $sc_to_slice"

        cc = complexity(branch_to_slice)
        if cc.sc ≤ slicer.sc_target
            push!(finished_slices, branch_to_slice)
            finished_count += 1
            if on_finished_slice !== nothing
                try
                    on_finished_slice(branch_to_slice)
                catch err
                    @warn "on_finished_slice callback failed" exception=(err, catch_backtrace())
                end
            end
            if iszero(cc.sc)
                feasible_solution = branch_to_slice.r
            else
                feasible_solution = QIP_value_gurobi(branch_to_slice.p.g, branch_to_slice.p.J, branch_to_slice.p.h, time_limit=604800.0) + branch_to_slice.r
            end
            primal_bound = max(primal_bound, feasible_solution)
            verbose ≥ 1 && @info "feasible solution: $feasible_solution, primal bound: $primal_bound"
       
        else
            verbose ≥ 1 && @info "cc.sc: $(cc.sc)"
            new_slices, new_scs, lp_scores = _slice_single_lp(branch_to_slice, primal_bound, slicer, size_dict, verbose)
            verbose ≥ 1 && @info "new_scs: $(new_scs)"
            verbose ≥ 1 && @info "lp_scores: $(lp_scores)"
            if !isempty(new_slices)
                for i in length(new_slices):-1:1
                    pushfirst!(unfinished_slices, new_slices[i])
                    pushfirst!(scs, new_scs[i])
                    verbose ≥ 1 && @info "lp_scores: $(lp_scores[i])"
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

function _slice_single_lp(slice::ST, primal_bound::Float64, slicer::ContractionTreeSlicer, size_dict::Dict{Int, Int}, verbose::Int) where ST

    uncompressed_code = uncompress(slice.code)
    region, loss = ob_region(slice.p.g, uncompressed_code, slicer, slicer.region_selector, size_dict, verbose)
    branches = optimal_branches_ground_counting_induced_sparsity(slice.p, uncompressed_code, slice.r, primal_bound, slicer, region, size_dict, verbose)
    temp_slices = branches
    
    new_slices = SlicedBranch[]
    new_scs = Int[]
    lp_scores = Float64[]
    for (i, slice) in enumerate(temp_slices)
        lp_score = QIP_gurobi_bound(slice.p.g, slice.p.J, slice.p.h, time_limit=100.0) + slice.r
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