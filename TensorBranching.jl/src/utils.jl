function _log2_einsize(eincode::ET, size_dict::Dict{LT, Int}) where {ET, LT}
    return foldl((x, y) -> x + log2(size_dict[y]), eincode.iy, init = 0.0)
end

function get_subtree_pre(code::CT, size_dict::Dict{LT, Int}, threshold::T) where {CT, LT, T}
    for subtree in PreOrderDFS(code)
        (subtree isa LeafString) && continue
        if _log2_einsize(subtree.eins, size_dict) ≤ threshold
            return subtree
        end
    end
    # if no subtree larger than threshold, return the whole code
    return code
end

function get_subtree_post(code::CT, size_dict::Dict{LT, Int}, threshold::T) where {CT, LT, T}
    for subtree in PostOrderDFS(code)
        (subtree isa LeafString) && continue
        if _log2_einsize(subtree.eins, size_dict) ≥ threshold
            return subtree
        end
    end
    # if no subtree larger than threshold, return the whole code
    return code
end

function list_subtree(code::CT, size_dict::Dict{LT, Int}, threshold::T) where {CT, LT, T}
    subtrees = Vector{CT}()
    for subtree in PostOrderDFS(code)
        (subtree isa LeafString) && continue
        if _log2_einsize(subtree.eins, size_dict) ≥ threshold
            push!(subtrees, subtree)
        end
    end
    return subtrees
end

function most_label_subtree(code::CT, size_dict::Dict{LT, Int}, threshold::T) where {CT, LT, T}
    subtrees = list_subtree(code, size_dict, threshold)
    num_of_labels = [length(OMEinsum.uniquelabels(subtree)) for subtree in subtrees]
    return subtrees[findmax(num_of_labels)[2]]
end

# remove tensors from a network and reformulate the contraction tree

function remove_tensors!(code::DynamicNestedEinsum{LT}, tids::Vector{Int}) where LT
    _remove_tensors!(code, code.eins.iy, Set(tids))
    reform_tree!(code)
    return code
end

function remove_tensors(code::Union{DynamicNestedEinsum{LT}, SlicedEinsum{LT}}, tids::Vector{Int}) where LT
    ccode = true_eincode(deepcopy(code))
    remove_tensors!(ccode, tids)
    return ccode
end

function is_removed(code::CT, tids) where CT
    return hasfield(CT, :tensorindex) ? (code.tensorindex ∈ tids) : false
end

function _remove_tensors!(code::DynamicNestedEinsum{LT}, iy::Vector{LT}, tids::Set{Int}) where LT
    OMEinsum.isleaf(code) && return is_removed(code, tids) ? (true, LT[]) : (false, iy)
    
    dels = Int[]
    for (i, ix) in enumerate(code.eins.ixs)
        flag, new_ix = _remove_tensors!(code.args[i], ix, tids)
        if flag
            push!(dels, i)
        else
            code.eins.ixs[i] = new_ix
        end
    end
    deleteat!(code.eins.ixs, dels)
    deleteat!(code.args, dels)

    niy = isempty(code.eins.ixs) ? LT[] : union(code.eins.ixs...)
    dely = findall(x -> !(x ∈ niy), code.eins.iy)
    deleteat!(code.eins.iy, dely)
    return isempty(code.eins.ixs), code.eins.iy
end

function tensors_removed(code::Union{DynamicNestedEinsum{LT}, SlicedEinsum{LT}}, vs::Vector{LT}) where LT
    tids = Int[]
    ixd = Dict(OMEinsum._flatten(code))
    for tid in keys(ixd)
        if any(x -> x ∈ vs, ixd[tid])
            push!(tids, tid)
        end
    end
    return tids
end

function is_binary(code::DynamicNestedEinsum{LT}) where LT
    for node in PreOrderDFS(code)
        (node isa LeafString) && continue
        length(node.eins.ixs) != 2 && return false
    end
    return true
end

@inline vec_replace!(old::Vector{T}, new::Vector{T}) where T = append!(empty!(old), new)

# reformulate tree as binary tree
function reform_tree!(code::DynamicNestedEinsum{LT}) where LT
    idx = Dict(OMEinsum._flatten(code))
    isempty(idx) && return code

    # a special case, when the root of the tree is non binary
    reformed_code, _ = _reform_tree!(code, idx)

    if reformed_code.eins.ixs != code.eins.ixs
        vec_replace!(code.eins.ixs, reformed_code.eins.ixs)
        vec_replace!(code.args, reformed_code.args)
    end

    @assert is_binary(code)

    return code
end

function _reform_tree!(code::DynamicNestedEinsum{LT}, idx::Dict{Int, Vector{LT}}) where LT
    OMEinsum.isleaf(code) && return (code, idx[code.tensorindex])

    if length(code.args) == 1
        return _reform_tree!(code.args[1], idx)
    end
    
    for (i, arg) in enumerate(code.args)
        ncode, nix = _reform_tree!(arg, idx)
        code.args[i] = ncode
        code.eins.ixs[i] = nix
    end
    return code, code.eins.iy
end

function unsafe_flatten(code::DynamicNestedEinsum{LT}) where LT
    ixd = Dict(OMEinsum._flatten(code))
    DynamicEinCode([haskey(ixd, i) ? ixd[i] : LT[] for i=1:maximum(keys(ixd))], collect(OMEinsum.getiy(code.eins)))
end

function rethermalize(code::Union{DynamicNestedEinsum{LT}, SlicedEinsum{LT}}, size_dict::Dict{LT, Int}, βs::IT, ntrials::Int, niters::Int, sc_target::Int) where {LT, IT}
    return optimize_code(code, size_dict, TreeSA(initializer = :specified, βs=βs, ntrials=ntrials, niters=niters, score = ScoreFunction(sc_target = sc_target)))
end


# this part is about reindex the tree with a vertex map
function inverse_vmap(vmap::Vector{Int})
    ivmap = zeros(Int, maximum(vmap))
    for (i, v) in enumerate(vmap)
        ivmap[v] = i
    end
    return ivmap
end
function inverse_vmap_dict(vmap::Vector{Int})
    ivmap = Dict{Int, Int}()
    for (i, v) in enumerate(vmap)
        ivmap[v] = i
    end
    return ivmap
end

# compute edge mapping from induced subgraph to original graph
# returns a dictionary mapping edges in g_new (as tuples) to edges in g_old (as tuples)
function compute_emap(g_old::SimpleGraph{Int}, g_new::SimpleGraph{Int}, vmap::Vector{Int})
    emap = Dict{Tuple{Int, Int}, Tuple{Int, Int}}()
    ivmap = inverse_vmap_dict(vmap)
    
    for e in edges(g_new)
        src_new = src(e)
        dst_new = dst(e)
        src_old = vmap[src_new]
        dst_old = vmap[dst_new]
        emap[(src_new, dst_new)] = (src_old, dst_old)
    end
    
    return emap
end

# compute edge index mapping for J values in SpinGlass/Ising models
# In GenericTensorNetworks, J is a vector where J[i] corresponds to the i-th edge in edges(g)
# This function returns a vector mapping edge indices in g_new to edge indices in g_old
function compute_emap_for_J(g_old::SimpleGraph{Int}, g_new::SimpleGraph{Int}, vmap::Vector{Int})
    # Create a mapping from edge tuples to edge indices in the original graph
    edge_dict = Dict{Tuple{Int, Int}, Int}()
    for (idx, e) in enumerate(edges(g_old))
        u, v = src(e), dst(e)
        # Store both (u,v) and (v,u) to handle undirected edges
        edge_dict[(u, v)] = idx
        edge_dict[(v, u)] = idx
    end
    
    emap = Int[]
    for e in edges(g_new)
        src_new = src(e)
        dst_new = dst(e)
        src_old = vmap[src_new]
        dst_old = vmap[dst_new]
        edge_tuple = (src_old, dst_old)
        if haskey(edge_dict, edge_tuple)
            push!(emap, edge_dict[edge_tuple])
        else
            error("Edge ($src_new, $dst_new) in new graph does not map to a valid edge in original graph")
        end
    end
    
    return emap
end

# Map J and h values for SpinGlass problem after induced_subgraph
# J_old: edge coupling values for original graph (length = ne(g_old))
# h_old: vertex field values for original graph (length = nv(g_old))
# Returns: (J_new, h_new) where J_new corresponds to edges in g_new, h_new corresponds to vertices in g_new
function map_spin_glass_weights(g_old::SimpleGraph{Int}, g_new::SimpleGraph{Int}, vmap::Vector{Int}, J_old::VT, h_old::VT) where VT
    # Map h: h_new[i] = h_old[vmap[i]]
    h_new = h_old[vmap]
    
    # Map J: need to find which edges in g_new correspond to which edges in g_old
    emap = compute_emap_for_J(g_old, g_new, vmap)
    J_new = J_old[emap]
    
    return J_new, h_new
end

function induced_spin_glass_subproblem(g_old::SimpleGraph{Int}, g_new::SimpleGraph{Int}, vmap::Vector{Int}, J_old::Vector{T}, h_old::Vector{T}, val, removed_vertices::Vector{Int}, region::Vector{Int}) where T
    # Create edge index mapping for efficient lookup
    # In undirected graphs, edges may be stored as (u, v) or (v, u), so we store both
    edge_idx_map = Dict{Tuple{Int, Int}, Int}()
    for (idx, e) in enumerate(edges(g_old))
        u, w = src(e), dst(e)
        edge_idx_map[(u, w)] = idx
        edge_idx_map[(w, u)] = idx  # Store both directions for undirected edges
    end
    # Initialize h with original values, will be updated based on removed vertices
    h = copy(h_old)
    J = copy(J_old)
    r = zero(T)  # Initialize energy contribution accumulator
    # val is a bitstring relative to region, so we need to find the position of v in region
    for v in removed_vertices
        k = findfirst(==(v), region)
        if k === nothing
            error("Vertex $v not found in region")
        end
        if readbit(val, k) == 1  #si=-1
            r += -h[v]
            for n in neighbors(g_old, v)
                edge_idx = edge_idx_map[(v, n)]
                h[n] += -J[edge_idx]
                J[edge_idx] = 0.0
            end
        else  #si=1
            r += h[v]
            for n in neighbors(g_old, v)
                edge_idx = edge_idx_map[(v, n)]
                h[n] += J[edge_idx]
                J[edge_idx] = 0.0
            end
        end
       
    end

    # Map J and h to the induced subgraph
    J_new, h_new = map_spin_glass_weights(g_old, g_new, vmap, J, h)
    return J_new, h_new, r
end


# Find edge index in J vector for a given edge (u, v)
# In undirected graphs, edges may be stored as (u, v) or (v, u), so we check both
# Returns the index in J vector, or throws error if edge not found
function find_edge_index(g::SimpleGraph{Int}, u::Int, v::Int)
    # Try both (u, v) and (v, u) since edges may be stored in either order
    for (idx, e) in enumerate(edges(g))
        if (src(e) == u && dst(e) == v) || (src(e) == v && dst(e) == u)
            return idx
        end
    end
    error("Edge ($u, $v) not found in graph")
end

# reindex the tree with a vertex map
function reindex_tree!(code::DynamicNestedEinsum{LT}, vmap::Vector{Int}) where LT
    _reindex_tree!(code, inverse_vmap(vmap))
    return code
end
function _reindex_tree!(code::DynamicNestedEinsum{LT}, ivmap::Vector{Int}) where LT
    OMEinsum.isleaf(code) && return nothing
    
    # notice that eins.ixs[i] is actually eins.args[i].eins.iy, the same in memory
    for (i, ix) in enumerate(code.eins.ixs)
        for (j, ixi) in enumerate(ix)
            ix[j] = ivmap[ixi]
        end
        _reindex_tree!(code.args[i], ivmap)
    end
    
    return nothing
end

# Personally, I think the design of SlicedEinsum is terrible, same function, different output type
function true_eincode(code::Union{DynamicNestedEinsum{LT}, SlicedEinsum{LT}}) where LT
    return code isa SlicedEinsum ? code.eins : code
end

function mis_complexity(code)
    return contraction_complexity(code, uniformsize(code, 2))
end

function random_ksg(m::Int, n::Int, rho::Float64, seed::Int)
    Random.seed!(seed)
    ksg = SimpleGraph(GenericTensorNetworks.random_diagonal_coupled_graph(m, n, rho))
    return ksg
end

function contraction_peak_memory(code::NestedEinsum, size_dict)
    ixs = getixsv(code)
    tscs = Float64[]
    initial_sc = sum(prod(Float64(size_dict[i]) for i in ix) for ix in ixs)
    push!(tscs, initial_sc)
    _tsc!(tscs, code, size_dict, ixs)
    return maximum(log2.(tscs))
end

function _tsc!(tscs, code, size_dict, ixs)
    isleaf(code) && return zero(Float64)

    freed_size = zero(Float64)
    for subcode in code.args
        freed_size += _tsc!(tscs, subcode, size_dict, ixs)
    end

    future_freed_size = isempty(code.eins.ixs) ? 0.0 : sum(isempty(ix) ? 1.0 : prod(Float64(size_dict[i]) for i in ix) for ix in code.eins.ixs)
    allocated_size = isempty(code.eins.iy) ? 1.0 : prod(Float64(size_dict[i]) for i in code.eins.iy)
    new_size = tscs[end] + allocated_size - freed_size
    push!(tscs, new_size)
    return future_freed_size
end

function contraction_all_memory(code::NestedEinsum, size_dict)
    return log2(_ssc!(code, size_dict))
end

function _ssc!(code, size_dict)
    isleaf(code) && return zero(Float64)
    t = isempty(code.eins.iy) ? 1.0 : prod(Float64(size_dict[i]) for i in code.eins.iy)
    return t + sum(_ssc!(subcode, size_dict) for subcode in code.args)
end

function show_status(scs, sc_target, num_unfinished, num_finished)
    @info "current num of unfinished slices: $num_unfinished, finished slices: $num_finished"
    counts = zeros(Int, Int(maximum(scs) - minimum(scs) + 1))
    for sc in scs
        counts[Int(sc - minimum(scs) + 1)] += 1
    end
    println(barplot(Int(minimum(scs)):Int(maximum(scs)), counts, xlabel = "num of slices", ylabel = "sc, target = $(sc_target)"))
end

function find_all_cliques(graph::AbstractGraph, min_size::Int=3)
    P = Set(vertices(graph))  
    R = Set{Int}()           
    X = Set{Int}()           
    cliques = Vector{Set{Int}}()
    
    bron_kerbosch(graph, R, P, X, cliques, min_size)
    return cliques
end

function bron_kerbosch(graph::AbstractGraph, R::Set{Int}, P::Set{Int}, X::Set{Int}, cliques::Vector{Set{Int}}, min_size::Int)
    if isempty(P) && isempty(X)

        if length(R) >= min_size
            push!(cliques, copy(R))
        end
        return
    end
    
    pivot = first(union(P, X))
    
    for v in setdiff(P, Set(neighbors(graph, pivot)))
        neighbors_v = Set(neighbors(graph, v))
        bron_kerbosch(graph, 
            union(R, Set([v])),
            intersect(P, neighbors_v),
            intersect(X, neighbors_v),
            cliques,
            min_size)
        P = setdiff(P, Set([v]))
        X = union(X, Set([v]))
    end
end

function LP_MWIS(graph::SimpleGraph,weights::Vector{T}; optimizer = SCIP.Optimizer) where T
    model = Model(optimizer)  
    set_silent(model)  
    nsc = nv(graph)
    @variable(model, 0 <= x[i = 1:nsc] <= 1)
    @objective(model, Max, sum(weights[i]*x[i] for i in 1:nsc))

    for e in edges(graph)
        i = src(e)
        j = dst(e)
        @constraint(model, x[i]+x[j] <= 1)
    end
    cliques = find_all_cliques(graph,3)
    for clique in cliques
        @constraint(model, sum(x[i] for i in clique) <= 1)
    end
    optimize!(model)
    
    return objective_bound(model)
end

function IP_MWIS(graph::SimpleGraph,weights::Vector{T}; optimizer = SCIP.Optimizer) where T
    model = Model(optimizer)  
    set_silent(model)
    nsc = nv(graph)
    @variable(model, 0 <= x[i = 1:nsc] <= 1, Int)
    @objective(model, Max, sum(weights[i]*x[i] for i in 1:nsc))

    for e in edges(graph)
        i = src(e)
        j = dst(e)
        @constraint(model, x[i]+x[j] <= 1)
    end
    optimize!(model)
    
    return objective_bound(model)
end

function quick_feasible_solution(graph::SimpleGraph,weights::Vector{T},time_limit::Float64; optimizer = SCIP.Optimizer) where T
    model = Model(optimizer)  
    set_silent(model)  
    set_time_limit_sec(model, time_limit)
    set_optimizer_attribute(model, "limits/gap", 0.1)
    nsc = nv(graph)
    @variable(model, 0 <= x[i = 1:nsc] <= 1, Int)
    @objective(model, Max, sum(weights[i]*x[i] for i in 1:nsc))
    for e in edges(graph)
        i = src(e)
        j = dst(e)
        @constraint(model, x[i]+x[j] <= 1)
    end
    cliques = find_all_cliques(graph,3)
    for clique in cliques
        @constraint(model, sum(x[i] for i in clique) <= 1)
    end
    optimize!(model)
    return objective_value(model)
end


function spinglass_linear_IP(g::SimpleGraph, J::Vector{T}, h::Vector{T}; optimizer = SCIP.Optimizer) where T
    model = Model(optimizer)  
    set_silent(model)
    nsc = nv(g)+ne(g)
    @variable(model, 0 <= x[i = 1:nsc] <= 1, Int)
    Q = [4*j for j in J]
    c = [-2*h[i] for i in 1:nv(g)]
    for (idx, e) in enumerate(edges(g))
        i = src(e)
        j = dst(e)
        c[i] += -2*J[idx]
        c[j] += -2*J[idx]
        @constraint(model, x[nv(g)+idx] >= x[i] + x[j] - 1)
        @constraint(model, x[nv(g)+idx] <= x[i])
        @constraint(model, x[nv(g)+idx] <= x[j])
    end
    @objective(model, Max, sum(Q[i-nv(g)]*x[i] for i in (nv(g)+1):nsc) + sum(c[i]*x[i] for i in 1:nv(g)))
    optimize!(model)
    return objective_value(model) + sum(J[i] for i in 1:ne(g)) + sum(h[i] for i in 1:nv(g))
end

function spinglass_linear_LP_bound(g::SimpleGraph, J::Vector{T}, h::Vector{T}; optimizer = SCIP.Optimizer) where T
    model = Model(optimizer)  
    set_silent(model)
    nsc = nv(g)+ne(g)
    @variable(model, 0 <= x[i = 1:nsc] <= 1)
    Q = [4*j for j in J]
    c = [-2*h[i] for i in 1:nv(g)]
    for (idx, e) in enumerate(edges(g))
        i = src(e)
        j = dst(e)
        c[i] += -2*J[idx]
        c[j] += -2*J[idx]
        @constraint(model, x[nv(g)+idx] >= x[i] + x[j] - 1)
        @constraint(model, x[nv(g)+idx] <= x[i])
        @constraint(model, x[nv(g)+idx] <= x[j])
    end
    @objective(model, Max, sum(Q[i-nv(g)]*x[i] for i in (nv(g)+1):nsc) + sum(c[i]*x[i] for i in 1:nv(g)))
    optimize!(model)
    return objective_value(model) + sum(J[i] for i in 1:ne(g)) + sum(h[i] for i in 1:nv(g))
end

function QP_bound(g::SimpleGraph, J::Vector{T}, h::Vector{T}; optimizer = SCIP.Optimizer, time_limit::Float64 = 10.0) where T
    n = nv(g)
    # Step 1: Convert J, h to Q matrix and c vector
    c = [-2*h[i] for i in 1:nv(g)]
    for (idx, e) in enumerate(edges(g))
        i = src(e)
        j = dst(e)
        c[i] += -2*J[idx]
        c[j] += -2*J[idx]
    end
    
    
    # lambda_min_estimate = Inf
    # for i in 1:n
    #     radius = sum(abs(Q[i, j]) for j in 1:n if j != i)
    #     center = Q[i, i]
    #     lambda_min_estimate = min(lambda_min_estimate, center - radius)
    # end
    
    # lambda_shift = max(zero(T), -lambda_min_estimate)
    # if lambda_shift > 0
    #     for i in 1:n
    #         Q[i, i] += lambda_shift
    #     end
    # end
    
    model = Model(optimizer)
    set_silent(model)
    # Set time limit using JuMP's built-in function
    set_time_limit_sec(model, time_limit)
    
    # Variables: continuous in [0, 1]
    @variable(model, 0 <= x[i = 1:n] <= 1)
    
    # Build quadratic objective efficiently: only include non-zero terms
    # Q is sparse (only edges have non-zero values), so we iterate over edges instead of all pairs
    # This avoids creating O(n²) terms for large graphs
    quadratic_terms = [4*J[idx] * x[src(e)] * x[dst(e)] for (idx, e) in enumerate(edges(g))]
    quadratic_obj = isempty(quadratic_terms) ? 0.0 : sum(quadratic_terms)
    
    linear_obj = sum(c[i] * x[i] for i in 1:n)
    @objective(model, Max, quadratic_obj + linear_obj)
    optimize!(model)
    # Check optimization status
    status = termination_status(model)
    
    # Handle empty collections in sum
    J_sum = isempty(J) ? zero(T) : sum(J)
    h_sum = isempty(h) ? zero(T) : sum(h)
    
    # If optimization completed successfully, return objective value
    if status == JuMP.MOI.OPTIMAL || status == JuMP.MOI.LOCALLY_SOLVED
       try
            obj_bound = objective_bound(model)
            return obj_bound + J_sum + h_sum
        catch e
            error("Failed to get objective value or bound from SCIP: $e")
        end
    else
        # If time limit reached or other issues, return upper bound
        # For maximization, objective_bound gives the upper bound
        try
            obj_bound = objective_bound(model)
            return obj_bound + J_sum + h_sum
        catch e
            error("QP optimization failed with status: $status, and failed to get bound: $e")
        end
    end
end 


function QIP_bound(g::SimpleGraph, J::Vector{T}, h::Vector{T}; optimizer = SCIP.Optimizer, time_limit::Float64 = 10.0) where T
    n = nv(g)
    # Step 1: Convert J, h to Q matrix and c vector
    c = [-2*h[i] for i in 1:nv(g)]
    for (idx, e) in enumerate(edges(g))
        i = src(e)
        j = dst(e)
        c[i] += -2*J[idx]
        c[j] += -2*J[idx]
    end
    
    
    # lambda_min_estimate = Inf
    # for i in 1:n
    #     radius = sum(abs(Q[i, j]) for j in 1:n if j != i)
    #     center = Q[i, i]
    #     lambda_min_estimate = min(lambda_min_estimate, center - radius)
    # end
    
    # lambda_shift = max(zero(T), -lambda_min_estimate)
    # if lambda_shift > 0
    #     for i in 1:n
    #         Q[i, i] += lambda_shift
    #     end
    # end
    
    model = Model(optimizer)
    set_silent(model)
    # Set time limit using JuMP's built-in function
    set_time_limit_sec(model, time_limit)
    
    # Variables: continuous in [0, 1]
    @variable(model, 0 <= x[i = 1:n] <= 1, Int)
    
    # Build quadratic objective efficiently: only include non-zero terms
    # Q is sparse (only edges have non-zero values), so we iterate over edges instead of all pairs
    # This avoids creating O(n²) terms for large graphs
    quadratic_terms = [4*J[idx] * x[src(e)] * x[dst(e)] for (idx, e) in enumerate(edges(g))]
    quadratic_obj = isempty(quadratic_terms) ? 0.0 : sum(quadratic_terms)
    
    linear_obj = sum(c[i] * x[i] for i in 1:n)
    @objective(model, Max, quadratic_obj + linear_obj)
    
    optimize!(model)
    
    # Check optimization status
    status = termination_status(model)
    
    # Handle empty collections in sum
    J_sum = isempty(J) ? zero(T) : sum(J)
    h_sum = isempty(h) ? zero(T) : sum(h)
    
    # If optimization completed successfully, return objective value
    if status == JuMP.MOI.OPTIMAL || status == JuMP.MOI.LOCALLY_SOLVED
        try
            obj_bound = objective_bound(model)
            return obj_bound + J_sum + h_sum
        catch e
            error("Failed to get objective value or bound from SCIP: $e")
        end
    else
        # If time limit reached or other issues, return upper bound
        # For maximization, objective_bound gives the upper bound
        try
            obj_bound = objective_bound(model)
            return obj_bound + J_sum + h_sum
        catch e
            error("QP optimization failed with status: $status, and failed to get bound: $e")
        end
    end
end 


function induced_sat_subproblem(g_old::SimpleGraph, clauses::Vector{Vector{Int}}, val, removed_vertices::Vector{Int}, region::Vector{Int})
    r = 0.0
    n_clauses_old = length(clauses)
    n_vars_old = nv(g_old) - n_clauses_old
    region_index_map = Dict{Int, Int}()
    for (idx, v) in enumerate(region)
        region_index_map[v] = idx
    end
    # Iterate through clauses, simultaneously check satisfaction and remove relevant literals
    new_clauses = Vector{Int}[]
    removed_vars_set = Set(removed_vertices)
    clause_to_be_removed = falses(n_clauses_old)  # Record which clauses are to be removed (satisfied or empty)
    # Precompute values for each removed_vertex
    removed_vars_values = Dict{Int, Int}()
    for v in removed_vertices
        k = get(region_index_map, v, nothing)
        if k !== nothing
            if val isa AbstractVector
                removed_vars_values[v] = val[k]
            else
                removed_vars_values[v] = readbit(val, k)
            end
        end
    end
    
    for (clause_idx, clause) in enumerate(clauses)
        removing_clause = false
        new_clause = Int[]
        
        for literal in clause
            var_idx = abs(literal)
            
            if var_idx in removed_vars_set
                # Variable is in removed_vertices, check if clause is satisfied
                val_bit = removed_vars_values[var_idx]
                is_positive = (literal > 0)
                
                # Check if satisfied: positive literal and val == 1, or negative literal and val == 0
                if (is_positive && val_bit == 1) || (!is_positive && val_bit == 0)
                    # Clause is satisfied, remove entirely (don't add to new_clause)
                    removing_clause = true
                    r += 1.0
                    break  # Clause is satisfied, no need to continue checking
                end
                # If not satisfied, don't add this literal to the new clause (equivalent to removal)
            else
                # Variable is not in removed_vertices, keep this literal
                push!(new_clause, literal)
            end
        end
        
        # Record clauses to be removed (satisfied or empty)
        if removing_clause || isempty(new_clause)
            clause_to_be_removed[clause_idx] = true
        else
            push!(new_clauses, new_clause)
        end
    end
    
    # Step 3: Count which variables are completely deleted (no longer appear in any clause)
    used_vars = Set{Int}()
    for clause in new_clauses
        for literal in clause
            push!(used_vars, abs(literal))
        end
    end
    
    # Find completely deleted variables (in region but not in used_vars)
    deleted_vars = setdiff(Set(region), used_vars)
    
    # Step 4: Find deleted clause vertices
    # Clause vertex numbering: n_vars_old + clause_idx
    deleted_clause_vertices = Int[]
    for (idx, removing_clause) in enumerate(clause_to_be_removed)
        if removing_clause
            push!(deleted_clause_vertices, n_vars_old + idx)
        end
    end
    
    # Step 5: Create new graph, remove deleted vertices
    # Vertices to remove: deleted_vars (variable vertices) and deleted_clause_vertices (clause vertices)
    all_removed_vertices = union(Set(deleted_vars), Set(deleted_clause_vertices))
    remaining_vertices = setdiff(Set(1:nv(g_old)), all_removed_vertices)
    
    # Create induced subgraph
    g_new, vmap = induced_subgraph(g_old, collect(remaining_vertices))
    
    # Step 6: Reorder graph vertices so variable vertices come first, clause vertices come after
    # vmap[i] is the old vertex index corresponding to the i-th vertex in the new graph
    # Create inverse mapping: old_vertex -> new_vertex_position (position in g_new)
    ivmap = Dict{Int, Int}()
    for (new_pos, old_vertex) in enumerate(vmap)
        ivmap[old_vertex] = new_pos
    end
    
    # Find all remaining variable vertices and clause vertices
    remaining_vars = sort([v for v in vmap if v <= n_vars_old])
    remaining_clauses = sort([v for v in vmap if v > n_vars_old])  # Clause vertices in old graph have index > n_vars_old
    
    # Create new vertex order: variables first, clauses after
    new_vertex_order = vcat(remaining_vars, remaining_clauses)
    n_vars_new = length(remaining_vars)
    n_clauses_new = length(remaining_clauses)
    
    # Create mapping from old vertex to new vertex position (in reordered graph)
    vertex_remap = Dict{Int, Int}()
    for (new_pos, old_vertex) in enumerate(new_vertex_order)
        vertex_remap[old_vertex] = new_pos
    end
    
    # Create mapping from position in g_new to position after reordering
    g_new_to_reordered = Dict{Int, Int}()
    for old_vertex in new_vertex_order
        g_new_pos = ivmap[old_vertex]
        reordered_pos = vertex_remap[old_vertex]
        g_new_to_reordered[g_new_pos] = reordered_pos
    end
    
    # Create reordered graph
    g_reordered = SimpleGraph(length(new_vertex_order))
    for e in edges(g_new)
        src_old = src(e)
        dst_old = dst(e)
        src_new = g_new_to_reordered[src_old]
        dst_new = g_new_to_reordered[dst_old]
        add_edge!(g_reordered, src_new, dst_new)
    end
    
    # Create variable mapping: old_var_index -> new_var_index (variables are continuously numbered from 1 after reordering)
    var_remap = Dict(v => new_idx for (new_idx, v) in enumerate(remaining_vars))
    
    # Remap variable indices in clauses
    new_clauses_remapped = Vector{Int}[]
    for clause in new_clauses
        new_clause = Int[]
        for literal in clause
            old_var = abs(literal)
            if haskey(var_remap, old_var)
                new_var = var_remap[old_var]
                # 保持正负号
                if literal > 0
                    push!(new_clause, new_var)
                else
                    push!(new_clause, -new_var)
                end
            end
        end
        if !isempty(new_clause)
            push!(new_clauses_remapped, new_clause)
        end
    end
    
    return g_reordered, new_clauses_remapped, remaining_vars, r
end