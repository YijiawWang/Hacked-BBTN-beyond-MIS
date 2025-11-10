"""
    alpha(g::SimpleGraph, weights::AbstractVector{WT}, openvertices::Vector{Int}) where WT      

Compute the alpha tensor for a given weighted sub-graph.

# Arguments
- `g::SimpleGraph`: The input sub-graph.
- `weights::AbstractVector{WT}`: The weights of the sub-graph.
- `openvertices::Vector{Int}`: The open vertices of the sub-graph.

# Returns
- The alpha tensor.
"""
function alpha(g::SimpleGraph, weights::AbstractVector{WT}, openvertices::Vector{Int}) where WT
	problem = GenericTensorNetwork(IndependentSet(g, weights); openvertices, optimizer = GreedyMethod())
	alpha_tensor = solve(problem, SizeMax())
    return alpha_tensor
end

"""
    reduced_alpha(g::SimpleGraph, weights::AbstractVector{WT}, openvertices::Vector{Int}) where WT      

Compute the reduced alpha tensor for a given weighted sub-graph.

# Arguments
- `g::SimpleGraph`: The input sub-graph.
- `weights::AbstractVector{WT}`: The weights of the sub-graph.
- `openvertices::Vector{Int}`: The open vertices of the sub-graph.

# Returns
- The reduced alpha tensor.
"""
function reduced_alpha(g::SimpleGraph, weights::AbstractVector{WT}, openvertices::Vector{Int}) where WT
	problem = GenericTensorNetwork(IndependentSet(g, weights); openvertices, optimizer = GreedyMethod())
	alpha_tensor = solve(problem, SizeMax())
	return mis_compactify!(alpha_tensor)
end

function _reduced_alpha_configs(g::SimpleGraph, weights::AbstractVector{WT}, openvertices::Vector{Int}, potential) where WT
	problem = GenericTensorNetwork(IndependentSet(g, weights); openvertices, optimizer = GreedyMethod())
	alpha_tensor = solve(problem, SizeMax())
	alpha_configs = solve(problem, ConfigsMax(; bounded=false))
	reduced_alpha_tensor = mis_compactify!(alpha_tensor; potential)
	# set the corresponding entries to 0.
	alpha_configs[map(iszero, reduced_alpha_tensor)] .= Ref(zero(eltype(alpha_configs)))
	# post processing
	configs = alpha_configs
	return configs
end

"""
    reduced_alpha_configs(solver::TensorNetworkSolver, graph::SimpleGraph, weights::AbstractVector{WT}, openvertices::Vector{Int}, potential=nothing) where WT

Compute the truth table according to the non-zero entries of the reduced alpha tensor for a given weighted sub-graph.

# Arguments
- `solver::TensorNetworkSolver`: The solver to use.
- `graph::SimpleGraph`: The input sub-graph.
- `weights::AbstractVector{WT}`: The weights of the sub-graph.
- `openvertices::Vector{Int}`: The open vertices of the sub-graph.
- `potential::Union{Nothing, Vector{WT}}`: The potential of the open vertices, defined as the sum of the weights of the nearestneighbors of the open vertices.

# Returns
- The truth table.
"""
function reduced_alpha_configs(::TensorNetworkSolver, graph::SimpleGraph, weights::AbstractVector{WT}, openvertices::Vector{Int}, potential=nothing) where WT
    configs = _reduced_alpha_configs(graph, weights, openvertices, potential)
    return BranchingTable(configs)
end

function OptimalBranchingCore.BranchingTable(arr::AbstractArray{<:CountingTropical{<:Real, <:ConfigEnumerator{N}}}) where N
    return BranchingTable(N, filter(!isempty, vec(map(collect_configs, arr))))
end

# New version: each (ov config, vs config) combination is a separate row
function OptimalBranchingCore.BranchingTable(arr::AbstractArray{<:CountingTropical{<:Real, <:ConfigEnumerator{N}}}, separate_rows::Bool) where N
    if separate_rows
        # Flatten: each config becomes its own row (one row per (ov, vs) combination)
        # collect_configs returns StaticBitVector, which BranchingTable constructor will convert to integers
        all_rows = Vector{Vector}()
        for elem in arr
            configs = collect_configs(elem)
            for config in configs
                if !isempty(config)
                    push!(all_rows, [config])
                end
            end
        end
        return BranchingTable(N, all_rows)
    else
        # Original behavior: group by open vertices (same ov config -> same row)
        return BranchingTable(N, filter(!isempty, vec(map(collect_configs, arr))))
    end
end

# Helper function to extract subset configuration from tn_vs config to vs config
function _extract_vs_config(config, vs::Vector{Int}, tn_vs::Vector{Int})
    # config corresponds to tn_vs configuration
    # Extract the bits corresponding to vs (which is a subset of tn_vs)
    vs_config = Int[]
    for v in vs
        idx = findfirst(==(v), tn_vs)
        if idx !== nothing
            # Access the bit at position idx (1-based) from config
            # config can be StaticBitVector or similar, which supports indexing
            bit_val = config[idx]
            push!(vs_config, Int(bit_val))
        end
    end
    return vs_config
end

function OptimalBranchingCore.BranchingTable(arr::AbstractArray{<:CountingTropical{<:Real, <:ConfigEnumerator{N}}}, separate_rows::Bool, vs::Vector{Int}, tn_vs::Vector{Int}) where N
    if separate_rows
        # Flatten: each config becomes its own row (one row per (ov, vs) combination)
        # collect_configs returns StaticBitVector, which BranchingTable constructor will convert to integers
        # Extract vs configurations from tn_vs configurations and remove duplicates
        n = length(vs)
        seen_configs = Set{Vector{Int}}()
        collected_rows = Vector{Vector{Vector{Int}}}()
        
        for elem in arr
            configs = collect_configs(elem)
            for config in configs
                if !isempty(config)
                    # Extract vs configuration from tn_vs configuration
                    vs_config = _extract_vs_config(config, vs, tn_vs)
                    # Remove duplicates by checking if we've seen this configuration
                    if !isempty(vs_config) && length(vs_config) == n && vs_config ∉ seen_configs
                        push!(seen_configs, vs_config)
                        # Each row contains one config: [vs_config] is Vector{Vector{Int}}
                        push!(collected_rows, [vs_config])
                    end
                end
            end
        end
        
        # Handle empty case
        if isempty(collected_rows)
            INT = BitBasis.longinttype(n, 2)
            return BranchingTable{INT}(n, Vector{Vector{INT}}())
        end
        
        # Use collected_rows directly - it's already Vector{Vector{Vector{Int}}}
        return BranchingTable(n, collected_rows)
    end
end

# Now we collect these configurations into a vector.
function collect_configs(cfg::CountingTropical{<:Real, <:ConfigEnumerator}, symbols::Union{Nothing, String}=nothing)
    cs = cfg.c.data
    symbols === nothing ? cs : [String([symbols[i] for (i, v) in enumerate(x) if v == 1]) for x in cs]
end

# Collect configs from ConfigEnumerator directly
function collect_configs(cfg::ConfigEnumerator, symbols::Union{Nothing, String}=nothing)
    cs = cfg.data
    symbols === nothing ? cs : [String([symbols[i] for (i, v) in enumerate(x) if v == 1]) for x in cs]
end

# New BranchingTable constructor for ConfigEnumerator arrays
function OptimalBranchingCore.BranchingTable(arr::AbstractArray{<:ConfigEnumerator{N}}, separate_rows::Bool) where N
    if separate_rows
        # Flatten: each config becomes its own row (one row per (ov, vs) combination)
        all_rows = Vector{Vector}()
        for elem in arr
            configs = collect_configs(elem)
            for config in configs
                if !isempty(config)
                    push!(all_rows, [config])
                end
            end
        end
        return BranchingTable(N, all_rows)
    else
        # Original behavior: group by open vertices (same ov config -> same row)
        return BranchingTable(N, filter(!isempty, vec(map(collect_configs, arr))))
    end
end

function OptimalBranchingCore.branching_table(p::MISProblem, solver::TensorNetworkSolver, vs::Vector{Int})
    ovs = open_vertices(p.g, vs)
    subg, vmap = induced_subgraph(p.g, vs)
	potential = [sum(p.weights[collect(setdiff(neighbors(p.g, v), vs))]) for v in ovs]
    tbl = reduced_alpha_configs(solver, subg, p.weights[vmap], Int[findfirst(==(v), vs) for v in ovs], potential)
    if solver.prune_by_env
        tbl = prune_by_env(tbl, p, vs)
    end
    return tbl
end

function OptimalBranchingCore.branching_table_ground_counting(p::MISProblem, solver::TensorNetworkSolver, vs::Vector{Int})
    ovs = open_vertices(p.g, vs)
    subg, vmap = induced_subgraph(p.g, vs)
	openvertices = Int[findfirst(==(v), vs) for v in ovs]
    problem = GenericTensorNetwork(IndependentSet(subg, p.weights[vmap]); openvertices, optimizer = GreedyMethod())
    alpha_tensor = solve(problem, SizeMax())
	alpha_configs = solve(problem, ConfigsMax(; bounded=false))
	alpha_configs[map(iszero, alpha_tensor)] .= Ref(zero(eltype(alpha_configs)))
	# post processing
	configs = alpha_configs
    
    return BranchingTable(configs, true)
end

function OptimalBranchingCore.branching_table_counting(p::MISProblem, solver::TensorNetworkSolver, vs::Vector{Int})
    ovs = open_vertices(p.g, vs)
    subg, vmap = induced_subgraph(p.g, vs)
	openvertices = Int[findfirst(==(v), vs) for v in ovs]
    problem = GenericTensorNetwork(IndependentSet(subg, p.weights[vmap]); openvertices, optimizer = GreedyMethod())
    alpha_tensor = solve(problem, SizeMax())
	all_configs = solve(problem, ConfigsAll())
	all_configs[map(iszero, alpha_tensor)] .= Ref(zero(eltype(all_configs)))
	# post processing
	configs = all_configs
    # Use separate_rows=true to put each (ov config, vs config) combination in a separate row
    return BranchingTable(configs, true)
end

function OptimalBranchingCore.branching_table_ground_counting_induced_sparsity(p::MISProblem, solver::TensorNetworkSolver, vs::Vector{Int}, primal_bound::T) where T
    tn_vs = union(OptimalBranchingMIS.open_neighbors(p.g, vs), vs)
    ovs = open_vertices(p.g, vs)
    tn_ovs = open_vertices(p.g, tn_vs)
    subg, vmap = induced_subgraph(p.g, vs)
    tn_subg, tn_vmap = induced_subgraph(p.g, tn_vs)
	tn_openvertices = Int[findfirst(==(v), tn_vs) for v in tn_ovs]
    problem = GenericTensorNetwork(IndependentSet(tn_subg, p.weights[tn_vmap]); openvertices=tn_openvertices, optimizer = GreedyMethod())
    alpha_tensor = solve(problem, SizeMax())
	alpha_configs = solve(problem, ConfigsMax(; bounded=false))
	alpha_configs[map(iszero, alpha_tensor)] .= Ref(zero(eltype(alpha_configs)))
	# post processing
	configs = alpha_configs
    
    return BranchingTable(configs, true, vs, tn_vs)
end

# Helper function to map J and h values for SpinGlass problem after induced_subgraph
# This is a copy of the function from TensorBranching to avoid circular dependencies
function _map_spin_glass_weights(g_old::SimpleGraph{Int}, g_new::SimpleGraph{Int}, vmap::Vector{Int}, J_old::VT, h_old::VT) where VT
    # Map h: h_new[i] = h_old[vmap[i]]
    h_new = h_old[vmap]
    
    # Map J: need to find which edges in g_new correspond to which edges in g_old
    # Create a mapping from edge tuples to edge indices in the original graph
    edge_dict = Dict{Tuple{Int, Int}, Int}()
    for (idx, e) in enumerate(edges(g_old))
        u, v = src(e), dst(e)
        edge_dict[(u, v)] = idx
        edge_dict[(v, u)] = idx  # Store both directions for undirected edges
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
    J_new = J_old[emap]
    
    return J_new, h_new
end

function OptimalBranchingCore.branching_table_ground_counting_induced_sparsity(p::SpinGlassProblem, solver::TensorNetworkSolver, vs::Vector{Int}, primal_bound::T) where T
    tn_vs = union(OptimalBranchingMIS.open_neighbors(p.g, vs), vs)
    ovs = open_vertices(p.g, vs)
    tn_ovs = open_vertices(p.g, tn_vs)
    subg, vmap = induced_subgraph(p.g, vs)
    tn_subg, tn_vmap = induced_subgraph(p.g, tn_vs)
	tn_openvertices = Int[findfirst(==(v), tn_vs) for v in tn_ovs]
    # Map J and h to the subgraph
    J_subg, h_subg = _map_spin_glass_weights(p.g, tn_subg, tn_vmap, p.J, p.h)
    problem = GenericTensorNetwork(SpinGlass(tn_subg, J_subg, h_subg); openvertices=tn_openvertices, optimizer = GreedyMethod())
    alpha_tensor = solve(problem, SizeMax())
	alpha_configs = solve(problem, ConfigsMax(; bounded=false))
	alpha_configs[map(iszero, alpha_tensor)] .= Ref(zero(eltype(alpha_configs)))
	# post processing
	configs = alpha_configs
    println("configs: ", length(configs))
    return BranchingTable(configs, true, vs, tn_vs)
end
"""
    branching_table_exhaustive(p::MISProblem, solver::AbstractTableSolver, vs::Vector{Int})

Generate a branching table that exhaustively enumerates all valid independent set configurations for the vertices in `vs`.

This function enumerates all 2^|vs| possible binary configurations and filters out those that are not valid independent sets
(i.e., configurations where adjacent vertices are both set to 1).

# Arguments
- `p::MISProblem`: The MIS problem.
- `solver::AbstractTableSolver`: The table solver (not used, but kept for interface compatibility).
- `vs::Vector{Int}`: The vertices to enumerate configurations for.

# Returns
- `BranchingTable`: A branching table where each row contains a single valid configuration (separate_rows=true format).
"""
function OptimalBranchingCore.branching_table_exhaustive(p::MISProblem, solver::AbstractTableSolver, vs::Vector{Int})
    n = length(vs)
    if n == 0
        return OptimalBranchingCore.BranchingTable(0, Vector{Vector{Int}}())
    end
    
    # Build adjacency matrix for vs (only edges within vs)
    subg, vmap = induced_subgraph(p.g, vs)
    adjacency = zeros(Bool, n, n)
    for edge in edges(subg)
        i, j = src(edge), dst(edge)
        adjacency[i, j] = true
        adjacency[j, i] = true
    end
    
    # Enumerate all 2^n configurations and check if they are valid independent sets
    valid_configs = Vector{Vector{Vector{Int}}}()
    
    for config_int in 0:(2^n - 1)
        # Convert integer to bit vector
        config_vec = Int[((config_int >> (i-1)) & 1) for i in 1:n]
        
        # Check if this is a valid independent set (no adjacent vertices both set to 1)
        is_valid = true
        for i in 1:n
            if config_vec[i] == 1
                for j in (i+1):n
                    if config_vec[j] == 1 && adjacency[i, j]
                        is_valid = false
                        break
                    end
                end
                !is_valid && break
            end
        end
        
        if is_valid
            # Each row contains one config vector (separate_rows format)
            push!(valid_configs, [config_vec])
        end
    end
    
    # Convert to BranchingTable format (each config as a separate row)
    return OptimalBranchingCore.BranchingTable(n, valid_configs)
end

"""
    clause_size(weights::Vector{WT}, bit_config, vertices::Vector) where WT

Compute the MIS size difference brought by the application of a clause for a given weighted graph.

# Arguments
- `weights::Vector{WT}`: The weights of the graph.
- `bit_config::Int`: The bit configuration of the clause.
- `vertices::Vector{Int}`: The vertices included in the clause.

# Returns
- The MIS size difference brought by the application of the clause.
"""
function clause_size(weights::Vector{WT}, bit_config, vertices::Vector) where WT
    weighted_size = zero(WT)
    for bit_pos in 1:length(vertices)
        if readbit(bit_config, bit_pos) == 1
            weighted_size += weights[vertices[bit_pos]]
        end
    end
    return weighted_size
end
clause_size(::UnitWeight, bit_config, vertices::Vector) = count_ones(bit_config)

# consider two different branching rule (A, and B) applied on the same set of vertices, with open vertices ovs.
# the neighbors of 1 vertices in A is label as NA1, and the neighbors of 1 vertices in B is label as NB1, and the pink_block is the set of vertices that are not in NB1 but in NA1.
# once mis(A) + mis(pink_block) ≤ mis(B), then A is not a good branching rule, and should be removed.
function prune_by_env(tbl::BranchingTable{INT}, p::MISProblem, vertices) where{INT<:Integer}
    g = p.g
    openvertices = open_vertices(g, vertices)
    ns = neighbors(g, vertices)
    so = Set(openvertices)

    new_table = Vector{Vector{INT}}()

    open_vertices_1 = [Int[] for i in 1:length(tbl.table)]
    neibs_0 = Set{Int}[]
    for i in 1:length(tbl.table)
        row = tbl.table[i]
        x = row[1]
        for n in 1:tbl.bit_length
            xn = (x >> (n-1)) & 1
            if (xn == 1) && (vertices[n] ∈ so)
                push!(open_vertices_1[i], vertices[n])
            end
        end
        push!(neibs_0, setdiff(ns, neighbors(g, open_vertices_1[i]) ∩ ns))
    end

    for i in 1:length(tbl.table)
        flag = true
        for j in 1:length(tbl.table)
            if i != j
                pink_block = setdiff(neibs_0[i], neibs_0[j])
                sg_pink, sg_vec = induced_subgraph(g, collect(pink_block))
                problem_pink = GenericTensorNetwork(IndependentSet(sg_pink, p.weights[collect(pink_block)]); optimizer = GreedyMethod())
                mis_pink = solve(problem_pink, SizeMax())[].n
                if (clause_size(p.weights, tbl.table[i][1], vertices) + mis_pink ≤ clause_size(p.weights, tbl.table[j][1], vertices)) && (!iszero(mis_pink))
                    flag = false
                    break
                end
            end
        end
        if flag
            push!(new_table, tbl.table[i])
        end
    end
    return BranchingTable(OptimalBranchingCore.nbits(tbl), new_table)
end

