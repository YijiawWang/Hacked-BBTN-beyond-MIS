# transform optimized eincode to elimination order
function eincode2order(code::NestedEinsum{L}) where {L}
    elimination_order = Vector{L}()
    OMEinsum.isleaf(code) && return elimination_order
    for node in PostOrderDFS(code)
        (node isa LeafString) && continue
        for id in setdiff(vcat(getixsv(node.eins)...), getiyv(node.eins))
            push!(elimination_order, id)
        end
    end
    return reverse!(elimination_order)
end

function eincode2graph(code::Union{NestedEinsum, EinCode})
    fcode = code isa NestedEinsum ? flatten(code) : code
    indices = uniquelabels(fcode)
    (indices isa Vector{Int}) && sort!(indices)
    g = SimpleGraph(length(indices))
    id_dict = Dict(id => i for (i, id) in enumerate(indices))
    for xs in [getixsv(fcode); getiyv(fcode)]
        for i in 1:length(xs)-1
            for j in i+1:length(xs)
                add_edge!(g, id_dict[xs[i]], id_dict[xs[j]])
            end
        end
    end
    return g, id_dict
end

function decompose(code::NestedEinsum{L}) where {L}
    g, id_dict = eincode2graph(code)
    labels = collect(keys(id_dict))[sortperm(collect(values(id_dict)))]
    return decomposition_tree(g, eincode2order(code), labels = labels)
end

function max_bag(tree::DecompositionTreeNode)
    max_bag = tree.bag
    max_size = length(max_bag)
    for node in PostOrderDFS(tree)
        if length(node.bag) > max_size
            max_bag = node.bag
            max_size = length(node.bag)
        end
    end
    return max_bag
end

# this function maps an elimination order on a old graph to a new graph with some vertices removed or reordered
function update_order(eo_old::Vector{Int}, vmap::Vector{Int})
    ivmap = inverse_vmap_dict(vmap)
    eo_new = Vector{Int}()
    for v in eo_old
        haskey(ivmap, v) && push!(eo_new, ivmap[v])
    end
    return eo_new
end
function update_tree(g_new::SimpleGraph{Int}, eo_old::Vector{Int}, vmap::Vector{Int})
    eo_new = update_order(eo_old, vmap)
    return decomposition_tree(g_new, eo_new)
end

# reconstruct the contraction order from the grouped elimination order
# if set use_tree to true, the decomposition tree will be constructed to get a better elimination order
function order2eincode(g::SimpleGraph{Int}, eo::Vector{Int}; use_tree::Bool = true)
    rcode = rawcode(IndependentSet(g))
    ixs = getixsv(rcode)
    incidence_list = IncidenceList(Dict([i=>ix for (i, ix) in enumerate(ixs)]))
    trees = Vector{Union{ContractionTree, Int}}()
    for sub_vs in connected_components(g)
        if length(sub_vs) == 1
            # a special corner case for the mis problem: the connected component is a single vertex has no edges
            push!(trees, incidence_list.e2v[sub_vs[1]][1])
        else
            if use_tree
                sub_g, sub_vmap = induced_subgraph(g, sub_vs)
                sub_ivmap = inverse_vmap_dict(sub_vmap)
                sub_eo = [sub_ivmap[i] for i in eo if i in sub_vs]
                tree = decomposition_tree(sub_g, sub_eo)
                grouped_eo = EliminationOrder(tree).order

                map!(x -> map!(y -> sub_vmap[y], x, x), grouped_eo, grouped_eo)
            else
                grouped_eo = [[i] for i in eo if i in sub_vs]
            end

            tree = eo2ct(grouped_eo, incidence_list, [1.0 for _ in 1:length(eo)])
            push!(trees, tree)
        end
    end
    tree = reduce((x,y) -> ContractionTree(x, y), trees)
    code = parse_eincode(incidence_list, tree, vertices = collect(1:length(ixs))) # this code is OMEinsumContractionOrders.NestedEinsum, not OMEinsum.NestedEinsum

    return decorate(code)
end

function order2eincode(g::SimpleGraph{Int}, clauses::Vector{Vector{Int}}, eo::Vector{Int}; use_tree::Bool = true)
    n_vars = nv(g) - length(clauses)
    
    # Step 1: Generate SAT problem from new graph and clauses
    sat = OptimalBranchingMIS.graph_clauses_to_sat(g, clauses, n_vars)
    rcode = rawcode(sat)
    ixs = getixsv(rcode)
    
    # Step 2: Create incidence list mapping tensor indices to their index sets
    # Only include tensors that actually exist in the new problem
    incidence_list = IncidenceList(Dict([i=>ix for (i, ix) in enumerate(ixs)]))
    
    # Step 3: Build the actual tensor network interaction graph from rcode
    # g is a bipartite graph (variables-clauses), not the TN interaction graph
    # We need to build the TN interaction graph from the tensor network structure
    # The vertices of tn_g correspond to variable indices (1 to n_vars)
    tn_g, id_dict = eincode2graph(rcode)
    
    # Step 4: Process connected components of the tensor network graph
    # eo is already mapped to the new graph (eo_new), containing variable indices 1 to n_vars
    trees = Vector{Union{ContractionTree, Int}}()
    
    for sub_vs_tn in connected_components(tn_g)
        # sub_vs_tn contains variable indices (1 to n_vars) that are in this TN component
        
        if use_tree
            # Build subgraph of the tensor network interaction graph
            sub_tn_g, sub_tn_vmap = induced_subgraph(tn_g, sub_vs_tn)
            sub_tn_ivmap = inverse_vmap_dict(sub_tn_vmap)
            
            # Filter eo to only include variables in this component and map to subgraph indices
            sub_eo = [sub_tn_ivmap[i] for i in eo if i in sub_vs_tn && haskey(sub_tn_ivmap, i)]
            
            # Check for variables in this component that are NOT in eo, and append them
            # This ensures we don't drop any variables/tensors
            existing_vars_set = Set(sub_eo)
            missing_vars = [sub_tn_ivmap[v] for v in sub_vs_tn if haskey(sub_tn_ivmap, v) && sub_tn_ivmap[v] ∉ existing_vars_set]
            if !isempty(missing_vars)
                append!(sub_eo, missing_vars)
            end
            
            if !isempty(sub_eo)
                tree = decomposition_tree(sub_tn_g, sub_eo)
                grouped_eo = EliminationOrder(tree).order
                
                # Map grouped_eo back to original variable indices (tn_g indices)
                map!(x -> map!(y -> sub_tn_vmap[y], x, x), grouped_eo, grouped_eo)
            else
                # Fallback: simple grouping (should be covered by missing_vars append, but just in case)
                grouped_eo = [[v] for v in sub_vs_tn]
            end
        else
            # Simple grouping: each variable gets its own group
            # Use order from eo if available
            ordered_vars = [v for v in eo if v in sub_vs_tn]
            existing_set = Set(ordered_vars)
            unordered_vars = [v for v in sub_vs_tn if v ∉ existing_set]
            
            grouped_eo = [[v] for v in ordered_vars]
            for v in unordered_vars
                push!(grouped_eo, [v])
            end
        end
        
        # Build contraction tree from grouped elimination order
        if !isempty(grouped_eo)
            # Pass a deepcopy of incidence_list because eo2ct might modify it
            tree = eo2ct(grouped_eo, deepcopy(incidence_list), [1.0 for _ in 1:length(eo)])
            push!(trees, tree)
        end
    end
    
    # Step 5: Combine all contraction trees
    if isempty(trees)
        error("No valid contraction trees generated")
    end
    
    # Use reduce to combine trees, ContractionTree constructor can accept Int as leaf nodes
    # This is the same approach as the SpinGlass version (line 90)
    tree = reduce((x,y) -> ContractionTree(x, y), trees)
    
    # Ensure tree is always ContractionTree type (not Int)
    # reduce with a single element returns that element directly, which could be Int
    # Also, eo2ct might return Int when there's only one tensor
    if tree isa Int
        # Special case: when there's only one tensor total (length(ixs) == 1)
        # We need to create a ContractionTree that represents a single tensor
        # For a single tensor, we can use eo2ct with a grouped elimination order containing just that tensor
        # This will create a proper ContractionTree that parse_eincode can handle
        if length(ixs) == 1
            # Create a grouped elimination order with just the single tensor
            grouped_eo = [[tree]]
            tree = eo2ct(grouped_eo, deepcopy(incidence_list), [1.0])
            # If eo2ct still returns Int (shouldn't happen), we have a problem
            if tree isa Int
                error("Failed to create ContractionTree for single tensor: tensor index $tree")
            end
        else
            error("Unexpected: tree is Int but length(ixs) > 1. This should not happen.")
        end
    end
    
    # Step 6: Parse contraction tree to generate NestedEinsum code
    # vertices should be indices 1 to length(ixs) (all tensors in the new problem)
    
    # parse_eincode expects an IncidenceList where vertices are Tensors (leaves of the tree).
    # The current incidence_list has vertices as Variables (v2e: Var -> Tensors).
    # We need to construct a new IncidenceList where v2e: Tensor -> Vars.
    # We can do this by passing the current v2e (Var -> Tensors) as the input map to IncidenceList constructor.
    # The constructor takes e2v (Edge -> Vertices). Here Edges=Vars, Vertices=Tensors.
    # So the resulting list will have v2e = Tensor -> Vars.
    il_for_parse = IncidenceList(incidence_list.v2e)
    
    code = parse_eincode(il_for_parse, tree, vertices = collect(1:length(ixs)))
    
    # Step 7: Decorate the code to convert from OMEinsumContractionOrders.NestedEinsum to OMEinsum.NestedEinsum
    return decorate(code)
end

function update_code(g_new::SimpleGraph{Int}, code_old::NestedEinsum, vmap::Vector{Int})
    eo_old = eincode2order(code_old)
    eo_new = update_order(eo_old, vmap)
    return order2eincode(g_new, eo_new)
end

function update_code(g_new::SimpleGraph{Int}, clauses::Vector{Vector{Int}}, code_old::NestedEinsum, vmap::Vector{Int})
    eo_old = eincode2order(code_old)
    eo_new = update_order(eo_old, vmap)
    return order2eincode(g_new, clauses, eo_new)
end

function ein2contraction_tree(code::NestedEinsum)
    @assert is_binary(code)
    return _ein2contraction_tree(code)
end

function _ein2contraction_tree(code)
    return isleaf(code) ? code.tensorindex : ContractionTree(_ein2contraction_tree(code.args[1]), _ein2contraction_tree(code.args[2]))
end