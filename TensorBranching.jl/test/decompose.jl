using TensorBranching
using TensorBranching.GenericTensorNetworks
using Graphs, OMEinsum, TropicalNumbers

using TensorBranching: eincode2order, eincode2graph, order2eincode, rethermalize, update_order
using OptimalBranching.OptimalBranchingMIS

using Test
using Random

Random.seed!(1234)

# @testset "converting graph and code" begin
#     for n in 30:10:100
#         g = random_regular_graph(n, 3)
#         code = GenericTensorNetwork(IndependentSet(g)).code

#         @test eincode2graph(code)[1] == g

#         eo = eincode2order(code)
#         code_new = order2eincode(g, eo)

#         sc_target = Int(contraction_complexity(code, uniformsize(code, 2)).sc)
#         re_code_new = rethermalize(code_new, uniformsize(code_new, 2), 100.0:100.0, 1, 10, sc_target)

#         for T in (Float64, Tropical{Float64})
#             tensors = GenericTensorNetworks.generate_tensors(T(1.0), IndependentSet(g))
#             @test code(tensors...) ≈ code_new(tensors...) ≈ re_code_new(tensors...)
#             # @show contraction_complexity(code, uniformsize(code, 2)).sc, contraction_complexity(code_new, uniformsize(code_new, 2)).sc, contraction_complexity(re_code_new, uniformsize(re_code_new, 2)).sc
#         end
#     end
# end

# @testset "reconstructing code" begin
#     for n in 30:10:100
#         for _ in 1:5
#             g = random_regular_graph(n, 3)
#             code = GenericTensorNetwork(IndependentSet(g)).code
#             removed_vertices = unique!(rand(1:n, 10))

#             g_new, vmap = induced_subgraph(g, setdiff(1:n, removed_vertices))

#             for _ in 1:10
#                 i = rand(1:nv(g_new))
#                 j = rand(1:nv(g_new))

#                 add_edge!(g_new, i, j)
#             end

#             code = GenericTensorNetwork(IndependentSet(g)).code
#             eo = eincode2order(code)
#             eo_new = update_order(eo, vmap)
#             code_new = order2eincode(g_new, eo_new)

#             net_direct = GenericTensorNetwork(IndependentSet(g_new))
#             net_new = GenericTensorNetwork(IndependentSet(g_new), code_new, Dict{Int, Int}())
#             @test solve(net_direct, SizeMax()) ≈ solve(net_new, SizeMax())
#         end
#     end
# end

# @testset "corner case: disconnected graph have 1 vertex" begin
#     g = random_regular_graph(30, 3)
#     removed_vertices = neighbors(g, 1)
#     g_new, vmap = induced_subgraph(g, setdiff(1:nv(g), removed_vertices))
#     code = GenericTensorNetwork(IndependentSet(g)).code
#     eo = eincode2order(code)
#     eo_new = update_order(eo, vmap)
#     code_new = order2eincode(g_new, eo_new)
#     @test !is_connected(g_new)
#     @test solve(GenericTensorNetwork(IndependentSet(g_new)), SizeMax()) ≈ solve(GenericTensorNetwork(IndependentSet(g_new), code_new, Dict{Int, Int}()), SizeMax())
# end

# @testset "corner case: disconnected graph have 2 vertices" begin
#     g = random_regular_graph(30, 3)
#     vs = [1, neighbors(g, 1)[1]]
#     removed_vertices = OptimalBranchingMIS.open_neighbors(g, vs)
#     g_new, vmap = induced_subgraph(g, setdiff(1:nv(g), removed_vertices))
#     code = GenericTensorNetwork(IndependentSet(g)).code
#     eo = eincode2order(code)
#     eo_new = update_order(eo, vmap)
#     code_new = order2eincode(g_new, eo_new)
#     @test !is_connected(g_new)
#     @test solve(GenericTensorNetwork(IndependentSet(g_new)), SizeMax()) ≈ solve(GenericTensorNetwork(IndependentSet(g_new), code_new, Dict{Int, Int}()), SizeMax())
# end

# @testset "compressing and uncompressing code" begin
#     for n in 30:10:100
#         g = random_regular_graph(n, 3)
#         code = GenericTensorNetwork(IndependentSet(g)).code
#         compressed_code = compress(code)
#         uncompressed_code = uncompress(compressed_code)

#         cc = mis_complexity(code)
#         ccc = mis_complexity(uncompressed_code)

#         @test cc.sc ≈ ccc.sc
#         @test cc.tc ≈ ccc.tc
#         @test cc.rwc ≈ ccc.rwc
#     end
# end

@testset "update_code for MaxSAT" begin
    using TensorBranching: update_code, induced_sat_subproblem
    
    # Helper function to generate a simple SAT problem
    function generate_simple_sat(n_vars::Int, n_clauses::Int)
        g = SimpleGraph(n_vars + n_clauses)
        clauses = Vector{Int}[]
        
        # Create simple clauses: each clause contains 2-3 variables
        for clause_idx in 1:n_clauses
            clause_vertex = n_vars + clause_idx
            clause = Int[]
            
            # Select 2-3 random variables
            vars_in_clause = unique(rand(1:n_vars, rand(2:3)))
            for var_idx in vars_in_clause
                is_negated = rand(Bool)
                push!(clause, is_negated ? -var_idx : var_idx)
                add_edge!(g, var_idx, clause_vertex)
            end
            push!(clauses, clause)
        end
        
        return g, clauses
    end
    
    # Test 1: Simple case with a few variables removed
    for _ in 1:3
        n_vars = 10
        n_clauses = 15
        g_old, clauses_old = generate_simple_sat(n_vars, n_clauses)
        
        # Create original SAT problem and code
        sat_old = OptimalBranchingMIS.graph_clauses_to_sat(g_old, clauses_old, n_vars)
        code_old = GenericTensorNetwork(sat_old).code
        
        # Create a subproblem by fixing some variables
        region = collect(1:min(5, n_vars))
        removed_vars = region[1:min(2, length(region))]
        val = rand([0, 1], length(region))
        
        g_new, clauses_new, remaining_vars, r = induced_sat_subproblem(g_old, clauses_old, val, removed_vars, region)
        
        # Build vmap: remaining_vars contains old variable indices that remain
        # In the new graph, variables are renumbered 1 to n_vars_new
        # vmap[i] should map old variable index to new variable index
        n_vars_new = nv(g_new) - length(clauses_new)
        
        if !isempty(remaining_vars) && n_vars_new > 0 && nv(g_new) > 0
            # Build mapping: old_var -> new_var
            # remaining_vars are sorted old variable indices
            # In new graph, they are numbered 1, 2, 3, ...
            old_to_new = Dict{Int, Int}()
            for (new_idx, old_var) in enumerate(sort(remaining_vars))
                old_to_new[old_var] = new_idx
            end
            
            # vmap: for each old variable in remaining_vars, map to new index
            vmap = [old_to_new[v] for v in sort(remaining_vars)]
            
            # Update code using update_code
            code_new = update_code(g_new, clauses_new, code_old, vmap)
            
            # Verify the new code works correctly
            sat_new = OptimalBranchingMIS.graph_clauses_to_sat(g_new, clauses_new, n_vars_new)
            net_direct = GenericTensorNetwork(sat_new)
            net_updated = GenericTensorNetwork(sat_new, code_new, Dict{Int, Int}())
            
            # Test that both networks produce the same result
            result_direct = solve(net_direct, CountingMax())
            result_updated = solve(net_updated, CountingMax())
            
            # Extract results: .n[1] is max satisfied clauses, .c[1] is count
            direct_n = result_direct[].n[1]
            direct_c = result_direct[].c[1]
            updated_n = result_updated[].n[1]
            updated_c = result_updated[].c[1]
            
            # Debug output
            println("Test iteration: n_vars_old=$n_vars, n_vars_new=$n_vars_new, removed_vars=$removed_vars")
            println("  Direct result: n=$direct_n, c=$direct_c")
            println("  Updated result: n=$updated_n, c=$updated_c")
            println("  Already satisfied clauses (r): $r")
            
            # Both should produce the same result since they solve the same problem
            # Note: The results should match exactly for the same SAT problem
            if direct_n != updated_n
                @error "Max satisfied clauses mismatch: direct=$direct_n, updated=$updated_n"
            end
            if direct_c != updated_c
                @error "Count mismatch: direct=$direct_c, updated=$updated_c"
            end
            @test direct_n == updated_n
            @test direct_c == updated_c
        end
    end
end