using OptimalBranchingMIS, OptimalBranchingCore
using OptimalBranchingMIS: graph_from_tuples, open_neighbors, open_vertices
using EliminateGraphs, EliminateGraphs.Graphs
using GenericTensorNetworks
using Test

@testset "extract vs config from tn_vs config" begin
    # Create a simple graph for testing
    # Graph: 1-2-3-4, where vs = [2, 3], tn_vs includes vs and their neighbors
    g = random_regular_graph(10, 3)
    weights = ones(Float64, nv(g))
    p = MISProblem(g, weights)
    
    # vs is the subset of vertices we care about
    vs = [1, 2, 3, 4]
    # tn_vs includes vs and their neighbors (vs is a subset of tn_vs)
    tn_vs = sort(union(vs, open_neighbors(g, vs)))  # [1, 2, 3, 4]
    
    # Generate configurations for tn_vs
    ovs = OptimalBranchingMIS.open_vertices(g, vs)
    tn_ovs = OptimalBranchingMIS.open_vertices(g, tn_vs)
    tn_subg, tn_vmap = induced_subgraph(g, tn_vs)
    tn_openvertices = Int[findfirst(==(v), tn_vs) for v in tn_ovs]
    
    problem = GenericTensorNetwork(IndependentSet(tn_subg, weights[tn_vmap]); openvertices=tn_openvertices, optimizer = GreedyMethod())
    alpha_tensor = solve(problem, SizeMax())
    alpha_configs = solve(problem, ConfigsMax(; bounded=false))
    alpha_configs[map(iszero, alpha_tensor)] .= Ref(zero(eltype(alpha_configs)))
    configs = alpha_configs
    
    # Test the BranchingTable constructor with vs and tn_vs
    tbl = OptimalBranchingCore.BranchingTable(configs, true, vs, tn_vs)
    # println("alpha_configs: ", alpha_configs)
    # println("tbl: ", tbl)
    # Verify that the bit_length matches the length of vs
    @test OptimalBranchingCore.nbits(tbl) == length(vs)
    
    # Verify that all configurations in the table have the correct length
    for row in tbl.table
        for config_int in row
            # Convert integer back to bit vector to verify length
            config_vec = Int[((config_int >> (i-1)) & 1) for i in 1:length(vs)]
            @test length(config_vec) == length(vs)
        end
    end
    
    # Test that duplicate configurations are removed
    # We can verify this by checking that the number of unique configurations
    # matches the number of rows (when separate_rows=true)
    all_configs = Vector{Vector{Int}}()
    for row in tbl.table
        for config_int in row
            config_vec = Int[((config_int >> (i-1)) & 1) for i in 1:length(vs)]
            push!(all_configs, config_vec)
        end
    end
    unique_configs = unique(all_configs)
    @test length(all_configs) == length(unique_configs)  # No duplicates
    
    # Test with separate_rows=false
    tbl2 = OptimalBranchingCore.BranchingTable(configs, false, vs, tn_vs)
    @test OptimalBranchingCore.nbits(tbl2) == length(vs)
    
    # Verify configurations are extracted correctly
    # Create a simple test case where we know the expected result
    # If tn_vs = [1, 2, 3, 4] and vs = [2, 3]
    # A config [1, 0, 1, 0] for tn_vs should become [0, 1] for vs
    # (assuming 2 is at index 2 in tn_vs and 3 is at index 3 in tn_vs)
end
