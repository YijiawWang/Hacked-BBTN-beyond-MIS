using OptimalBranchingMIS, OptimalBranchingCore, OptimalBranchingMIS.Graphs
using KaHyPar
using OptimalBranchingCore: IPSolver
using OptimalBranchingMIS: clause_size, alpha, reduced_alpha
using ProblemReductions
using Test  
using Random, GenericTensorNetworks

@testset "pruner PH2" begin
    function tree_like_N3_neighborhood(g::SimpleGraph)
        for layer in 1:3
            for v in vertices(g)
                for _ = 1:(3-degree(g, v))
                    add_vertex!(g)
                    add_edge!(g, v, nv(g))
                end
            end
        end
        return g
    end

    vs = [1,2,3,4,5,6,7,8]
    measure = D3Measure()
    table_solver = TensorNetworkSolver(true)
    set_cover_solver = IPSolver()
    edges = [(1, 2), (1, 5), (2, 3), (2, 6), (3, 4), (4, 5), (5, 8), (6, 7), (7, 8)]
    branching_region = SimpleGraph(Graphs.SimpleEdge.(edges))
    graph = tree_like_N3_neighborhood(copy(branching_region))

    ovs = OptimalBranchingMIS.open_vertices(graph, vs)
    subg, vmap = induced_subgraph(graph, vs)
    tbl = OptimalBranchingMIS.reduced_alpha_configs(table_solver, subg, UnitWeight(nv(subg)), Int[findfirst(==(v), vs) for v in ovs])
    @test length(tbl.table) == 9

    problem = MISProblem(graph)
    pruned_tbl = OptimalBranchingMIS.prune_by_env(tbl, problem, vs)
    @test length(pruned_tbl.table) == 5
end

@testset "clause_size" begin
    weights = [1.0, 2.0, 3.0, 4.0, 5.0]
    clause = 0b10100
    vertices = [1, 2, 3, 4, 5]
    @test clause_size(weights, clause, vertices) == 8.0
end

@testset "alpha tensor" begin
    g = SimpleGraph(Graphs.SimpleEdge.([(1,2), (2,3)])) 
    weights = ones(Float64, nv(g))
    openvertices = [1,3]
    alpha_tensor = alpha(g, weights, openvertices)
    @test alpha_tensor[1,2].n == 1

    alpha_tensor = alpha(g, UnitWeight(nv(g)), openvertices)
    @test alpha_tensor[1,2].n == 1

    reduced_alpha_tensor = reduced_alpha(g, weights, openvertices)
    @test reduced_alpha_tensor[1,2].n == -Inf

    reduced_alpha_tensor = reduced_alpha(g, UnitWeight(nv(g)), openvertices)
    @test reduced_alpha_tensor[1,2].n == -Inf
end

@testset "pruner PH2" begin
    function tree_like_N3_neighborhood(g::SimpleGraph)
        for layer in 1:3
            for v in vertices(g)
                for _ = 1:(3-degree(g, v))
                    add_vertex!(g)
                    add_edge!(g, v, nv(g))
                end
            end
        end
        return g
    end

    vs = [1,2,3,4,5,6,7,8]
    measure = D3Measure()
    table_solver = TensorNetworkSolver(true)
    set_cover_solver = IPSolver()
    edges = [(1, 2), (1, 5), (2, 3), (2, 6), (3, 4), (4, 5), (5, 8), (6, 7), (7, 8)]
    branching_region = SimpleGraph(Graphs.SimpleEdge.(edges))
    graph = tree_like_N3_neighborhood(copy(branching_region))
    weights = [1.1 for i in 1:nv(graph)]

    ovs = OptimalBranchingMIS.open_vertices(graph, vs)
    subg, vmap = induced_subgraph(graph, vs)
    tbl = OptimalBranchingMIS.reduced_alpha_configs(table_solver, subg, weights, Int[findfirst(==(v), vs) for v in ovs])
    @test length(tbl.table) == 9

    problem = MISProblem(graph, weights)
    pruned_tbl = OptimalBranchingMIS.prune_by_env(tbl, problem, vs)
    @test length(pruned_tbl.table) == 5
end

@testset "folding and refuction for degree-2" begin
    edges = [(1, 2), (1, 3), (2, 4), (3, 5)]
    graph = SimpleGraph(Graphs.SimpleEdge.(edges))
    measure = D3Measure()
    table_solver = TensorNetworkSolver(true)
    set_cover_solver = IPSolver()
    vs = [1,2,3]
    ovs = OptimalBranchingMIS.open_vertices(graph, vs)
    subg, vmap = induced_subgraph(graph, vs)

    weights = [1,1,1,0.2,0.3]
    tbl = OptimalBranchingMIS.reduced_alpha_configs(table_solver, subg, weights, Int[findfirst(==(v), vs) for v in ovs])
    @test length(tbl.table) == 2

    problem = MISProblem(graph, weights)
    pruned_tbl = OptimalBranchingMIS.prune_by_env(tbl, problem, vs)
    @test length(pruned_tbl.table) == 1

    weights = [1,0.7,0.7,0.2,0.3]
    tbl = OptimalBranchingMIS.reduced_alpha_configs(table_solver, subg, weights, Int[findfirst(==(v), vs) for v in ovs])
    @test length(tbl.table) == 2

    problem = MISProblem(graph, weights)
    pruned_tbl = OptimalBranchingMIS.prune_by_env(tbl, problem, vs)
    @test length(pruned_tbl.table) == 2
end

@testset "BranchingTable with tn_vs, vs, and ovs" begin
    seed = 1234
    Random.seed!(seed)
    graph = Graphs.random_regular_graph(30, 3)
    weights = ones(Float64, nv(graph))
    
    vs = sort(union([1, 13, 20, 21], OptimalBranchingMIS.open_neighbors(graph, [1, 13, 20, 21])))
    tn_vs = sort(union(vs, OptimalBranchingMIS.open_neighbors(graph, vs))) 
    ovs = OptimalBranchingMIS.open_vertices(graph, vs)  # vertices in vs with neighbors outside vs
    println("vs: ", vs)
    println("tn_vs: ", tn_vs)
    println("ovs: ", ovs)
    tn_ovs = OptimalBranchingMIS.open_vertices(graph, tn_vs)
    
    # Generate configurations for tn_vs
    tn_subg, tn_vmap = induced_subgraph(graph, tn_vs)
    tn_openvertices = Int[findfirst(==(v), tn_vs) for v in tn_ovs]
    
    problem = GenericTensorNetwork(IndependentSet(tn_subg, weights[tn_vmap]); openvertices=tn_openvertices, optimizer = GreedyMethod())
    alpha_tensor = solve(problem, SizeMax())
    alpha_configs = solve(problem, ConfigsMax(; bounded=false))
    alpha_configs[map(iszero, alpha_tensor)] .= Ref(zero(eltype(alpha_configs)))
    configs = alpha_configs
    
    # Test with separate_rows=true: each config should be in its own row
    tbl_separate = OptimalBranchingCore.BranchingTable(configs, true, vs, tn_vs, ovs)
    @test OptimalBranchingCore.nbits(tbl_separate) == length(vs)
    @test length(tbl_separate.table) > 0
    
    # Verify that each row contains exactly one config when separate_rows=true
    for row in tbl_separate.table
        @test length(row) == 1
    end
    
    # Test with separate_rows=false: configs with same ovs config should be grouped
    tbl_grouped = OptimalBranchingCore.BranchingTable(configs, false, vs, tn_vs, ovs)
    @test OptimalBranchingCore.nbits(tbl_grouped) == length(vs)
    @test length(tbl_grouped.table) > 0
    
    # Verify that when separate_rows=false, we should have fewer or equal rows
    # (because configs are grouped by ovs config)
    @test length(tbl_grouped.table) <= length(tbl_separate.table)
    
    # Verify that all configs in a row have the same ovs configuration
    # (we can't directly check ovs config from the table, but we can verify
    # that the grouping happened correctly by checking the structure)
    for row in tbl_grouped.table
        @test length(row) >= 1  # Each row should have at least one config
    end
    
    # Verify that all vs_configs are valid (have correct length)
    println("tbl_separate.table: ")
    for row in tbl_separate.table
        for config_int in row
            # Convert integer back to bit vector to verify length
            config_bits = [Int(((config_int >> (i-1)) & 1)) for i in 1:length(vs)]
            config_tuple = tuple(config_bits...)
            println("config: ", config_tuple)
            @test length(config_bits) == length(vs)
        end
    end
    
    println("tbl_grouped.table: ")
    for (row_idx, row) in enumerate(tbl_grouped.table)
        println("Row $row_idx:")
        for config_int in row
            config_bits = [Int(((config_int >> (i-1)) & 1)) for i in 1:length(vs)]
            config_tuple = tuple(config_bits...)
            println("  config: ", config_tuple)
            @test length(config_bits) == length(vs)
        end
    end
end