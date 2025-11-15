using TensorBranching
using Graphs, TropicalNumbers, OMEinsum

using OptimalBranching
using OptimalBranching.OptimalBranchingCore, OptimalBranching.OptimalBranchingMIS
using OptimalBranching.OptimalBranchingMIS.EliminateGraphs

using GenericTensorNetworks
using GenericTensorNetworks: generate_tensors

using Test
using Random
Random.seed!(1234)

using TensorBranching: remove_tensors, remove_tensors!, tensors_removed, unsafe_flatten, rethermalize, reindex_tree!
using TensorBranching: induced_spin_glass_subproblem
using BitBasis
using TensorBranching: spinglass_linear_IP, spinglass_linear_LP_bound, QP_bound, QIP_bound
using GenericTensorNetworks: SpinGlass
using JuMP, SCIP


# @testset "tree reform" begin
#     for i in 1:1000
#         Random.seed!(i)
#         n = 30
#         g = random_regular_graph(n, 3)
#         net = GenericTensorNetwork(IndependentSet(g))
#         order = net.code
#         tensors = generate_tensors(TropicalF32(1.0), net)

#         # check if the result is still correct after removing for more than one rounds
#         for _ in 1:4
#             remove = rand(1:nv(g), rand(1:5))
#             subg, vmap = induced_subgraph(g, setdiff(1:nv(g), remove))
#             mis = mis2(EliminateGraph(subg))

#             tids = tensors_removed(order, remove)
#             sub_order = remove_tensors(order, tids)
#             @test sub_order(tensors...)[].n ≈ mis

#             ri_order = reindex_tree!(sub_order, vmap)
#             @test ri_order(tensors...)[].n ≈ mis
#             g = subg
#             order = ri_order
#         end
#     end
# end

# @testset "tree rethermalize" begin
#     for i in 1:10
#         n = 60
#         g = random_regular_graph(n, 3)
#         net = GenericTensorNetwork(IndependentSet(g), optimizer=TreeSA())
#         order = net.code
#         tensors = GenericTensorNetworks.generate_tensors(TropicalF32(1.0), net)
        
#         n1 = neighbors(g, 1) ∪ [1]
#         n2 = union([neighbors(g, x) for x in n1]...) ∪ n1
#         subg, vmap = induced_subgraph(g, setdiff(1:n, n2))
#         sub_order = remove_tensors(order, tensors_removed(order, n2))
#         rt_order = rethermalize(deepcopy(sub_order), uniformsize(sub_order, 2), 100.0:100.0, 1, 10, 25)

#         ri_order = reindex_tree!(deepcopy(sub_order), vmap)
#         rt_ri_order = rethermalize(deepcopy(ri_order), uniformsize(ri_order, 2), 100.0:100.0, 1, 10, 25)
        
#         @test rt_order(tensors...)[].n ≈ sub_order(tensors...)[].n ≈ ri_order(tensors...)[].n ≈ rt_ri_order(tensors...)[].n ≈ mis2(EliminateGraph(subg))
#     end
# end

# @testset "spinglass_linear_bound vs SizeMax" begin
#     Random.seed!(1234)
    
#     # Test with different graph sizes and configurations
#     for test_idx in 1:3
#         Random.seed!(test_idx * 1000)
        
#         if test_idx == 1
#             g = random_regular_graph(10, 3)
#         elseif test_idx == 2
#             g = random_regular_graph(20, 3)
#         else
#             g = random_regular_graph(15, 4)
#         end
        
#         J = randn(Float64, ne(g))
#         h = randn(Float64, nv(g))
        
#         # Compute SizeMax using GenericTensorNetworks
#         problem = GenericTensorNetwork(SpinGlass(g, J, h); optimizer=GreedyMethod())
#         size_max_result = solve(problem, SizeMax())[].n
        
#         # Compute linear bound using spinglass_linear_bound
#         linear_ip_result = spinglass_linear_IP(g, J, h; optimizer=SCIP.Optimizer)
#         linear_lp_bound_result = spinglass_linear_LP_bound(g, J, h; optimizer=SCIP.Optimizer)
        
#         println("Test case $test_idx: Graph nv=$(nv(g)), ne=$(ne(g))")
#         println("  SizeMax result: $size_max_result")
#         println("  Linear IP result: $linear_ip_result")
#         println("  Linear LP bound result: $linear_lp_bound_result")
       
#         # The linear bound should be >= SizeMax (since it's an upper bound)
#         # But ideally they should be equal for exact solutions
#         @test (linear_ip_result - size_max_result) <= 1e-6
#         @test linear_lp_bound_result >= size_max_result - 1e-6
#     end
# end

@testset "induced_spin_glass_subproblem" begin
    using TensorBranching: induced_spin_glass_subproblem
    using GenericTensorNetworks: SpinGlass
    using BitBasis
    
    Random.seed!(1234)
    
    # Test case 1: Simple 3-vertex path graph
    # Graph: 1-2-3, remove vertex 2
    g_old = SimpleGraph(3)
    add_edge!(g_old, 1, 2)
    add_edge!(g_old, 2, 3)
    
    J_old = [1.0, 2.0]  # J for edges (1,2) and (2,3)
    h_old = [0.5, 1.0, 0.3]  # h for vertices 1, 2, 3
    
    # Remove vertex 2, keep vertices 1 and 3
    removed_vertices = [2]
    region = [2]  # Region contains only vertex 2
    remaining_vertices = [1, 3]
    
    g_new, vmap = induced_subgraph(g_old, remaining_vertices)
    
    # Test with s_2 = -1 (bit = 1)
    INT = BitBasis.longinttype(length(region), 2)
    val_s_minus1 = bmask(INT, 1)  # First bit is 1, meaning s_2 = -1
    
    J_new1, h_new1, r1 = induced_spin_glass_subproblem(
        g_old, g_new, vmap, J_old, h_old, val_s_minus1, removed_vertices, region
    )
    
    # Expected results for s_2 = -1:
    # r should accumulate: -h[2] = -1.0
    # h[1] should be updated: h[1] + (-J[1]) = 0.5 - 1.0 = -0.5
    # h[3] should be updated: h[3] + (-J[2]) = 0.3 - 2.0 = -1.7
    # J[1] and J[2] should be set to 0.0
    # Since g_new has no edges (vertices 1 and 3 are not connected), J_new should be empty
    
    @test r1 ≈ -1.0 atol=1e-10
    @test length(J_new1) == 0  # No edges in induced subgraph
    @test length(h_new1) == 2
    @test h_new1[1] ≈ -0.5 atol=1e-10  # h[1] - J[1]
    @test h_new1[2] ≈ -1.7 atol=1e-10  # h[3] - J[2]
    
    # Test with s_2 = +1 (bit = 0)
    val_s_plus1 = INT(0)  # First bit is 0, meaning s_2 = +1
    
    J_new2, h_new2, r2 = induced_spin_glass_subproblem(
        g_old, g_new, vmap, J_old, h_old, val_s_plus1, removed_vertices, region
    )
    
    # Expected results for s_2 = +1:
    # r should accumulate: h[2] = 1.0
    # h[1] should be updated: h[1] + J[1] = 0.5 + 1.0 = 1.5
    # h[3] should be updated: h[3] + J[2] = 0.3 + 2.0 = 2.3
    
    @test r2 ≈ 1.0 atol=1e-10
    @test length(J_new2) == 0
    @test length(h_new2) == 2
    @test h_new2[1] ≈ 1.5 atol=1e-10  # h[1] + J[1]
    @test h_new2[2] ≈ 2.3 atol=1e-10  # h[3] + J[2]
    
    # Test case 2: Triangle graph, remove one vertex
    # Graph: triangle 1-2-3-1, remove vertex 1
    g_old2 = SimpleGraph(3)
    add_edge!(g_old2, 1, 2)
    add_edge!(g_old2, 2, 3)
    add_edge!(g_old2, 3, 1)
    J_old2 = [1.0, 2.0, 3.0]  # J for edges (1,2), (1,3), (3,2)
    h_old2 = [0.5, 1.0, 0.3]
    
    removed_vertices2 = [1]
    region2 = [1]
    remaining_vertices2 = [2, 3]
    
    g_new2, vmap2 = induced_subgraph(g_old2, remaining_vertices2)
    
    # Test with s_1 = -1
    INT2 = BitBasis.longinttype(length(region2), 2)
    val2 = bmask(INT2, 1)
    
    J_new3, h_new3, r3 = induced_spin_glass_subproblem(
        g_old2, g_new2, vmap2, J_old2, h_old2, val2, removed_vertices2, region2
    )
    
    # Expected: g_new2 has edge (2,3) with J = 3.0
    # h[2] should be: h[2] + (-J[1]) = 1.0 - 1.0 = 0.0
    # h[3] should be: h[3] + (-J[3]) = 0.3 - 2.0 = -1.7
    # r should be: -h[1] = -0.5
    # J_new should have one element: J[2] = 2.0 (edge (2,3) remains)
    
    @test r3 ≈ -0.5 atol=1e-10
    @test length(J_new3) == 1
    @test J_new3[1] ≈ 3.0 atol=1e-10  # Edge (2,3) with original J[2]
    @test length(h_new3) == 2
    @test h_new3[1] ≈ 0.0 atol=1e-10  # h[2] - J[1]
    @test h_new3[2] ≈ -1.7 atol=1e-10  # h[3] - J[3]
    
    # Test case 3: Remove multiple vertices
    g_old3 = SimpleGraph(4)
    add_edge!(g_old3, 1, 2)
    add_edge!(g_old3, 2, 3)
    add_edge!(g_old3, 3, 4)
    
    J_old3 = [1.0, 2.0, 3.0]
    h_old3 = [0.1, 0.2, 0.3, 0.4]
    
    removed_vertices3 = [2, 3]
    region3 = [2, 3]
    remaining_vertices3 = [1, 4]
    
    g_new3, vmap3 = induced_subgraph(g_old3, remaining_vertices3)
    
    # Test with s_2 = -1, s_3 = +1
    # val: bit 1 = 1 (s_2 = -1), bit 2 = 0 (s_3 = +1)
    INT3 = BitBasis.longinttype(length(region3), 2)
    val3 = bmask(INT3, 1)  # Only first bit set
    
    J_new4, h_new4, r4 = induced_spin_glass_subproblem(
        g_old3, g_new3, vmap3, J_old3, h_old3, val3, removed_vertices3, region3
    )
    
    # Expected:
    # For vertex 2 (s_2 = -1): r += -h[2] = -0.2, h[1] += -J[1] = 0.1 - 1.0 = -0.9, h[3] += -J[2] = 0.3 - 2.0 = -1.7
    # For vertex 3 (s_3 = +1): r += h[3] = -1.7 + 0.3 = -1.4 (but h[3] was already updated), h[2] += J[2] = -1.7 + 2.0 = 0.3, h[4] += J[3] = 0.4 + 3.0 = 3.4
    # But wait, vertex 3 was already processed, so h[3] is now -1.7, not 0.3
    # Actually, we need to track the updates more carefully
    
    # The function processes vertices in order, so:
    # After processing vertex 2: h[1] = 0.1 - 1.0 = -0.9, h[3] = 0.3 - 2.0 = -1.7, r = -0.2
    # After processing vertex 3: h[2] = 0.2 + 2.0 = 2.2 (but vertex 2 is removed, so this doesn't matter),
    #                            h[4] = 0.4 + 3.0 = 3.4, r = -0.2 + (-1.7) = -1.9 (using updated h[3])
    
    # Actually, looking at the code more carefully: when processing vertex 3, it uses h[3] which is already updated to -1.7
    # So r = -0.2 + (-1.7) = -1.9
    
    @test length(J_new4) == 0  # No edges in induced subgraph (vertices 1 and 4 are not connected)
    @test length(h_new4) == 2
    @test h_new4[1] ≈ -0.9 atol=1e-10  # h[1] - J[1]
    @test h_new4[2] ≈ 3.4 atol=1e-10  # h[4] + J[3]
    @test r4 ≈ -1.9 atol=1e-10  # -h[2] + h[3]_updated = -0.2 + (-1.7) = -1.9
    
    # Test case 4: Empty removed vertices (should return original values)
    removed_vertices4 = Int[]
    region4 = Int[]
    remaining_vertices4 = [1, 2, 3, 4]
    
    g_new4, vmap4 = induced_subgraph(g_old3, remaining_vertices4)
    
    INT4 = BitBasis.longinttype(0, 2)
    val4 = INT4(0)
    
    J_new5, h_new5, r5 = induced_spin_glass_subproblem(
        g_old3, g_new4, vmap4, J_old3, h_old3, val4, removed_vertices4, region4
    )
    
    @test r5 ≈ 0.0 atol=1e-10
    @test length(J_new5) == length(J_old3)
    @test J_new5 ≈ J_old3 atol=1e-10
    @test length(h_new5) == length(h_old3)
    @test h_new5 ≈ h_old3 atol=1e-10
    
    println("✓ All tests for induced_spin_glass_subproblem passed!")
end


@testset "QP_bound and QIP_bound" begin
    using TensorBranching: QIP_bound
    using GenericTensorNetworks: SpinGlass
    using OMEinsumContractionOrders: TreeSA
    
    Random.seed!(1234)
    
    # Test case 1: Small graph with known structure
    println("\n=== Testing QIP_bound ===")
    
    # Test case 1: Simple 3-vertex path graph
    g1 = SimpleGraph(3)
    add_edge!(g1, 1, 2)
    add_edge!(g1, 2, 3)
    
    J1 = [1.0, 2.0]  # J for edges (1,2) and (2,3)
    h1 = [0.5, 1.0, 0.3]  # h for vertices 1, 2, 3
    
    # Compute exact solution using GenericTensorNetworks
    problem1 = GenericTensorNetwork(SpinGlass(g1, J1, h1); optimizer=TreeSA())
    exact_result1 = solve(problem1, SizeMax())[].n
    
    # Compute QIP bound
    qip_bound1 = QIP_bound(g1, J1, h1; optimizer=SCIP.Optimizer)
    
    # Compute LP bound and QP bound for comparison
    lp_bound1 = spinglass_linear_LP_bound(g1, J1, h1; optimizer=SCIP.Optimizer)
    qp_bound1 = QP_bound(g1, J1, h1; optimizer=SCIP.Optimizer)
    
    println("Test case 1: Path graph (nv=3, ne=2)")
    println("  Exact result: $exact_result1")
    println("  QIP bound: $qip_bound1")
    println("  QP bound: $qp_bound1")
    println("  LP bound: $lp_bound1")
    
    # QIP bound should be >= exact result (upper bound for maximization)
    @test qip_bound1 >= exact_result1 - 1e-6
    # QIP bound should be <= QP bound (integer constraint makes it tighter or equal)
    @test qip_bound1 <= qp_bound1 + 1e-6
    println("  ✓ QIP bound is valid upper bound")
    
    # Test case 2: Random regular graph
    g2 = random_regular_graph(10, 3)
    J2 = randn(Float64, ne(g2))
    h2 = randn(Float64, nv(g2))
    
    problem2 = GenericTensorNetwork(SpinGlass(g2, J2, h2); optimizer=TreeSA())
    exact_result2 = solve(problem2, SizeMax())[].n
    
    qip_bound2 = QIP_bound(g2, J2, h2; optimizer=SCIP.Optimizer)
    qp_bound2 = QP_bound(g2, J2, h2; optimizer=SCIP.Optimizer)
    lp_bound2 = spinglass_linear_LP_bound(g2, J2, h2; optimizer=SCIP.Optimizer)
    
    println("\nTest case 2: Random regular graph (nv=10, ne=$(ne(g2)))")
    println("  Exact result: $exact_result2")
    println("  QIP bound: $qip_bound2")
    println("  QP bound: $qp_bound2")
    println("  LP bound: $lp_bound2")
    
    @test qip_bound2 >= exact_result2 - 1e-6
    @test qip_bound2 <= qp_bound2 + 1e-6
    println("  ✓ QIP bound is valid upper bound")
    
    # Test case 3: Larger graph
    g3 = random_regular_graph(14, 3)
    J3 = randn(Float64, ne(g3))
    h3 = randn(Float64, nv(g3))
    
    problem3 = GenericTensorNetwork(SpinGlass(g3, J3, h3); optimizer=TreeSA())
    exact_result3 = solve(problem3, SizeMax())[].n
    
    qip_bound3 = QIP_bound(g3, J3, h3; optimizer=SCIP.Optimizer)
    qp_bound3 = QP_bound(g3, J3, h3; optimizer=SCIP.Optimizer)
    lp_bound3 = spinglass_linear_LP_bound(g3, J3, h3; optimizer=SCIP.Optimizer)
    
    println("\nTest case 3: Random regular graph (nv=14, ne=$(ne(g3)))")
    println("  Exact result: $exact_result3")
    println("  QIP bound: $qip_bound3")
    println("  QP bound: $qp_bound3")
    println("  LP bound: $lp_bound3")
    
    @test qip_bound3 >= exact_result3 - 1e-6
    @test qip_bound3 <= qp_bound3 + 1e-6
    println("  ✓ QIP bound is valid upper bound")
    
    # Test case 4: Test with all zeros (should return 0)
    g4 = SimpleGraph(5)
    J4 = zeros(Float64, ne(g4))
    h4 = zeros(Float64, nv(g4))
    
    qip_bound4 = QIP_bound(g4, J4, h4; optimizer=SCIP.Optimizer)
    
    println("\nTest case 4: Graph with all zeros")
    println("  QIP bound: $qip_bound4")
    
    @test abs(qip_bound4) < 1e-6
    println("  ✓ QIP bound is approximately zero")
    
    # Test case 5: Test with positive J and h
    g5 = SimpleGraph(4)
    add_edge!(g5, 1, 2)
    add_edge!(g5, 2, 3)
    add_edge!(g5, 3, 4)
    
    J5 = [1.0, 1.0, 1.0]
    h5 = [1.0, 1.0, 1.0, 1.0]
    
    problem5 = GenericTensorNetwork(SpinGlass(g5, J5, h5); optimizer=TreeSA())
    exact_result5 = solve(problem5, SizeMax())[].n
    
    qip_bound5 = QIP_bound(g5, J5, h5; optimizer=SCIP.Optimizer)
    qp_bound5 = QP_bound(g5, J5, h5; optimizer=SCIP.Optimizer)
    
    println("\nTest case 5: Path graph with positive weights")
    println("  Exact result: $exact_result5")
    println("  QIP bound: $qip_bound5")
    println("  QP bound: $qp_bound5")
    
    @test qip_bound5 >= exact_result5 - 1e-6
    @test qip_bound5 <= qp_bound5 + 1e-6
    println("  ✓ QIP bound is valid upper bound")
    
    println("\n✓ All tests for QIP_bound passed!")
end