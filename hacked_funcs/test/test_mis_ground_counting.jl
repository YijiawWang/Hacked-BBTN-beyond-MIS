using OptimalBranching
using OptimalBranching.OptimalBranchingCore, OptimalBranching.OptimalBranchingMIS
using GenericTensorNetworks, ProblemReductions
using Graphs, TropicalNumbers, OMEinsum, OMEinsumContractionOrders,Random
using Base.Threads
using TensorBranching: ob_region, optimal_branches, ContractionTreeSlicer, uncompress, SlicedBranch, complexity
using TensorBranching: optimal_branches_ground_counting, initialize_code, GreedyBrancher, ScoreRS
using OptimalBranchingMIS: TensorNetworkSolver
using OMEinsumContractionOrders: TreeSA
using OMEinsum: DynamicNestedEinsum
using OMEinsumContractionOrders: uniformsize
include("../src/mis_ground_counting.jl")



function test_slice_bfs(g, weights,sc_target)
    println("=" ^ 60)
    println("Testing slice_bfs function")
    println("=" ^ 60)
    
    # Create test problem
    p = MISProblem(g, weights)
    g_problem = GenericTensorNetwork(IndependentSet(g, weights); optimizer=TreeSA())
    # count_all_independent_sets = solve(g_problem, CountingAll())[]
    count_maximum_independent_sets = solve(g_problem, CountingMax())[]
    mis_size = count_maximum_independent_sets.n[1]
    mis_count = count_maximum_independent_sets.c[1]
    println("Maximum independent set size: ", mis_size)
    println("Count maximum independent sets: ", mis_count)

    println("\nGraph info:")
    println("  Vertices: ", nv(g))
    println("  Edges: ", ne(g))
    println("  Edge list: ", collect(edges(g)))
    
    # Create a contraction code for the independent set problem using initialize_code
    code = initialize_code(g, TreeSA())
    
    # Create a slicer with reasonable parameters
    
    slicer = ContractionTreeSlicer(
        sc_target = sc_target,
        table_solver = TensorNetworkSolver(),
        region_selector = ScoreRS(n_max=10),
        brancher = GreedyBrancher()
    )
    
    println("\nSlicer parameters:")
    println("  sc_target: ", slicer.sc_target)
    println("  table_solver: ", typeof(slicer.table_solver))
    
    # Test slice_bfs
    println("\n" * "-" ^ 60)
    println("Running slice_bfs...")
    println("-" ^ 60)
    
    try
        finished_slices = slice_bfs(p, slicer, code, 1)
        
        println("\nResults:")
        println("  Number of finished slices: ", length(finished_slices))
        
        if !isempty(finished_slices)
            mis_count_sum = 0.0
            for (i, slice) in enumerate(finished_slices[1:length(finished_slices)])
                sub_g_problem = GenericTensorNetwork(IndependentSet(slice.p.g, slice.p.weights); optimizer=TreeSA())
                sub_count_maximum_independent_sets = solve(sub_g_problem, CountingMax())[]
                sub_mis_size = sub_count_maximum_independent_sets.n[1] + slice.r
                
            
                sub_mis_count = sub_count_maximum_independent_sets.c[1]
                if abs(sub_mis_size - mis_size) < 1e-12
                    mis_count_sum += sub_mis_count
                end
            end
            println("Count maximum independent sets: ", mis_count_sum)
            @assert mis_count_sum == mis_count "MIS count mismatch: expected $mis_count, got $mis_count_sum"
            
        end
        
        return finished_slices
        
    catch e
        println("\n❌ Error during slice_bfs:")
        println("  ", e)
        rethrow(e)
    end
end

# Run the test
if abspath(PROGRAM_FILE) == @__FILE__
    seed = 12345
    Random.seed!(seed)
    g = random_regular_graph(50, 3)
    weights = UnitWeight(nv(g))
    test_slice_bfs(g, weights, 4)
    test_slice_bfs(g, weights, 3)

    g = random_regular_graph(50, 3)
    weights = rand(Float64, nv(g))
    test_slice_bfs(g, weights, 4)
    test_slice_bfs(g, weights, 3)

    g = SimpleGraph(GenericTensorNetworks.random_diagonal_coupled_graph(10, 10, 0.8))
    weights = UnitWeight(nv(g))
    test_slice_bfs(g, weights, 5)
    test_slice_bfs(g, weights, 4)


    g = SimpleGraph(GenericTensorNetworks.random_diagonal_coupled_graph(10, 10, 0.8))
    weights = rand(Float64, nv(g))
    test_slice_bfs(g, weights, 5)
    test_slice_bfs(g, weights, 4)

end

