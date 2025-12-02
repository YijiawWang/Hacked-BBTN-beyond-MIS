using OptimalBranching
using OptimalBranching.OptimalBranchingCore, OptimalBranching.OptimalBranchingMIS
using GenericTensorNetworks, ProblemReductions
using Graphs, TropicalNumbers, OMEinsum, OMEinsumContractionOrders,Random
using Base.Threads
using TensorBranching: ob_region, optimal_branches, ContractionTreeSlicer, uncompress, SlicedBranch, complexity
using TensorBranching: optimal_branches_counting, initialize_code, GreedyBrancher, ScoreRS
using OptimalBranchingMIS: TensorNetworkSolver
using OMEinsumContractionOrders: TreeSA
using OMEinsum: DynamicNestedEinsum
using OMEinsumContractionOrders: uniformsize
include("../src/mis_counting.jl")



function test_slice_bfs(g, weights, sc_target)
    println("=" ^ 60)
    println("Testing slice_bfs function (counting)")
    println("=" ^ 60)
    
    p = MISProblem(g, weights)
    g_problem = GenericTensorNetwork(IndependentSet(g, weights); optimizer=TreeSA())
    count_all_independent_sets = solve(g_problem, CountingAll())[]
    println("Count all independent sets: ", count_all_independent_sets)

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
            all_count_sum = 0.0
            for (i, slice) in enumerate(finished_slices)
                sub_g_problem = GenericTensorNetwork(IndependentSet(slice.p.g, slice.p.weights); optimizer=TreeSA())
                sub_count_all_independent_sets = solve(sub_g_problem, CountingAll())[]
                all_count_sum += sub_count_all_independent_sets
            end
            println("Count all independent sets (sum): ", all_count_sum)
            @assert all_count_sum == count_all_independent_sets "All independent sets count mismatch: expected $count_all_independent_sets, got $all_count_sum"
            
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
    g = SimpleGraph(GenericTensorNetworks.random_diagonal_coupled_graph(20, 20, 0.8))
    weights = UnitWeight(nv(g))
    test_slice_bfs(g, weights, 10)
    test_slice_bfs(g, weights, 6)
end
