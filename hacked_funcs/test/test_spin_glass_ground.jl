using OptimalBranching
using OptimalBranching.OptimalBranchingCore, OptimalBranching.OptimalBranchingMIS
using GenericTensorNetworks, ProblemReductions
using Graphs, TropicalNumbers, OMEinsum, OMEinsumContractionOrders,Random
using Base.Threads
using TensorBranching: ob_region, optimal_branches, ContractionTreeSlicer, uncompress, SlicedBranch, complexity
using TensorBranching: optimal_branches_ground_induced_sparsity, initialize_code, GreedyBrancher, ScoreRS
using OptimalBranchingMIS: TensorNetworkSolver
using OMEinsumContractionOrders: TreeSA
using OMEinsum: DynamicNestedEinsum
using OMEinsumContractionOrders: uniformsize
include("../src/spin_glass_ground.jl")



function test_slice_dfs(g, J, h,sc_target)
    println("=" ^ 60)
    println("Testing slice_dfs function")
    println("=" ^ 60)
    
    # Create test problem
    p = SpinGlassProblem(g, J, h)
    g_problem = GenericTensorNetwork(SpinGlass(g, J, h); optimizer=TreeSA())
    maximum_spin_glass_result = solve(g_problem, SizeMax())[].n
    println("Maximum spin glass size: ", maximum_spin_glass_result)

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
    println("Running slice_dfs...")
    println("-" ^ 60)
    
    try
        finished_slices = slice_dfs_lp(p, slicer, code, true, 1)
        
        println("\nResults:")
        println("  Number of finished slices: ", length(finished_slices))
        
        if !isempty(finished_slices)
            maximum_spin_glass_result_sum = 0.0
            for (i, slice) in enumerate(finished_slices[1:length(finished_slices)])
                sub_g_problem = GenericTensorNetwork(SpinGlass(slice.p.g, slice.p.J, slice.p.h); optimizer=TreeSA())
                sub_maximum_spin_glass_result = solve(sub_g_problem, SizeMax())[].n + slice.r
                if sub_maximum_spin_glass_result > maximum_spin_glass_result_sum
                    maximum_spin_glass_result_sum = sub_maximum_spin_glass_result
                end
            end
            println("Maximum spin glass result: ", maximum_spin_glass_result_sum,", maximum_spin_glass_result: $maximum_spin_glass_result")
            @assert (maximum_spin_glass_result_sum - maximum_spin_glass_result) < 1e-12 "Maximum spin glass result mismatch: expected $maximum_spin_glass_result, got $maximum_spin_glass_result_sum"
        end
        
        return finished_slices
        
    catch e
        println("\n❌ Error during slice_dfs:")
        println("  ", e)
        rethrow(e)
    end
end
# Run the test
if abspath(PROGRAM_FILE) == @__FILE__
    seed = 12345
    Random.seed!(seed)
    # g = SimpleGraph(GenericTensorNetworks.random_diagonal_coupled_graph(20, 20, 0.8))
    bit_length = 7
    n10 = rand(2^(bit_length-1):2^bit_length)
    n20 = rand(2^(bit_length-1):2^bit_length)
    n1 = prevprime(2^bit_length, 1)
    n2 = prevprime(2^bit_length, 2)
    # n1 = prevprime(n10, 1)
    # n2 = prevprime(n20, 1)
    factoring = Factoring(bit_length, bit_length, n1*n2) # initialize the Factoring problem
    println("n1: ", n1, " n2: ", n2, " n1*n2: ", n1*n2)
    g = reduction_graph()
    paths = reduction_paths(g,Factoring,SpinGlass)
    reduction_result = reduceto(paths[1], factoring)
    target = target_problem(reduction_result)
    println("Target problem vertices: ", nv(target.graph))
    println("Target problem edges: ", ne(target.graph))
    
    # Calculate degree statistics
    degrees = [degree(target.graph, v) for v in 1:nv(target.graph)]
    avg_degree = sum(degrees) / length(degrees)
    max_degree = maximum(degrees)
    min_degree = minimum(degrees)
    println("Target problem average degree: ", avg_degree)
    println("Target problem max degree: ", max_degree)
    println("Target problem min degree: ", min_degree)
    
    println("Target problem J: ", target.J)
    println("Target problem h: ", target.h)

    p = SpinGlassProblem(target.graph, target.J, target.h)
    g = target.graph
    J = target.J
    h = target.h
    test_slice_dfs(g, J, h, 10)
    


    # #g = random_regular_graph(200, 3)
    # g = Graphs.grid([15, 15])
    # # J = randn(Float64, ne(g))
    # J = 2.0 * rand(Bool, ne(g)) .- 1.0  # Random ±1
    # h = ones(Float64, nv(g))
    # test_slice_dfs(g, J, h, 8)
   
end

