using OptimalBranching
using OptimalBranching.OptimalBranchingCore, OptimalBranching.OptimalBranchingMIS
using GenericTensorNetworks, ProblemReductions
using Graphs, TropicalNumbers, OMEinsum, OMEinsumContractionOrders,Random
using Base.Threads
using TensorBranching: ob_region, optimal_branches, ContractionTreeSlicer, uncompress, SlicedBranch, complexity
using TensorBranching: optimal_branches_ground_counting, optimal_branches_ground_counting_induced_sparsity, initialize_code, GreedyBrancher, ScoreRS
using OptimalBranchingMIS: TensorNetworkSolver
using OMEinsumContractionOrders: TreeSA
using OMEinsum: DynamicNestedEinsum
using OMEinsumContractionOrders: uniformsize
include("../src/spin_glass_ground_counting.jl")



function test_slice_dfs(g, J, h,sc_target)
    println("=" ^ 60)
    println("Testing slice_dfs function")
    println("=" ^ 60)
    
    # Create test problem
    p = SpinGlassProblem(g, J, h)
    g_problem = GenericTensorNetwork(SpinGlass(g, J, h); optimizer=TreeSA())
    # count_all_independent_sets = solve(g_problem, CountingAll())[]
    count_maximum_spin_glass = solve(g_problem, CountingMax())[]
    spin_glass_size = count_maximum_spin_glass.n[1]
    spin_glass_count = count_maximum_spin_glass.c[1]
    println("Maximum spin glass size: ", spin_glass_size)
    println("Count maximum spin glass: ", spin_glass_count)

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
            spin_glass_count_sum = 0.0
            for (i, slice) in enumerate(finished_slices[1:length(finished_slices)])
                sub_g_problem = GenericTensorNetwork(SpinGlass(slice.p.g, slice.p.J, slice.p.h); optimizer=TreeSA())
                sub_count_maximum_spin_glass = solve(sub_g_problem, CountingMax())[]
                sub_spin_glass_size = sub_count_maximum_spin_glass.n[1] + slice.r
                sub_spin_glass_count = sub_count_maximum_spin_glass.c[1]
                if abs(sub_spin_glass_size - spin_glass_size) < 1e-4
                    spin_glass_count_sum += sub_spin_glass_count
                end
                
            end
            println("Count maximum spin glass: ", spin_glass_count_sum,", spin_glass_size: $spin_glass_size, spin_glass_count: $spin_glass_count")
            @assert spin_glass_count_sum == spin_glass_count "Spin glass count mismatch: expected $spin_glass_count, got $spin_glass_count_sum"
            
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
    seed = 123456
    Random.seed!(seed)
    g = Graphs.grid([20, 20])
    J = Float32.(2.0 * rand(Bool, ne(g)) .- 1.0)  # Random ±1
    # h = Float32.(ones(Float32, nv(g)) * 0.5)
    # h = randn(Float32, nv(g))
    h = Float32.(2.0 * rand(Bool, nv(g)) .- 1.0)
    test_slice_dfs(g, J, h, 10)
    test_slice_dfs(g, J, h, 6)
end

