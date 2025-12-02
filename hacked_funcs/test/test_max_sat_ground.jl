using OptimalBranching
using OptimalBranching.OptimalBranchingCore, OptimalBranching.OptimalBranchingMIS
using GenericTensorNetworks, ProblemReductions
using Graphs, TropicalNumbers, OMEinsum, OMEinsumContractionOrders, Random
using Base.Threads
using TensorBranching: ob_region, optimal_branches, ContractionTreeSlicer, uncompress, SlicedBranch, complexity
using TensorBranching: optimal_branches_ground_induced_sparsity, initialize_code, GreedyBrancher, ScoreRS
using OptimalBranchingMIS: TensorNetworkSolver, graph_clauses_to_sat
using OMEinsumContractionOrders: TreeSA
using OMEinsum: DynamicNestedEinsum
using OMEinsumContractionOrders: uniformsize
include("../src/max_sat_ground.jl")

"""
    generate_simple_sat(n_vars::Int, n_clauses::Int)

Generate a simple SAT problem with random clauses.
Returns (g, clauses) where:
- g is a bipartite graph with n_vars variable vertices and n_clauses clause vertices
- clauses is a list of clauses, each clause is a vector of integers (positive for positive literals, negative for negative)
"""
function generate_simple_sat(n_vars::Int, n_clauses::Int; seed::Int=1234)
    Random.seed!(seed)
    g = SimpleGraph(n_vars + n_clauses)
    clauses = Vector{Vector{Int}}()
    
    for clause_idx in 1:n_clauses
        # Each clause has 2-3 literals
        clause_size = rand(2:3)
        clause = Int[]
        used_vars = Set{Int}()
        
        for _ in 1:clause_size
            var = rand(1:n_vars)
            while var in used_vars
                var = rand(1:n_vars)
            end
            push!(used_vars, var)
            # Randomly choose positive or negative literal
            literal = rand(Bool) ? var : -var
            push!(clause, literal)
            
            # Add edge between variable vertex and clause vertex
            clause_vertex = n_vars + clause_idx
            add_edge!(g, var, clause_vertex)
        end
        
        push!(clauses, clause)
    end
    
    return g, clauses
end

function test_slice_dfs(g, clauses, sc_target)
    println("=" ^ 60)
    println("Testing slice_dfs function for MaxSAT")
    println("=" ^ 60)
    
    n_vars = nv(g) - length(clauses)
    
    # Create test problem
    p = MaxSatProblem(g, clauses)
    sat = graph_clauses_to_sat(g, clauses, n_vars)
    g_problem = GenericTensorNetwork(sat; optimizer=TreeSA())
    t0 = time()
    result = solve(g_problem, SizeMax())[]
    maximum_satisfied_clauses = result.n
    println("Maximum satisfied clauses: ", maximum_satisfied_clauses)
    t1 = time()
    println("Time taken to solve the problem: ", t1 - t0)
    exit()
    println("\nProblem info:")
    println("  Variables: ", n_vars)
    println("  Clauses: ", length(clauses))
    println("  Graph vertices: ", nv(g))
    println("  Graph edges: ", ne(g))
    
    # Create a contraction code for the SAT problem
    # We need to create code from the SAT problem
    code = GenericTensorNetwork(sat; optimizer=TreeSA()).code
    
    # Create a slicer with reasonable parameters
    slicer = ContractionTreeSlicer(
        sc_target = sc_target,
        table_solver = TensorNetworkSolver(),
        region_selector = ScoreRS(n_max=5),
        brancher = GreedyBrancher()
    )
    
    println("\nSlicer parameters:")
    println("  sc_target: ", slicer.sc_target)
    println("  table_solver: ", typeof(slicer.table_solver))
    
    # Test slice_dfs
    println("\n" * "-" ^ 60)
    println("Running slice_dfs...")
    println("-" ^ 60)
    
    try
        finished_slices = slice_dfs(p, slicer, code, 1)
        
        println("\nResults:")
        println("  Number of finished slices: ", length(finished_slices))
        
        if !isempty(finished_slices)
            maximum_satisfied_clauses_sum = 0.0
            for (i, slice) in enumerate(finished_slices)
                n_vars_slice = nv(slice.p.g) - length(slice.p.clauses)
                n_clauses_slice = length(slice.p.clauses)
                println("\nSlice $i:")
                println("  n_vars_slice: $n_vars_slice")
                println("  n_clauses_slice: $n_clauses_slice")
                println("  slice.r: $(slice.r)")
                println("  slice.p.clauses: $(slice.p.clauses)")
                
                sub_result_n = 0.0
                if n_clauses_slice == 0
                    sub_result_n = 0.0
                else
                    sat_slice = graph_clauses_to_sat(slice.p.g, slice.p.clauses, n_vars_slice)
                    sub_g_problem = GenericTensorNetwork(sat_slice; optimizer=TreeSA())
                    sub_result = solve(sub_g_problem, SizeMax())[]
                    sub_result_n = sub_result.n
                end
                println("  sub_result.n (max satisfied in subproblem): $(sub_result_n)")
                
                sub_maximum_satisfied_clauses = sub_result_n + slice.r
                println("  total (sub_result.n + slice.r): $sub_maximum_satisfied_clauses")
                
                if sub_maximum_satisfied_clauses > maximum_satisfied_clauses_sum
                    maximum_satisfied_clauses_sum = sub_maximum_satisfied_clauses
                end
            end
            println("Maximum satisfied clauses (from slices): ", maximum_satisfied_clauses_sum)
            println("Maximum satisfied clauses (direct): ", maximum_satisfied_clauses)
            @assert abs(maximum_satisfied_clauses_sum - maximum_satisfied_clauses) < 1e-12 "Maximum satisfied clauses mismatch: expected $maximum_satisfied_clauses, got $maximum_satisfied_clauses_sum"
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
    for seed in 1:10
        Random.seed!(seed)
        
        
        # Test 1: Medium problem
        g, clauses = generate_simple_sat(30, 800, seed=seed+1)
        test_slice_dfs(g, clauses, 8)
    end
end

