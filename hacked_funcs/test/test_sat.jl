using OptimalBranching
using OptimalBranching.OptimalBranchingCore, OptimalBranching.OptimalBranchingMIS
using GenericTensorNetworks, ProblemReductions
using GenericTensorNetworks: ∧, ∨, ¬, GenericTensorNetwork, GreedyMethod, SizeMax, solve
using GenericTensorNetworks.ProblemReductions: CNF, CNFClause, BoolVar, Satisfiability
using Graphs, Random
using OMEinsumContractionOrders: TreeSA
using TensorBranching: ob_region, optimal_branches, ContractionTreeSlicer, uncompress, SlicedBranch, complexity
using TensorBranching: optimal_branches_ground_induced_sparsity, initialize_code, GreedyBrancher, ScoreRS
using OptimalBranchingMIS: TensorNetworkSolver
using OMEinsum: DynamicNestedEinsum
using OMEinsumContractionOrders: uniformsize
include("../src/sat.jl")

"""
    graph_clauses_to_sat(g::SimpleGraph, clauses::Vector{Vector{Int}}, n_vars::Int, use_constraint::Bool = false)

Convert a graph and clauses to a Satisfiability problem.
"""
function graph_clauses_to_sat(g::SimpleGraph, clauses::Vector{Vector{Int}}, n_vars::Int, use_constraint::Bool = false)
    n_clauses = length(clauses)
    
    # Validate the number of vertices in the graph
    @assert nv(g) == n_vars + n_clauses "Graph should have $(n_vars + n_clauses) vertices, but has $(nv(g))"
    
    # Create variable symbols (using x1, x2, ...)
    var_symbols = [Symbol("x$i") for i in 1:n_vars]
    
    # Build CNF clauses
    cnf_clauses = CNFClause{Symbol}[]
    
    for (clause_idx, clause) in enumerate(clauses)
        bool_vars = BoolVar{Symbol}[]
        
        # Process each literal in the clause
        for literal in clause
            # Get the variable index (absolute value)
            var_idx = abs(literal)
            @assert 1 <= var_idx <= n_vars "Variable index $var_idx out of range [1, $n_vars]"
            
            # Determine if it's a positive or negative literal
            is_negated = literal < 0
            
            # Create BoolVar
            push!(bool_vars, BoolVar(var_symbols[var_idx], is_negated))
        end
        
        # Create CNFClause
        push!(cnf_clauses, CNFClause(bool_vars))
    end
    
    # Create CNF object
    cnf = CNF(cnf_clauses)
   
    return Satisfiability(cnf; use_constraints=use_constraint)
end

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
    println("Testing slice_dfs function for SAT")
    println("=" ^ 60)
    
    n_vars = nv(g) - length(clauses)
    
    # Create test problem
    # Use MaxSatProblem for slice_dfs (since slice_dfs only supports MaxSatProblem)
    p = MaxSatProblem(g, clauses, true)
    sat = graph_clauses_to_sat(g, clauses, n_vars, p.use_constraint)
    g_problem = GenericTensorNetwork(sat; optimizer=TreeSA())
    t0 = time()
    result = solve(g_problem, SizeMax())[]
    num_satisfying_assignments = result.n
    println("Number of satisfying assignments: ", num_satisfying_assignments)
    t1 = time()
    println("Time taken to solve the problem: ", t1 - t0)
    
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
        total_satisfying_assignments = -Inf
       
        # For SAT, we sum up the counts from all slices (since SAT is a counting problem)
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
                # If no clauses, all assignments are satisfying
                sub_result_n = 0.0
            else
                sat_slice = graph_clauses_to_sat(slice.p.g, slice.p.clauses, n_vars_slice, slice.p.use_constraint)
                sub_g_problem = GenericTensorNetwork(sat_slice; optimizer=TreeSA())
                sub_result = solve(sub_g_problem, SizeMax())[]
                sub_result_n = sub_result.n
            end
            println("  sub_result.n (satisfying assignments in subproblem): $(sub_result_n)")
            
            slice_contrib = sub_result_n
            println("  slice contribution: $slice_contrib")
            
            total_satisfying_assignments = max(total_satisfying_assignments, slice_contrib)
        end
        println("\nTotal satisfying assignments (from slices): ", total_satisfying_assignments)
        println("Total satisfying assignments (direct): ", num_satisfying_assignments)
        # Check if both are -inf, both are 0, or both are finite and equal
        if isinf(total_satisfying_assignments) && isinf(num_satisfying_assignments)
            # Both are -inf, check if they're both negative infinity
            @assert total_satisfying_assignments == num_satisfying_assignments "Satisfying assignments mismatch: both are inf but different signs"
        elseif total_satisfying_assignments == 0 && num_satisfying_assignments == 0
            # Both are 0, pass
        elseif isfinite(total_satisfying_assignments) && isfinite(num_satisfying_assignments)
            # Both are finite, check if they're equal
            @assert abs(total_satisfying_assignments - num_satisfying_assignments) < 1e-12 "Satisfying assignments mismatch: expected $num_satisfying_assignments, got $total_satisfying_assignments"
        else
            @assert false "Satisfying assignments mismatch: expected $num_satisfying_assignments, got $total_satisfying_assignments (one is inf/zero and the other is not)"
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
    for m in [2,4,8]
        for seed in 1:5
            Random.seed!(seed)
        
            # Test with different problem sizes
            # - g::SimpleGraph: A bipartite graph with n_vars variable vertices and n_clauses clause vertices.
            #   Edges connect variable vertices to clause vertices only (no edges between variables or between clauses).
            # - clauses::Vector{Vector{Int}}: A list of clauses, where each clause is a vector of integers.
            #   Positive integers represent positive literals (unnegated variables), negative integers represent negated literals.
            g, clauses = generate_simple_sat(15, 15*m, seed=seed)
            test_slice_dfs(g, clauses, 3)
        end
    end
end
