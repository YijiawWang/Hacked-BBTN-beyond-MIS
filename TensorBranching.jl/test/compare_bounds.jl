using TensorBranching
using TensorBranching: spinglass_linear_LP_bound, QP_bound, QIP_bound
using Graphs
using GenericTensorNetworks
using Random
using JuMP, SCIP

"""
    compare_bounds(; seed=1234, time_limit=10.0, verbose=true)

Compare three upper bound methods (LP, QP, QIP) on three graph types:
1. n=600, degree=3 Random Regular Graph (RRG)
2. 70×70 RKSG graph
3. 70×70 Lattice graph

Returns a dictionary with results for each graph type.
"""
function compare_bounds(; seed::Int=1234, time_limit::Float64=10.0, verbose::Bool=true)
    Random.seed!(seed)
    
    results = Dict{String, Dict{String, Any}}()
    
    # Test case 1: n=600, degree=3 Random Regular Graph
    if verbose
        println("="^80)
        println("Test Case 1: Random Regular Graph (n=600, degree=3)")
        println("="^80)
    end
    
    g1 = random_regular_graph(600, 3)
    J1 = randn(Float64, ne(g1))
    h1 = randn(Float64, nv(g1))
    
    if verbose
        println("Graph: nv=$(nv(g1)), ne=$(ne(g1))")
        println("Computing bounds...")
    end
    
    rrg_results = compute_bounds(g1, J1, h1, time_limit, verbose)
    rrg_results["graph_type"] = "RRG_n600_d3"
    rrg_results["nv"] = nv(g1)
    rrg_results["ne"] = ne(g1)
    results["RRG_n600_d3"] = rrg_results
    
    # Test case 2: 70×70 RKSG graph
    if verbose
        println("\n" * "="^80)
        println("Test Case 2: RKSG Graph (70×70)")
        println("="^80)
    end
    
    g2 = SimpleGraph(GenericTensorNetworks.random_diagonal_coupled_graph(70, 70, 0.8))
    J2 = randn(Float64, ne(g2))
    h2 = randn(Float64, nv(g2))
    
    if verbose
        println("Graph: nv=$(nv(g2)), ne=$(ne(g2))")
        println("Computing bounds...")
    end
    
    rksg_results = compute_bounds(g2, J2, h2, time_limit, verbose)
    rksg_results["graph_type"] = "RKSG_70x70"
    rksg_results["nv"] = nv(g2)
    rksg_results["ne"] = ne(g2)
    results["RKSG_70x70"] = rksg_results
    
    # Test case 3: 70×70 Lattice graph
    if verbose
        println("\n" * "="^80)
        println("Test Case 3: Lattice Graph (70×70)")
        println("="^80)
    end
    
    g3 = Graphs.grid([70, 70])
    J3 = randn(Float64, ne(g3))
    h3 = randn(Float64, nv(g3))
    
    if verbose
        println("Graph: nv=$(nv(g3)), ne=$(ne(g3))")
        println("Computing bounds...")
    end
    
    lattice_results = compute_bounds(g3, J3, h3, time_limit, verbose)
    lattice_results["graph_type"] = "Lattice_70x70"
    lattice_results["nv"] = nv(g3)
    lattice_results["ne"] = ne(g3)
    results["Lattice_70x70"] = lattice_results
    
    # Print summary
    if verbose
        print_summary(results)
    end
    
    return results
end

"""
    compute_bounds(g, J, h, time_limit, verbose)

Compute all three bounds for a given graph and problem instance.
"""
function compute_bounds(g::SimpleGraph, J::Vector{Float64}, h::Vector{Float64}, 
                        time_limit::Float64, verbose::Bool)
    results = Dict{String, Any}()
    
    # LP bound
    if verbose
        print("  Computing LP bound... ")
    end
    try
        start_time = time()
        lp_bound = spinglass_linear_LP_bound(g, J, h; optimizer=SCIP.Optimizer)
        lp_time = time() - start_time
        results["LP_bound"] = lp_bound
        results["LP_time"] = lp_time
        results["LP_status"] = "success"
        if verbose
            println("✓ $lp_bound (time: $(round(lp_time, digits=3))s)")
        end
    catch e
        results["LP_bound"] = nothing
        results["LP_time"] = nothing
        results["LP_status"] = "failed: $e"
        if verbose
            println("✗ Failed: $e")
        end
    end
    
    # QP bound
    if verbose
        print("  Computing QP bound... ")
    end
    try
        start_time = time()
        qp_bound = QP_bound(g, J, h; optimizer=SCIP.Optimizer, time_limit=time_limit)
        qp_time = time() - start_time
        results["QP_bound"] = qp_bound
        results["QP_time"] = qp_time
        results["QP_status"] = "success"
        if verbose
            println("✓ $qp_bound (time: $(round(qp_time, digits=3))s)")
        end
    catch e
        results["QP_bound"] = nothing
        results["QP_time"] = nothing
        results["QP_status"] = "failed: $e"
        if verbose
            println("✗ Failed: $e")
        end
    end
    
    # QIP bound
    if verbose
        print("  Computing QIP bound... ")
    end
    try
        start_time = time()
        qip_bound = QIP_bound(g, J, h; optimizer=SCIP.Optimizer, time_limit=time_limit)
        qip_time = time() - start_time
        results["QIP_bound"] = qip_bound
        results["QIP_time"] = qip_time
        results["QIP_status"] = "success"
        if verbose
            println("✓ $qip_bound (time: $(round(qip_time, digits=3))s)")
        end
    catch e
        results["QIP_bound"] = nothing
        results["QIP_time"] = nothing
        results["QIP_status"] = "failed: $e"
        if verbose
            println("✗ Failed: $e")
        end
    end
    
    # Compute comparisons
    if results["LP_status"] == "success" && results["QP_status"] == "success"
        results["LP_vs_QP"] = results["LP_bound"] - results["QP_bound"]
    end
    if results["LP_status"] == "success" && results["QIP_status"] == "success"
        results["LP_vs_QIP"] = results["LP_bound"] - results["QIP_bound"]
    end
    if results["QP_status"] == "success" && results["QIP_status"] == "success"
        results["QP_vs_QIP"] = results["QP_bound"] - results["QIP_bound"]
    end
    
    return results
end

"""
    print_summary(results)

Print a formatted summary of all results.
"""
function print_summary(results::Dict{String, Dict{String, Any}})
    println("\n" * "="^80)
    println("SUMMARY")
    println("="^80)
    
    for (graph_name, graph_results) in results
        println("\n$(graph_results["graph_type"]): nv=$(graph_results["nv"]), ne=$(graph_results["ne"])")
        println("-"^80)
        
        if graph_results["LP_status"] == "success"
            println("  LP bound:  $(graph_results["LP_bound"])  (time: $(round(graph_results["LP_time"], digits=3))s)")
        else
            println("  LP bound:  FAILED - $(graph_results["LP_status"])")
        end
        
        if graph_results["QP_status"] == "success"
            println("  QP bound:  $(graph_results["QP_bound"])  (time: $(round(graph_results["QP_time"], digits=3))s)")
        else
            println("  QP bound:  FAILED - $(graph_results["QP_status"])")
        end
        
        if graph_results["QIP_status"] == "success"
            println("  QIP bound: $(graph_results["QIP_bound"])  (time: $(round(graph_results["QIP_time"], digits=3))s)")
        else
            println("  QIP bound: FAILED - $(graph_results["QIP_status"])")
        end
        
        # Print comparisons
        if haskey(graph_results, "LP_vs_QP")
            diff = graph_results["LP_vs_QP"]
            tighter = diff > 0 ? "QP" : "LP"
            println("  LP vs QP:  $(round(diff, digits=6))  ($tighter is tighter)")
        end
        
        if haskey(graph_results, "LP_vs_QIP")
            diff = graph_results["LP_vs_QIP"]
            tighter = diff > 0 ? "QIP" : "LP"
            println("  LP vs QIP: $(round(diff, digits=6))  ($tighter is tighter)")
        end
        
        if haskey(graph_results, "QP_vs_QIP")
            diff = graph_results["QP_vs_QIP"]
            tighter = diff > 0 ? "QIP" : "QP"
            println("  QP vs QIP: $(round(diff, digits=6))  ($tighter is tighter)")
        end
    end
    
    println("\n" * "="^80)
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    results = compare_bounds(seed=1234, time_limit=20.0, verbose=true)
end

