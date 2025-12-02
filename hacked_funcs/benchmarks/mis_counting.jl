using OptimalBranching
using OptimalBranching.OptimalBranchingCore, OptimalBranching.OptimalBranchingMIS
using GenericTensorNetworks, ProblemReductions
using Graphs, TropicalNumbers, OMEinsum, OMEinsumContractionOrders, Random
using Base.Threads
using TensorBranching: ob_region, optimal_branches, ContractionTreeSlicer, uncompress, SlicedBranch, complexity
using TensorBranching: optimal_branches_counting, initialize_code, GreedyBrancher, ScoreRS
using OptimalBranchingMIS: TensorNetworkSolver
using OMEinsumContractionOrders: TreeSA
using OMEinsum: DynamicNestedEinsum
using OMEinsumContractionOrders: uniformsize
using CSV, DataFrames
using OrderedCollections
include("../src/mis_counting.jl")

# Set random seed
Random.seed!(12345)

# Input and output directories
input_dir = "models/mis_graphs"
output_dir = "results/mis_counting"
mkpath(output_dir)  # Create output directory

# Get all graph files
graph_files = filter(x -> endswith(x, ".graph"), readdir(input_dir, join=true))

# Results file
results_file = joinpath(output_dir, "mis_counting_results.csv")

# Create or open results file
if isfile(results_file)
    # Append to existing file
    results_df = CSV.read(results_file, DataFrame)
else
    # Create new DataFrame with headers
    results_df = DataFrame(
        graph_name = String[],
        vertices = Int[],
        edges = Int[],
        sc_target = Int[],
        total_tc = Float32[],
        slice_num = Int[],
        total_tc_slicing = Float32[],
        sc_slicing = Float32[],
        slice_num_slicing = Int[]
    )
    CSV.write(results_file, results_df)  # Write header
end

# Fixed space complexity target
sc_target = 31

for graph_file in graph_files
    # Read graph file
    graph = SimpleGraph()
    open(graph_file, "r") do io
        # Skip metadata lines
        while !eof(io)
            line = readline(io)
            if startswith(line, "vertices: ")
                n_vertices = parse(Int, split(line)[2])
                graph = SimpleGraph(n_vertices)
            elseif line == "edges:"
                break
            end
        end
        
        # Add edges
        while !eof(io)
            line = readline(io)
            if isempty(line) || startswith(line, "#")
                continue
            end
            parts = split(line)
            if length(parts) >= 2
                u = parse(Int, parts[1])
                v = parse(Int, parts[2])
                add_edge!(graph, u, v)
            end
        end
    end
    
    # Set weights to 1 (Float32)
    weights = ones(Float32, nv(graph))
    p = MISProblem(graph, weights)
    
    # Extract graph name
    graph_name = basename(graph_file)
    
    println("\nProcessing graph: $graph_name")
    println("  Vertices: ", nv(graph))
    println("  Edges: ", ne(graph))
    
    # Create contraction code
    seed = 1
    Random.seed!(seed)
    code = initialize_code(graph, TreeSA())
    cc = contraction_complexity(code, uniformsize(code, 2))
    println("  Code complexity: ", cc)
    
    println("  " * "="^40)
    println("  sc_target: $sc_target")
    
    # Set up slicer
    slicer = ContractionTreeSlicer(
        sc_target = sc_target,
        table_solver = TensorNetworkSolver(),
        region_selector = ScoreRS(n_max=10),
        brancher = GreedyBrancher()
    )
    
    # Compute slicing results (for the entire graph)
    edge_ixs = [[minmax(e.src,e.dst)...] for e in Graphs.edges(graph)]
    vertex_ixs = [[v] for v in 1:nv(graph)]
    ixs = vcat(edge_ixs, vertex_ixs)
    code0 = OMEinsumContractionOrders.EinCode([ixs...], Int[])
    Random.seed!(seed)
    optcode_sliced = optimize_code(code0, uniformsize(code0, 2), TreeSA(), slicer=TreeSASlicer(score=ScoreFunction(sc_target=sc_target)))
    total_tc_slicing, sc_slicing = OMEinsum.timespace_complexity(optcode_sliced, uniformsize(code0, 2))
    slice_num_slicing = length(optcode_sliced.slicing)
    println("  Slicing results:")
    println("    Total tc (slicing): ", total_tc_slicing)
    println("    Space complexity (slicing): ", sc_slicing)
    println("    Number of slices (slicing): ", slice_num_slicing)
    
    # Run counting
    finished_slices = slice_bfs(p, slicer, code, 1)
    
    # Calculate total time complexity (using Float32)
    total_tc = -Inf32  # Initialize to negative infinity (Float32)
    if !isempty(finished_slices)
        for slice in finished_slices
            cc_val = complexity(slice).tc
            if total_tc == -Inf32
                total_tc = Float32(cc_val)
            else
                total_tc = log2(2^total_tc + 2^cc_val)
            end
        end
        println("  Total tc (branching): ", total_tc)
        println("  Slice num (branching): ", length(finished_slices))
        
        # Create a row of data
        row = DataFrame(
            graph_name = [graph_name],
            vertices = [nv(graph)],
            edges = [ne(graph)],
            sc_target = [sc_target],
            total_tc = [total_tc],
            slice_num = [length(finished_slices)],
            total_tc_slicing = [Float32(total_tc_slicing)],
            sc_slicing = [Float32(sc_slicing)],
            slice_num_slicing = [slice_num_slicing]
        )
        
        # Immediately append to CSV file
        CSV.write(results_file, row, append=true)
        println("  Results saved to CSV for graph: $graph_name")
    end
end

println("\nAll results saved to: $results_file")