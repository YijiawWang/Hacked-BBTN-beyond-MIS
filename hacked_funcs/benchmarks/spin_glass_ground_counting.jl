using OptimalBranching
using OptimalBranching.OptimalBranchingCore, OptimalBranching.OptimalBranchingMIS
using GenericTensorNetworks, ProblemReductions
using Graphs, TropicalNumbers, OMEinsum, OMEinsumContractionOrders, Random
using Base.Threads
using TensorBranching: ob_region, optimal_branches, ContractionTreeSlicer, uncompress, SlicedBranch, complexity
using TensorBranching: optimal_branches_ground_counting, initialize_code, GreedyBrancher, ScoreRS
using OptimalBranchingMIS: TensorNetworkSolver
using OMEinsumContractionOrders: TreeSA
using OMEinsum: DynamicNestedEinsum
using OMEinsumContractionOrders: uniformsize
using CSV, DataFrames
using OrderedCollections
include("../src/spin_glass_ground_counting.jl")

# Set random seed
Random.seed!(12345)

# Input and output directories
input_dir = "models/spin_glass_models"
output_dir = "results/spin_glass_ground_counting_scip"
mkpath(output_dir)  # Create output directory

# Get all model files
model_files = filter(x -> endswith(x, ".model"), readdir(input_dir, join=true))

# Results file
results_file = joinpath(output_dir, "spin_glass_ground_counting_results.csv")

# Create or open results file
if isfile(results_file)
    # Append to existing file
    results_df = CSV.read(results_file, DataFrame)
else
    # Create new DataFrame with headers
    results_df = DataFrame(
        model_name = String[],
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

for model_file in model_files
    # Read model file
    graph = SimpleGraph()
    edge_weights = Dict{Tuple{Int,Int}, Float64}()
    open(model_file, "r") do io
        # Skip metadata lines until vertex coordinates
        while !eof(io)
            line = readline(io)
            if startswith(line, "vertices: ")
                n_vertices = parse(Int, split(line)[2])
                graph = SimpleGraph(n_vertices)
            elseif startswith(line, "edges: ")
                n_edges = parse(Int, split(line)[2])
            elseif line == "vertex_coordinates:"
                break
            end
        end
        
        # Skip vertex coordinates
        while !eof(io)
            line = readline(io)
            if line == ""
                break
            end
        end
        
        # Read edges with weights
        if readline(io) == "edges_with_weights:"  # Skip this line
            while !eof(io)
                line = readline(io)
                if isempty(line)
                    continue
                end
                parts = split(line)
                if length(parts) >= 3
                    u = parse(Int, parts[1])
                    v = parse(Int, parts[2])
                    weight = parse(Float64, parts[3])
                    add_edge!(graph, u, v)
                    edge_weights[(min(u,v), max(u,v))] = weight
                end
            end
        end
    end
    
    edge_weights_vec = Vector{Float32}(undef, ne(graph))
    for (idx, e) in enumerate(edges(graph))
        u, v = src(e), dst(e)
        key = (min(u, v), max(u, v))
        edge_weights_vec[idx] = Float32(edge_weights[key])
    end
    h = Float32.(ones(Float32, nv(graph)) * 0.5)
    
    p = SpinGlassProblem(graph, edge_weights_vec, h)
    
    # Extract model name
    model_name = basename(model_file)
    
    println("\nProcessing model: $model_name")
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
    
    # Run ground state counting
    finished_slices = slice_dfs_lp(p, slicer, code, true, 1)

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
            model_name = [model_name],
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
        println("  Results saved to CSV for model: $model_name")
    end
end

println("\nAll results saved to: $results_file")