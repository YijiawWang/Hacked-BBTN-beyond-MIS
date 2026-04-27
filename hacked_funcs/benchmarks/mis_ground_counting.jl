using OptimalBranching
using OptimalBranching.OptimalBranchingCore, OptimalBranching.OptimalBranchingMIS
using GenericTensorNetworks, ProblemReductions
using Graphs, TropicalNumbers, OMEinsum, OMEinsumContractionOrders, Random
using Base.Threads
using TensorBranching: ob_region, optimal_branches, ContractionTreeSlicer, uncompress, SlicedBranch, complexity
using TensorBranching: optimal_branches_ground_counting, optimal_branches_ground_counting_induced_sparsity, initialize_code, GreedyBrancher, ScoreRS
using OptimalBranchingMIS: TensorNetworkSolver
using OMEinsumContractionOrders: TreeSA
using OMEinsum: DynamicNestedEinsum
using OMEinsumContractionOrders: uniformsize
using CSV, DataFrames
using OrderedCollections
include("../src/mis_ground_counting.jl")
include(joinpath(@__DIR__, "..", "..", "contractors", "mis_slice_contract.jl"))

# Wrapped in `main()` so that this file can be `include`d by other
# drivers (e.g. `beyond_mis/scripts/_lib/slice_mis.jl`) without
# triggering the benchmark loop. The guard below preserves the
# `julia mis_ground_counting.jl` CLI behaviour.
function main()
# Set random seed
Random.seed!(12345)

# Input and output directories
input_dir = "models/mis_graphs/bigksg"
output_dir = "results/mis_ground_counting"
mkpath(output_dir)  # Create output directory
mkpath(SLICE_RESULTS_ROOT)   # central slice dump (beyond_mis/branch_results)
mis_ground_counting_graph_type = basename(rstrip(input_dir, '/'))

# Get all graph files
graph_files = filter(x -> endswith(x, ".graph"), readdir(input_dir, join=true))

# Results file
results_file = joinpath(output_dir, "mis_ground_counting_results.csv")

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
    # weights = randn(Float32, nv(graph))
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
    
    # Stream-persist every finished slice (graph + weights + r + saved
    # contraction code) into its own directory under
    # `beyond_mis/branch_results/` as soon as it is produced, so a
    # crashed / interrupted run still leaves usable on-disk slices.
    slice_subdir = "mis_ground_counting_$(replace(graph_name, ".graph" => ""))_seed=$(seed)"
    slice_writer = init_mis_slice_writer(slice_subdir;
        original   = (graph, weights),
        graph_name = graph_name,
        graph_type = mis_ground_counting_graph_type,
        overwrite  = true,
        meta = Dict(
            "graph"      => graph_name,
            "seed"       => seed,
            "sc_target"  => sc_target,
            "vertices"   => nv(graph),
            "edges"      => ne(graph),
        ))
    println("  Streaming slices to $(slice_writer.dirname)")

    # Run ground state counting
    finished_slices = slice_dfs_lp(p, slicer, code, true, 1;
        on_finished_slice = slice -> begin
            sid = append_mis_slice!(slice_writer, slice; flush_summary = true)
            cc_s = complexity(slice)
            println("  [slice $sid saved] sc=$(cc_s.sc) tc=$(cc_s.tc) " *
                    "nv=$(nv(slice.p.g)) ne=$(ne(slice.p.g)) r=$(slice.r) " *
                    "(total saved: $(length(slice_writer.ids)))")
            flush(stdout)
        end)
    slice_dir = finalize_mis_slice_writer!(slice_writer)
    println("  Saved $(length(finished_slices)) slice(s) to $slice_dir")

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
        
        # Build a row of data
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
        
        # Append immediately to the CSV file
        CSV.write(results_file, row, append=true)
        println("  Results saved to CSV for graph: $graph_name")
    end
end

println("\nAll results saved to: $results_file")
end  # function main

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end