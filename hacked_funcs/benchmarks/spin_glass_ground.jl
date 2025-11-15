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
using CSV, DataFrames
using OrderedCollections
include("../src/spin_glass_ground.jl")



seed = 12345
Random.seed!(seed)
# g = random_regular_graph(200, 3)
# g = SimpleGraph(GenericTensorNetworks.random_diagonal_coupled_graph(30, 30, 0.8))
# g = SimpleGraph(GenericTensorNetworks.random_diagonal_coupled_graph(60, 60, 0.8))



lattice_rows = 70
lattice_cols = 70
# g = Graphs.grid([lattice_rows, lattice_cols])
# g = SimpleGraph(GenericTensorNetworks.random_diagonal_coupled_graph(lattice_rows, lattice_cols, 0.8))
g = random_regular_graph(600, 3)
# J = ones(Float32, ne(g)) # Edge couplings
# h = zeros(Float32, nv(g)) # Vertex fields
# J = randn(Float64, ne(g))  # Standard normal distribution (mean=0, std=1)
# J = 2.0 * rand(Bool, ne(g)) .- 1.0
# h = randn(Float64, nv(g))
J = randn(Float64, ne(g))
h = ones(Float64, nv(g))
p = SpinGlassProblem(g, J, h)

filename = "rrg_n600_d3_randn_dfs_lp.csv"
# filename = "rksg_n30filling80_rand_dfs.csv"
# filename = "lattice_$(lattice_rows)x$(lattice_cols)_zf1_dfs_lp.csv"
# filename = "rksg_n$(lattice_rows)filling80_randn_dfs_lp.csv"

println("\nGraph info:")
println("  Vertices: ", nv(g))
println("  Edges: ", ne(g))


# Create a contraction code for the spin glass problem using initialize_code
seed = 1
Random.seed!(seed)
code = initialize_code(g, TreeSA())
cc = contraction_complexity(code, uniformsize(code, 2))
println("Code complexity: ", cc)


for sc_target in [31]
    println("=" ^ 60)
    println("sc_target: ", sc_target)
    cc = contraction_complexity(code, uniformsize(code, 2))
    println("Code complexity: ", cc)
    println("=" ^ 60)
    slicer = ContractionTreeSlicer(
    sc_target = sc_target,
    table_solver = TensorNetworkSolver(),
    region_selector = ScoreRS(n_max=10),
    brancher = GreedyBrancher()
    )

    finished_slices = slice_dfs_lp(p, slicer, code, true, 1)
   

    edge_ixs = [[minmax(e.src,e.dst)...] for e in Graphs.edges(g)]
    vertex_ixs = [[v] for v in 1:nv(g)]
    ixs = vcat(edge_ixs, vertex_ixs)
    code0 = OMEinsumContractionOrders.EinCode([ixs...], Int[])
    seed = 1
    Random.seed!(seed)
    optcode_sliced = optimize_code(code0, uniformsize(code0, 2), TreeSA(), slicer=TreeSASlicer(score=ScoreFunction(sc_target=sc_target)))
    total_tc_slicing, sc_slicing = OMEinsum.timespace_complexity(optcode_sliced, uniformsize(code0, 2))
    slice_num_slicing = length(optcode_sliced.slicing)
    println("total_tc_slicing: ", total_tc_slicing)
    println("sc_slicing: ", sc_slicing)
    println("nslices: ", slice_num_slicing)
    
    global total_tc = 0.0
    if !isempty(finished_slices)
        for (i, slice) in enumerate(finished_slices[1:length(finished_slices)])
            global total_tc
            cc = complexity(slice)
            total_tc = log2(2^total_tc + 2^(cc.tc) - 1)
        end
        println("Total tc: ", total_tc)
        println("slice num: ", length(finished_slices))
        
        # Save results to CSV file
        results_dir = "./results/spin_glass_ground"
        mkpath(results_dir)  # Create directory if it doesn't exist
        
        results_file = joinpath(results_dir, filename)
        
        # Prepare data row with OrderedDict to preserve column order
        row_data = OrderedDict(
            "sc_target" => sc_target,
            "total_tc_slicing" => total_tc_slicing,
            "slice_num_slicing" => 2.0^slice_num_slicing,
            "total_tc" => total_tc,
            "slice_num" => length(finished_slices)
        )
        
        # Check if file exists to determine if we need to write header
        file_exists = isfile(results_file)
        
        # Create DataFrame from the row (OrderedDict preserves order)
        df = DataFrame([row_data])
        
        # Append to CSV file
        CSV.write(results_file, df, append=file_exists, writeheader=!file_exists)
        
        println("\nResults saved to: ", results_file)
    end

end

