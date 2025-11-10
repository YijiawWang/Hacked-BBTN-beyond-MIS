using OptimalBranching
using OptimalBranching.OptimalBranchingCore, OptimalBranching.OptimalBranchingMIS
using GenericTensorNetworks, ProblemReductions
using Graphs, TropicalNumbers, OMEinsum, OMEinsumContractionOrders, Random
using TensorBranching  # Import the full module to access internal functions
using TensorBranching: ContractionTreeSlicer, SlicedBranch, initialize_code, GreedyBrancher, ScoreRS
using OptimalBranchingMIS: TensorNetworkSolver, SpinGlassProblem
using OMEinsumContractionOrders: TreeSA
using OMEinsum: DynamicNestedEinsum
using OMEinsumContractionOrders: uniformsize
using BitBasis

# Access the internal function via getfield
const generate_branch_no_reduction_spin_glass = getfield(TensorBranching, :generate_branch_no_reduction_spin_glass)

function test_fix_vertex_branching(g, J, h, fixed_vertex::Int)
    """
    Test that fixing a vertex and creating two branches (s=-1 and s=+1)
    results in branch counts that sum to the total count.
    """
    println("=" ^ 60)
    println("Testing fix vertex branching")
    println("=" ^ 60)
    println("Graph: $(nv(g)) vertices, $(ne(g)) edges")
    println("Fixed vertex: $fixed_vertex")
    
    # Create original problem
    p = SpinGlassProblem(g, J, h)
    
    # Calculate total ground state count
    g_problem = GenericTensorNetwork(SpinGlass(g, J, h); optimizer=TreeSA())
    count_maximum_spin_glass = solve(g_problem, CountingMax())[]
    total_size = count_maximum_spin_glass.n[1]
    total_count = count_maximum_spin_glass.c[1]
    
    println("\nOriginal problem:")
    println("  Ground state size: $total_size")
    println("  Ground state count: $total_count")
    
    # Create code and size_dict
    code = initialize_code(g, TreeSA())
    size_dict = uniformsize(code, 2)
    
    # Create slicer (minimal, just for the function call)
    slicer = ContractionTreeSlicer(
        sc_target = 100,  # Large enough to not refine
        table_solver = TensorNetworkSolver(),
        region_selector = ScoreRS(n_max=10),
        brancher = GreedyBrancher()
    )
    
    # Region is just the fixed vertex
    region = [fixed_vertex]
    
    # Determine INT type
    INT = BitBasis.longinttype(length(region), 2)
    
    # Create two branches: s=-1 (bit=1) and s=+1 (bit=0)
    removed_vertices_list = [fixed_vertex]
    
    # Branch 1: s = -1 (bit = 1)
    val1 = INT(1)  # First bit is 1, meaning s = -1
    branch1 = generate_branch_no_reduction_spin_glass(p, code, removed_vertices_list, val1, slicer, size_dict, region)
    
    # Branch 2: s = +1 (bit = 0)
    val2 = INT(0)  # First bit is 0, meaning s = +1
    branch2 = generate_branch_no_reduction_spin_glass(p, code, removed_vertices_list, val2, slicer, size_dict, region)
    
    println("\nBranch 1 (s=$fixed_vertex = -1):")
    println("  Subgraph: $(nv(branch1.p.g)) vertices, $(ne(branch1.p.g)) edges")
    println("  Fixed energy contribution r: $(branch1.r)")
    
    println("\nBranch 2 (s=$fixed_vertex = +1):")
    println("  Subgraph: $(nv(branch2.p.g)) vertices, $(ne(branch2.p.g)) edges")
    println("  Fixed energy contribution r: $(branch2.r)")
    
    # Calculate ground state counts for each branch
    branch1_count = 0.0
    branch2_count = 0.0
    
    if nv(branch1.p.g) > 0
        branch1_problem = GenericTensorNetwork(SpinGlass(branch1.p.g, branch1.p.J, branch1.p.h); optimizer=TreeSA())
        branch1_result = solve(branch1_problem, CountingMax())[]
        branch1_size = branch1_result.n[1] + branch1.r
        branch1_count = branch1_result.c[1]
        println("\nBranch 1 results:")
        println("  Ground state size: $branch1_size (subgraph: $(branch1_result.n[1]), fixed: $(branch1.r))")
        println("  Ground state count: $branch1_count")
    else
        # Empty subgraph, check if r matches total_size
        println("\nBranch 1: Empty subgraph, fixed energy: $(branch1.r)")
        if abs(branch1.r - total_size) < 1e-12
            branch1_count = 1.0  # Only one configuration (the fixed one)
        end
    end
    
    if nv(branch2.p.g) > 0
        branch2_problem = GenericTensorNetwork(SpinGlass(branch2.p.g, branch2.p.J, branch2.p.h); optimizer=TreeSA())
        branch2_result = solve(branch2_problem, CountingMax())[]
        branch2_size = branch2_result.n[1] + branch2.r
        branch2_count = branch2_result.c[1]
        println("\nBranch 2 results:")
        println("  Ground state size: $branch2_size (subgraph: $(branch2_result.n[1]), fixed: $(branch2.r))")
        println("  Ground state count: $branch2_count")
    else
        # Empty subgraph, check if r matches total_size
        println("\nBranch 2: Empty subgraph, fixed energy: $(branch2.r)")
        if abs(branch2.r - total_size) < 1e-12
            branch2_count = 1.0  # Only one configuration (the fixed one)
        end
    end
    
    # Verify counts sum to total
    sum_count = branch1_count + branch2_count
    println("\n" * "=" ^ 60)
    println("Verification:")
    println("  Total count: $total_count")
    println("  Branch 1 count: $branch1_count")
    println("  Branch 2 count: $branch2_count")
    println("  Sum: $sum_count")
    
    if abs(sum_count - total_count) < 1e-6
        println("  ✓ PASS: Sum equals total count!")
        return true
    else
        println("  ✗ FAIL: Sum does not equal total count!")
        println("  Difference: $(abs(sum_count - total_count))")
        return false
    end
end

function test_fix_three_adjacent_vertices(g, J, h, region::Vector{Int})
    """
    Test that fixing 3 adjacent vertices and creating all 8 branches (2^3 = 8)
    results in branch counts that sum to the total count.
    """
    println("=" ^ 60)
    println("Testing fix 3 adjacent vertices branching")
    println("=" ^ 60)
    println("Graph: $(nv(g)) vertices, $(ne(g)) edges")
    println("Region (3 adjacent vertices): $region")
    
    if length(region) != 3
        error("Region must contain exactly 3 vertices")
    end
    
    # Create original problem
    p = SpinGlassProblem(g, J, h)
    
    # Calculate total ground state count
    g_problem = GenericTensorNetwork(SpinGlass(g, J, h); optimizer=TreeSA())
    count_maximum_spin_glass = solve(g_problem, CountingMax())[]
    total_size = count_maximum_spin_glass.n[1]
    total_count = count_maximum_spin_glass.c[1]
    
    println("\nOriginal problem:")
    println("  Ground state size: $total_size")
    println("  Ground state count: $total_count")
    
    # Create code and size_dict
    code = initialize_code(g, TreeSA())
    size_dict = uniformsize(code, 2)
    
    # Create slicer (minimal, just for the function call)
    slicer = ContractionTreeSlicer(
        sc_target = 100,  # Large enough to not refine
        table_solver = TensorNetworkSolver(),
        region_selector = ScoreRS(n_max=10),
        brancher = GreedyBrancher()
    )
    
    # Determine INT type
    INT = BitBasis.longinttype(length(region), 2)
    
    # Generate all 8 possible configurations (2^3 = 8)
    # Each configuration is represented by a bitstring where bit i represents vertex region[i]
    # bit = 1 means s = -1, bit = 0 means s = +1
    branches = []
    branch_counts = Float64[]
    branch_sizes = Float64[]
    
    println("\nGenerating all 8 branches...")
    for config in 0:7  # 0 to 7 = 000 to 111 in binary
        val = INT(config)
        
        # Create branch
        branch = generate_branch_no_reduction_spin_glass(p, code, region, val, slicer, size_dict, region)
        push!(branches, branch)
        
        # Calculate ground state count for this branch
        if nv(branch.p.g) > 0
            branch_problem = GenericTensorNetwork(SpinGlass(branch.p.g, branch.p.J, branch.p.h); optimizer=TreeSA())
            branch_result = solve(branch_problem, CountingMax())[]
            branch_size = branch_result.n[1] + branch.r
            branch_count = branch_result.c[1]
            push!(branch_sizes, branch_size)
            push!(branch_counts, branch_count)
            
            # Print configuration
            config_str = join([readbit(val, i) == 1 ? "-1" : "+1" for i in 1:3], ", ")
            println("  Config $config [$(config_str)]: size=$branch_size, count=$branch_count, r=$(branch.r)")
        else
            # Empty subgraph
            push!(branch_sizes, branch.r)
            if abs(branch.r - total_size) < 1e-12
                push!(branch_counts, 1.0)
            else
                push!(branch_counts, 0.0)
            end
            config_str = join([readbit(val, i) == 1 ? "-1" : "+1" for i in 1:3], ", ")
            println("  Config $config [$(config_str)]: empty subgraph, r=$(branch.r)")
        end
    end
    
    # Verify counts sum to total
    sum_count = sum(branch_counts)
    
    # Check that all branches with maximum size contribute
    max_branch_size = maximum(branch_sizes)
    valid_branches = [i for i in 1:8 if abs(branch_sizes[i] - max_branch_size) < 1e-12]
    sum_count_max_size = sum([branch_counts[i] for i in valid_branches])
    
    println("\n" * "=" ^ 60)
    println("Verification:")
    println("  Total count: $total_count")
    println("  Total size: $total_size")
    println("  Max branch size: $max_branch_size")
    println("  Sum of all branch counts: $sum_count")
    println("  Sum of max-size branch counts: $sum_count_max_size")
    println("  Valid branches (max size): $valid_branches")
    
    # Check if total_size matches max_branch_size
    size_match = abs(total_size - max_branch_size) < 1e-12
    
    if size_match && abs(sum_count_max_size - total_count) < 1e-6
        println("  ✓ PASS: Sum of max-size branch counts equals total count!")
        return true
    else
        println("  ✗ FAIL:")
        if !size_match
            println("    Size mismatch: total_size=$total_size, max_branch_size=$max_branch_size")
        end
        if abs(sum_count_max_size - total_count) >= 1e-6
            println("    Count mismatch: expected $total_count, got $sum_count_max_size")
            println("    Difference: $(abs(sum_count_max_size - total_count))")
        end
        return false
    end
end

# Run tests
if abspath(PROGRAM_FILE) == @__FILE__
    Random.seed!(12345)
    
    println("\n" * "=" ^ 80)
    println("Test 1: Small random regular graph")
    println("=" ^ 80)
    g1 = random_regular_graph(10, 3)
    J1 = ones(Float32, ne(g1))
    h1 = zeros(Float32, nv(g1))
    test_fix_vertex_branching(g1, J1, h1, 1)
    
    println("\n" * "=" ^ 80)
    println("Test 2: Small random regular graph with random J and h")
    println("=" ^ 80)
    g2 = random_regular_graph(10, 3)
    J2 = rand(Float64, ne(g2)) .- 0.5  # Random values between -0.5 and 0.5
    h2 = rand(Float64, nv(g2)) .- 0.5
    test_fix_vertex_branching(g2, J2, h2, 5)
    
    println("\n" * "=" ^ 80)
    println("Test 3: Larger graph")
    println("=" ^ 80)
    g3 = random_regular_graph(20, 3)
    J3 = ones(Float32, ne(g3))
    h3 = zeros(Float32, nv(g3))
    test_fix_vertex_branching(g3, J3, h3, 1)
    
    println("\n" * "=" ^ 80)
    println("Test 4: Fix 3 adjacent vertices (exhaustive 8 configurations)")
    println("=" ^ 80)
    g4 = random_regular_graph(16, 3)
    J4 = ones(Float32, ne(g4))
    h4 = zeros(Float32, nv(g4))
    # Find 3 adjacent vertices: pick vertex 1 and its first 2 neighbors
    v1 = 1
    neighbors_v1 = collect(neighbors(g4, v1))
    if length(neighbors_v1) >= 2
        region4 = [v1, neighbors_v1[1], neighbors_v1[2]]
        test_fix_three_adjacent_vertices(g4, J4, h4, region4)
    else
        println("  Skipped: vertex 1 doesn't have enough neighbors")
    end
    
    println("\n" * "=" ^ 80)
    println("Test 5: Fix 3 adjacent vertices with random J and h")
    println("=" ^ 80)
    g5 = random_regular_graph(16, 3)
    J5 = rand(Float64, ne(g5)) .- 0.5
    h5 = rand(Float64, nv(g5)) .- 0.5
    v1 = 1
    neighbors_v1 = collect(neighbors(g5, v1))
    if length(neighbors_v1) >= 2
        region5 = [v1, neighbors_v1[1], neighbors_v1[2]]
        test_fix_three_adjacent_vertices(g5, J5, h5, region5)
    else
        println("  Skipped: vertex 1 doesn't have enough neighbors")
    end
    
    println("\n" * "=" ^ 80)
    println("All tests completed!")
    println("=" ^ 80)
end

