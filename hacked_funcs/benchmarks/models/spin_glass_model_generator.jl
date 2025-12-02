using Graphs, Random

# Create output directory
output_dir = "spin_glass_models"
!isdir(output_dir) && mkdir(output_dir)

# Set random seed
Random.seed!(1)

# Graph sizes to generate
ns = 55:5:55

for n in ns
    for seed in 1:10
        Random.seed!(seed)
    # Create n×n 2D grid graph
        graph = Graphs.grid([n, n])
        
        # Assign random +1 or -1 weights to each edge
        edge_weights = Dict()
        for e in edges(graph)
            edge_weights[(src(e), dst(e))] = rand([-1, 1])
        end
        
        # Create filename with J type
        filename = "spin_glass_J±1_grid_n=$(n)_seed=$(seed).model"
        filepath = joinpath(output_dir, filename)
        
        # Save model to file
        open(filepath, "w") do io
            # Write model metadata
            println(io, "# 2D Grid Spin Glass Model with ±1 couplings")
            println(io, "n = $n")
            println(io, "seed = $seed")
            println(io, "J_type = ±1")
            println(io, "vertices: ", nv(graph))
            println(io, "edges: ", ne(graph))
            println(io)
            
            # Write vertex coordinates (optional, for visualization)
            println(io, "vertex_coordinates:")
            for i in 1:n, j in 1:n
                vertex_id = (i-1)*n + j
                println(io, "$vertex_id: $i, $j")
            end
            println(io)
            
            # Write edges with weights
            println(io, "edges_with_weights:")
            for e in edges(graph)
                s, d = src(e), dst(e)
                weight = edge_weights[(min(s,d), max(s,d))]  # Ensure consistent undirected edge representation
                println(io, "$s $d $weight")
            end
        end
        
        println("Generated: $filepath")
    end
end

println("\nAll models generated and saved to $output_dir directory")