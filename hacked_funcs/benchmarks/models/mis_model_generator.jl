using Graphs, GenericTensorNetworks, Random

# Create output directory
output_dir = "mis_graphs"
!isdir(output_dir) && mkdir(output_dir)

# Set random seed
Random.seed!(1)

# Graph sizes to generate
ns = 64:10:64

for n in ns
    for seed in 1:10
        Random.seed!(seed)
        # Generate random KSG graph
        graph = GenericTensorNetworks.random_diagonal_coupled_graph(n, n, 0.8)
        graph = SimpleGraph(graph)  # Convert to simple graph
        
        # Create filename: includes n, random seed, and graph type
        filename = "random_ksg_n=$(n)_seed=$(seed).graph"
        filepath = joinpath(output_dir, filename)
        
        # Save graph to file
        open(filepath, "w") do io
            # Write basic graph information
            println(io, "# Random KSG graph")
            println(io, "n = $n")
            println(io, "seed = $seed")
            println(io, "type = random_ksg")
            println(io)
            
            # Write number of vertices
            println(io, "vertices: ", nv(graph))
            
            # Write edge list
            println(io, "edges:")
            for e in edges(graph)
                println(io, e.src, " ", e.dst)
            end
        end
        
        println("Generated: $filepath")
    end
end

println("\nAll graphs generated and saved to $output_dir directory")