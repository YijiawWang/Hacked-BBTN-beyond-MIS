using Graphs, GenericTensorNetworks, Random

# Create output directory
output_dir = "mis_graphs/fact_mis_graphs"
!isdir(output_dir) && mkdir(output_dir)

seed = 1
Random.seed!(seed)
k1 = 14
k2 = 14
k10 = 14
k20 = 14
n10 = rand(2^(k10-1):2^k10)
n20 = rand(2^(k20-1):2^k20)
n1 = prevprime(n10)
n2 = prevprime(n20)

# n1 = prevprime(2^k10)
# n2 = prevprime(2^k20,2)
println("n1: $n1, n2: $n2")

n = Int(n1*n2)
mg, mw, vmap = map_factoring(k1, k2, n)
graph = copy(mg)
graph = SimpleGraph(graph)
weights = copy(mw)
weights = [Float64(w) for w in weights]