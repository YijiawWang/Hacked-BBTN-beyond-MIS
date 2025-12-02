"""
    mutable struct MISProblem{INT <: Integer, VT<:AbstractVector} <: AbstractProblem

Represents a Maximum (Weighted) Independent Set (M(W)IS) problem.

# Fields
- `g::SimpleGraph`: The graph associated with the M(W)IS problem.
- `weights::VT`: The weights of the vertices in the graph. It's set to be UnitWeight if the graph is not weighted.

# Methods
- `copy(p::MISProblem)`: Creates a copy of the given `MISProblem`.
- `Base.show(io::IO, p::MISProblem)`: Displays the number of vertices in the `MISProblem`.
"""
mutable struct MISProblem{INT <: Integer, VT<:AbstractVector} <: AbstractProblem
    g::SimpleGraph{Int}
    weights::VT
    function MISProblem(g::SimpleGraph{Int}, weights::VT) where VT
        new{BitBasis.longinttype(nv(g), 2), VT}(g, weights)
    end
    function MISProblem(g::SimpleGraph{Int})
        new{BitBasis.longinttype(nv(g), 2), UnitWeight}(g, UnitWeight(nv(g)))
    end
end
Base.copy(p::MISProblem) = MISProblem(copy(p.g), copy(p.weights))
Base.show(io::IO, p::MISProblem) = print(io, "MISProblem($(nv(p.g)))")
OptimalBranchingCore.has_zero_size(p::MISProblem) = nv(p.g) == 0
Base.:(==)(p1::MISProblem{T1}, p2::MISProblem{T2}) where {T1, T2} = 
    (T1 == T2) && (p1.g == p2.g) && (p1.weights == p2.weights)

"""
    TensorNetworkSolver
    TensorNetworkSolver(; prune_by_env::Bool = true)

A struct representing a solver for tensor network problems. 
This struct serves as a specific implementation of the `AbstractTableSolver` type.
"""
@kwdef struct TensorNetworkSolver <: AbstractTableSolver
    prune_by_env::Bool = true
end

"""
    NumOfVertices

A struct representing a measure that counts the number of vertices in a graph. 
Each vertex is counted as 1.

# Fields
- None
"""
struct NumOfVertices <: AbstractMeasure end

"""
    measure(p::MISProblem, ::NumOfVertices)

Calculates the number of vertices in the given `MISProblem`.

# Arguments
- `p::MISProblem`: The problem instance containing the graph.

# Returns
- `Int`: The number of vertices in the graph.
"""
OptimalBranchingCore.measure(p::MISProblem{INT}, ::NumOfVertices) where {INT} = nv(p.g)

"""
    size_reduction(p::MISProblem{INT}, ::NumOfVertices, cl::Clause, variables::Vector) where {INT}

Calculates the size reduction for the given `MISProblem` using the number of vertices after applying the clause.

# Arguments
- `p::MISProblem{INT}`: The problem instance containing the graph.
- `cl::Clause`: The clause to be applied.
- `variables::Vector`: The variables included in the clause.

# Returns
- `sum`: The size reduction value.
"""
function OptimalBranchingCore.size_reduction(p::MISProblem{INT}, ::NumOfVertices, cl::Clause, variables::Vector) where {INT}
    return count_ones(removed_mask(INT, variables, p.g, cl))
end

"""
    D3Measure

A struct representing a measure that calculates the sum of the maximum degree minus 2 for each vertex in the graph.

# Fields
- None
"""
struct D3Measure <: AbstractMeasure end

"""
    measure(p::MISProblem, ::D3Measure)

Calculates the D3 measure for the given `MISProblem`, which is defined as the sum of 
the maximum degree of each vertex minus 2, for all vertices in the graph.

# Arguments
- `p::MISProblem`: The problem instance containing the graph.

# Returns
- `Int`: The computed D3 measure value.
"""
function OptimalBranchingCore.measure(p::MISProblem{INT}, ::D3Measure) where {INT}
    g = p.g
    if nv(g) == 0
        return 0
    else
        dg = degree(g)
        return Int(sum(max(d - 2, 0) for d in dg))
    end
end

"""
    size_reduction(p::MISProblem{INT}, ::D3Measure, cl::Clause, variables::Vector) where {INT}

Calculates the size reduction for the given `MISProblem` using the D3 measure after applying the clause.

# Arguments
- `p::MISProblem{INT}`: The problem instance containing the graph.
- `cl::Clause`: The clause to be applied.
- `variables::Vector`: The variables included in the clause.

# Returns
- `sum`: The size reduction value.
"""
function OptimalBranchingCore.size_reduction(p::MISProblem{INT}, ::D3Measure, cl::Clause, variables::Vector) where {INT}
    remove_mask = removed_mask(INT, variables, p.g, cl)
    iszero(remove_mask) && return 0
    sum = 0
    for i in 1:nv(p.g)
        deg = degree(p.g, i)
        deg <= 2 && continue
        if readbit(remove_mask, i) == 1
            sum += max(deg - 2, 0)
        else
            countneighbor = count(v -> readbit(remove_mask, v) == 0, neighbors(p.g, i)) 
            sum += max(deg - 2, 0) - max(countneighbor - 2, 0)
        end
    end
    return sum
end

"""
    mutable struct SpinGlassProblem{INT <: Integer, VT<:AbstractVector} <: AbstractProblem

Represents a Spin Glass problem.

# Fields
- `g::SimpleGraph`: The graph associated with the Spin Glass problem.
- `weights::VT`: The weights/couplings of the edges in the graph.

# Methods
- `copy(p::SpinGlassProblem)`: Creates a copy of the given `SpinGlassProblem`.
- `Base.show(io::IO, p::SpinGlassProblem)`: Displays the number of vertices in the `SpinGlassProblem`.
"""
mutable struct SpinGlassProblem{INT <: Integer, VT<:AbstractVector} <: AbstractProblem
    g::SimpleGraph{Int}
    J::VT
    h::VT
    function SpinGlassProblem(g::SimpleGraph{Int}, J::VT, h::VT) where VT
        new{BitBasis.longinttype(nv(g), 2), VT}(g, J, h)
    end
    function SpinGlassProblem(g::SimpleGraph{Int})
        new{BitBasis.longinttype(nv(g), 2), UnitWeight}(g, UnitWeight(nv(g)), UnitWeight(nv(g)))
    end
end
Base.copy(p::SpinGlassProblem) = SpinGlassProblem(copy(p.g), copy(p.J), copy(p.h))
Base.show(io::IO, p::SpinGlassProblem) = print(io, "SpinGlassProblem($(nv(p.g)))")
OptimalBranchingCore.has_zero_size(p::SpinGlassProblem) = nv(p.g) == 0
Base.:(==)(p1::SpinGlassProblem{T1}, p2::SpinGlassProblem{T2}) where {T1, T2} = 
(T1 == T2) && (p1.g == p2.g) && (p1.J == p2.J) && (p1.h == p2.h)


"""
    mutable struct MaxSatProblem{INT <: Integer, VT} <: AbstractProblem

Represents a Maximum Satisfiability (Max SAT) problem.

# Fields
- `g::SimpleGraph`: The graph associated with the Max SAT problem.
  - The first n_vars vertices (1 to n_vars) correspond to variables
  - The remaining vertices (n_vars+1 to n_vars+n_clauses) correspond to clauses
  - An edge connects a variable vertex to a clause vertex if the variable appears in that clause
- `clauses::Vector{Vector{Int}}`: The list of clauses, where each clause is a vector of integers.
  - Positive integers represent positive literals (variables)
  - Negative integers represent negative literals (negated variables)

# Type Parameters
- `INT`: Integer type for bit basis operations
- `VT`: Placeholder type parameter for compatibility with SlicedBranch (not used, set to Nothing)

# Methods
- `copy(p::MaxSatProblem)`: Creates a copy of the given `MaxSatProblem`.
- `Base.show(io::IO, p::MaxSatProblem)`: Displays the number of variables and clauses in the `MaxSatProblem`.
"""
mutable struct MaxSatProblem{INT <: Integer, VT} <: AbstractProblem
    g::SimpleGraph{Int}
    clauses::Vector{Vector{Int}}
    use_constraint::Bool
    function MaxSatProblem(g::SimpleGraph{Int}, clauses::Vector{Vector{Int}}, use_constraint::Bool = false)
        new{BitBasis.longinttype(nv(g), 2), Nothing}(g, clauses, use_constraint)
    end
end
Base.copy(p::MaxSatProblem) = MaxSatProblem(copy(p.g), [copy(c) for c in p.clauses], p.use_constraint)
function Base.show(io::IO, p::MaxSatProblem)
    n_vars = nv(p.g) - length(p.clauses)
    n_clauses = length(p.clauses)
    print(io, "MaxSatProblem($n_vars vars, $n_clauses clauses, use_constraint = $(p.use_constraint))")
end
OptimalBranchingCore.has_zero_size(p::MaxSatProblem) = nv(p.g) == 0
Base.:(==)(p1::MaxSatProblem{T1, V1}, p2::MaxSatProblem{T2, V2}) where {T1, T2, V1, V2} = 
    (T1 == T2) && (V1 == V2) && (p1.g == p2.g) && (p1.clauses == p2.clauses) && (p1.use_constraint == p2.use_constraint)