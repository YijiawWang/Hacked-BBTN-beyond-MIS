"""
    spin_glass_slice_contract.jl

I/O + exact contraction utilities for spin-glass slices produced by
`slice_dfs_lp` (see `beyond_mis/hacked_funcs/src/spin_glass_ground_counting.jl`).

The slicer turns one large spin glass into many smaller `SlicedBranch`
objects, each carrying:

  * a sub-graph `g_i` (induced subgraph after branching/reductions),
  * its couplings `J_i` and local fields `h_i`,
  * a contraction code `code_i` that fits within the requested space
    complexity budget (`sc_target`),
  * an additive constant `r_i` (energy already accounted for by the
    fixed spins outside the sub-graph).

Each slice contributes one ground-state energy / degeneracy pair via
`(E_i + r_i, c_i)` where `(E_i, c_i) = solve(... CountingMax())`. The
overall ground-state energy is `E_g = max_i (E_i + r_i)` and the total
ground-state degeneracy is `sum_{i: E_i + r_i ≈ E_g} c_i`.

Save layout (one directory per benchmark instance):

    <dir>/
        slices.csv                # id, r, sc, tc, nv, ne, has_code
        meta.txt                  # free-form model info (optional)
        graph_<id>.lg             # SimpleGraph (Graphs.savegraph LG format)
        eincode_<id>.json         # OMEinsum DynamicNestedEinsum (writejson)
        J_<id>.txt                # 1st line eltype tag, then ne(g) entries
        h_<id>.txt                # 1st line eltype tag, then nv(g) entries
        r_<id>.txt                # 1st line eltype tag, then a single value

Trivial slices with `code === nothing` (i.e. the empty graph after all
spins were fixed) skip `eincode_<id>.json`; their contribution is taken
to be `(r_i, 1)`.
"""

using Graphs
using OMEinsum                                # writejson, readjson
using OMEinsumContractionOrders: TreeSA, uniformsize
using GenericTensorNetworks
import GenericTensorNetworks: contractx, _x   # bypass the `solve` API for
                                              # the finitefield / CRT path,
                                              # which needs to control the
                                              # element type explicitly.
import GenericTensorNetworks.Mods: Mod, value  # vendored modular arithmetic
using Primes                                   # prevprime
using ProblemReductions: SpinGlass
using CSV, DataFrames
using TensorBranching: uncompress, complexity

# ----------------------------------------------------------------------
# Central slice-results root
# ----------------------------------------------------------------------

"""
    SLICE_RESULTS_ROOT

Default top-level directory in which every slice run is dumped:
`beyond_mis/branch_results/`. Each instance lives in its own subdir and
a single `summary.csv` at the root indexes them.
"""
const SLICE_RESULTS_ROOT =
    abspath(joinpath(@__DIR__, "..", "branch_results"))

"""
    slice_results_dir(subdir; root=SLICE_RESULTS_ROOT) -> String

Compose the absolute path of an individual slice directory inside
`SLICE_RESULTS_ROOT`. If `subdir` is already absolute or already exists
as-given, it is returned unchanged.
"""
function slice_results_dir(subdir::AbstractString; root::AbstractString = SLICE_RESULTS_ROOT)
    isabspath(subdir) && return String(subdir)
    isdir(subdir)     && return String(abspath(subdir))
    return joinpath(root, String(subdir))
end

_summary_path(root::AbstractString = SLICE_RESULTS_ROOT) = joinpath(root, "summary.csv")

# ----------------------------------------------------------------------
# Low-level vector / scalar (de)serialisation
# ----------------------------------------------------------------------

const _NUMERIC_TYPE_TAGS = Dict{String, DataType}(
    "Float64" => Float64,
    "Float32" => Float32,
    "Float16" => Float16,
    "Int64"   => Int64,
    "Int32"   => Int32,
    "Int16"   => Int16,
    "Bool"    => Bool,
)

function _save_numeric_vector(filename::String, v::AbstractVector)
    T = eltype(v)
    haskey(_NUMERIC_TYPE_TAGS, string(T)) || error("unsupported numeric eltype $(T)")
    open(filename, "w") do io
        println(io, string(T))
        for x in v
            println(io, string(x))
        end
    end
    return nothing
end

function _load_numeric_vector(filename::String)
    lines = readlines(filename)
    isempty(lines) && error("empty vector file: $filename")
    tag = strip(lines[1])
    haskey(_NUMERIC_TYPE_TAGS, tag) || error("unknown numeric type tag '$tag' in $filename")
    T = _NUMERIC_TYPE_TAGS[tag]
    n = length(lines) - 1
    v = Vector{T}(undef, n)
    @inbounds for i in 1:n
        v[i] = parse(T, strip(lines[i + 1]))
    end
    return v
end

function _save_numeric_scalar(filename::String, x::Number)
    T = typeof(x)
    haskey(_NUMERIC_TYPE_TAGS, string(T)) || error("unsupported numeric type $(T) for scalar")
    open(filename, "w") do io
        println(io, string(T))
        println(io, string(x))
    end
    return nothing
end

function _load_numeric_scalar(filename::String)
    lines = readlines(filename)
    length(lines) >= 2 || error("scalar file too short: $filename")
    tag = strip(lines[1])
    haskey(_NUMERIC_TYPE_TAGS, tag) || error("unknown numeric type tag '$tag' in $filename")
    T = _NUMERIC_TYPE_TAGS[tag]
    return parse(T, strip(lines[2]))
end

# ----------------------------------------------------------------------
# Slice paths
# ----------------------------------------------------------------------

_graph_path(dir, id) = joinpath(dir, "graph_$(id).lg")
_code_path(dir, id)  = joinpath(dir, "eincode_$(id).json")
_J_path(dir, id)     = joinpath(dir, "J_$(id).txt")
_h_path(dir, id)     = joinpath(dir, "h_$(id).txt")
_r_path(dir, id)     = joinpath(dir, "r_$(id).txt")
_csv_path(dir)       = joinpath(dir, "slices.csv")
_meta_path(dir)      = joinpath(dir, "meta.txt")

# Original (pre-slicing) problem files. Stored alongside the slices so a
# verifier can run a strict GTN contraction without having to rebuild
# the input from scratch.
_orig_graph_path(dir) = joinpath(dir, "original_graph.lg")
_orig_J_path(dir)     = joinpath(dir, "original_J.txt")
_orig_h_path(dir)     = joinpath(dir, "original_h.txt")

# ----------------------------------------------------------------------
# Save
# ----------------------------------------------------------------------

"""
    save_spin_glass_slices(subdir, slices;
                            root = SLICE_RESULTS_ROOT,
                            original = nothing,
                            model_name = "",
                            graph_type = "",
                            overwrite = false,
                            meta = Dict(),
                            update_summary = true) -> String

Persist a vector of `SlicedBranch` (spin glass) objects produced by
`slice_dfs_lp` to `joinpath(root, subdir)` so they can later be
re-contracted exactly via [`contract_spin_glass_slices`](@ref).

`slices` must be iterable of objects with the fields `.p.g`, `.p.J`,
`.p.h`, `.r`, `.code` (i.e. `SlicedBranch{INT,VT,RT,SpinGlassProblem{INT,VT}}`).

If `original = (g, J, h)` is supplied, the **pre-slicing** spin glass
is also dumped (`original_graph.lg`, `original_J.txt`,
`original_h.txt`); this lets external verifiers reproduce the strict
GTN contraction without having to rebuild the input.

The convention is to encode the graph type, size and seed into `subdir`
(e.g. `rrg_n=600_d=3_seed=1`); pass an absolute path if you really want
to write somewhere outside `SLICE_RESULTS_ROOT`.

If `overwrite=true` the directory is cleared first.

If `update_summary=true` (default) one row is appended to / updated in
`<root>/summary.csv` describing this run (graph_type, vertices, edges,
sc_target, slice_num, total_tc, ...).

Returns the absolute path of the per-instance directory.
"""
function save_spin_glass_slices(subdir::AbstractString, slices;
                                 root::AbstractString = SLICE_RESULTS_ROOT,
                                 original = nothing,
                                 model_name::AbstractString = "",
                                 graph_type::AbstractString = "",
                                 overwrite::Bool = false,
                                 meta::AbstractDict = Dict{String,Any}(),
                                 update_summary::Bool = true)
    dirname = slice_results_dir(subdir; root = root)

    if overwrite && isdir(dirname)
        rm(dirname; recursive = true)
    end
    mkpath(dirname)

    if original !== nothing
        og, oJ, oh = original
        Graphs.savegraph(_orig_graph_path(dirname), og)
        _save_numeric_vector(_orig_J_path(dirname), Vector(oJ))
        _save_numeric_vector(_orig_h_path(dirname), Vector(oh))
    end

    ids   = Int[]
    rs    = Float64[]
    scs   = Float64[]
    tcs   = Float64[]
    nvs   = Int[]
    nes   = Int[]
    has_c = Bool[]

    for (id, slice) in enumerate(slices)
        _save_one_slice(dirname, slice, id)
        cc = complexity(slice)
        push!(ids, id)
        push!(rs, Float64(slice.r))
        push!(scs, Float64(cc.sc))
        push!(tcs, Float64(cc.tc))
        push!(nvs, nv(slice.p.g))
        push!(nes, ne(slice.p.g))
        push!(has_c, !isnothing(slice.code))
    end

    df = DataFrame(id = ids, r = rs, sc = scs, tc = tcs,
                   nv = nvs, ne = nes, has_code = has_c)
    CSV.write(_csv_path(dirname), df)

    if !isempty(model_name) || !isempty(graph_type) || !isempty(meta)
        open(_meta_path(dirname), "w") do io
            isempty(model_name) || println(io, "model_name = ", model_name)
            isempty(graph_type) || println(io, "graph_type = ", graph_type)
            for (k, v) in meta
                println(io, k, " = ", v)
            end
        end
    end

    if update_summary
        info = Dict{Symbol,Any}(
            :subdir       => relpath(dirname, root),
            :model_name   => model_name,
            :graph_type   => graph_type,
            :slice_num    => length(ids),
            :max_sc       => isempty(scs) ? 0.0 : maximum(scs),
            :total_tc     => _logsumexp_tc(tcs),
        )
        for (k, v) in meta
            sym = Symbol(string(k))
            haskey(info, sym) || (info[sym] = v)
        end
        update_slice_summary(info; root = root)
    end

    return dirname
end

# log2(sum_i 2^tc_i), the natural way to combine per-slice tc.
function _logsumexp_tc(tcs::AbstractVector{<:Real})
    isempty(tcs) && return -Inf
    m = maximum(tcs)
    return m + log2(sum(2.0 .^ (tcs .- m)))
end

"""
    update_slice_summary(info::AbstractDict; root = SLICE_RESULTS_ROOT, key = :subdir)

Insert (or replace, when `info[key]` is already present) one row into
`<root>/summary.csv`. Missing columns are added on the fly so different
benchmark families can share the same summary file.
"""
function update_slice_summary(info::AbstractDict; root::AbstractString = SLICE_RESULTS_ROOT,
                               key::Symbol = :subdir)
    mkpath(root)
    path = _summary_path(root)

    info_sym = Dict{Symbol,Any}(Symbol(string(k)) => v for (k, v) in info)
    haskey(info_sym, key) ||
        error("update_slice_summary: info dict is missing key $(key)")
    new_row = DataFrame([info_sym])

    df = if isfile(path)
        try
            CSV.read(path, DataFrame)
        catch err
            @warn "summary.csv at $path is unreadable, starting fresh ($(err))"
            DataFrame()
        end
    else
        DataFrame()
    end

    if isempty(df)
        df = new_row
    else
        # Drop any existing row with the same `key` value, then append.
        if hasproperty(df, key)
            df = filter(row -> row[key] != info_sym[key], df)
        end
        df = vcat(df, new_row; cols = :union)
    end

    CSV.write(path, df)
    return path
end

function _save_one_slice(dirname, slice, id::Int)
    g = slice.p.g
    Graphs.savegraph(_graph_path(dirname, id), g)
    _save_numeric_vector(_J_path(dirname, id), Vector(slice.p.J))
    _save_numeric_vector(_h_path(dirname, id), Vector(slice.p.h))
    _save_numeric_scalar(_r_path(dirname, id), slice.r)
    if !isnothing(slice.code)
        # `slice.code` is a `CompressedEinsum`; uncompress to a
        # DynamicNestedEinsum that OMEinsum.writejson can serialise.
        writejson(_code_path(dirname, id), uncompress(slice.code))
    end
    return nothing
end

# ----------------------------------------------------------------------
# Load
# ----------------------------------------------------------------------

"""
    load_spin_glass_slice(subdir, id; root = SLICE_RESULTS_ROOT) -> NamedTuple

Reload a single slice. `subdir` may be either an absolute path, an
existing path, or a name relative to `root`. Returns
`(id, g, J, h, r, code)`. `code === nothing` means the slice was the
trivial all-fixed branch.
"""
function load_spin_glass_slice(subdir::AbstractString, id::Integer;
                                root::AbstractString = SLICE_RESULTS_ROOT)
    dirname = slice_results_dir(subdir; root = root)
    g = Graphs.loadgraph(_graph_path(dirname, id))
    J = _load_numeric_vector(_J_path(dirname, id))
    h = _load_numeric_vector(_h_path(dirname, id))
    r = _load_numeric_scalar(_r_path(dirname, id))
    code_path = _code_path(dirname, id)
    code = isfile(code_path) ? readjson(code_path) : nothing
    return (id = Int(id), g = g, J = J, h = h, r = r, code = code)
end

"""
    list_spin_glass_slices(subdir; root = SLICE_RESULTS_ROOT) -> DataFrame

Read back the `slices.csv` summary written by [`save_spin_glass_slices`](@ref).
"""
list_spin_glass_slices(subdir::AbstractString; root::AbstractString = SLICE_RESULTS_ROOT) =
    CSV.read(_csv_path(slice_results_dir(subdir; root = root)), DataFrame)

"""
    load_original_spin_glass(subdir; root = SLICE_RESULTS_ROOT) -> NamedTuple

Reload the **pre-slicing** spin glass `(g, J, h)` saved alongside the
slices when the dump was created with `original = (g, J, h)`. Throws if
the original is not present.
"""
function load_original_spin_glass(subdir::AbstractString; root::AbstractString = SLICE_RESULTS_ROOT)
    dirname = slice_results_dir(subdir; root = root)
    has_original_spin_glass(dirname; root = "") ||
        error("no original problem stored in $dirname (expected " *
              "$(_orig_graph_path(dirname)) etc.)")
    g = Graphs.loadgraph(_orig_graph_path(dirname))
    J = _load_numeric_vector(_orig_J_path(dirname))
    h = _load_numeric_vector(_orig_h_path(dirname))
    return (; g, J, h)
end

"""
    has_original_spin_glass(subdir; root = SLICE_RESULTS_ROOT) -> Bool
"""
function has_original_spin_glass(subdir::AbstractString; root::AbstractString = SLICE_RESULTS_ROOT)
    dirname = isempty(root) ? String(subdir) : slice_results_dir(subdir; root = root)
    return isfile(_orig_graph_path(dirname)) &&
           isfile(_orig_J_path(dirname)) &&
           isfile(_orig_h_path(dirname))
end

"""
    list_slice_runs(; root = SLICE_RESULTS_ROOT) -> DataFrame

Read the top-level `summary.csv` aggregating every benchmark instance
that has been dumped under `root`.
"""
function list_slice_runs(; root::AbstractString = SLICE_RESULTS_ROOT)
    path = _summary_path(root)
    isfile(path) || return DataFrame()
    return CSV.read(path, DataFrame)
end

# ----------------------------------------------------------------------
# Contract one slice
# ----------------------------------------------------------------------

"""
    contract_one_spin_glass_slice(slice;
                                  usecuda            = false,
                                  fallback_optimizer = TreeSA(),
                                  count_eltype       = :finitefield,
                                  energy_scale       = 2,
                                  max_crt_iter       = 8,
                                  verbose            = false)
        -> (energy, count, runtime,
            primes,        # only set in :finitefield mode
            crt_history)   # only set in :finitefield mode

Run one CountingMax-style contraction on a previously-loaded slice and
return the **shifted** ground-state energy `E_i + r_i`, the slice's local
degeneracy `c_i`, and the elapsed wall-time.

`count_eltype` selects the algebra used inside the contraction:

* `:finitefield` (default, GPU-safe, exact):
    Repeats the contraction with eltype
    `CountingTropical{Float64, Mods.Mod{p, Int32}}` for several primes
    `p` (decreasing from `typemax(Int32)`) and combines the per-prime
    counts via the Chinese-Remainder Theorem.  Mirrors the
    `big_integer_solve` mechanism used by
    `GenericTensorNetworks.graph_polynomial(:finitefield)` (see
    `GenericTensorNetworks/src/graph_polynomials.jl`).  Iteration stops
    after the recovered count is identical for two consecutive primes
    (i.e. the BigInt result has stabilised) or when `max_crt_iter` is
    reached.  Energy stays in `Float64` (tropical max well-defined and
    GPU-friendly via `CuTropicalGEMM`); only the **count** moves to
    32-bit modular arithmetic, sidestepping the LLVM NVPTX ≥ 256-bit
    by-value codegen problems hit by `CountingTropical{Int128,Int128}`.

* `Float64` (legacy direct path):
    `CountingTropical{Float64,Float64}`. Count is exact only up to
    `2^53 ≈ 9e15`; beyond that the result is still finite but no longer
    an exact integer.

* `Int128` (legacy direct path, **broken on GPU**):
    `CountingTropical{Int128, Int128}`.  Crashes the LLVM NVPTX backend
    on most non-trivial einsums (256-bit struct passed by value as a
    kernel argument).  Kept only for CPU runs / historical comparison.

* `BigInt` (CPU only):
    `CountingTropical{BigInt, BigInt}`. Arbitrary precision but cannot
    be uploaded to the GPU (`BigInt` is not `isbits`).

Energy scaling
--------------
For all integer-valued contraction modes (everything except `Float64`)
the energy exponents inside the SpinGlass tensors must be integers.
We convert the slice's couplings on the fly:
    `J → round.(Int, energy_scale .* J)`
    `h → round.(Int, energy_scale .* h)`
The contracted `result.n` is then `energy_scale × E_subgraph`; we divide
out the scale before adding the (unscaled) additive constant `r`.
`energy_scale = 2` covers the common `J ∈ {±1}, h ∈ {±0.5, ±1}` case.

Trivial slices (`nv(g) == 0`) short-circuit; slices without saved code
fall back to `fallback_optimizer`.

Returned NamedTuple
-------------------
Always carries `(energy, count::BigInt, runtime)`. The `:finitefield`
mode additionally fills:
* `primes :: Vector{Int}`       — primes used, in order of use
* `crt_history :: Vector{BigInt}` — running CRT result after each prime

For the legacy paths these two extra fields are empty.
"""
function contract_one_spin_glass_slice(slice; usecuda::Bool = false,
                                        fallback_optimizer = TreeSA(),
                                        count_eltype = :finitefield,
                                        energy_scale::Integer = 2,
                                        max_crt_iter::Int = 8,
                                        verbose::Bool = false)
    g, J, h, r, code = slice.g, slice.J, slice.h, slice.r, slice.code

    if nv(g) == 0
        return (energy = Float64(r), count = BigInt(1), runtime = 0.0,
                primes = Int[], crt_history = BigInt[])
    end

    if count_eltype === :finitefield
        return _contract_one_finitefield(g, J, h, r, code;
                                          usecuda = usecuda,
                                          fallback_optimizer = fallback_optimizer,
                                          energy_scale = energy_scale,
                                          max_crt_iter = max_crt_iter,
                                          verbose = verbose)
    elseif count_eltype isa DataType
        return _contract_one_direct(g, J, h, r, code;
                                     usecuda = usecuda,
                                     fallback_optimizer = fallback_optimizer,
                                     count_eltype = count_eltype,
                                     energy_scale = energy_scale)
    else
        error("unknown count_eltype $(repr(count_eltype)); use :finitefield " *
              "(default, GPU-safe, exact) or one of Float64 / Int128 / BigInt.")
    end
end

# ---------------------------------------------------------------------------
# Direct (legacy) path: solve(problem, CountingMax(); T = count_eltype)
# ---------------------------------------------------------------------------
function _contract_one_direct(g, J, h, r, code;
                              usecuda::Bool, fallback_optimizer,
                              count_eltype::DataType, energy_scale::Integer)
    usecuda && count_eltype === BigInt &&
        error("BigInt is not isbits and cannot be used on the GPU; pass " *
              "usecuda=false or count_eltype=:finitefield.")

    Jc, hc, scale_used = _scale_couplings(J, h; want_int = (count_eltype !== Float64),
                                                 energy_scale = energy_scale)
    problem = isnothing(code) ?
        GenericTensorNetwork(SpinGlass(g, Jc, hc); optimizer = fallback_optimizer) :
        GenericTensorNetwork(SpinGlass(g, Jc, hc), code, Dict{Int, Int}())

    t = @elapsed begin
        raw = solve(problem, CountingMax(); T = count_eltype, usecuda = usecuda)
        result = Array(raw)[]
    end

    energy = Float64(result.n) / scale_used + Float64(r)
    return (energy = energy, count = _to_bigint(result.c), runtime = t,
            primes = Int[], crt_history = BigInt[])
end

# ---------------------------------------------------------------------------
# Finitefield (CRT) path
#   Each iteration: contract with CountingTropical{Float64, Mod{p, Int32}}.
#   Combine per-prime counts via CRT until the BigInt result stabilises.
# ---------------------------------------------------------------------------
function _contract_one_finitefield(g, J, h, r, code;
                                   usecuda::Bool, fallback_optimizer,
                                   energy_scale::Integer, max_crt_iter::Int,
                                   verbose::Bool)
    Jc, hc, scale_used = _scale_couplings(J, h; want_int = true,
                                                 energy_scale = energy_scale)
    problem = isnothing(code) ?
        GenericTensorNetwork(SpinGlass(g, Jc, hc); optimizer = fallback_optimizer) :
        GenericTensorNetwork(SpinGlass(g, Jc, hc), code, Dict{Int, Int}())

    primes_used = Int[]
    remainders  = Int[]
    energies_p  = Float64[]
    crt_history = BigInt[]
    runtimes    = Float64[]

    # Walk primes downward from `typemax(Int32) = 2^31 - 1` (itself a
    # Mersenne prime, M31). `prevprime(N)` returns the largest prime ≤ N,
    # so the first iteration uses 2147483647, the second uses
    # `prevprime(2147483646) = 2147483629`, etc., until the CRT count
    # stabilises or `max_crt_iter` is reached.
    N = Int(typemax(Int32))
    last_x = BigInt(-1)
    for k in 1:max_crt_iter
        N <= 2 && error("[finitefield] ran out of Int32 primes after $(k-1) iteration(s)")
        p = Int(prevprime(N))
        N = p - 1                      # next iteration starts strictly below p
        push!(primes_used, p)

        Tp = Mod{Int32(p), Int32}
        CT = CountingTropical{Float64, Tp}
        x  = _x(CT; invert = false)

        t = @elapsed begin
            raw = contractx(problem, x; usecuda = usecuda)
            res = Array(raw)[]
        end
        energy_p = Float64(res.n)
        c_modp   = Int(value(res.c))
        push!(remainders, c_modp)
        push!(energies_p, energy_p)
        push!(runtimes,   t)

        if k > 1 && !isapprox(energy_p, energies_p[1]; atol = 1e-6)
            @warn "[finitefield] energy disagrees across primes ($(energies_p)); " *
                  "treating $(energies_p[1]) as the truth."
        end

        x_now = _crt_combine(remainders, primes_used)
        push!(crt_history, x_now)
        verbose && @info "[finitefield] iter $k  p=$p  count mod p = $c_modp  " *
                          "CRT = $x_now  t = $(round(t, digits=3))s"

        if x_now == last_x
            break
        end
        last_x = x_now
    end

    energy = energies_p[1] / scale_used + Float64(r)
    count  = isempty(crt_history) ? BigInt(0) : crt_history[end]
    return (energy = energy, count = count, runtime = sum(runtimes),
            primes = primes_used, crt_history = crt_history)
end

# Common preprocessor: cast J,h to integer (×energy_scale) when required.
function _scale_couplings(J, h; want_int::Bool, energy_scale::Integer)
    if !want_int
        return (collect(J), collect(h), 1)
    end
    energy_scale > 0 ||
        error("energy_scale must be a positive integer, got $energy_scale")
    Jf = Float64.(collect(J)) .* energy_scale
    hf = Float64.(collect(h)) .* energy_scale
    Ji = round.(Int, Jf)
    hi = round.(Int, hf)
    err = max(maximum(abs, Jf .- Ji; init = 0.0),
              maximum(abs, hf .- hi; init = 0.0))
    err < 1e-10 ||
        error("J/h cannot be represented as Int after scaling by $(energy_scale) " *
              "(max rounding error = $(err)); pick a larger energy_scale.")
    return (Ji, hi, energy_scale)
end

# Robust BigInt conversion regardless of whether the count component is an
# Integer (Int128, Int64, ...) or a Float64 holding an integer-valued double.
_to_bigint(x::Integer) = BigInt(x)
_to_bigint(x::AbstractFloat) = isfinite(x) ?
    BigInt(round(x)) :
    error("non-finite count $(x) — use a wider count eltype (:finitefield / BigInt)")

# Standard incremental CRT.  All primes are coprime (distinct primes).
# Result lies in `0 .. prod(primes)-1`.
function _crt_combine(remainders::Vector{<:Integer}, primes::Vector{<:Integer})
    @assert length(remainders) == length(primes) && !isempty(primes)
    M = BigInt(primes[1])
    x = BigInt(remainders[1])
    @inbounds for k in 2:length(primes)
        p    = BigInt(primes[k])
        r    = BigInt(remainders[k])
        Minv = invmod(M, p)
        x    = x + M * mod((r - x) * Minv, p)
        M   *= p
        x    = mod(x, M)
    end
    return x
end

# ----------------------------------------------------------------------
# Contract every slice + sum to ground-state degeneracy
# ----------------------------------------------------------------------

"""
    contract_spin_glass_slices(subdir;
                                root               = SLICE_RESULTS_ROOT,
                                usecuda            = false,
                                atol               = 1e-6,
                                verbose            = false,
                                fallback_optimizer = TreeSA(),
                                count_eltype       = :finitefield,
                                energy_scale       = 2,
                                max_crt_iter       = 8,
                                ids                = nothing)

Iterate over every slice stored in `joinpath(root, subdir)` (`subdir`
may also be absolute / pre-existing), contract it via the saved `code`,
shift its energy by `r`, and combine the per-slice results into the
**exact** ground-state energy and degeneracy of the original spin
glass.

`count_eltype` defaults to `:finitefield`, which is the only mode
that gives **exact** counting on the GPU without overflow risk. See
[`contract_one_spin_glass_slice`](@ref) for the full menu (`:finitefield`,
`Float64`, `Int128`, `BigInt`) and the reasoning behind each.

Returns a `NamedTuple`:

    (energy, count, per_slice :: DataFrame, total_runtime)

`per_slice` carries one row per slice with `(id, r, energy, count,
runtime, n_primes, last_prime, used_in_total)`. `n_primes` /
`last_prime` are populated for the `:finitefield` mode (number of
primes consumed before CRT stabilised, and the last prime used).
`used_in_total` flags rows contributing to the final degeneracy
(i.e. `|E_i + r_i - E_g| ≤ atol`).
"""
function contract_spin_glass_slices(subdir::AbstractString;
                                     root::AbstractString = SLICE_RESULTS_ROOT,
                                     usecuda::Bool = false,
                                     atol::Real = 1e-6,
                                     verbose::Bool = false,
                                     fallback_optimizer = TreeSA(),
                                     count_eltype = :finitefield,
                                     energy_scale::Integer = 2,
                                     max_crt_iter::Int = 8,
                                     ids = nothing)
    dirname = slice_results_dir(subdir; root = root)
    summary = list_spin_glass_slices(dirname)
    slice_ids = ids === nothing ? Vector{Int}(summary.id) : Vector{Int}(ids)

    energies     = Float64[]
    counts       = BigInt[]
    runtimes     = Float64[]
    rs           = Float64[]
    n_primes     = Int[]      # only meaningful for :finitefield
    last_prime   = Int[]

    total_t = @elapsed for (k, id) in enumerate(slice_ids)
        slice = load_spin_glass_slice(dirname, id)
        res   = contract_one_spin_glass_slice(slice; usecuda = usecuda,
                                              fallback_optimizer = fallback_optimizer,
                                              count_eltype = count_eltype,
                                              energy_scale = energy_scale,
                                              max_crt_iter = max_crt_iter,
                                              verbose = verbose)
        push!(energies, res.energy)
        push!(counts,   res.count)
        push!(runtimes, res.runtime)
        push!(rs,       Float64(slice.r))
        push!(n_primes, length(res.primes))
        push!(last_prime, isempty(res.primes) ? 0 : res.primes[end])
        if verbose
            extra = isempty(res.primes) ? "" :
                "  (CRT: $(length(res.primes)) prime(s), last p = $(res.primes[end]))"
            @info "slice $id ($k/$(length(slice_ids))): nv=$(nv(slice.g)) ne=$(ne(slice.g)) " *
                  "E+r=$(res.energy) count=$(res.count) t=$(round(res.runtime, digits=3))s" *
                  extra
        end
    end

    Eg = isempty(energies) ? -Inf : maximum(energies)
    used = falses(length(energies))
    total_count = BigInt(0)
    @inbounds for i in eachindex(energies)
        if abs(energies[i] - Eg) ≤ atol
            total_count += counts[i]
            used[i] = true
        end
    end

    per_slice = DataFrame(id = slice_ids,
                          r = rs,
                          energy = energies,
                          count = counts,
                          runtime = runtimes,
                          n_primes = n_primes,
                          last_prime = last_prime,
                          used_in_total = used)

    return (energy = Eg,
            count  = total_count,
            per_slice = per_slice,
            total_runtime = total_t)
end
