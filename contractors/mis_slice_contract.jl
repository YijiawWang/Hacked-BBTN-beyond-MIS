"""
    mis_slice_contract.jl

I/O + contraction utilities for MIS / MWIS slices produced by the MIS
variants of `slice_bfs` / `slice_dfs_lp` (see
`beyond_mis/hacked_funcs/src/mis_counting.jl` and
`beyond_mis/hacked_funcs/src/mis_ground_counting.jl`).

Two complementary contractor families live in this file:

  * **Ground counting** (`CountingMax()`, used by
    `mis_ground_counting.jl`): per slice we get
    `(size_i, c_i) = solve(IndependentSet(g_i, w_i), CountingMax())`,
    combined with an `r_i` shift into
    `S_g = max_i (size_i + r_i)` and
    `deg = Σ_{i: size_i + r_i ≈ S_g} c_i`.
    See [`contract_one_mis_slice`](@ref),
    [`contract_mis_slices`](@ref).
  * **Total counting** (`CountingAll()`, used by `mis_counting.jl`):
    per slice we get the count of *every* IS in `g_i` and the slices'
    counts sum directly to the total IS count of the original graph
    (no `r`-shift, no max-filtering — each slice corresponds to a
    disjoint sub-tree of the original IS enumeration). See
    [`contract_one_mis_counting_slice`](@ref),
    [`contract_mis_counting_slices`](@ref).

A `SlicedBranch{INT, VT, RT, MISProblem{INT, VT}}` carries:

  * a sub-graph `g_i` (induced subgraph after branching/reductions),
  * its vertex weights `w_i` (`Vector{<:Number}` for MWIS, or
    `ProblemReductions.UnitWeight` for unweighted MIS),
  * a contraction code `code_i` whose space complexity fits the slicer's
    `sc_target`,
  * an additive constant `r_i` (the maximum-IS contribution of the
    vertices already fixed outside `g_i`).

Every slice is dumped under

    <SLICE_RESULTS_ROOT>/<subdir>/
        slices.csv                  # id, r, sc, tc, nv, ne, has_code, weights_kind
        meta.txt                    # free-form (graph_name, graph_type, ...)
        graph_<id>.lg               # SimpleGraph (Graphs.savegraph LG format)
        eincode_<id>.json           # OMEinsum DynamicNestedEinsum (writejson)
        weights_<id>.txt            # only when slice.p.weights is a numeric vector
        r_<id>.txt                  # numeric scalar
        original_graph.lg           # only when `original = (g, weights)` is given
        original_weights.txt        # only when `original` weights are non-unit
        original_weights_kind.txt   # 'unit' | 'vector'

and a row is upserted in `<SLICE_RESULTS_ROOT>/summary.csv` describing
the run (slice_num, max_sc, total_tc, ...).

The `SLICE_RESULTS_ROOT` constant and the low-level helpers
(`slice_results_dir`, `_save_numeric_vector`, `_save_numeric_scalar`,
`update_slice_summary`, `_logsumexp_tc`, ...) are shared with
`spin_glass_slice_contract.jl` via `include` so that MIS-ground and
spin-glass-ground runs land under the same `beyond_mis/branch_results/`
root and share a single `summary.csv`.
"""

using Graphs
using OMEinsum                                # writejson, readjson
using OMEinsumContractionOrders: TreeSA, uniformsize
using GenericTensorNetworks
using ProblemReductions: IndependentSet, UnitWeight
using CSV, DataFrames
using TensorBranching: uncompress, complexity

# Pull in the spin-glass file's shared infrastructure (SLICE_RESULTS_ROOT,
# slice_results_dir, _NUMERIC_TYPE_TAGS, _save_numeric_vector,
# _save_numeric_scalar, _load_numeric_vector, _load_numeric_scalar,
# update_slice_summary, _logsumexp_tc, _csv_path, _meta_path, ...).
include(joinpath(@__DIR__, "spin_glass_slice_contract.jl"))

# ----------------------------------------------------------------------
# MIS-specific paths and constants
# ----------------------------------------------------------------------

_mis_weights_path(dir, id) = joinpath(dir, "weights_$(id).txt")
_mis_orig_weights_path(dir) = joinpath(dir, "original_weights.txt")
_mis_orig_weights_kind_path(dir) = joinpath(dir, "original_weights_kind.txt")

const _MIS_SLICE_CSV_HEADER = "id,r,sc,tc,nv,ne,has_code,weights_kind"
const _MIS_WEIGHTS_KIND_UNIT = "unit"
const _MIS_WEIGHTS_KIND_VEC  = "vector"

_weights_kind_tag(w) = (w isa UnitWeight) ? _MIS_WEIGHTS_KIND_UNIT :
                                            _MIS_WEIGHTS_KIND_VEC

# ----------------------------------------------------------------------
# Streaming writer
# ----------------------------------------------------------------------

"""
    MISSliceWriter

Mutable handle returned by [`init_mis_slice_writer`](@ref). Callers feed
finished MIS slices to [`append_mis_slice!`](@ref); each call writes the
slice's files to disk and appends one row to `slices.csv` immediately,
so partial progress survives a crash. Call
[`finalize_mis_slice_writer!`](@ref) at the end to refresh the
top-level `summary.csv`.

The same writer powers both `mis_ground_counting.jl` and
`mis_counting.jl`; downstream you choose between
[`contract_mis_slices`](@ref) (CountingMax / ground) or
[`contract_mis_counting_slices`](@ref) (CountingAll / total).
"""
mutable struct MISSliceWriter
    dirname::String
    root::String
    csv_path::String
    next_id::Int
    ids::Vector{Int}
    rs::Vector{Float64}
    scs::Vector{Float64}
    tcs::Vector{Float64}
    nvs::Vector{Int}
    nes::Vector{Int}
    has_c::Vector{Bool}
    weights_kinds::Vector{String}
    graph_name::String
    graph_type::String
    meta::Dict{String,Any}
    update_summary::Bool
end

"""
    init_mis_slice_writer(subdir; root, original, graph_name,
                          graph_type, overwrite, meta,
                          update_summary) -> MISSliceWriter

Prepare `joinpath(root, subdir)` for streaming MIS slice writes:

  * (optionally) clears the directory (`overwrite=true`),
  * creates it,
  * dumps the pre-slicing problem if `original = (g, weights)`. Both
    unit-weight and explicit-vector weights are supported; a one-line
    `original_weights_kind.txt` records which it was.
  * writes `meta.txt` (graph_name / graph_type / extra metadata),
  * starts an empty `slices.csv` with the standard MIS header.
"""
function init_mis_slice_writer(subdir::AbstractString;
                                root::AbstractString = SLICE_RESULTS_ROOT,
                                original = nothing,
                                graph_name::AbstractString = "",
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
        og, ow = original
        Graphs.savegraph(joinpath(dirname, "original_graph.lg"), og)
        kind = _weights_kind_tag(ow)
        if kind == _MIS_WEIGHTS_KIND_VEC
            _save_numeric_vector(_mis_orig_weights_path(dirname), Vector(ow))
        end
        open(_mis_orig_weights_kind_path(dirname), "w") do io
            println(io, kind)
        end
    end

    if !isempty(graph_name) || !isempty(graph_type) || !isempty(meta)
        open(_meta_path(dirname), "w") do io
            isempty(graph_name) || println(io, "graph_name = ", graph_name)
            isempty(graph_type) || println(io, "graph_type = ", graph_type)
            for (k, v) in meta
                println(io, k, " = ", v)
            end
        end
    end

    csv_path = _csv_path(dirname)
    open(csv_path, "w") do io
        println(io, _MIS_SLICE_CSV_HEADER)
    end

    meta_str = Dict{String,Any}(string(k) => v for (k, v) in meta)

    return MISSliceWriter(dirname, String(root), csv_path, 1,
                          Int[], Float64[], Float64[], Float64[],
                          Int[], Int[], Bool[], String[],
                          String(graph_name), String(graph_type),
                          meta_str, update_summary)
end

"""
    append_mis_slice!(writer, slice; id=nothing,
                      flush_summary=false) -> Int

Persist one finished MIS `SlicedBranch` immediately:

  * dumps `graph_<id>.lg`, `r_<id>.txt`, and (when applicable)
    `weights_<id>.txt` and `eincode_<id>.json`;
  * appends one row to `slices.csv`.

If `id` is omitted, the next sequential id (starting from 1) is used.
When `flush_summary=true` the top-level `summary.csv` is rewritten with
the running totals after this slice — useful so even a crashed run
leaves a usable `summary.csv` behind.

Returns the id used for this slice.
"""
function append_mis_slice!(writer::MISSliceWriter, slice;
                            id::Union{Nothing,Integer} = nothing,
                            flush_summary::Bool = false)
    sid = id === nothing ? writer.next_id : Int(id)
    writer.next_id = max(writer.next_id, sid + 1)

    g = slice.p.g
    w = slice.p.weights
    Graphs.savegraph(_graph_path(writer.dirname, sid), g)
    weights_kind = _weights_kind_tag(w)
    if weights_kind == _MIS_WEIGHTS_KIND_VEC
        _save_numeric_vector(_mis_weights_path(writer.dirname, sid), Vector(w))
    end
    _save_numeric_scalar(_r_path(writer.dirname, sid), Float64(slice.r))
    if !isnothing(slice.code)
        # `slice.code` is a CompressedEinsum; uncompress to a
        # DynamicNestedEinsum that OMEinsum.writejson can serialise.
        writejson(_code_path(writer.dirname, sid), uncompress(slice.code))
    end

    cc = complexity(slice)
    push!(writer.ids,   sid)
    push!(writer.rs,    Float64(slice.r))
    push!(writer.scs,   Float64(cc.sc))
    push!(writer.tcs,   Float64(cc.tc))
    push!(writer.nvs,   nv(g))
    push!(writer.nes,   ne(g))
    push!(writer.has_c, !isnothing(slice.code))
    push!(writer.weights_kinds, weights_kind)

    open(writer.csv_path, "a") do io
        println(io, sid, ",",
                    Float64(slice.r), ",",
                    Float64(cc.sc),   ",",
                    Float64(cc.tc),   ",",
                    nv(g),            ",",
                    ne(g),            ",",
                    !isnothing(slice.code), ",",
                    weights_kind)
    end

    if flush_summary && writer.update_summary
        _flush_mis_slice_summary(writer)
    end

    return sid
end

"""
    finalize_mis_slice_writer!(writer) -> String

Flush the top-level `summary.csv` (if `update_summary=true`) and return
the per-instance directory path. Safe to call multiple times.
"""
function finalize_mis_slice_writer!(writer::MISSliceWriter)
    if writer.update_summary
        _flush_mis_slice_summary(writer)
    end
    return writer.dirname
end

function _flush_mis_slice_summary(writer::MISSliceWriter)
    info = Dict{Symbol,Any}(
        :subdir     => relpath(writer.dirname, writer.root),
        :graph_name => writer.graph_name,
        :graph_type => writer.graph_type,
        :slice_num  => length(writer.ids),
        :max_sc     => isempty(writer.scs) ? 0.0 : maximum(writer.scs),
        :total_tc   => _logsumexp_tc(writer.tcs),
        :family     => "mis",
    )
    for (k, v) in writer.meta
        sym = Symbol(string(k))
        haskey(info, sym) || (info[sym] = v)
    end
    update_slice_summary(info; root = writer.root)
    return nothing
end

# ----------------------------------------------------------------------
# Convenience: dump a vector of slices in one go
# ----------------------------------------------------------------------

"""
    save_mis_slices(subdir, slices; root, original, graph_name,
                    graph_type, overwrite, meta, update_summary) -> String

Persist a vector of finished MIS `SlicedBranch` objects to
`joinpath(root, subdir)` so they can later be reloaded via
[`load_mis_slice`](@ref) / [`list_mis_slices`](@ref). Equivalent to
opening a writer, calling `append_mis_slice!` for every slice, and then
finalising. Returns the absolute path of the per-instance directory.
"""
function save_mis_slices(subdir::AbstractString, slices;
                          root::AbstractString = SLICE_RESULTS_ROOT,
                          original = nothing,
                          graph_name::AbstractString = "",
                          graph_type::AbstractString = "",
                          overwrite::Bool = false,
                          meta::AbstractDict = Dict{String,Any}(),
                          update_summary::Bool = true)
    writer = init_mis_slice_writer(subdir;
        root           = root,
        original       = original,
        graph_name     = graph_name,
        graph_type     = graph_type,
        overwrite      = overwrite,
        meta           = meta,
        update_summary = update_summary,
    )
    for slice in slices
        append_mis_slice!(writer, slice; flush_summary = false)
    end
    finalize_mis_slice_writer!(writer)
    return writer.dirname
end

# ----------------------------------------------------------------------
# Loaders
# ----------------------------------------------------------------------

"""
    load_mis_slice(subdir, id; root = SLICE_RESULTS_ROOT) -> NamedTuple

Reload a single MIS slice. `subdir` may be either an absolute path, an
existing path, or a name relative to `root`. Returns
`(id, g, weights, r, code)`. `code === nothing` means the slice was the
trivial all-fixed branch; `weights` is a `UnitWeight(nv(g))` when the
slice was unweighted.
"""
function load_mis_slice(subdir::AbstractString, id::Integer;
                         root::AbstractString = SLICE_RESULTS_ROOT)
    dirname = slice_results_dir(subdir; root = root)
    g = Graphs.loadgraph(_graph_path(dirname, id))
    r = _load_numeric_scalar(_r_path(dirname, id))
    wpath = _mis_weights_path(dirname, id)
    weights = isfile(wpath) ? _load_numeric_vector(wpath) : UnitWeight(nv(g))
    code_path = _code_path(dirname, id)
    code = isfile(code_path) ? readjson(code_path) : nothing
    return (id = Int(id), g = g, weights = weights, r = r, code = code)
end

"""
    list_mis_slices(subdir; root = SLICE_RESULTS_ROOT) -> DataFrame

Read back the `slices.csv` summary written by [`save_mis_slices`](@ref).
"""
list_mis_slices(subdir::AbstractString; root::AbstractString = SLICE_RESULTS_ROOT) =
    CSV.read(_csv_path(slice_results_dir(subdir; root = root)), DataFrame)

"""
    load_original_mis(subdir; root = SLICE_RESULTS_ROOT) -> NamedTuple

Reload the **pre-slicing** MIS problem `(g, weights)` saved alongside
the slices when the dump was created with `original = (g, weights)`.
`weights` is reloaded as `UnitWeight(nv(g))` when the original problem
was unweighted, otherwise as a numeric `Vector`.
"""
function load_original_mis(subdir::AbstractString; root::AbstractString = SLICE_RESULTS_ROOT)
    dirname = isabspath(subdir) ? String(subdir) : slice_results_dir(subdir; root = root)
    has_original_mis(dirname; root = "") ||
        error("no original MIS problem stored in $dirname (expected " *
              "original_graph.lg + original_weights_kind.txt; create it " *
              "with `init_mis_slice_writer(...; original = (g, weights))`)")
    g = Graphs.loadgraph(joinpath(dirname, "original_graph.lg"))
    kind = strip(read(_mis_orig_weights_kind_path(dirname), String))
    weights = if kind == _MIS_WEIGHTS_KIND_UNIT
        UnitWeight(nv(g))
    else
        _load_numeric_vector(_mis_orig_weights_path(dirname))
    end
    return (; g, weights)
end

"""
    has_original_mis(subdir; root = SLICE_RESULTS_ROOT) -> Bool
"""
function has_original_mis(subdir::AbstractString; root::AbstractString = SLICE_RESULTS_ROOT)
    dirname = isempty(root) ? String(subdir) : slice_results_dir(subdir; root = root)
    isfile(joinpath(dirname, "original_graph.lg")) || return false
    isfile(_mis_orig_weights_kind_path(dirname))   || return false
    kind = strip(read(_mis_orig_weights_kind_path(dirname), String))
    return kind == _MIS_WEIGHTS_KIND_UNIT ||
           (kind == _MIS_WEIGHTS_KIND_VEC && isfile(_mis_orig_weights_path(dirname)))
end

# ----------------------------------------------------------------------
# Weight pre-processor (shared by the finitefield CountingMax path)
# ----------------------------------------------------------------------

# Cast `weights` into a form acceptable by `IndependentSet(g, w)` when
# the contraction is going to run in modular (Mod{p, Int32}) arithmetic.
#
# * `UnitWeight` is forwarded as-is (the IS tensor is then `[1, x]`).
# * Numeric weights are scaled by `weight_scale` and rounded to `Int`;
#   the function errors out when the rounding error exceeds `1e-10`.
#
# Returns `(weights_used, scale_used)` where `scale_used == 1` for the
# UnitWeight pass-through case.
function _scale_weights(w; weight_scale::Integer)
    if w isa UnitWeight
        return (w, 1)
    end
    weight_scale > 0 ||
        error("weight_scale must be a positive integer, got $weight_scale")
    wv = collect(w)
    if weight_scale == 1 && eltype(wv) <: Integer
        return (Vector{Int}(wv), 1)
    end
    wf = Float64.(wv) .* weight_scale
    wi = round.(Int, wf)
    err = maximum(abs, wf .- wi; init = 0.0)
    err < 1e-10 ||
        error("MIS weights cannot be represented as Int after scaling by " *
              "$(weight_scale) (max rounding error = $(err)); pick a larger " *
              "weight_scale.")
    return (wi, weight_scale)
end

# ----------------------------------------------------------------------
# Contract one MIS slice — CountingMax (ground counting / MWIS)
# ----------------------------------------------------------------------

"""
    contract_one_mis_slice(slice;
                            usecuda            = false,
                            fallback_optimizer = TreeSA(),
                            count_eltype       = :finitefield,
                            weight_scale       = 1,
                            max_crt_iter       = 8,
                            verbose            = false)
        -> (size, count, runtime,
            primes,        # only set in :finitefield mode
            crt_history)   # only set in :finitefield mode

Re-contract a single previously-loaded MIS slice for the **CountingMax**
property (= maximum-(weighted-)IS size and its degeneracy) and return
the *shifted* size `size_i + r_i`, the slice's local degeneracy `c_i`
and the elapsed wall-time.

  * If the slice is trivial (`nv(g) == 0`) we short-circuit with
    `(r, 1)` (the single empty IS).
  * Otherwise we build `GenericTensorNetwork(IndependentSet(g, w_used),
    code, Dict{Int,Int}())` — i.e. **reusing the saved contraction
    order**. Slices without saved code fall back to `fallback_optimizer`.

`count_eltype` selects the algebra used inside the contraction (mirrors
[`contract_one_spin_glass_slice`](@ref)):

* `:finitefield` (default, GPU-safe, exact):
    Per-prime contraction with eltype
    `CountingTropical{Float64, Mods.Mod{p, Int32}}` for several primes
    `p` (decreasing from `typemax(Int32)`); per-prime counts are
    CRT-combined into an exact `BigInt`. Iteration stops once the
    recovered count repeats (or after `max_crt_iter`). Sidesteps the
    LLVM NVPTX ≥ 256-bit by-value codegen problems hit by
    `CountingTropical{Int128,Int128}`.

* `Float64` (legacy direct path):
    `CountingTropical{Float64,Float64}`. Count is exact only up to
    `2^53 ≈ 9e15`; beyond that the result is no longer an exact
    integer.

* `Int128` (legacy direct path, **broken on GPU**):
    `CountingTropical{Int128,Int128}`. Crashes the LLVM NVPTX backend
    on most non-trivial einsums (256-bit struct passed by value as a
    kernel argument). Kept for CPU runs / historical comparison.

* `BigInt` (CPU only):
    `CountingTropical{BigInt,BigInt}`. Arbitrary precision but cannot
    be uploaded to the GPU (`BigInt` is not `isbits`).

Weight scaling
--------------
The integer-valued contraction modes (everything except `Float64`)
require integer vertex weights. `UnitWeight` is forwarded as-is; for
explicit numeric weights we scale by `weight_scale` and round to `Int`
on the fly. The contracted `result.n` is then
`weight_scale × size_subgraph`; we divide out the scale before adding
the (unscaled) additive constant `r`. `weight_scale = 1` covers the
common unit-weight / integer-weight MIS case; pick a larger value
(e.g. `2`) when MWIS weights live on a half-integer lattice.

Returned NamedTuple
-------------------
Always carries `(size, count::BigInt, runtime)`. The `:finitefield`
mode additionally fills:
* `primes :: Vector{Int}`         — primes used, in order of use
* `crt_history :: Vector{BigInt}` — running CRT result after each prime

The legacy paths leave both extra fields empty.

This routine is shared by the slice dumps produced by both
`mis_counting.jl` and `mis_ground_counting.jl`: each slice dump only
records the sub-problem `(g_i, w_i, r_i, code_i)`, and the property of
interest after re-contraction is the same — count the maximum
(weighted) independent sets within that sub-problem.
"""
function contract_one_mis_slice(slice; usecuda::Bool = false,
                                  fallback_optimizer = TreeSA(),
                                  count_eltype = :finitefield,
                                  weight_scale::Integer = 1,
                                  max_crt_iter::Int = 8,
                                  verbose::Bool = false)
    g, w, r, code = slice.g, slice.weights, slice.r, slice.code

    if nv(g) == 0
        return (size = Float64(r), count = BigInt(1), runtime = 0.0,
                primes = Int[], crt_history = BigInt[])
    end

    if count_eltype === :finitefield
        return _contract_one_mis_finitefield(g, w, r, code;
                                              usecuda = usecuda,
                                              fallback_optimizer = fallback_optimizer,
                                              weight_scale = weight_scale,
                                              max_crt_iter = max_crt_iter,
                                              verbose = verbose)
    elseif count_eltype isa DataType
        return _contract_one_mis_direct(g, w, r, code;
                                         usecuda = usecuda,
                                         fallback_optimizer = fallback_optimizer,
                                         count_eltype = count_eltype,
                                         weight_scale = weight_scale)
    else
        error("unknown count_eltype $(repr(count_eltype)); use :finitefield " *
              "(default, GPU-safe, exact) or one of Float64 / Int128 / BigInt.")
    end
end

# ---------------------------------------------------------------------------
# Direct (legacy) path: solve(problem, CountingMax(); T = count_eltype)
# ---------------------------------------------------------------------------
function _contract_one_mis_direct(g, w, r, code;
                                   usecuda::Bool, fallback_optimizer,
                                   count_eltype::DataType,
                                   weight_scale::Integer)
    usecuda && count_eltype === BigInt &&
        error("BigInt is not isbits and cannot be used on the GPU; pass " *
              "usecuda=false or count_eltype=:finitefield.")

    wc, scale_used = _scale_weights(w; weight_scale = (count_eltype === Float64 ? 1 : weight_scale))
    problem = isnothing(code) ?
        GenericTensorNetwork(IndependentSet(g, wc); optimizer = fallback_optimizer) :
        GenericTensorNetwork(IndependentSet(g, wc), code, Dict{Int, Int}())

    t = @elapsed begin
        raw = solve(problem, CountingMax(); T = count_eltype, usecuda = usecuda)
        result = Array(raw)[]
    end

    size_val = Float64(result.n) / scale_used + Float64(r)
    return (size = size_val,
            count = _to_bigint(result.c),
            runtime = t,
            primes = Int[], crt_history = BigInt[])
end

# ---------------------------------------------------------------------------
# Finitefield (CRT) path
#   Each iteration: contract with CountingTropical{Float64, Mod{p, Int32}}.
#   Combine per-prime counts via CRT until the BigInt result stabilises.
# ---------------------------------------------------------------------------
function _contract_one_mis_finitefield(g, w, r, code;
                                        usecuda::Bool, fallback_optimizer,
                                        weight_scale::Integer, max_crt_iter::Int,
                                        verbose::Bool)
    wc, scale_used = _scale_weights(w; weight_scale = weight_scale)
    problem = isnothing(code) ?
        GenericTensorNetwork(IndependentSet(g, wc); optimizer = fallback_optimizer) :
        GenericTensorNetwork(IndependentSet(g, wc), code, Dict{Int, Int}())

    primes_used = Int[]
    remainders  = Int[]
    sizes_p     = Float64[]
    crt_history = BigInt[]
    runtimes    = Float64[]

    # Walk primes downward from `typemax(Int32) = 2^31 - 1` (itself a
    # Mersenne prime, M31), exactly like `_contract_one_finitefield` for
    # spin glasses.  See that function for the rationale.
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
        size_p = Float64(res.n)
        c_modp = Int(value(res.c))
        push!(remainders, c_modp)
        push!(sizes_p,    size_p)
        push!(runtimes,   t)

        if k > 1 && !isapprox(size_p, sizes_p[1]; atol = 1e-6)
            @warn "[finitefield] size disagrees across primes ($(sizes_p)); " *
                  "treating $(sizes_p[1]) as the truth."
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

    size_val = sizes_p[1] / scale_used + Float64(r)
    count    = isempty(crt_history) ? BigInt(0) : crt_history[end]
    return (size = size_val, count = count, runtime = sum(runtimes),
            primes = primes_used, crt_history = crt_history)
end

# ----------------------------------------------------------------------
# Contract every slice + sum to maximum-IS degeneracy
# ----------------------------------------------------------------------

"""
    contract_mis_slices(subdir;
                         root               = SLICE_RESULTS_ROOT,
                         usecuda            = false,
                         atol               = 1e-6,
                         verbose            = false,
                         fallback_optimizer = TreeSA(),
                         count_eltype       = :finitefield,
                         weight_scale       = 1,
                         max_crt_iter       = 8,
                         ids                = nothing)

Iterate over every MIS slice stored in `joinpath(root, subdir)`
(`subdir` may also be absolute / pre-existing), contract it via the
saved `code`, shift its size by `r`, and combine the per-slice results
into the **exact** maximum (weighted) independent set size and
degeneracy of the original problem.

`count_eltype` defaults to `:finitefield`, the only mode that gives
**exact** counting on the GPU without overflow risk. See
[`contract_one_mis_slice`](@ref) for the full menu (`:finitefield`,
`Float64`, `Int128`, `BigInt`) and the reasoning behind each.

Returns a `NamedTuple`:

    (size, count, per_slice :: DataFrame, total_runtime)

`per_slice` carries one row per slice with `(id, r, size, count,
runtime, n_primes, last_prime, used_in_total)`. `n_primes` /
`last_prime` are populated for the `:finitefield` mode (number of
primes consumed before CRT stabilised, and the last prime used).
`used_in_total` flags rows contributing to the final degeneracy
(i.e. `|size_i + r_i - S_g| ≤ atol`).
"""
function contract_mis_slices(subdir::AbstractString;
                              root::AbstractString = SLICE_RESULTS_ROOT,
                              usecuda::Bool = false,
                              atol::Real = 1e-6,
                              verbose::Bool = false,
                              fallback_optimizer = TreeSA(),
                              count_eltype = :finitefield,
                              weight_scale::Integer = 1,
                              max_crt_iter::Int = 8,
                              ids = nothing)
    dirname = slice_results_dir(subdir; root = root)
    summary = list_mis_slices(dirname)
    slice_ids = ids === nothing ? Vector{Int}(summary.id) : Vector{Int}(ids)

    sizes      = Float64[]
    counts     = BigInt[]
    runtimes   = Float64[]
    rs         = Float64[]
    n_primes   = Int[]
    last_prime = Int[]

    total_t = @elapsed for (k, id) in enumerate(slice_ids)
        slice = load_mis_slice(dirname, id)
        res   = contract_one_mis_slice(slice; usecuda = usecuda,
                                       fallback_optimizer = fallback_optimizer,
                                       count_eltype = count_eltype,
                                       weight_scale = weight_scale,
                                       max_crt_iter = max_crt_iter,
                                       verbose = verbose)
        push!(sizes,      res.size)
        push!(counts,     res.count)
        push!(runtimes,   res.runtime)
        push!(rs,         Float64(slice.r))
        push!(n_primes,   length(res.primes))
        push!(last_prime, isempty(res.primes) ? 0 : res.primes[end])
        if verbose
            extra = isempty(res.primes) ? "" :
                "  (CRT: $(length(res.primes)) prime(s), last p = $(res.primes[end]))"
            @info "slice $id ($k/$(length(slice_ids))): nv=$(nv(slice.g)) ne=$(ne(slice.g)) " *
                  "size+r=$(res.size) count=$(res.count) t=$(round(res.runtime, digits=3))s" *
                  extra
        end
    end

    Sg = isempty(sizes) ? -Inf : maximum(sizes)
    used = falses(length(sizes))
    total_count = BigInt(0)
    @inbounds for i in eachindex(sizes)
        if abs(sizes[i] - Sg) ≤ atol
            total_count += counts[i]
            used[i] = true
        end
    end

    per_slice = DataFrame(id = slice_ids,
                          r = rs,
                          size = sizes,
                          count = counts,
                          runtime = runtimes,
                          n_primes = n_primes,
                          last_prime = last_prime,
                          used_in_total = used)

    return (size = Sg,
            count  = total_count,
            per_slice = per_slice,
            total_runtime = total_t)
end

# ----------------------------------------------------------------------
# Total-counting (CountingAll) — for the `mis_counting.jl` benchmark
# ----------------------------------------------------------------------

"""
    contract_one_mis_counting_slice(slice;
                                    usecuda            = false,
                                    fallback_optimizer = TreeSA(),
                                    count_eltype       = :finitefield,
                                    max_crt_iter       = 8,
                                    verbose            = false)
        -> (count, runtime, primes, crt_history)

Re-contract a single previously-loaded MIS slice for the **CountingAll**
property and return the slice's local **total IS count** of `g_i` (i.e.
the number of *every* independent set, not just the maximum-weight
ones) plus elapsed wall-time.

This is the per-slice routine matching `mis_counting.jl`'s `slice_bfs`:
each slice carries a sub-problem `(g_i, w_i)` whose IS configurations
form a **disjoint** subset of the original graph's IS, so the total IS
count of the original graph is recovered by summing this routine over
every slice — `r_i` is irrelevant here and is therefore ignored.

  * Trivial slice (`nv(g) == 0`) returns `count = BigInt(1)` (the empty
    IS).
  * Missing-code slices fall back to `fallback_optimizer`.

`count_eltype` selects the algebra used inside the contraction:

* `:finitefield` (default, GPU-safe, exact):
    Per-prime contraction with eltype `Mods.Mod{p, Int32}` for several
    primes `p` near `typemax(Int32)`; per-prime counts are CRT-combined
    into an exact `BigInt`. Stops once the recovered count is identical
    across two consecutive primes (or after `max_crt_iter`). For
    `CountingAll` the polynomial variable is just `1`, so each iteration
    contracts `one(Mod{p, Int32})` instead of `_x(...)`.

* `Float64` / `Int128` / `BigInt` (legacy direct paths):
    Defer to `solve(problem, CountingAll(); usecuda = usecuda)`.
    Note that `solve` ignores `T` for `CountingAll` and always falls
    back to its built-in `big_integer_solve(Int32, 100)` CRT loop, so
    these three values behave identically — they are kept for API
    parity with [`contract_one_mis_slice`](@ref).

Returned NamedTuple
-------------------
Always carries `(count::BigInt, runtime)`. The `:finitefield` mode
additionally fills:
* `primes :: Vector{Int}`         — primes used, in order of use
* `crt_history :: Vector{BigInt}` — running CRT result after each prime

The legacy paths leave both extra fields empty.

Weights play no role in the IS count (the constraint is purely
topological), so this routine accepts both `UnitWeight` and explicit
numeric weights without scaling.
"""
function contract_one_mis_counting_slice(slice; usecuda::Bool = false,
                                          fallback_optimizer = TreeSA(),
                                          count_eltype = :finitefield,
                                          max_crt_iter::Int = 8,
                                          verbose::Bool = false)
    g, w, code = slice.g, slice.weights, slice.code

    if nv(g) == 0
        return (count = BigInt(1), runtime = 0.0,
                primes = Int[], crt_history = BigInt[])
    end

    if count_eltype === :finitefield
        return _contract_one_mis_counting_finitefield(g, w, code;
                                                       usecuda = usecuda,
                                                       fallback_optimizer = fallback_optimizer,
                                                       max_crt_iter = max_crt_iter,
                                                       verbose = verbose)
    elseif count_eltype isa DataType
        return _contract_one_mis_counting_direct(g, w, code;
                                                  usecuda = usecuda,
                                                  fallback_optimizer = fallback_optimizer)
    else
        error("unknown count_eltype $(repr(count_eltype)); use :finitefield " *
              "(default, GPU-safe, exact) or one of Float64 / Int128 / BigInt.")
    end
end

# ---------------------------------------------------------------------------
# Direct (legacy) path: solve(problem, CountingAll(); usecuda).
# `solve` already runs its own internal big_integer_solve(Int32, 100) CRT
# loop for CountingAll, so the explicit `count_eltype` is informational
# only here.
# ---------------------------------------------------------------------------
function _contract_one_mis_counting_direct(g, w, code; usecuda::Bool, fallback_optimizer)
    # CountingAll() is **weight-independent** (it counts every IS, not the
    # MWIS-weighted polynomial), so we deliberately drop `w` and use the
    # default `UnitWeight(nv(g))` (eltype `Int`). This avoids the
    # `_pow(::Mod{Int32}, ::Float32)` ambiguity that otherwise blows up
    # the internal `big_integer_solve(Int32, 100)` CRT loop when callers
    # save slices with `Vector{Float32}` weights (which happens by
    # default — see `mis_counting.jl`).
    problem = isnothing(code) ?
        GenericTensorNetwork(IndependentSet(g); optimizer = fallback_optimizer) :
        GenericTensorNetwork(IndependentSet(g), code, Dict{Int, Int}())

    t = @elapsed begin
        raw = solve(problem, CountingAll(); usecuda = usecuda)
        result = Array(raw)[]
    end

    return (count = _to_bigint(result), runtime = t,
            primes = Int[], crt_history = BigInt[])
end

# ---------------------------------------------------------------------------
# Finitefield (CRT) path for CountingAll
#   Each iteration: contract with x = one(Mod{p, Int32}).  Combine
#   per-prime counts via CRT until the BigInt result stabilises.
# ---------------------------------------------------------------------------
function _contract_one_mis_counting_finitefield(g, w, code;
                                                 usecuda::Bool, fallback_optimizer,
                                                 max_crt_iter::Int, verbose::Bool)
    # CountingAll() is weight-independent (see notes in
    # `_contract_one_mis_counting_direct`); using `IndependentSet(g)` here
    # also lets `Mod{Int32, Int32}^Int` survive `generate_tensors`,
    # which would fail if `w :: Vector{Float32}` were forwarded.
    problem = isnothing(code) ?
        GenericTensorNetwork(IndependentSet(g); optimizer = fallback_optimizer) :
        GenericTensorNetwork(IndependentSet(g), code, Dict{Int, Int}())

    primes_used = Int[]
    remainders  = Int[]
    crt_history = BigInt[]
    runtimes    = Float64[]

    N = Int(typemax(Int32))
    last_x = BigInt(-1)
    for k in 1:max_crt_iter
        N <= 2 && error("[finitefield] ran out of Int32 primes after $(k-1) iteration(s)")
        p = Int(prevprime(N))
        N = p - 1
        push!(primes_used, p)

        Tp = Mod{Int32(p), Int32}
        x  = one(Tp)                  # CountingAll <=> evaluate at x = 1

        t = @elapsed begin
            raw = contractx(problem, x; usecuda = usecuda)
            res = Array(raw)[]
        end
        c_modp = Int(value(res))
        push!(remainders, c_modp)
        push!(runtimes,   t)

        x_now = _crt_combine(remainders, primes_used)
        push!(crt_history, x_now)
        verbose && @info "[finitefield CountingAll] iter $k  p=$p  count mod p = $c_modp  " *
                          "CRT = $x_now  t = $(round(t, digits=3))s"

        if x_now == last_x
            break
        end
        last_x = x_now
    end

    count = isempty(crt_history) ? BigInt(0) : crt_history[end]
    return (count = count, runtime = sum(runtimes),
            primes = primes_used, crt_history = crt_history)
end

"""
    contract_mis_counting_slices(subdir;
                                  root               = SLICE_RESULTS_ROOT,
                                  usecuda            = false,
                                  verbose            = false,
                                  fallback_optimizer = TreeSA(),
                                  count_eltype       = :finitefield,
                                  max_crt_iter       = 8,
                                  ids                = nothing)

Iterate over every MIS slice in `joinpath(root, subdir)`
(`subdir` may also be absolute / pre-existing), contract each one with
`CountingAll()` reusing the saved code, and **sum** the per-slice
counts into the **exact** total independent-set count of the original
graph.

`count_eltype` defaults to `:finitefield`, which contracts each slice
with `Mod{p, Int32}` arithmetic and CRT-combines the per-prime counts
into an exact `BigInt` (GPU-safe, no overflow risk). See
[`contract_one_mis_counting_slice`](@ref) for the full menu.

Returns a `NamedTuple`:

    (count, per_slice :: DataFrame, total_runtime)

`per_slice` carries one row per slice with `(id, r, count, runtime,
n_primes, last_prime)`. `n_primes` / `last_prime` are populated for the
`:finitefield` mode. Unlike [`contract_mis_slices`](@ref) there is no
`used_in_total` column — every slice contributes to the sum.
"""
function contract_mis_counting_slices(subdir::AbstractString;
                                       root::AbstractString = SLICE_RESULTS_ROOT,
                                       usecuda::Bool = false,
                                       verbose::Bool = false,
                                       fallback_optimizer = TreeSA(),
                                       count_eltype = :finitefield,
                                       max_crt_iter::Int = 8,
                                       ids = nothing)
    dirname = slice_results_dir(subdir; root = root)
    summary = list_mis_slices(dirname)
    slice_ids = ids === nothing ? Vector{Int}(summary.id) : Vector{Int}(ids)

    counts     = BigInt[]
    runtimes   = Float64[]
    rs         = Float64[]
    n_primes   = Int[]
    last_prime = Int[]

    total_t = @elapsed for (k, id) in enumerate(slice_ids)
        slice = load_mis_slice(dirname, id)
        res   = contract_one_mis_counting_slice(slice; usecuda = usecuda,
                                                fallback_optimizer = fallback_optimizer,
                                                count_eltype = count_eltype,
                                                max_crt_iter = max_crt_iter,
                                                verbose = verbose)
        push!(counts,     res.count)
        push!(runtimes,   res.runtime)
        push!(rs,         Float64(slice.r))
        push!(n_primes,   length(res.primes))
        push!(last_prime, isempty(res.primes) ? 0 : res.primes[end])
        if verbose
            extra = isempty(res.primes) ? "" :
                "  (CRT: $(length(res.primes)) prime(s), last p = $(res.primes[end]))"
            @info "slice $id ($k/$(length(slice_ids))): nv=$(nv(slice.g)) ne=$(ne(slice.g)) " *
                  "count=$(res.count) t=$(round(res.runtime, digits=3))s" * extra
        end
    end

    total_count = isempty(counts) ? BigInt(0) : sum(counts)

    per_slice = DataFrame(id = slice_ids,
                          r = rs,
                          count = counts,
                          runtime = runtimes,
                          n_primes = n_primes,
                          last_prime = last_prime)

    return (count = total_count,
            per_slice = per_slice,
            total_runtime = total_t)
end
