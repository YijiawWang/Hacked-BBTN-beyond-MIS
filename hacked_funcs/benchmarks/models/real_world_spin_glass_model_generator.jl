using Graphs, Random, Downloads, Printf

# ============================================================================
# Real-world spin-glass model generator. For each requested base graph the
# script
#
#   1. Downloads (or reuses a cached copy of) the SuiteSparse Matrix
#      Collection `.tar.gz` archive into `raw_data/<key>/`,
#   2. Parses the Matrix Market file into a `SimpleGraph` (1-based,
#      undirected, self-loops dropped, deduplicated),
#   3. Generates i.i.d. random ±1 couplings with `MersenneTwister(seed)`,
#   4. Writes a `.model` file in the exact format produced by
#      `spin_glass_model_generator.jl` and consumed by
#      `read_spin_glass_model` in
#      `hacked_funcs/benchmarks/spin_glass_ground_counting.jl`.
#
# Matches the on-disk layout of the existing `cora_citation`,
# `openflights_routes`, and `power_grid_western_us` real-world models:
#
#   models/
#   ├── raw_data/<key>/<key>.mtx                # cached source MTX
#   └── spin_glass_models/real-world/
#       └── spin_glass_J±1_<key>_h=<h>_seed=<seed>.model
#
# Run from `beyond_mis/hacked_funcs/benchmarks/models/`. The default field
# `h = 0.5` matches the existing real-world models (it is *not* baked into
# the file body — it is recorded in the header and applied at consume
# time, exactly like for the existing real-world models).
# ============================================================================

const RAW_DIR    = "raw_data"
const OUTPUT_DIR = joinpath("spin_glass_models", "real-world")

const _USAGE = """
Usage:

    julia real_world_spin_glass_model_generator.jl
        [--networks=<key>[,<key>...]]   # default: all (road_minnesota,netscience,bcspwr09,pajek_yeast)
        [--seeds=lo[:hi]]               # default: 1:1
        [--h=<float>]                   # default: 0.5
        [--k-core=<int>]                # default: 0 (no peeling)
                                        # If > 0, take the K-core of the loaded
                                        # graph (drop vertices iteratively until
                                        # min degree >= K) and renumber 1..n.
                                        # Use this to dial sc / tw down on
                                        # otherwise-too-dense networks.
        [--drop-top-hubs=<int>]         # default: 0 (no hub removal)
                                        # Remove the N highest-degree vertices
                                        # from the loaded graph, then keep
                                        # whatever remains. Useful for
                                        # hub-and-spoke networks (airline
                                        # routes, dictionary cross-refs)
                                        # where a handful of super-nodes blow
                                        # up max-degree and tw.
        [--keep-lcc]                    # after k-core / hub-drop, keep only
                                        # the largest connected component.
        [--raw-dir=<path>]              # default: $(RAW_DIR)
        [--output-dir=<path>]           # default: $(OUTPUT_DIR)
        [--no-download]                 # error out if a cached MTX is missing
        [--force]                       # re-download / overwrite existing models

Available <key>s:
    road_minnesota   Minnesota road network (Gleich, MatlabBGL via SuiteSparse)
                       2,642 V / 3,303 E / max deg 5     (very low tw)
    netscience       Coauthorship of network scientists (Newman 2006)
                       1,589 V / 2,742 E / max deg 34    (low tw, fragmented)
    bcspwr09         Western US power grid (Dembart-Lewis 1981, HB)
                       1,723 V / 2,394 E / max deg 14    (very low tw)
    bus_494          IEEE 494-bus AC power-flow admittance (HB)
    bus_662          IEEE 662-bus AC power-flow admittance (HB)
    bus_685          IEEE 685-bus AC power-flow admittance (HB)
    bus_1138         IEEE 1138-bus AC power-flow admittance (HB)
    pajek_yeast      Yeast PPI (Bu et al. 2003, Pajek)
                       2,361 V / 6,646 E / max deg 64    (degeneracy 10; tw <~ 30)
    odlis            ODLIS dictionary cross-references (Reitz 2002, Pajek)
                       2,909 V / ~9,100 E / mid-density  (mid-tw candidate)
"""


# ---------------------------------------------------------------------------
# Network registry
# ---------------------------------------------------------------------------

struct NetworkSpec
    key                ::String                                # output filename token
    suitesparse_group  ::String
    suitesparse_name   ::String
    archive_member     ::String                                # `.mtx` file inside the tarball
    expected_vertices  ::Int                                   # for sanity warnings only
    expected_edges     ::Int                                   # for sanity warnings only
    description        ::String
    source_url         ::String
    citation           ::String
end

const NETWORK_REGISTRY = Dict{String,NetworkSpec}(
    # ----- "low-tw" infrastructure / sparse / fragmented -------------------
    "road_minnesota" => NetworkSpec(
        "road_minnesota",
        "Gleich",
        "minnesota",
        "minnesota.mtx",
        2642,
        3303,
        "Minnesota road network (D. Gleich, MatlabBGL graph library)",
        "https://sparse.tamu.edu/Gleich/minnesota",
        "D. Gleich, MatlabBGL; redistributed via the SuiteSparse Matrix Collection (T. Davis et al., 2010).",
    ),
    "netscience" => NetworkSpec(
        "netscience",
        "Newman",
        "netscience",
        "netscience.mtx",
        1589,
        2742,
        "Coauthorship network of scientists working on network theory and experiment",
        "https://sparse.tamu.edu/Newman/netscience",
        "M. E. J. Newman, \"Finding community structure in networks using the eigenvectors of matrices\", Phys. Rev. E 74, 036104 (2006).",
    ),
    "bcspwr09" => NetworkSpec(
        "bcspwr09",
        "HB",
        "bcspwr09",
        "bcspwr09.mtx",
        1723,
        2394,
        "Western US power grid admittance structure (Harwell-Boeing collection)",
        "https://sparse.tamu.edu/HB/bcspwr09",
        "B. Dembart and J. Lewis, 1981; Harwell-Boeing sparse matrix test collection (I. Duff, R. Grimes, J. Lewis).",
    ),
    # Classic Harwell-Boeing AC power-flow admittance matrices (per-bus
    # impedance + line connectivity). Slightly denser than the pure
    # `bcspwr` topology because they record transformer / generator
    # couplings on top of the line graph.
    "bus_494" => NetworkSpec(
        "bus_494",
        "HB",
        "494_bus",
        "494_bus.mtx",
        494,
        1666,                                   # MTX nnz; will dedup down
        "IEEE 494-bus AC power-flow admittance matrix (Harwell-Boeing)",
        "https://sparse.tamu.edu/HB/494_bus",
        "Harwell-Boeing sparse matrix test collection; F. L. Alvarado, IEEE 494-bus power flow test case.",
    ),
    "bus_662" => NetworkSpec(
        "bus_662",
        "HB",
        "662_bus",
        "662_bus.mtx",
        662,
        2474,
        "IEEE 662-bus AC power-flow admittance matrix (Harwell-Boeing)",
        "https://sparse.tamu.edu/HB/662_bus",
        "Harwell-Boeing sparse matrix test collection; F. L. Alvarado, IEEE 662-bus power flow test case.",
    ),
    "bus_685" => NetworkSpec(
        "bus_685",
        "HB",
        "685_bus",
        "685_bus.mtx",
        685,
        3249,
        "IEEE 685-bus AC power-flow admittance matrix (Harwell-Boeing)",
        "https://sparse.tamu.edu/HB/685_bus",
        "Harwell-Boeing sparse matrix test collection; F. L. Alvarado, IEEE 685-bus power flow test case.",
    ),
    "bus_1138" => NetworkSpec(
        "bus_1138",
        "HB",
        "1138_bus",
        "1138_bus.mtx",
        1138,
        4054,
        "IEEE 1138-bus AC power-flow admittance matrix (Harwell-Boeing)",
        "https://sparse.tamu.edu/HB/1138_bus",
        "Harwell-Boeing sparse matrix test collection; F. L. Alvarado, IEEE 1138-bus power flow test case.",
    ),

    # ---- Politics / social ------------------------------------------------
    # Adamic & Glance's directed political-blog citation network around the
    # 2004 US election. Hub-rich (max deg ~351), so likely needs
    # `--drop-top-hubs` to be useful for spin-glass contraction.
    "polblogs" => NetworkSpec(
        "polblogs",
        "Newman",
        "polblogs",
        "polblogs.mtx",
        1490,
        19025,
        "Political blogosphere snapshot (Feb 2005, US 2004 election context)",
        "https://sparse.tamu.edu/Newman/polblogs",
        "L. A. Adamic and N. Glance, \"The political blogosphere and the 2004 US Election\", WWW 2005 Workshop on Weblogging Ecosystem.",
    ),

    # ---- Citation networks ------------------------------------------------
    # Garfield's SciMet citation network: papers from / citing the journal
    # \"Scientometrics\", 1978-2000.
    "scimet" => NetworkSpec(
        "scimet",
        "Pajek",
        "SciMet",
        "SciMet.mtx",
        3084,
        10413,
        "SciMet citation network: papers from/citing Scientometrics 1978-2000",
        "https://sparse.tamu.edu/Pajek/SciMet",
        "E. Garfield, Scientometrics citation data 1978-2000; Pajek redistribution by V. Batagelj (2002).",
    ),

    # ---- Engineering meshes (planar / 2D FEM, real-world meaningful) ------
    # Alan George's L-shape thermal-conduction FEM mesh family (1978).
    # 2D triangulated, max deg ~6, treewidth ~ sqrt(n). The lshp1270 /
    # lshp1561 / lshp1882 entries cover the n in [1.2k, 1.9k] band and
    # bracket the user's target tw range [30, 40] tightly:
    #   n=1270 -> tw ~ 36   (sqrt(1270) = 35.6)
    #   n=1561 -> tw ~ 39   (sqrt(1561) = 39.5)
    #   n=1882 -> tw ~ 43   (sqrt(1882) = 43.4, TreeSA-confirmed sc=45)
    "lshp1270" => NetworkSpec(
        "lshp1270",
        "HB",
        "lshp1270",
        "lshp1270.mtx",
        1270,
        8668,
        "Alan George's L-shape thermal-conduction 2D FEM mesh (n=1270, 1978)",
        "https://sparse.tamu.edu/HB/lshp1270",
        "A. George, \"An automatic nested dissection algorithm for irregular finite element problems\", SIAM J. Numer. Anal. 15(5), 1053-1069 (1978); Harwell-Boeing redistribution.",
    ),
    "lshp1561" => NetworkSpec(
        "lshp1561",
        "HB",
        "lshp1561",
        "lshp1561.mtx",
        1561,
        10681,
        "Alan George's L-shape thermal-conduction 2D FEM mesh (n=1561, 1978)",
        "https://sparse.tamu.edu/HB/lshp1561",
        "A. George, \"An automatic nested dissection algorithm for irregular finite element problems\", SIAM J. Numer. Anal. 15(5), 1053-1069 (1978); Harwell-Boeing redistribution.",
    ),
    "lshp1882" => NetworkSpec(
        "lshp1882",
        "HB",
        "lshp1882",
        "lshp1882.mtx",
        1882,
        12904,
        "Alan George's L-shape thermal-conduction 2D FEM mesh (n=1882, 1978)",
        "https://sparse.tamu.edu/HB/lshp1882",
        "A. George, \"An automatic nested dissection algorithm for irregular finite element problems\", SIAM J. Numer. Anal. 15(5), 1053-1069 (1978); Harwell-Boeing redistribution.",
    ),
    # US Navy DTNSRDC structural connection table at n=1242. dwt_2680
    # (TreeSA sc=44) and dwt_1242 should bracket [30, 40]:
    #   n=1242 -> tw ~ 35   (sqrt(1242) = 35.2)
    "dwt_1242" => NetworkSpec(
        "dwt_1242",
        "HB",
        "dwt_1242",
        "dwt_1242.mtx",
        1242,
        10426,
        "DTNSRDC structural connection table (US Navy, 1980, n=1242)",
        "https://sparse.tamu.edu/HB/dwt_1242",
        "G. Everstine and D. Taylor, David Taylor Naval Ship R&D Center connection table (1980); Harwell-Boeing redistribution.",
    ),
    # DTNSRDC (David Taylor Naval Ship R&D Center) connection table -- real
    # ship/submarine structural connectivity from the US Navy in 1980.
    "dwt_2680" => NetworkSpec(
        "dwt_2680",
        "HB",
        "dwt_2680",
        "dwt_2680.mtx",
        2680,
        25026,
        "DTNSRDC structural connection table (US Navy, 1980)",
        "https://sparse.tamu.edu/HB/dwt_2680",
        "G. Everstine and D. Taylor, David Taylor Naval Ship R&D Center connection table (1980); Harwell-Boeing redistribution.",
    ),
    # Boeing ore-car stiffness matrix -- real industrial structural problem
    # but DOF coupling makes it relatively dense (avg deg ~22).
    "bcsstk11" => NetworkSpec(
        "bcsstk11",
        "HB",
        "bcsstk11",
        "bcsstk11.mtx",
        1473,
        34241,
        "Boeing ore-car stiffness matrix (lumped masses, 1982)",
        "https://sparse.tamu.edu/HB/bcsstk11",
        "J. Lewis (Boeing), 1982; Harwell-Boeing sparse matrix test collection.",
    ),

    # ----- "mid-/high-tw" candidates (avg deg ~5-15, more clustered) -------
    # Maniu et al. 2019 (arXiv:1901.06862) measured tw of "Yeast"
    # (2284 nodes, 6646 edges -- a slightly different snapshot of the same
    # Bu et al. 2003 PPI) at lower=54, upper=255. The Pajek/yeast variant
    # below has 2361 nodes / 13828 edges and is similarly hub-rich, so its
    # full sc is expected to be > 50; combine with `--k-core=K` to dial
    # density down.
    "pajek_yeast" => NetworkSpec(
        "pajek_yeast",
        "Pajek",
        "yeast",
        "yeast.mtx",
        2361,
        13828,
        "Yeast protein-protein interaction network (Bu et al. 2003, Pajek)",
        "https://sparse.tamu.edu/Pajek/yeast",
        "D. Bu et al., \"Topological structure analysis of the protein-protein interaction network in budding yeast\", Nucleic Acids Research 31(9), 2443-2450 (2003); Pajek redistribution by V. Batagelj.",
    ),
    # Joan Reitz's ODLIS (Online Dictionary for Library and Information
    # Science) cross-reference network. Each entry is a vertex and a
    # directed edge (u->v) means the definition of u references term v;
    # we treat it as undirected for the spin-glass model. avg deg ~6 with
    # bounded but non-trivial hubs -- a reasonable mid-density "real
    # informational graph" candidate.
    "odlis" => NetworkSpec(
        "odlis",
        "Pajek",
        "ODLIS",
        "ODLIS.mtx",
        2909,
        18246,                                     # MTX nnz; undirected edge count comes out smaller after dedup
        "ODLIS online dictionary of library & information science cross-references",
        "https://sparse.tamu.edu/Pajek/ODLIS",
        "J. M. Reitz, ODLIS (Online Dictionary for Library and Information Science), 2002; Pajek redistribution by V. Batagelj and A. Mrvar (2006).",
    ),
    # The Pajek `EVA` corporate ownership graph (4475 vertices) is too
    # large; the KONECT Hamsterster (1858 nodes, max deg 273) would be a
    # natural denser candidate but is not on the SuiteSparse mirror, so
    # we skip it here.
)


# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------

function _parse_int_range(s::AbstractString)
    parts = split(s, ':')
    if length(parts) == 1
        v = parse(Int, parts[1])
        return v:v
    elseif length(parts) == 2
        return parse(Int, parts[1]):parse(Int, parts[2])
    else
        error("invalid range spec: $s (expected lo or lo:hi)")
    end
end

function _parse_args(args)
    networks       = collect(keys(NETWORK_REGISTRY))    # default: all
    seeds          = 1:1
    h              = 0.5
    k_core         = 0
    drop_top_hubs  = 0
    keep_lcc       = false
    raw_dir        = RAW_DIR
    output_dir     = OUTPUT_DIR
    no_download    = false
    force          = false

    networks_set_explicitly = false

    for a in args
        if startswith(a, "--networks=")
            spec = String(split(a, "="; limit = 2)[2])
            networks = String.(strip.(split(spec, ',')))
            filter!(!isempty, networks)
            networks_set_explicitly = true
        elseif startswith(a, "--seeds=")
            seeds = _parse_int_range(split(a, "="; limit = 2)[2])
        elseif startswith(a, "--h=")
            h = parse(Float64, split(a, "="; limit = 2)[2])
        elseif startswith(a, "--k-core=")
            k_core = parse(Int, split(a, "="; limit = 2)[2])
            k_core >= 0 || error("--k-core must be >= 0 (got $k_core)")
        elseif startswith(a, "--drop-top-hubs=")
            drop_top_hubs = parse(Int, split(a, "="; limit = 2)[2])
            drop_top_hubs >= 0 || error("--drop-top-hubs must be >= 0 (got $drop_top_hubs)")
        elseif a == "--keep-lcc"
            keep_lcc = true
        elseif startswith(a, "--raw-dir=")
            raw_dir = String(split(a, "="; limit = 2)[2])
        elseif startswith(a, "--output-dir=")
            output_dir = String(split(a, "="; limit = 2)[2])
        elseif a == "--no-download"
            no_download = true
        elseif a == "--force"
            force = true
        elseif a in ("-h", "--help")
            println(_USAGE)
            exit(0)
        else
            error("unknown / unsupported argument: $a\n$_USAGE")
        end
    end

    # Stable order: respect the user's order if explicit, else use a
    # deterministic ordering for the "all" default.
    if !networks_set_explicitly
        networks = ["road_minnesota", "netscience", "bcspwr09", "pajek_yeast"]
    end

    known_keys_str = join(sort(collect(keys(NETWORK_REGISTRY))), ", ")
    for key in networks
        haskey(NETWORK_REGISTRY, key) ||
            error("unknown network key: $key (known: $known_keys_str)\n$_USAGE")
    end

    return (; networks, seeds, h, k_core, drop_top_hubs, keep_lcc,
              raw_dir, output_dir, no_download, force)
end


# ---------------------------------------------------------------------------
# Download + extract a single SuiteSparse MTX archive
# ---------------------------------------------------------------------------

# SuiteSparse moved off the Heroku-hosted mirror; the TAMU and UFL mirrors
# are the current canonical hosts. Try them in order and use the first
# reachable one.
const _SUITESPARSE_MIRRORS = [
    "https://sparse.tamu.edu/MM",
    "https://www.cise.ufl.edu/research/sparse/MM",
    "https://suitesparse-collection-website.herokuapp.com/MM",
]

function _archive_urls(spec::NetworkSpec)
    return ["$mirror/$(spec.suitesparse_group)/$(spec.suitesparse_name).tar.gz"
            for mirror in _SUITESPARSE_MIRRORS]
end

"""
    _try_download(urls, dest) -> url

Try each URL in turn. Returns the URL that succeeded, or rethrows the
last error if every mirror fails.
"""
function _try_download(urls::Vector{String}, dest::AbstractString)
    last_err = nothing
    for url in urls
        try
            println("[download] $url\n           -> $dest")
            Downloads.download(url, dest; timeout = 60.0)
            return url
        catch err
            @warn "mirror failed; trying next" url=url err=sprint(showerror, err)
            last_err = err
        end
    end
    last_err === nothing && error("no mirrors configured")
    throw(last_err)
end

"""
    ensure_mtx(spec, raw_dir; allow_download, force)

Return the absolute path to `<raw_dir>/<spec.key>/<spec.archive_member>`,
downloading and extracting the SuiteSparse `.tar.gz` archive on demand.
Errors out if `allow_download` is `false` and the file is not already
cached.
"""
function ensure_mtx(spec::NetworkSpec, raw_dir::AbstractString;
                    allow_download::Bool, force::Bool)
    target_dir = joinpath(raw_dir, spec.key)
    target    = joinpath(target_dir, spec.archive_member)
    mkpath(target_dir)

    if isfile(target) && !force
        return abspath(target)
    end

    if !allow_download
        urls_str = join(_archive_urls(spec), "\n  ")
        error("Cached MTX missing: $target\n" *
              "Re-run without --no-download, or place the file manually.\n" *
              "Mirrors:\n  $urls_str")
    end

    tar_gz = joinpath(target_dir, "$(spec.suitesparse_name).tar.gz")
    _try_download(_archive_urls(spec), tar_gz)

    # SuiteSparse archives extract to `<suitesparse_name>/<suitesparse_name>.mtx`
    # (plus auxiliary files such as `<name>_coord.mtx` for Gleich/minnesota).
    # Extract into `target_dir` and then move the relevant `.mtx` files
    # one level up so they live at `<target_dir>/<spec.archive_member>`.
    extract_root = joinpath(target_dir, "_extract")
    mkpath(extract_root)
    try
        run(`tar -xzf $tar_gz -C $extract_root`)
    catch err
        error("Failed to extract $tar_gz: $err")
    end

    inner_dir = joinpath(extract_root, spec.suitesparse_name)
    if !isdir(inner_dir)
        # Some archives don't have a wrapping directory; fall back to the
        # extract root itself.
        inner_dir = extract_root
    end

    extracted = joinpath(inner_dir, spec.archive_member)
    if !isfile(extracted)
        # As a last resort, scan the extract tree for *.mtx files.
        candidates = String[]
        for (root, _, files) in walkdir(extract_root)
            for f in files
                endswith(f, ".mtx") && push!(candidates, joinpath(root, f))
            end
        end
        if length(candidates) == 1
            extracted = candidates[1]
        elseif !isempty(candidates)
            # Prefer the one whose basename matches the requested member.
            match_idx = findfirst(p -> basename(p) == spec.archive_member, candidates)
            if match_idx !== nothing
                extracted = candidates[match_idx]
            else
                found_str = join(basename.(candidates), ", ")
                error("Could not locate $(spec.archive_member) inside $tar_gz; " *
                      "found: $found_str")
            end
        else
            error("No .mtx file found inside $tar_gz after extraction.")
        end
    end

    cp(extracted, target; force = true)

    # Also copy any auxiliary .mtx files (e.g. coordinates) for archival
    # purposes; the parser ignores them.
    for (root, _, files) in walkdir(inner_dir)
        for f in files
            if endswith(f, ".mtx") && f != spec.archive_member
                aux_dst = joinpath(target_dir, f)
                isfile(aux_dst) || cp(joinpath(root, f), aux_dst; force = false)
            end
        end
    end

    rm(extract_root; recursive = true, force = true)
    rm(tar_gz; force = true)

    println("[download] cached -> $target")
    return abspath(target)
end


# ---------------------------------------------------------------------------
# Matrix Market parser (coordinate format only)
# ---------------------------------------------------------------------------

"""
    parse_mtx_coordinate(path) -> (n, edges)

Read an MM coordinate-format file and return `(n, edges)` where `n` is
`max(nrows, ncols)` and `edges` is a `Vector{Tuple{Int,Int}}` of
deduplicated, self-loop-free, 1-based undirected edges. The header is
expected to be `%%MatrixMarket matrix coordinate <pattern|real|integer|complex> <symmetric|general|...>`.

For symmetric matrices we mirror nothing (an undirected `SimpleGraph` is
itself symmetric); for `general` we de-duplicate so each unordered pair
shows up at most once.
"""
function parse_mtx_coordinate(path::AbstractString)
    open(path, "r") do io
        # Header line
        header = ""
        while !eof(io)
            line = readline(io)
            isempty(line) && continue
            header = line
            break
        end
        startswith(header, "%%MatrixMarket") ||
            error("Not a Matrix Market file: $path (first line: $header)")
        toks = split(header)
        length(toks) >= 5 ||
            error("Malformed MatrixMarket header in $path: $header")
        toks[2] == "matrix" ||
            error("Expected `matrix` object in $path, got: $(toks[2])")
        toks[3] == "coordinate" ||
            error("Only coordinate format is supported, got: $(toks[3]) in $path")
        field   = toks[4]                              # pattern | real | integer | complex
        symtype = toks[5]                              # symmetric | general | skew-symmetric | hermitian

        has_value = field != "pattern"
        symmetric = symtype in ("symmetric", "skew-symmetric", "hermitian")

        # Skip comment / blank lines, find size line
        nrows = ncols = nnz = 0
        while !eof(io)
            line = strip(readline(io))
            (isempty(line) || startswith(line, "%")) && continue
            sz = split(line)
            length(sz) >= 3 ||
                error("Malformed size line in $path: $line")
            nrows = parse(Int, sz[1])
            ncols = parse(Int, sz[2])
            nnz   = parse(Int, sz[3])
            break
        end

        seen   = Set{Tuple{Int,Int}}()
        edges  = Vector{Tuple{Int,Int}}()
        sizehint!(edges, nnz)

        read_count = 0
        while !eof(io) && read_count < nnz
            line = strip(readline(io))
            (isempty(line) || startswith(line, "%")) && continue
            parts = split(line)
            length(parts) >= 2 ||
                error("Malformed entry in $path: $line")
            i = parse(Int, parts[1])
            j = parse(Int, parts[2])
            if has_value
                length(parts) >= 3 ||
                    error("Expected value column in $path: $line")
            end
            read_count += 1
            i == j && continue                        # drop self-loops
            key = i < j ? (i, j) : (j, i)
            if !(key in seen)
                push!(seen, key)
                push!(edges, key)
            end
        end
        read_count == nnz ||
            error("Truncated MTX: read $read_count entries, header promised $nnz")

        # Even for `general` matrices we treat (i,j) and (j,i) as the same
        # undirected edge, so the dedup above already covers the asymmetric
        # case. `symmetric` is handled identically.
        n = max(nrows, ncols)
        return n, edges
    end
end


# ---------------------------------------------------------------------------
# Build SimpleGraph + ±1 weights, write out `.model`
# ---------------------------------------------------------------------------

function _build_graph(n::Int, edge_list::Vector{Tuple{Int,Int}})
    g = SimpleGraph(n)
    for (u, v) in edge_list
        add_edge!(g, u, v)
    end
    return g
end

function _max_degree(g::SimpleGraph)
    return isempty(vertices(g)) ? 0 : maximum(degree(g))
end

function _model_filename(spec::NetworkSpec, h::Float64, seed::Int;
                         k_core::Int = 0,
                         drop_top_hubs::Int = 0,
                         keep_lcc::Bool = false)
    parts = String[]
    drop_top_hubs > 0 && push!(parts, "drop$(drop_top_hubs)")
    k_core         > 0 && push!(parts, "kcore=$(k_core)")
    keep_lcc           && push!(parts, "lcc")
    suffix = isempty(parts) ? "" : "_" * join(parts, "_")
    return "spin_glass_J±1_$(spec.key)$(suffix)_h=$(h)_seed=$(seed).model"
end

"""
    _peel_k_core(graph, k) -> (subgraph, vmap)

Iteratively drop vertices of degree < k until no such vertex remains
(this is the standard k-core peel). Returns the induced subgraph
relabelled to `1:nv(subgraph)` together with `vmap[i] = original_id`
for each surviving vertex `i` in the new graph. When `k <= 0` returns
the input untouched (and `vmap = 1:nv(graph)`).
"""
function _peel_k_core(graph::SimpleGraph, k::Int)
    if k <= 0
        return graph, collect(1:nv(graph))
    end
    cn  = Graphs.core_number(graph)
    keep = findall(c -> c >= k, cn)
    isempty(keep) &&
        error("k-core $k peel left zero vertices; pick a smaller --k-core")
    sub, vmap = induced_subgraph(graph, keep)
    return sub, vmap
end

"""
    _drop_top_hubs(graph, n) -> (subgraph, vmap)

Remove the `n` highest-degree vertices (ties broken by vertex id) and
return the induced subgraph on the remaining vertices, relabelled to
`1:nv(subgraph)`. Useful for hub-and-spoke real-world networks where a
handful of super-nodes (e.g. JFK / ORD in OpenFlights, "library" /
"dictionary" in ODLIS) inflate max degree and tw without adding much
combinatorial structure.
"""
function _drop_top_hubs(graph::SimpleGraph, n::Int)
    if n <= 0
        return graph, collect(1:nv(graph))
    end
    n >= nv(graph) &&
        error("--drop-top-hubs $n removes all $(nv(graph)) vertices")
    degs = degree(graph)
    # Sort vertex ids by descending degree; drop the first `n`.
    order  = sortperm(degs; rev = true)
    drop   = Set(order[1:n])
    keep   = Int[v for v in 1:nv(graph) if v ∉ drop]
    sub, vmap = induced_subgraph(graph, keep)
    return sub, vmap
end

"""
    _keep_largest_cc(graph) -> (subgraph, vmap)

Keep only the largest connected component, relabelled to
`1:nv(subgraph)`. `vmap[i]` gives the original id of new vertex `i`.
"""
function _keep_largest_cc(graph::SimpleGraph)
    nv(graph) == 0 && return graph, Int[]
    ccs = connected_components(graph)
    isempty(ccs) && return graph, collect(1:nv(graph))
    largest = ccs[argmax(length.(ccs))]
    sub, vmap = induced_subgraph(graph, largest)
    return sub, vmap
end

"""
    write_real_world_model(filepath, spec, graph, edge_weights, h, seed; k_core, full_nv, full_ne)

Write `graph` plus i.i.d. ±1 `edge_weights` into a `.model` file in the
same format `spin_glass_model_generator.jl` produces. `h` is recorded in
the header (and applied at consume-time by
`spin_glass_ground_counting.jl`); `seed` is the RNG seed used to draw the
edge signs. When `k_core > 0` the header also records the k-core
parameter and the pre-peel `(full_nv, full_ne)` so the provenance is
unambiguous.
"""
function write_real_world_model(filepath::AbstractString,
                                spec::NetworkSpec,
                                graph::SimpleGraph,
                                edge_weights::AbstractDict,
                                h::Float64,
                                seed::Int;
                                k_core::Int = 0,
                                drop_top_hubs::Int = 0,
                                keep_lcc::Bool = false,
                                full_nv::Int = nv(graph),
                                full_ne::Int = ne(graph))
    open(filepath, "w") do io
        println(io, "# Real-world spin glass: J = ±1 i.i.d., uniform field h = $(h)")
        println(io, "# Base graph: $(spec.key) ($(spec.description))")
        println(io, "# SuiteSparse: $(spec.suitesparse_group)/$(spec.suitesparse_name)")
        println(io, "# Source URL : $(spec.source_url)")
        println(io, "# Citation   : $(spec.citation)")
        println(io, "# Source file: $(joinpath(RAW_DIR, spec.key, spec.archive_member))")
        any_pp = drop_top_hubs > 0 || k_core > 0 || keep_lcc
        if any_pp
            steps = String[]
            drop_top_hubs > 0 && push!(steps, "drop top $(drop_top_hubs) hubs")
            k_core         > 0 && push!(steps, "$(k_core)-core peel")
            keep_lcc           && push!(steps, "keep LCC")
            println(io, "# Preprocess  : ", join(steps, " -> "),
                       " (renumbered 1..$(nv(graph)))")
            println(io, "# Full graph  : $(full_nv) vertices, $(full_ne) edges before preprocessing")
        end
        println(io, "# Vertices    : $(nv(graph))")
        println(io, "# Edges       : $(ne(graph))  (self-loops dropped, undirected, deduplicated)")
        println(io, "# Max degree  : $(_max_degree(graph))")
        println(io, "seed = $seed")
        println(io, "h = $h    # uniform external field (applied at consume-time)")
        println(io, "J_type = ±1")
        if drop_top_hubs > 0
            println(io, "drop_top_hubs = $drop_top_hubs")
        end
        if k_core > 0
            println(io, "k_core = $k_core")
        end
        if keep_lcc
            println(io, "keep_lcc = true")
        end
        println(io, "vertices: ", nv(graph))
        println(io, "edges: ", ne(graph))
        println(io)
        println(io, "edges_with_weights:")
        for e in edges(graph)
            s, d = src(e), dst(e)
            w = edge_weights[(min(s, d), max(s, d))]
            println(io, "$s $d $w")
        end
    end
    println("[generate] $filepath")
end

function generate_one(spec::NetworkSpec, mtx_path::AbstractString;
                      seed::Int, h::Float64, output_dir::AbstractString,
                      force::Bool, k_core::Int = 0,
                      drop_top_hubs::Int = 0, keep_lcc::Bool = false)
    out_path = joinpath(output_dir,
                        _model_filename(spec, h, seed;
                                        k_core = k_core,
                                        drop_top_hubs = drop_top_hubs,
                                        keep_lcc = keep_lcc))
    if isfile(out_path) && !force
        println("[generate] skip (exists): $out_path  (pass --force to overwrite)")
        return out_path
    end

    n, edge_list = parse_mtx_coordinate(mtx_path)
    full_graph   = _build_graph(n, edge_list)

    if nv(full_graph) != spec.expected_vertices
        @warn "Vertex count mismatch" key=spec.key got=nv(full_graph) expected=spec.expected_vertices
    end
    if ne(full_graph) != spec.expected_edges
        @info "Edge count differs from registry expectation" key=spec.key got=ne(full_graph) expected=spec.expected_edges
    end

    full_nv = nv(full_graph)
    full_ne = ne(full_graph)

    # Apply preprocessing in order: drop hubs -> k-core peel -> keep LCC.
    graph = full_graph
    if drop_top_hubs > 0
        graph, _ = _drop_top_hubs(graph, drop_top_hubs)
        @printf("[drop-top-hubs %d] -> %d vertices, %d edges, max_deg=%d\n",
                drop_top_hubs, nv(graph), ne(graph), _max_degree(graph))
    end
    if k_core > 0
        graph, _ = _peel_k_core(graph, k_core)
        @printf("[k-core %d]        -> %d vertices, %d edges, max_deg=%d\n",
                k_core, nv(graph), ne(graph), _max_degree(graph))
    end
    if keep_lcc
        graph, _ = _keep_largest_cc(graph)
        @printf("[keep-lcc]         -> %d vertices, %d edges, max_deg=%d\n",
                nv(graph), ne(graph), _max_degree(graph))
    end

    rng = MersenneTwister(seed)
    edge_weights = Dict{Tuple{Int,Int}, Int}()
    for e in edges(graph)
        edge_weights[(min(src(e), dst(e)), max(src(e), dst(e)))] = rand(rng, (-1, 1))
    end

    mkpath(output_dir)
    write_real_world_model(out_path, spec, graph, edge_weights, h, seed;
                           k_core = k_core,
                           drop_top_hubs = drop_top_hubs,
                           keep_lcc = keep_lcc,
                           full_nv = full_nv,
                           full_ne = full_ne)
    @printf("           vertices = %d  edges = %d  max_degree = %d\n",
            nv(graph), ne(graph), _max_degree(graph))
    return out_path
end


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

function main(args)
    cfg = _parse_args(args)
    mkpath(cfg.raw_dir)
    mkpath(cfg.output_dir)

    networks_str = join(cfg.networks, ", ")
    println("="^72)
    println("[real_world_spin_glass_model_generator]")
    println("  networks       : $networks_str")
    println("  seeds          : $(collect(cfg.seeds))")
    println("  h              : $(cfg.h)")
    println("  drop_top_hubs  : $(cfg.drop_top_hubs)")
    println("  k_core         : $(cfg.k_core)")
    println("  keep_lcc       : $(cfg.keep_lcc)")
    println("  raw_dir        : $(cfg.raw_dir)")
    println("  output_dir     : $(cfg.output_dir)")
    println("  download       : $(!cfg.no_download)")
    println("  force          : $(cfg.force)")
    println("="^72)

    written = String[]
    for key in cfg.networks
        spec = NETWORK_REGISTRY[key]
        println("\n--- $(spec.key) ($(spec.description)) ---")
        mtx = ensure_mtx(spec, cfg.raw_dir;
                         allow_download = !cfg.no_download,
                         force          = cfg.force)
        for seed in cfg.seeds
            push!(written,
                  generate_one(spec, mtx; seed = seed, h = cfg.h,
                               output_dir = cfg.output_dir,
                               force = cfg.force,
                               k_core = cfg.k_core,
                               drop_top_hubs = cfg.drop_top_hubs,
                               keep_lcc = cfg.keep_lcc))
        end
    end

    println("\n[done] wrote $(length(written)) model file(s):")
    for p in written
        println("  $p")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
