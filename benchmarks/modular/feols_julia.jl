#!/usr/bin/env julia

using JSON3
using DataFrames
using Parquet2
using FixedEffectModels
using StatsModels
using Printf
using Statistics

# ── table formatting ──
function fmt_time(t::Float64)
    if t < 1.0
        return @sprintf("%.1fms", t * 1000)
    else
        return @sprintf("%.3fs", t)
    end
end

DGP_W = Ref(16)

function print_header(name::String)
    w = DGP_W[]
    hdr = "  " * rpad("dgp", w) * @sprintf(" %12s %4s %10s %10s %10s  %s", "n_obs", "n_fe", "min", "median", "max", "status")
    sep = "  " * "-"^(length(hdr) - 2)
    println(stderr, "\n  ", name)
    println(stderr, sep)
    println(stderr, hdr)
    println(stderr, sep)
    flush(stderr)
end

function print_row(dgp::String, n_obs::Int, n_fe::Int, times::Vector{Float64})
    w = DGP_W[]
    if isempty(times)
        println(stderr, "  ", rpad(dgp, w),
            @sprintf(" %12s %4d %10s %10s %10s  %s", format_number(n_obs), n_fe, "—", "—", "—", "FAIL"))
    else
        mn = fmt_time(minimum(times))
        md = fmt_time(median(times))
        mx = fmt_time(maximum(times))
        println(stderr, "  ", rpad(dgp, w),
            @sprintf(" %12s %4d %10s %10s %10s  %s", format_number(n_obs), n_fe, mn, md, mx, "ok"))
    end
    flush(stderr)
end

function format_number(n::Int)
    s = string(n)
    result = ""
    for (i, c) in enumerate(reverse(s))
        if i > 1 && (i - 1) % 3 == 0
            result = "," * result
        end
        result = c * result
    end
    return result
end

# ── parse normalized vcov_type: "iid", "hetero", or "cluster:<colname>" ──
function parse_vcov(vcov_type::String)
    if startswith(vcov_type, "cluster:")
        cluster_col = replace(vcov_type, "cluster:" => "")
        return Vcov.cluster(Symbol(cluster_col))
    elseif vcov_type == "iid"
        return Vcov.simple()
    elseif vcov_type == "hetero"
        return Vcov.robust()
    else
        error("Unknown vcov_type: $vcov_type")
    end
end

# ── main ──
function main()
    if length(ARGS) != 1
        error("Expected exactly one argument: path to JSON config.")
    end

    config = JSON3.read(read(ARGS[1], String))
    manifest = config[:manifest]
    formula_str = String(config[:formula])
    fe_cols = String.(config[:fe_cols])
    n_fe = length(fe_cols)
    vcov_type = String(config[:vcov_type])
    vcov_spec = parse_vcov(vcov_type)

    DGP_W[] = max(16, maximum(length(String(entry[:dgp])) for entry in manifest))

    # Build the reg formula: y ~ x1 + fe(indiv_id) + fe(year)
    # Parse from the fixest-style formula
    depvar = String(config[:depvar])
    covariates = String.(config[:covariates])

    lhs_term = term(Symbol(depvar))
    rhs_expr = foldl(+, [term(Symbol(c)) for c in covariates])
    fe_expr = foldl(+, [fe(Symbol(col)) for col in fe_cols])
    formula = lhs_term ~ rhs_expr + fe_expr

    print_header("julia.FixedEffectModels (feols)")

    prev_dgp = nothing
    prev_nobs = nothing
    group_times = Float64[]

    for entry in manifest
        cur_dgp = String(entry[:dgp])
        cur_nobs = Int(entry[:n_obs])

        # flush previous group when key changes
        if prev_dgp !== nothing && (cur_dgp != prev_dgp || cur_nobs != prev_nobs)
            print_row(prev_dgp, prev_nobs, n_fe, group_times)
            group_times = Float64[]
        end
        prev_dgp = cur_dgp
        prev_nobs = cur_nobs

        dataset_id = String(entry[:dataset_id])
        iter_type = String(entry[:iter_type])
        iter_num = Int(entry[:iter_num])
        data_path = String(entry[:data_path])

        elapsed = nothing
        success = true
        error_msg = nothing

        try
            df = DataFrame(Parquet2.Dataset(data_path))
            start_time = time()
            reg(df, formula, vcov_spec, nthreads=Sys.CPU_THREADS, progress_bar = false)
            elapsed = time() - start_time
        catch e
            success = false
            error_msg = string(e)
        end

        # collect trial times (skip burnin)
        if iter_type != "burnin" && elapsed !== nothing
            push!(group_times, elapsed)
        end

        # JSON result to stdout
        result = Dict(
            "dataset_id" => dataset_id,
            "dgp" => cur_dgp,
            "n_obs" => cur_nobs,
            "iter_type" => iter_type,
            "iter_num" => iter_num,
            "time" => elapsed,
            "success" => success,
            "error" => error_msg,
        )
        println(stdout, JSON3.write(result))
    end

    # flush last group
    if prev_dgp !== nothing
        print_row(prev_dgp, prev_nobs, n_fe, group_times)
    end
end

main()
