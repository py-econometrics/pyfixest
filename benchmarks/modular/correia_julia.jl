#!/usr/bin/env julia

using CSV
using DataFrames
using FixedEffectModels
using JSON3
using StatsModels

function main()
    if length(ARGS) != 1
        error("Expected exactly one argument: path to JSON config.")
    end

    config = JSON3.read(read(ARGS[1], String))
    manifest = config[:manifest]
    depvar = String(config[:depvar])
    covariates = String.(config[:covariates])
    fe_cols = String.(config[:fe_cols])
    tolerance = Float64(config[:tolerance])

    lhs_term = term(Symbol(depvar))
    rhs_expr = foldl(+, [term(Symbol(c)) for c in covariates])
    fe_expr = foldl(+, [fe(Symbol(col)) for col in fe_cols])
    formula = lhs_term ~ rhs_expr + fe_expr

    for entry in manifest
        elapsed = nothing
        success = true
        error_msg = nothing
        n_obs = Int(entry[:n_obs])

        try
            df = CSV.read(String(entry[:data_path]), DataFrame)
            n_obs = nrow(df)
            start_time = time()
            reg(df, formula; tol=tolerance, progress_bar=false)
            elapsed = time() - start_time
        catch e
            success = false
            error_msg = sprint(showerror, e)
        end

        result = Dict(
            "dataset_id" => String(entry[:dataset_id]),
            "iter_num" => Int(entry[:iter_num]),
            "n_obs" => n_obs,
            "time" => elapsed,
            "success" => success,
            "error" => error_msg,
        )
        println(stdout, JSON3.write(result))
        flush(stdout)
        GC.gc()
    end
end

main()
