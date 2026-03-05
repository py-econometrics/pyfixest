#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(arrow)
  library(fixest)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1) {
  stop("Expected exactly one argument: path to JSON config.")
}

config <- fromJSON(args[[1]], simplifyVector = FALSE)
manifest <- config$manifest
demean_cols <- unlist(config$demean_cols, use.names = FALSE)
fe_cols <- unlist(config$fe_cols, use.names = FALSE)

for (entry in manifest) {
  elapsed <- NULL
  success <- TRUE
  error_msg <- NULL

  tryCatch(
    {
      df <- as.data.frame(read_parquet(entry$data_path))
      x <- as.matrix(df[, demean_cols, drop = FALSE])
      fe_list <- lapply(fe_cols, function(col) as.factor(df[[col]]))
      elapsed <- unname(system.time({
        demean(x, f = fe_list)
      })[["elapsed"]])
    },
    error = function(e) {
      success <<- FALSE
      error_msg <<- conditionMessage(e)
      elapsed <<- NULL
    }
  )

  cat(
    toJSON(
      list(
        dataset_id = entry$dataset_id,
        dgp = entry$dgp,
        n_obs = entry$n_obs,
        iter_type = entry$iter_type,
        iter_num = entry$iter_num,
        time = elapsed,
        success = success,
        error = error_msg
      ),
      auto_unbox = TRUE,
      null = "null"
    ),
    "\n"
  )
}
