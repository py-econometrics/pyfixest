#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(fixest)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1) {
  stop("Expected exactly one argument: path to JSON config.")
}

config <- fromJSON(args[[1]], simplifyVector = FALSE)
manifest <- config$manifest
formula <- as.formula(config$formula)
tolerance <- config$tolerance

for (entry in manifest) {
  elapsed <- NULL
  success <- TRUE
  error_msg <- NULL
  n_obs <- entry$n_obs

  tryCatch(
    {
      df <- utils::read.csv(entry$data_path)
      n_obs <- nrow(df)
      elapsed <- unname(system.time({
        suppressMessages(feols(formula, data = df, fixef.tol = tolerance))
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
        iter_num = entry$iter_num,
        n_obs = n_obs,
        time = elapsed,
        success = success,
        error = error_msg
      ),
      auto_unbox = TRUE,
      null = "null"
    ),
    "\n"
  )

  if (exists("df")) {
    rm(df)
  }
  gc(verbose = FALSE)
}
