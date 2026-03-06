#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(arrow)
  library(fixest)
  library(jsonlite)
})

setFixest_nthreads(parallel::detectCores())

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1) {
  stop("Expected exactly one argument: path to JSON config.")
}

config <- fromJSON(args[[1]], simplifyVector = FALSE)
manifest <- config$manifest
formula_str <- config$formula
fe_cols <- unlist(config$fe_cols, use.names = FALSE)
n_fe <- length(fe_cols)
vcov_type <- config$vcov

# Parse vcov: either a string like "iid"/"hetero" or a named list like {"CRV1": "col"}
if (is.list(vcov_type)) {
  vcov_arg <- vcov_type
} else {
  vcov_arg <- vcov_type
}

formula <- as.formula(formula_str)

# ── table formatting helpers ──
fmt_time <- function(t) {
  if (t < 1) sprintf("%.1fms", t * 1000) else sprintf("%.3fs", t)
}

print_header <- function() {
  hdr <- sprintf("  %-16s %12s %4s %10s %10s %10s  %s", "dgp", "n_obs", "n_fe", "min", "median", "max", "status")
  sep <- paste0("  ", paste(rep("-", nchar(hdr) - 2), collapse = ""))
  message(sep)
  message(hdr)
  message(sep)
}

print_row <- function(dgp, n_obs, n_fe, times) {
  times <- times[!is.na(times)]
  if (length(times) > 0) {
    mn <- fmt_time(min(times))
    md <- fmt_time(median(times))
    mx <- fmt_time(max(times))
    status <- "ok"
  } else {
    mn <- "\u2014"; md <- "\u2014"; mx <- "\u2014"
    status <- "FAIL"
  }
  message(sprintf("  %-16s %12s %4d %10s %10s %10s  %s",
                  dgp, format(n_obs, big.mark = ","), n_fe, mn, md, mx, status))
}

# ── main loop ──
message(sprintf("\n  r.fixest (feols)"))
print_header()

prev_dgp <- NULL
prev_nobs <- NULL
group_times <- c()

for (idx in seq_along(manifest)) {
  entry <- manifest[[idx]]
  cur_dgp <- entry$dgp
  cur_nobs <- entry$n_obs

  # flush previous group when key changes
  if (!is.null(prev_dgp) && (cur_dgp != prev_dgp || cur_nobs != prev_nobs)) {
    print_row(prev_dgp, prev_nobs, n_fe, group_times)
    group_times <- c()
  }
  prev_dgp <- cur_dgp
  prev_nobs <- cur_nobs

  elapsed <- NULL
  success <- TRUE
  error_msg <- NULL

  tryCatch(
    {
      df <- as.data.frame(read_parquet(entry$data_path))
      elapsed <- unname(system.time({
        fit <- feols(formula, data = df, vcov = vcov_arg, nthreads = parallel::detectCores())
      })[["elapsed"]])
    },
    error = function(e) {
      success <<- FALSE
      error_msg <<- conditionMessage(e)
      elapsed <<- NULL
    }
  )

  # collect trial times (skip burnin)
  if (entry$iter_type != "burnin" && !is.null(elapsed)) {
    group_times <- c(group_times, elapsed)
  }

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

# flush last group
if (!is.null(prev_dgp)) {
  print_row(prev_dgp, prev_nobs, n_fe, group_times)
}
