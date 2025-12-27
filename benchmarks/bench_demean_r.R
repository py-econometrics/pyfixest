#!/usr/bin/env Rscript
# Benchmark fixest demeaning directly in R
# Usage: Rscript bench_demean_r.R [n_obs] [dgp_type] [n_fe]

library(fixest)

args <- commandArgs(trailingOnly = TRUE)
n_obs <- if (length(args) >= 1) as.integer(args[1]) else 100000L
dgp_type <- if (length(args) >= 2) args[2] else "difficult"
n_fe <- if (length(args) >= 3) as.integer(args[3]) else 3L

# Use 2 threads to match fixest_benchmarks settings
setFixest_nthreads(2)

# Generate data matching Python benchmark DGP
set.seed(42)
n_year <- 10L
n_indiv_per_firm <- 23L
n_indiv <- max(1L, round(n_obs / n_year))
n_firm <- max(1L, round(n_indiv / n_indiv_per_firm))

indiv_id <- rep(1:n_indiv, each = n_year)[1:n_obs]
year <- rep(1:n_year, times = n_indiv)[1:n_obs]

if (dgp_type == "simple") {
  firm_id <- sample(1:n_firm, n_obs, replace = TRUE)
} else {
  # difficult: sequential assignment
  firm_id <- rep(1:n_firm, length.out = n_obs)
}

# Generate outcome
x1 <- rnorm(n_obs)
firm_fe <- rnorm(n_firm)[firm_id]
unit_fe <- rnorm(n_indiv)[indiv_id]
year_fe <- rnorm(n_year)[year]
y <- x1 + firm_fe + unit_fe + year_fe + rnorm(n_obs)

df <- data.frame(
  y = y,
  x1 = x1,
  indiv_id = indiv_id,
  year = year,
  firm_id = firm_id
)

# Build formula based on n_fe
if (n_fe == 2) {
  fml <- y ~ x1 | indiv_id + year
} else {
  fml <- y ~ x1 | indiv_id + year + firm_id
}

# Warm up
invisible(feols(fml, data = df, notes = FALSE, warn = FALSE, nthreads = 2L))

# Benchmark
n_runs <- 5L
times <- numeric(n_runs)

for (i in 1:n_runs) {
  start <- Sys.time()
  fit <- feols(fml, data = df, notes = FALSE, warn = FALSE, nthreads = 2L)
  end <- Sys.time()
  times[i] <- as.numeric(end - start, units = "secs") * 1000  # ms
}

cat(sprintf("fixest (R native) - n=%d, type=%s, %dFE\n", n_obs, dgp_type, n_fe))
cat(sprintf("  Times (ms): %s\n", paste(round(times, 2), collapse = ", ")))
cat(sprintf("  Median: %.2f ms\n", median(times)))
cat(sprintf("  Min: %.2f ms\n", min(times)))
