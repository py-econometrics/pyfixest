# Load necessary libraries
library(fixest)
library(ritest)
library(reticulate)

set.seed(1232)

# Import the pyfixest package
pyfixest <- import("pyfixest")
data <- pyfixest$get_data(N = as.integer(1000), seed = as.integer(2999))
# unbalanced treatments: 20% of rows and 5 of 17 clusters are treated
data$D_unit <- as.numeric((seq_len(nrow(data)) - 1) %% 5 == 0)
data$D_cluster <- as.numeric(data$group_id %% 4 == 0)

# Define the function to run the tests
run_tests_ritest <- function(data) {
  # Print the column names of the data
  print(names(data))

  # Define the formula, resampling variable, and cluster combinations.
  # Clustered cases need a treatment that is constant within clusters.
  fixef <- c("", "|f1", "|f1+f2")
  cases <- rbind(
    expand.grid(
      formula = paste0("Y~X1+f3", fixef),
      resampvar = c("X1", "f3", "X1=-0.75", "f3>0.05"),
      cluster = NA,
      stringsAsFactors = FALSE
    ),
    expand.grid(
      formula = paste0("Y~D_unit+f3", fixef),
      resampvar = c("D_unit", "D_unit=0.2"),
      cluster = NA,
      stringsAsFactors = FALSE
    ),
    expand.grid(
      formula = paste0("Y~D_cluster+f3", fixef),
      resampvar = c("D_cluster", "D_cluster=0.2"),
      cluster = "group_id",
      stringsAsFactors = FALSE
    )
  )
  reps <- 10000
  seed <- 123

  # Initialize an empty data frame to store results
  results <- data.frame()

  for (i in seq_len(nrow(cases))) {
    fml <- cases$formula[i]
    resampvar <- cases$resampvar[i]
    cluster <- cases$cluster[i]
    fit <- feols(as.formula(fml), data = data)

    # ritest() captures `resampvar` unevaluated, so pass it via do.call()
    args <- list(object = fit, resampvar = resampvar, reps = reps, seed = seed)
    if (!is.na(cluster)) {
      args$cluster <- cluster
    }
    res_r <- do.call(ritest, args)

    results <- rbind(results, data.frame(
      formula = fml,
      resampvar = resampvar,
      cluster = ifelse(is.na(cluster), "none", cluster),
      pval = res_r$pval,
      se = res_r$se,
      ci_lower = res_r$ci[1]
    ))
  }

  # Save the results to a CSV file
  write.csv(results, "tests/data/ritest_results.csv", row.names = FALSE)
}

# Run the tests
run_tests_ritest(data)

# Stratified randomization inference. `ritest` looks up a strata variable
# outside the model in the data named in the fit's call and drops incomplete
# rows, so the data are complete. Strata nest the `group_id` clusters.
data_strata <- data[complete.cases(data), ]
data_strata$stratum <- data_strata$group_id %% 3

run_tests_ritest_strata <- function() {
  fixef <- c("", "|f1")
  cases <- rbind(
    expand.grid(
      formula = paste0("Y~X1+f3", fixef),
      resampvar = "X1",
      cluster = NA,
      stringsAsFactors = FALSE
    ),
    expand.grid(
      formula = paste0("Y~D_unit+f3", fixef),
      resampvar = "D_unit",
      cluster = NA,
      stringsAsFactors = FALSE
    ),
    expand.grid(
      formula = paste0("Y~D_cluster+f3", fixef),
      resampvar = "D_cluster",
      cluster = "group_id",
      stringsAsFactors = FALSE
    )
  )
  reps <- 10000
  seed <- 123

  results <- data.frame()

  for (i in seq_len(nrow(cases))) {
    fml <- cases$formula[i]
    resampvar <- cases$resampvar[i]
    cluster <- cases$cluster[i]
    fit <- feols(as.formula(fml), data = data_strata)

    # ritest() captures `resampvar` unevaluated, so pass it via do.call()
    args <- list(
      object = fit, resampvar = resampvar, strata = "stratum",
      reps = reps, seed = seed
    )
    if (!is.na(cluster)) {
      args$cluster <- cluster
    }
    res_r <- do.call(ritest, args)

    results <- rbind(results, data.frame(
      formula = fml,
      resampvar = resampvar,
      cluster = ifelse(is.na(cluster), "none", cluster),
      pval = res_r$pval,
      se = res_r$se,
      ci_lower = res_r$ci[1]
    ))
  }

  write.csv(results, "tests/data/ritest_strata_results.csv", row.names = FALSE)
}

run_tests_ritest_strata()
