# Load necessary libraries
library(fixest)
library(ritest)
library(reticulate)

set.seed(1232)

# Import the pyfixest package
pyfixest <- import("pyfixest")
data <- pyfixest$get_data(N = as.integer(1000), seed = as.integer(2999))

# Define the function to run the tests
run_tests_ritest <- function(data) {
  # Print the column names of the data
  print(names(data))

  # Define the formulas, resampling variables, and clusters
  formulas <- c("Y~X1+f3", "Y~X1+f3|f1", "Y~X1+f3|f1+f2")
  resampvars <- c("X1", "f3", "X1=-0.75", "f3>0.05")
  clusters <- c(NA, "group_id")
  reps <- 10000
  seed <- 123

  # Initialize an empty data frame to store results
  results <- data.frame()

  # Loop through each combination of formula, resampvar, and cluster
  for (fml in formulas) {
    for (resampvar in resampvars) {
      for (cluster in clusters) {
        fit <- feols(as.formula(fml), data = data)

        if (!is.na(cluster)) {
          res_r <- ritest(object = fit, resampvar = resampvar, cluster = cluster, reps = reps, seed = seed)
        } else {
          res_r <- ritest(object = fit, resampvar = resampvar, reps = reps, seed = seed)
        }

        results <- rbind(results, data.frame(
          formula = fml,
          resampvar = resampvar,
          cluster = ifelse(is.na(cluster), "none", cluster),
          pval = res_r$pval,
          se = res_r$se,
          ci_lower = res_r$ci[1]
        ))
      }
    }
  }

  # Save the results to a CSV file
  write.csv(results, "tests/data/ritest_results.csv", row.names = FALSE)
}

# Run the tests
run_tests_ritest(data)
