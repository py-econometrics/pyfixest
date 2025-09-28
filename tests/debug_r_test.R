# Debug script for R test issues
library(fixest)
library(jsonlite)
library(reticulate)
library(broom)

# Import pyfixest to get data
pyfixest <- import("pyfixest")

# Source the main functions
source("r_cache/run_feols_tests.R")

# Create a simple test case to debug
test_params <- list(
  test_id = "debug_test",
  formula = "Y~X1",
  data_params = list(
    N = 100, 
    seed = 123, 
    beta_type = "2", 
    error_type = "2", 
    model = "Feols",
    f3_type = "str"
  ),
  estimation_params = list(
    vcov = "iid", 
    weights = NULL, 
    dropna = FALSE, 
    demeaner_backend = "numba",
    ssc = list(adj = TRUE, cluster_adj = TRUE)
  ),
  hash = "debug"
)

cat("Testing with debug parameters...\n")
print(test_params)

# Now you can use browser() here and it will work
browser()

# Run the test
results <- run_single_feols_test(test_params)
print(results)
