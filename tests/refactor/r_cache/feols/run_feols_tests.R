# Standalone R script for running FEOLS tests and caching results
# This script reads test parameters from JSON and outputs results to JSON

library(fixest)
library(jsonlite)
library(reticulate)
library(broom)

# Import pyfixest to get data
pyfixest <- import("pyfixest")

# Helper functions (from original test_vs_fixest.py)
convert_f3 <- function(data, f3_type) {
  if (f3_type == "categorical") {
    data$f3 <- as.factor(data$f3)
  } else if (f3_type == "int") {
    data$f3 <- as.integer(data$f3)
  } else if (f3_type == "str") {
    data$f3 <- as.character(data$f3)
  } else if (f3_type == "object") {
    data$f3 <- as.character(data$f3)  # R doesn't have object type
  } else if (f3_type == "float") {
    data$f3 <- as.numeric(data$f3)
  }
  return(data)
}

c_to_as_factor <- function(py_fml) {
  # Transform formulaic C-syntax for categorical variables into R's as.factor
  pattern <- "C\\((.*?)\\)"
  replacement <- "factor(\\1, exclude = NA)"
  r_fml <- gsub(pattern, replacement, py_fml)
  return(r_fml)
}

get_data_r <- function(fml, data) {
  # Extract factor variables and filter NAs
  vars <- strsplit(strsplit(fml, "~")[[1]][2], "\\|")[[1]][1]
  vars <- strsplit(vars, "\\+")[[1]]

  factor_vars <- c()
  for (var in vars) {
    if (grepl("C\\(", var)) {
      var <- gsub(" ", "", var)
      var <- substr(var, 3, nchar(var) - 1)
      factor_vars <- c(factor_vars, var)
    }
  }

  # Filter out NAs if factor_vars exist
  if (length(factor_vars) > 0) {
    complete_rows <- complete.cases(data[factor_vars])
    data_r <- data[complete_rows, ]
  } else {
    data_r <- data
  }

  return(data_r)
}

get_r_inference <- function(inference) {
  if (is.list(inference) && "CRV1" %in% names(inference)) {
    return(as.formula(paste("~", inference$CRV1)))
  } else {
    return(inference)
  }
}

run_single_feols_test <- function(test_params) {
  tryCatch({
    # Extract parameters
    test_id <- test_params$test_id
    formula <- test_params$formula
    data_params <- test_params$data_params
    estimation_params <- test_params$estimation_params

    # Generate data using pyfixest
    f3_type <- data_params$f3_type
    data_params$f3_type <- NULL  # Remove before passing to get_data

    # Convert numeric parameters to integers (R passes floats, Python needs ints)
    data_params$seed <- as.integer(data_params$seed)
    data_params$N <- as.integer(data_params$N)

    data <- do.call(pyfixest$get_data, data_params)

    # Apply dropna if needed
    if (estimation_params$dropna) {
      data <- data[complete.cases(data), ]
    }

    # Handle categorical conversion
    data[data == "nan"] <- NA

    # Convert f3 type
    data <- convert_f3(data, f3_type)

    # Prepare data for R
    data_r <- get_data_r(formula, data)

    # Convert formula for R
    r_formula <- c_to_as_factor(formula)

    # Get R inference specification
    r_inference <- get_r_inference(estimation_params$vcov)

    # Extract SSC parameters
    ssc_params <- estimation_params$ssc
      adj <- ssc_params$adj
      cluster_adj <- ssc_params$cluster_adj

      # Run R estimation
      if (!is.null(estimation_params$weights)) {
        r_fit <- feols(
          as.formula(r_formula),
          vcov = r_inference,
          data = data_r,
          ssc = ssc(adj, "nested", cluster_adj, "min", "min", FALSE),
          weights = as.formula(paste("~", estimation_params$weights))
        )
      } else {
        r_fit <- feols(
          as.formula(r_formula),
          vcov = r_inference,
          data = data_r,
          ssc = ssc(adj, "nested", cluster_adj, "min", "min", FALSE)
        )
      }

    # Extract basic results
    coef_table <- coeftable(r_fit)
    confint_table <- confint(r_fit)
    coef_table_x1 <- coef_table[rownames(coef_table) == "X1", ]
    confint_table_x1 <- confint_table[rownames(confint_table) == "X1", ]

    r_coef <- as.numeric(coef(r_fit)["X1"])
    r_se <- as.numeric(se(r_fit)["X1"])  # Use se() function for full precision
    r_tstat <- tstat(r_fit)["X1"]  # Calculate t-stat directly for consistency
    r_pval <- pvalue(r_fit)["X1"]
    r_confint_low <- as.numeric(confint_table_x1["2.5 %"])
    r_confint_high <- as.numeric(confint_table_x1["97.5 %"])

    # Get additional statistics
    r_vcov <- as.numeric(vcov(r_fit)[1, 1])
    r_nobs <- as.numeric(nobs(r_fit))
    r_n_coefs <- length(coef(r_fit))

    # Get degrees of freedom (handle NULL values)
    r_dof_k <- attr(r_fit$cov.scaled, "dof.K")
    r_df_t <- attr(r_fit$cov.scaled, "df.t")

    # Convert NULL to NA for JSON serialization
    if (is.null(r_dof_k)) r_dof_k <- NA
    if (is.null(r_df_t)) r_df_t <- NA

    # Get residuals and predictions for iid case
    r_resid <- NULL
    r_predict <- NULL
    if (estimation_params$vcov == "iid" && adj && cluster_adj) {
      r_resid <- as.numeric(residuals(r_fit))[1:5]  # First 5 for comparison
      r_predict <- as.numeric(predict(r_fit))[1:5]  # First 5 for comparison
    }


    # Return results
    results <- list(
      test_id = test_id,
      formula = formula,
      test_group = "feols",
      hash = test_params$hash,

      # Main results
      r_coef = r_coef,
      r_n_coefs = r_n_coefs,
      r_se = r_se,
      r_pval = r_pval,
      r_tstat = r_tstat,
      r_confint = c(r_confint_low, r_confint_high),
      r_vcov = r_vcov,
      r_nobs = r_nobs,
      r_dof_k = r_dof_k,
      r_df_t = r_df_t,

      # Optional results (for iid case)
      r_resid = r_resid,
      r_predict = r_predict,

      # Test parameters for reference
      inference = estimation_params$vcov,
      weights = estimation_params$weights,
      dropna = estimation_params$dropna,
      f3_type = f3_type,
      demeaner_backend = estimation_params$demeaner_backend,

      success = TRUE
    )

    return(results)

  }, error = function(e) {
    return(list(
      test_id = test_params$test_id,
      error = as.character(e),
      hash = test_params$hash,
      success = FALSE
    ))
  })
}

# Main execution
main <- function() {
  # Read command line arguments
  args <- commandArgs(trailingOnly = TRUE)

  if (length(args) != 2) {
    cat("Usage: Rscript run_feols_tests.R <input_json> <output_json>\n")
    quit(status = 1)
  }

  input_file <- args[1]
  output_file <- args[2]

  # Read test parameters from JSON
  if (!file.exists(input_file)) {
    cat("Error: Input file", input_file, "does not exist\n")
    quit(status = 1)
  }

  test_params <- fromJSON(input_file)

  # Run the test
  cat("Running R test:", test_params$test_id, "\n")
  results <- run_single_feols_test(test_params)

    # Write results to JSON with proper formatting
    write_json(results, output_file, pretty = TRUE, auto_unbox = TRUE, digits = 10)

  if (results$success) {
    cat("Test completed successfully\n")
  } else {
    cat("Test failed:", results$error, "\n")
  }
}

# Run main function
main()
