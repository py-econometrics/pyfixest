library(fixest)
library(jsonlite)
library(reticulate)

# Import pyfixest to get data
pyfixest <- import("pyfixest")

# Helper functions (adapted from FEOLS script)
convert_f3 <- function(data, f3_type) {
  if (f3_type == "str") {
    data$f3 <- as.character(data$f3)
  } else if (f3_type == "object") {
    data$f3 <- as.character(data$f3)
  } else if (f3_type == "int") {
    data$f3 <- as.integer(data$f3)
  } else if (f3_type == "categorical") {
    data$f3 <- as.factor(data$f3)
  } else if (f3_type == "float") {
    data$f3 <- as.numeric(data$f3)
  }
  return(data)
}

c_to_as_factor <- function(formula) {
  # Convert C() to as.factor() for R
  formula <- gsub("C\\(([^)]+)\\)", "as.factor(\\1)", formula)
  return(formula)
}

get_data_r <- function(formula, data) {
  # Extract variables from formula and prepare data for R
  # This is a simplified version - the original is more complex
  vars_in_formula <- all.vars(as.formula(formula))
  data_r <- data[, colnames(data) %in% c(vars_in_formula, "weights", "group_id"), drop = FALSE]
  return(data_r)
}

get_r_inference <- function(inference) {
  if (is.list(inference) && "CRV1" %in% names(inference)) {
    cluster_var <- inference[["CRV1"]]
    return(as.formula(paste("~", cluster_var)))
  } else {
    return(inference)
  }
}

run_single_fepois_test <- function(test_params) {
  tryCatch({
    # Extract parameters
    test_id <- test_params$test_id
    formula <- test_params$formula
    data_params <- test_params$data_params
    estimation_params <- test_params$estimation_params

    cat("Running FEPOIS test:", test_id, "\n")
    cat("Formula:", formula, "\n")

    # Generate data using pyfixest
    f3_type <- data_params$f3_type
    data_params$f3_type <- NULL  # Remove before passing to get_data

    # Convert numeric parameters to integers (R passes floats, Python needs ints)
    data_params$seed <- as.integer(data_params$seed)
    data_params$N <- as.integer(data_params$N)

    cat("Generating data with params:", toString(data_params), "\n")
    data <- do.call(pyfixest$get_data, data_params)

    # Apply dropna if needed (should always be FALSE for FEPOIS)
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

    # Extract FEPOIS-specific parameters
    glm_tol <- estimation_params$iwls_tol  # R uses glm_tol, Python uses iwls_tol
    glm_maxiter <- estimation_params$iwls_maxiter  # R uses glm_maxiter, Python uses iwls_maxiter

    cat("Running R FEPOIS estimation...\n")

    # Run R estimation (FEPOIS uses fepois instead of feols)
    r_fit <- fepois(
      as.formula(r_formula),
      vcov = r_inference,
      data = data_r,
      ssc = ssc(adj, "nested", cluster_adj, "min", "min", FALSE),
      glm_tol = glm_tol,
      glm_maxiter = glm_maxiter
    )

    cat("Extracting results...\n")

    # Extract results using the same structure as FEOLS
    coef_table <- coeftable(r_fit)
    confint_table <- confint(r_fit)
    coef_table_x1 <- coef_table[rownames(coef_table) == "X1", ]
    confint_table_x1 <- confint_table[rownames(confint_table) == "X1", ]

    r_coef <- as.numeric(coef_table_x1["Estimate"])
    r_se <- as.numeric(coef_table_x1["Std. Error"])
    r_tstat <- as.numeric(coef_table_x1["t value"])
    r_pval <- as.numeric(coef_table_x1["Pr(>|t|)"])
    r_confint_low <- as.numeric(confint_table_x1["2.5 %"])
    r_confint_high <- as.numeric(confint_table_x1["97.5 %"])
    r_confint <- c(r_confint_low, r_confint_high)

    # Get additional statistics (matching FEOLS structure)
    r_vcov <- as.numeric(vcov(r_fit)[1, 1])
    r_nobs <- as.numeric(nobs(r_fit))
    r_n_coefs <- length(coef(r_fit))

    # Get degrees of freedom
    r_dof_k <- attr(r_fit$cov.scaled, "dof.K")
    r_df_t <- attr(r_fit$cov.scaled, "df.t")

    # FEPOIS-specific statistics
    r_deviance <- as.numeric(r_fit$deviance)
    r_irls_weights <- as.numeric(r_fit$irls_weights)

    # Get residuals and predictions for iid case (matching FEOLS structure)
    r_resid <- NULL
    r_predict_response <- NULL
    r_predict_link <- NULL
    if (estimation_params$vcov == "iid" && adj && cluster_adj) {
      r_resid <- as.numeric(residuals(r_fit))[1:5]  # First 5 for comparison
      r_predict_response <- as.numeric(predict(r_fit, type = "response"))[1:5]  # First 5 for comparison
      r_predict_link <- as.numeric(predict(r_fit, type = "link"))[1:5]  # First 5 for comparison
    }

    cat("Preparing results...\n")

    results <- list(
      test_id = test_id,
      formula = formula,
      hash = test_params$hash,

      # Main coefficient results
      coef = r_coef,
      se = r_se,
      tstat = r_tstat,
      pval = r_pval,
      confint = r_confint,

      # Model statistics
      nobs = r_nobs,
      vcov = r_vcov,
      deviance = r_deviance,
      dof_k = r_dof_k,
      df_t = r_df_t,
      n_coefs = r_n_coefs,

      # FEPOIS-specific results
      irls_weights = r_irls_weights,

      # Residuals and predictions
      resid = r_resid,
      predict_response = r_predict_response,
      predict_link = r_predict_link,

      success = TRUE
    )

    cat("FEPOIS test completed successfully for", test_id, "\n")
    return(results)

  }, error = function(e) {
    cat("Error in FEPOIS test", test_id, ":", as.character(e), "\n")
    return(list(
      test_id = test_id,
      error = as.character(e),
      hash = test_params$hash,
      success = FALSE
    ))
  })
}

# Main execution block
if (length(commandArgs(trailingOnly = TRUE)) >= 2) {
  input_file <- commandArgs(trailingOnly = TRUE)[1]
  output_file <- commandArgs(trailingOnly = TRUE)[2]

  cat("Reading test parameters from:", input_file, "\n")
  test_params <- fromJSON(input_file)

  cat("Running FEPOIS test...\n")
  results <- run_single_fepois_test(test_params)

  cat("Writing results to:", output_file, "\n")
  write_json(results, output_file, pretty = TRUE, auto_unbox = TRUE)

  cat("FEPOIS test script completed.\n")
} else {
  cat("Usage: Rscript run_fepois_tests.R <input_file> <output_file>\n")
  quit(status = 1)
}
