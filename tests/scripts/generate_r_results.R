#!/usr/bin/env Rscript

# Load necessary libraries
library(fixest)
library(jsonlite)
library(reticulate)

cat("Starting R result generation for pyfixest comparison tests...\n")

# Function to safely execute code and handle errors
safe_execute <- function(expr, default_value = NA) {
  tryCatch(expr, error = function(e) {
    cat("Error:", conditionMessage(e), "\n")
    return(default_value)
  })
}

# Function to convert Python formula to R formula
py_fml_to_r_fml <- function(fml) {
  # Convert C() to as.factor()
  r_fml <- gsub("C\\(([^)]+)\\)", "as.factor(\\1)", fml)

  # Convert i() syntax - this is more complex and might need refinement
  # For now, keep the basic structure and handle in fixest directly
  # TODO: Add more sophisticated i() conversion if needed

  return(r_fml)
}

# Function to get inference settings for R
get_r_inference <- function(inference_py) {
  if (is.character(inference_py)) {
    if (inference_py == "iid") {
      return("iid")
    } else if (inference_py == "hetero") {
      return("hetero")
    }
  } else if (is.list(inference_py) && "CRV1" %in% names(inference_py)) {
    return(paste0("cluster = ", inference_py$CRV1))
  }
  return("iid")  # default
}

# Function to extract key results from R fixest model
extract_results <- function(fit_r, formula, inference, weights, dropna, test_type, family = NULL) {
  result <- tryCatch({
    # Get basic model summary
    tidy_result <- broom::tidy(fit_r, conf.int = TRUE)

    # Focus on X1 coefficient (main coefficient of interest)
    x1_row <- which(tidy_result$term == "X1")

    if (length(x1_row) == 0) {
      # If X1 not found, return NA results
      return(data.frame(
        formula = formula,
        inference = ifelse(is.character(inference), inference, "CRV1_group_id"),
        weights = ifelse(is.null(weights), "none", weights),
        dropna = dropna,
        test_type = test_type,
        family = ifelse(is.null(family), "none", family),
        coef = NA,
        se = NA,
        pvalue = NA,
        tstat = NA,
        confint_low = NA,
        confint_high = NA,
        vcov_00 = NA,
        nobs = safe_execute(nobs(fit_r), NA),
        n_coefs = safe_execute(length(coef(fit_r)), NA),
        dof_k = safe_execute(attr(fit_r$cov.scaled, "dof.K"), NA),
        df_t = safe_execute(attr(fit_r$cov.scaled, "df.t"), NA),
        resid_1 = NA,
        resid_2 = NA,
        resid_3 = NA,
        resid_4 = NA,
        resid_5 = NA,
        predict_1 = NA,
        predict_2 = NA,
        predict_3 = NA,
        predict_4 = NA,
        predict_5 = NA,
        stringsAsFactors = FALSE
      ))
    }

    # Extract X1 results
    x1_coef <- tidy_result$estimate[x1_row]
    x1_se <- tidy_result$std.error[x1_row]
    x1_pvalue <- tidy_result$p.value[x1_row]
    x1_tstat <- tidy_result$statistic[x1_row]
    x1_confint_low <- tidy_result$conf.low[x1_row]
    x1_confint_high <- tidy_result$conf.high[x1_row]

    # Get variance-covariance matrix
    vcov_matrix <- safe_execute(vcov(fit_r), matrix(NA))
    vcov_00 <- ifelse(is.matrix(vcov_matrix), vcov_matrix[1, 1], NA)

    # Get model diagnostics
    n_obs <- safe_execute(nobs(fit_r), NA)
    n_coefs <- safe_execute(length(coef(fit_r)), NA)
    dof_k <- safe_execute(attr(fit_r$cov.scaled, "dof.K"), NA)
    df_t <- safe_execute(attr(fit_r$cov.scaled, "df.t"), NA)

    # Get residuals and predictions (first 5 values)
    residuals_r <- safe_execute(residuals(fit_r)[1:5], rep(NA, 5))
    predictions_r <- safe_execute(predict(fit_r)[1:5], rep(NA, 5))

    # Ensure we have exactly 5 values
    if (length(residuals_r) < 5) {
      residuals_r <- c(residuals_r, rep(NA, 5 - length(residuals_r)))
    }
    if (length(predictions_r) < 5) {
      predictions_r <- c(predictions_r, rep(NA, 5 - length(predictions_r)))
    }

    data.frame(
      formula = formula,
      inference = ifelse(is.character(inference), inference, "CRV1_group_id"),
      weights = ifelse(is.null(weights), "none", weights),
      dropna = dropna,
      test_type = test_type,
      family = ifelse(is.null(family), "none", family),
      coef = x1_coef,
      se = x1_se,
      pvalue = x1_pvalue,
      tstat = x1_tstat,
      confint_low = x1_confint_low,
      confint_high = x1_confint_high,
      vcov_00 = vcov_00,
      nobs = n_obs,
      n_coefs = n_coefs,
      dof_k = dof_k,
      df_t = df_t,
      resid_1 = residuals_r[1],
      resid_2 = residuals_r[2],
      resid_3 = residuals_r[3],
      resid_4 = residuals_r[4],
      resid_5 = residuals_r[5],
      predict_1 = predictions_r[1],
      predict_2 = predictions_r[2],
      predict_3 = predictions_r[3],
      predict_4 = predictions_r[4],
      predict_5 = predictions_r[5],
      stringsAsFactors = FALSE
    )
  }, error = function(e) {
    cat("Error extracting results for formula:", formula, "\n")
    cat("Error message:", conditionMessage(e), "\n")
    return(data.frame(
      formula = formula,
      inference = ifelse(is.character(inference), inference, "CRV1_group_id"),
      weights = ifelse(is.null(weights), "none", weights),
      dropna = dropna,
      test_type = test_type,
      family = ifelse(is.null(family), "none", family),
      coef = NA,
      se = NA,
      pvalue = NA,
      tstat = NA,
      confint_low = NA,
      confint_high = NA,
      vcov_00 = NA,
      nobs = NA,
      n_coefs = NA,
      dof_k = NA,
      df_t = NA,
      resid_1 = NA,
      resid_2 = NA,
      resid_3 = NA,
      resid_4 = NA,
      resid_5 = NA,
      predict_1 = NA,
      predict_2 = NA,
      predict_3 = NA,
      predict_4 = NA,
      predict_5 = NA,
      stringsAsFactors = FALSE
    ))
  })

  return(result)
}

# Main function to generate results
generate_all_results <- function() {
  # Read configuration
  config_path <- "tests/config/test_specifications.json"
  if (!file.exists(config_path)) {
    stop("Configuration file not found: ", config_path)
  }

  config <- fromJSON(config_path)
  cat("Loaded configuration file\n")

  # Import pyfixest to get data
  cat("Importing pyfixest...\n")
  pyfixest <- import("pyfixest")

  # Generate different datasets
  cat("Generating datasets...\n")
  data_feols <- pyfixest$get_data(
    N = as.integer(config$data_generation$feols_params$N),
    seed = as.integer(config$data_generation$feols_params$seed),
    model = config$data_generation$feols_params$model
  )

  data_fepois <- pyfixest$get_data(
    N = as.integer(config$data_generation$fepois_params$N),
    seed = as.integer(config$data_generation$fepois_params$seed),
    model = config$data_generation$fepois_params$model
  )

  data_glm <- pyfixest$get_data(
    N = as.integer(config$data_generation$glm_params$N),
    seed = as.integer(config$data_generation$glm_params$seed)
  )
  # Convert Y to binary for GLM
  data_glm$Y <- ifelse(data_glm$Y > 0, 1, 0)

  # Initialize results storage
  all_results <- list()

  # Generate FEOLS results
  cat("Generating FEOLS results...\n")
  feols_results <- data.frame()

  for (formula in config$test_configurations$feols$formulas) {
    for (inference in config$test_configurations$feols$inference_types) {
      for (weights in config$test_configurations$feols$weights) {
        for (dropna in config$test_configurations$feols$dropna) {

          cat("Processing:", formula, "with inference:",
              ifelse(is.character(inference), inference, "CRV1"), "\n")

          # Prepare data
          data_current <- if (dropna) na.omit(data_feols) else data_feols

          # Convert formula
          r_fml <- py_fml_to_r_fml(formula)

          # Convert inference
          r_inference <- get_r_inference(inference)

          # Fit model
          fit_r <- tryCatch({
            if (is.null(weights)) {
              feols(
                as.formula(r_fml),
                data = data_current,
                vcov = r_inference,
                ssc = ssc(
                  adj = config$estimation_settings$ssc_settings$adj,
                  cluster.adj = config$estimation_settings$ssc_settings$cluster_adj,
                  fixef.K = "nested",
                  fixef.force.exact = FALSE,
                  t.df = "min",
                  dof.K = "min"
                )
              )
            } else {
              feols(
                as.formula(r_fml),
                data = data_current,
                vcov = r_inference,
                weights = as.formula(paste0("~", weights)),
                ssc = ssc(
                  adj = config$estimation_settings$ssc_settings$adj,
                  cluster.adj = config$estimation_settings$ssc_settings$cluster_adj,
                  fixef.K = "nested",
                  fixef.force.exact = FALSE,
                  t.df = "min",
                  dof.K = "min"
                )
              )
            }
          }, error = function(e) {
            cat("Error fitting model for formula:", formula, "\n")
            cat("Error message:", conditionMessage(e), "\n")
            return(NULL)
          })

          if (!is.null(fit_r)) {
            result <- extract_results(fit_r, formula, inference, weights, dropna, "feols")
            feols_results <- rbind(feols_results, result)
          }
        }
      }
    }
  }

  all_results$feols <- feols_results

  # Generate IV results
  cat("Generating IV results...\n")
  iv_results <- data.frame()

  for (formula in config$test_configurations$iv$formulas) {
    for (inference in config$test_configurations$iv$inference_types) {
      for (weights in config$test_configurations$iv$weights) {
        for (dropna in config$test_configurations$iv$dropna) {

          cat("Processing IV:", formula, "\n")

          # Prepare data
          data_current <- if (dropna) na.omit(data_feols) else data_feols

          # Convert formula
          r_fml <- py_fml_to_r_fml(formula)

          # Convert inference
          r_inference <- get_r_inference(inference)

          # Fit IV model
          fit_r <- tryCatch({
            if (is.null(weights)) {
              feols(
                as.formula(r_fml),
                data = data_current,
                vcov = r_inference,
                ssc = ssc(
                  adj = config$estimation_settings$ssc_settings$adj,
                  cluster.adj = config$estimation_settings$ssc_settings$cluster_adj,
                  fixef.K = "nested",
                  fixef.force.exact = FALSE,
                  t.df = "min",
                  dof.K = "min"
                )
              )
            } else {
              feols(
                as.formula(r_fml),
                data = data_current,
                vcov = r_inference,
                weights = as.formula(paste0("~", weights)),
                ssc = ssc(
                  adj = config$estimation_settings$ssc_settings$adj,
                  cluster.adj = config$estimation_settings$ssc_settings$cluster_adj,
                  fixef.K = "nested",
                  fixef.force.exact = FALSE,
                  t.df = "min",
                  dof.K = "min"
                )
              )
            }
          }, error = function(e) {
            cat("Error fitting IV model for formula:", formula, "\n")
            return(NULL)
          })

          if (!is.null(fit_r)) {
            result <- extract_results(fit_r, formula, inference, weights, dropna, "iv")
            iv_results <- rbind(iv_results, result)
          }
        }
      }
    }
  }

  all_results$iv <- iv_results

  # Generate GLM results
  cat("Generating GLM results...\n")
  glm_results <- data.frame()

  for (formula in config$test_configurations$glm$formulas) {
    for (family in config$test_configurations$glm$families) {
      for (inference in config$test_configurations$glm$inference_types) {
        for (dropna in config$test_configurations$glm$dropna) {

          cat("Processing GLM:", formula, "family:", family, "\n")

          # Prepare data
          data_current <- if (dropna) na.omit(data_glm) else data_glm

          # Convert formula
          r_fml <- py_fml_to_r_fml(formula)

          # Convert inference
          r_inference <- get_r_inference(inference)

          # Set family
          family_r <- switch(family,
            "probit" = binomial(link = "probit"),
            "logit" = binomial(link = "logit"),
            "gaussian" = gaussian()
          )

          # Fit GLM model
          fit_r <- tryCatch({
            feglm(
              as.formula(r_fml),
              data = data_current,
              family = family_r,
              vcov = r_inference
            )
          }, error = function(e) {
            cat("Error fitting GLM model for formula:", formula, "family:", family, "\n")
            return(NULL)
          })

          if (!is.null(fit_r)) {
            result <- extract_results(fit_r, formula, inference, NULL, dropna, "glm", family)
            glm_results <- rbind(glm_results, result)
          }
        }
      }
    }
  }

  all_results$glm <- glm_results

  # Generate FEPOIS results
  cat("Generating FEPOIS results...\n")
  fepois_results <- data.frame()

  for (formula in config$test_configurations$fepois$formulas) {
    for (inference in config$test_configurations$fepois$inference_types) {
      for (weights in config$test_configurations$fepois$weights) {
        for (dropna in config$test_configurations$fepois$dropna) {

          cat("Processing FEPOIS:", formula, "\n")

          # Prepare data
          data_current <- if (dropna) na.omit(data_fepois) else data_fepois

          # Convert formula
          r_fml <- py_fml_to_r_fml(formula)

          # Convert inference
          r_inference <- get_r_inference(inference)

          # Fit Poisson model
          fit_r <- tryCatch({
            if (is.null(weights)) {
              fepois(
                as.formula(r_fml),
                data = data_current,
                vcov = r_inference,
                ssc = ssc(
                  adj = config$estimation_settings$ssc_settings$adj,
                  cluster.adj = config$estimation_settings$ssc_settings$cluster_adj,
                  fixef.K = "nested",
                  fixef.force.exact = FALSE,
                  t.df = "min",
                  dof.K = "min"
                )
              )
            } else {
              fepois(
                as.formula(r_fml),
                data = data_current,
                vcov = r_inference,
                weights = as.formula(paste0("~", weights)),
                ssc = ssc(
                  adj = config$estimation_settings$ssc_settings$adj,
                  cluster.adj = config$estimation_settings$ssc_settings$cluster_adj,
                  fixef.K = "nested",
                  fixef.force.exact = FALSE,
                  t.df = "min",
                  dof.K = "min"
                )
              )
            }
          }, error = function(e) {
            cat("Error fitting FEPOIS model for formula:", formula, "\n")
            return(NULL)
          })

          if (!is.null(fit_r)) {
            result <- extract_results(fit_r, formula, inference, weights, dropna, "fepois")
            fepois_results <- rbind(fepois_results, result)
          }
        }
      }
    }
  }

  all_results$fepois <- fepois_results

  return(all_results)
}

# Save results to CSV files
save_results <- function(results) {
  output_dir <- "tests/data/cached_results"

  # Ensure output directory exists
  if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
  }

  # Save each test type to separate CSV
  for (test_type in names(results)) {
    if (nrow(results[[test_type]]) > 0) {
      output_file <- file.path(output_dir, paste0(test_type, "_results.csv"))
      write.csv(results[[test_type]], output_file, row.names = FALSE)
      cat("Saved", nrow(results[[test_type]]), "results to", output_file, "\n")
    } else {
      cat("No results generated for", test_type, "\n")
    }
  }

  # Save metadata
  metadata <- data.frame(
    generated_at = Sys.time(),
    r_version = R.version.string,
    fixest_version = packageVersion("fixest"),
    config_version = "1.0.0",
    total_feols = nrow(results$feols),
    total_iv = nrow(results$iv),
    total_glm = nrow(results$glm),
    total_fepois = nrow(results$fepois),
    stringsAsFactors = FALSE
  )

  write.csv(metadata, file.path(output_dir, "metadata.csv"), row.names = FALSE)
  cat("Saved metadata to", file.path(output_dir, "metadata.csv"), "\n")
}

# Main execution
main <- function() {
  cat("===== R Result Generation Script =====\n")
  cat("Started at:", as.character(Sys.time()), "\n\n")

  # Working directory should already be project root when run via pixi
  cat("Working directory:", getwd(), "\n")

  # Check if we're in the right place
  if (!file.exists("tests/config/test_specifications.json")) {
    cat("Warning: test_specifications.json not found in tests/config/\n")
    cat("Looking for it in current directory structure...\n")

    # Try to find the project root
    if (file.exists("pyproject.toml") && dir.exists("tests")) {
      cat("Found project root, continuing...\n")
    } else {
      stop("Cannot find project root. Please run from pyfixest project directory.")
    }
  }

  cat("\n")

  # Generate all results
  results <- generate_all_results()

  # Save results
  save_results(results)

  cat("\n===== Generation Complete =====\n")
  cat("Finished at:", as.character(Sys.time()), "\n")
  cat("Results saved to tests/data/cached_results/\n")
}

# Run if script is executed directly
if (sys.nframe() == 0) {
  main()
}
