#' Calculate two-stage difference-in-differences following Gardner (2021)
#'
#' @import fixest
#'
#' @param data The dataframe containing all the variables
#' @param yname Outcome variable
#' @param first_stage Fixed effects and other covariates you want to residualize
#'   with in first stage.
#'   Formula following \code{\link[fixest:feols]{fixest::feols}}.
#'   Fixed effects specified after "`|`".
#' @param second_stage Second stage, these should be the treatment indicator(s)
#'   (e.g. treatment variable or event-study leads/lags).
#'   Formula following \code{\link[fixest:feols]{fixest::feols}}.
#'   Use `i()` for factor variables, see \code{\link[fixest:i]{fixest::i}}.
#' @param treatment A variable that = 1 if treated, = 0 otherwise
#' @param cluster_var What variable to cluster standard errors. This can be IDs
#'   or a higher aggregate level (state for example)
#' @param weights Optional. Variable name for regression weights.
#' @param bootstrap Optional. Should standard errors be calculated using bootstrap?
#'   Default is `FALSE`.
#' @param n_bootstraps Optional. How many bootstraps to run.
#'   Default is `250`.
#' @param return_bootstrap Optional. Logical. Will return each bootstrap second-stage
#'   estimate to allow for manual use, e.g. percentile standard errors and empirical
#'   confidence intervals.
#' @param verbose Optional. Logical. Should information about the two-stage
#'   procedure be printed back to the user?
#'   Default is `TRUE`.
#'
#' @return `fixest` object with adjusted standard errors
#'   (either by formula or by bootstrap). All the methods from `fixest` package
#'   will work, including \code{\link[fixest:esttable]{fixest::esttable}} and
#'   \code{\link[fixest:coefplot]{fixest::coefplot}}
#'
#' @section Examples:
#'
#' Load example dataset which has two treatment groups and homogeneous treatment effects
#'
#' ```{r, comment = "#>", collapse = TRUE}
#' # Load Example Dataset
#' data("df_hom")
#' ```
#'
#' ### Static TWFE
#'
#' You can run a static TWFE fixed effect model for a simple treatment indicator
#' ```{r, comment = "#>", collapse = TRUE}
#' static <- did2s(df_hom,
#'     yname = "dep_var", treatment = "treat", cluster_var = "state",
#'     first_stage = ~ 0 | unit + year,
#'     second_stage = ~ i(treat, ref=FALSE))
#'
#' fixest::esttable(static)
#' ```
#'
#' ### Event Study
#'
#' Or you can use relative-treatment indicators to estimate an event study estimate
#' ```{r, comment = "#>", collapse = TRUE}
#' es <- did2s(df_hom,
#'     yname = "dep_var", treatment = "treat", cluster_var = "state",
#'     first_stage = ~ 0 | unit + year,
#'     second_stage = ~ i(rel_year, ref=c(-1, Inf)))
#'
#' fixest::esttable(es)
#' ```
#'
#' ```{r, eval = F}
#' # plot rel_year coefficients and standard errors
#' fixest::coefplot(es, keep = "rel_year::(.*)")
#' ```
#'
#' ### Example from Cheng and Hoekstra (2013)
#'
#' Here's an example using data from Cheng and Hoekstra (2013)
#' ```{r, comment = "#>", collapse = TRUE}
#' # Castle Data
#' castle <- haven::read_dta("https://github.com/scunning1975/mixtape/raw/master/castle.dta")
#'
#' did2s(
#' 	data = castle,
#' 	yname = "l_homicide",
#' 	first_stage = ~ 0 | sid + year,
#' 	second_stage = ~ i(post, ref=0),
#' 	treatment = "post",
#' 	cluster_var = "state", weights = "popwt"
#' )
#' ```
#'
#' @export
did2s <- function(data, yname, first_stage, second_stage, treatment, cluster_var,
                  weights = NULL, bootstrap = FALSE, n_bootstraps = 250,
                  return_bootstrap = FALSE, verbose = TRUE) {
  
  # Check Parameters ---------------------------------------------------------
  
  if (!inherits(data, "data.frame")) stop("`did2s` requires a data.frame like object for analysis.")
  
  # Extract vars from formula
  if (inherits(first_stage, "formula")) first_stage <- as.character(first_stage)[[2]]
  if (inherits(second_stage, "formula")) second_stage <- as.character(second_stage)[[2]]
  
  # Check that treatment is a 0/1 or T/F variable
  if (!all(
    unique(data[[treatment]]) %in% c(1, 0, T, F)
  )) {
    stop(sprintf(
      "'%s' must be a 0/1 or T/F variable indicating which observations are untreated/not-yet-treated.",
      treatment
    ))
  }
  
  
  # Print --------------------------------------------------------------------
  if (verbose) {
    if (!bootstrap) cluster_msg <- paste0("- Standard errors will be clustered by `", cluster_var, "`\n")
    if (bootstrap) cluster_msg <- paste0("- Standard errors will be block bootstrapped with cluster `", cluster_var, "`\n")
    message(
      paste(
        "Running Two-stage Difference-in-Differences\n",
        paste0("- first stage formula `", paste0("~ ", first_stage), "`\n"),
        paste0("- second stage formula `", paste0("~ ", second_stage), "`\n"),
        paste0("- The indicator variable that denotes when treatment is on is `", treatment, "`\n"),
        cluster_msg,
        collapse = "\n"
      )
    )
  }
  
  # Point Estimates ----------------------------------------------------------
  
  est <- did2s_estimate(
    data = data,
    yname = yname,
    first_stage = first_stage,
    second_stage = second_stage,
    treatment = treatment,
    weights = weights,
    bootstrap = bootstrap
  )
  
  # Analytic Standard Errors -------------------------------------------------
  
  if (!bootstrap) {
    # Subset data to the observations used in the second stage
    # obsRemoved have - in front of rows, so they are deleted
    removed_rows <- est$second_stage$obs_selection$obsRemoved
    if (!is.null(removed_rows)) data <- data[removed_rows, ]
    
    # Extract weights
    if (is.null(weights)) {
      weights_vector <- rep.int(1L, nrow(data))
    } else {
      weights_vector <- sqrt(data[[weights]])
    }
    
    # Extract first stage
    first_u <- est$first_u
    if (!is.null(removed_rows)) first_u <- first_u[removed_rows]
    
    # x1 is matrix used to predict Y(0)
    x1 <- did2s_sparse(data, est$first_stage, weights_vector)
    
    # Extract second stage
    second_u <- stats::residuals(est$second_stage)
    x2 <- did2s_sparse(data, est$second_stage, weights_vector)
    
    # multiply by weights
    first_u <- weights_vector * first_u
    x1 <- weights_vector * x1
    second_u <- weights_vector * second_u
    x2 <- weights_vector * x2
    
    # x10 is matrix used to estimate first stage (zero out rows with D_it = 1)
    x10 <- copy(x1)
    # treated rows. Note dgcMatrix is 0-index !!
    treated_rows = which(data[[treatment]] == 1L) - 1
    idx = x10@i %in% treated_rows
    x10@x[idx] <- 0
    
    # x2'x1 (x10'x10)^-1
    # Note: MatrixExtra makes transposing sparse matrices easy
    # Note: SparseM relies on A (x10'x10) being positive symmetric for solving
    V <- MatrixExtra::t_deep(
      SparseM::solve(
        Matrix::crossprod(x10),
        MatrixExtra::t_shallow(Matrix::crossprod(x2, x1))
      )
    )
    
    # Unique values of cluster variable
    cl = data[[cluster_var]]
    cls = split(1:length(cl), as.factor(cl))
    
    for (i in 1:length(cls)) {
      in_cl = cls[[i]]
      
      x2_g = x2[in_cl, , drop = FALSE]
      x10_g = x10[in_cl, , drop = FALSE]
      first_u_g = first_u[in_cl]
      second_u_g = second_u[in_cl]
      
      W = Matrix::crossprod(x2_g, second_u_g) - V %*% Matrix::crossprod(x10_g, first_u_g)
      
      # W' W
      if(i == 1) { 
        meat_sum = Matrix::tcrossprod(W)
      } else {
        meat_sum = meat_sum + Matrix::tcrossprod(W)
      }
    }
    
    # (X_2'X_2)^-1 (sum W_g W_g') (X_2'X_2)^-1
    bread = SparseM::solve(Matrix::crossprod(x2))
    cov <- as.matrix(bread %*% meat_sum %*% bread)
  }
  
  
  # Bootstrap Standard Errors ------------------------------------------------
  if (bootstrap) {
    message(paste0("Starting ", n_bootstraps, " bootstraps at cluster level: ", cluster_var, "\n"))
    
    # Unique values of cluster variable
    cl <- unique(data[[cluster_var]])
    
    stat <- function(x, i) {
      # select the observations to subset based on the cluster var
      block_obs <- unlist(lapply(i, function(n) which(x[n] == data[[cluster_var]])))
      # run regression for given replicate, return estimated coefficients
      stats::coefficients(
        did2s_estimate(
          data = data[block_obs, ],
          yname = yname,
          first_stage = first_stage,
          second_stage = second_stage,
          treatment = treatment,
          weights = weights,
          bootstrap = TRUE
        )$second_stage
      )
    }
    
    boot <- boot::boot(cl, stat, n_bootstraps)
    
    # Get estimates and fix names
    estimates <- boot$t
    colnames(estimates) <- names(stats::coef(est$second_stage))
    
    # Bootstrap Var-Cov Matrix
    cov <- stats::cov(estimates)
    
    if (return_bootstrap) {
      return(estimates)
    }
  }
  
  # summary creates fixest object with correct standard errors and vcov
  
  # Once fixest updates on CRAN
  # rescale cov by G/(G-1) and use t(G-1) distribution
  # G = length(cl)
  # cov = cov * G/(G-1)
  
  return(base::suppressWarnings(
    # summary(
    #   est$second_stage,
    #   .vcov = list("Two-stage Adjusted" = cov),
    #   ssc = ssc(adj = FALSE, t.df = G-1)
    # )
    summary(est$second_stage, .vcov = cov)
  ))
}


# Point estimate for did2s
did2s_estimate <- function(data, yname, first_stage, second_stage, treatment,
                           weights = NULL, bootstrap = FALSE) {
  ## We'll use fixest's formula expansion macros to swap out first and second
  ## stages (see: ?fixest::xpd)
  fixest::setFixest_fml(
    ..first_stage = first_stage,
    ..second_stage = second_stage
  )
  
  
  # First stage among untreated
  untreat <- data[data[[treatment]] == 0, ]
  if (is.null(weights)) {
    weights_vector <- NULL
  } else {
    weights_vector <- untreat[[weights]]
  }
  
  first_stage <- fixest::feols(fixest::xpd(~ 0 + ..first_stage, lhs = yname),
                               data = untreat,
                               weights = weights_vector,
                               combine.quick = FALSE, # allows var1^var2 in FEs
                               warn = FALSE,
                               notes = FALSE
  )
  
  # Residualize outcome variable but keep same yname
  first_u <- data[[yname]] - stats::predict(first_stage, newdata = data)
  data[[yname]] <- first_u
  
  # Zero out residual rows with D_it = 1 (for analytical SEs later on)
  if (!bootstrap) first_u[data[[treatment]] == 1] <- 0
  
  # Second stage
  
  if (!is.null(weights)) weights_vector <- data[[weights]]
  
  second_stage <- fixest::feols(fixest::xpd(~ 0 + ..second_stage, lhs = yname),
                                data = data,
                                weights = weights_vector,
                                warn = FALSE,
                                notes = FALSE
  )
  
  ret <- list(
    first_stage = first_stage,
    second_stage = second_stage
  )
  
  if (!bootstrap) {
    ret <- list(
      first_stage = first_stage,
      second_stage = second_stage,
      first_u = first_u
    )
  } else {
    ret <- list(second_stage = second_stage)
  }
  
  return(ret)
}