suppressPackageStartupMessages(library(fixest))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly = TRUE)
case <- fromJSON(args[[2]], simplifyVector = FALSE)
data <- read.csv(file.path(args[[1]], case$data))
reps <- as.integer(args[[4]])
threads <- as.integer(args[[5]])
setFixest_nthreads(threads)
fit <- function(label) {
  formulas <- if (label == "separate") case$separate else list(case$r_multi)
  options <- list(data = data, vcov = "iid",
    fixef.rm = if (case$scenario == "singletons") "singleton" else "none",
    fixef.tol = 1e-8, fixef.iter = 10000, nthreads = threads,
    lean = as.logical(as.integer(args[[6]])),
    data.save = !as.logical(as.integer(args[[7]])), notes = FALSE
  )
  if (case$scenario == "weights") options$weights <- ~weight
  result <- lapply(formulas, function(f) do.call(feols, c(list(fml = as.formula(f)), options)))
  if (label == "multi") unname(as.list(result[[1]])) else result
}
estimates <- function(models) lapply(models, function(m) list(
  names = I(names(coef(m))), coef = I(unname(coef(m))),
  se = I(unname(se(m))), vcov = matrix(as.numeric(vcov(m)), nrow = length(coef(m))), nobs = nobs(m)
))
a <- estimates(fit("separate"))
b <- estimates(fit("multi"))
stopifnot(length(a) == case$models, length(b) == case$models)
for (i in seq_along(a)) stopifnot(isTRUE(all.equal(a[[i]], b[[i]], tolerance = 1e-7)))
rm(a)
times <- list(separate = c(), multi = c())
for (rep in seq_len(reps)) {
  order <- if (rep %% 2 == 1) c("separate", "multi") else c("multi", "separate")
  for (label in order) {
    gc()
    start <- proc.time()[["elapsed"]]
    result <- fit(label)
    times[[label]] <- c(times[[label]], proc.time()[["elapsed"]] - start)
    stopifnot(length(result) == case$models)
    rm(result)
  }
}
write_json(list(case = case, backend = "fixest", status = "passed",
                times = lapply(times, I), estimates = b,
                environment = list(R = R.version.string, fixest = as.character(packageVersion("fixest")),
                                   platform = R.version$platform, threads = threads)),
           args[[3]], auto_unbox = TRUE, pretty = TRUE, digits = 16)
