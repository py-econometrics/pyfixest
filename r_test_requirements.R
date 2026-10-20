# R and the canonical reference packages are installed by Pixi.
args <- commandArgs(trailingOnly = TRUE)
required <- c("avlm", "did2s", "ivDiag", "wildrwolf")
if (!"--extended-only" %in% args) {
    required <- c(required, "collapse", "summclust", "ritest")
}

# Use a separate, cacheable user library in CI, without modifying Pixi's prefix.
user_library <- Sys.getenv("R_LIBS_USER")
if (nzchar(user_library) && !grepl("%", user_library, fixed = TRUE)) {
    dir.create(user_library, recursive = TRUE, showWarnings = FALSE)
    .libPaths(c(user_library, .libPaths()))
}

missing_packages <- function() {
    required[!vapply(required, requireNamespace, logical(1), quietly = TRUE)]
}
missing <- missing_packages()
if (length(missing) && !"--check" %in% args) {
    install.packages(
        missing,
        repos = c(
            "https://s3alfisc.r-universe.dev",
            "https://grantmcdermott.r-universe.dev",
            "https://fastverse.r-universe.dev",
            "https://cloud.r-project.org"
        ),
        Ncpus = 2L
    )
    missing <- missing_packages()
}
if (length(missing)) {
    stop(
        "Required R reference packages could not be loaded: ",
        paste(missing, collapse = ", "),
        ". Run pixi run -e py312-r Rscript r_test_requirements.R --extended-only."
    )
}
for (package in required) {
    message(package, " ", as.character(packageVersion(package)))
}
