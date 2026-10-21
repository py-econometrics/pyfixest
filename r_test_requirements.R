# R and the canonical reference packages are installed by Pixi.
args <- commandArgs(trailingOnly = TRUE)
required <- c("avlm", "did2s", "ivDiag", "wildrwolf")
if (!"--extended-only" %in% args) {
    required <- c(required, "collapse", "summclust", "ritest")
}

missing <- required[!vapply(required, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing)) {
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
}
missing <- required[!vapply(required, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing)) {
    stop(
        "Required R reference packages could not be loaded: ",
        paste(missing, collapse = ", "),
        ". Run pixi run -e py312-r Rscript r_test_requirements.R --extended-only."
    )
}
