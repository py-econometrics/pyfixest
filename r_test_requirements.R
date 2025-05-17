# note: R, fixest, sandwich, broom are installed via conda
install.packages(
    c('did2s', 'wildrwolf', 'reticulate', 'ivDiag', 'car'),
    repos='https://cran.rstudio.com'
);
install.packages(
    'ritest',
    repos = c('https://grantmcdermott.r-universe.dev', 'https://cloud.r-project.org')
);
