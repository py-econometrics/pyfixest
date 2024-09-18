# note: fixest and broom are installed via conda
install.packages(
    c('fixest', 'broom','clubSandwich', 'did2s', 'wildrwolf', 'reticulate', 'ivDiag'),
    repos='https://cran.rstudio.com',
);
install.packages(
    'ritest',
    repos = c('https://grantmcdermott.r-universe.dev', 'https://cloud.r-project.org'),
);
