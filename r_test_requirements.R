# note: R, fixest, sandwich, broom are installed via conda
install.packages(
    c('did2s', 'reticulate', 'ivDiag'),
    repos='https://cran.rstudio.com'
);
install.packages(
   c('collapse', 'summclust', 'wildrwolf'),
    repos = c('https://s3alfisc.r-universe.dev', 'https://cloud.r-project.org', 'https://fastverse.r-universe.dev')
);
install.packages(
    'ritest',
    repos = c('https://grantmcdermott.r-universe.dev', 'https://cloud.r-project.org')
);
