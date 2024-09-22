import rpy2.robjects as robjects
from rpy2.robjects.packages import importr

# Define the list of R library paths in Python
new_lib_paths = robjects.StrVector(
    [
        "/home/runner/work/pyfixest/pyfixest/.pixi/envs/dev/lib/R/library",
        "/usr/local/lib/R/site-library",
        "/usr/lib/R/site-library",
        "/usr/lib/R/library",
    ]
)

# Set the library paths in the R environment
robjects.r[".libPaths"](new_lib_paths)

# Verify that the paths have been set correctly
print("Updated R library paths:", robjects.r(".libPaths()"))

# Now attempt to import the did2s package
did2s = importr("did2s")
