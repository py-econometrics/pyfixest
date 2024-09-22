import rpy2.robjects as robjects
from rpy2.robjects.packages import importr

# Set the R library paths again
new_lib_paths = robjects.StrVector(
    [
        "/home/runner/work/pyfixest/pyfixest/.pixi/envs/dev/lib/R/library",
        "/usr/local/lib/R/site-library",
        "/usr/lib/R/site-library",
        "/usr/lib/R/library",
    ]
)
robjects.r[".libPaths"](new_lib_paths)

# Check if did2s is installed
try:
    did2s = importr("did2s")
    print("did2s package imported successfully.")
except Exception as e:
    print(f"Error importing did2s: {e}")
