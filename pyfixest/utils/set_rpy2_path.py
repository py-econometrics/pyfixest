import rpy2.robjects as robjects
from rpy2.robjects.packages import importr


def update_r_paths():
    "Get current R library paths."
    current_r_paths = robjects.r(".libPaths()")

    # Define your custom paths
    custom_paths = robjects.StrVector(
        [
            "/home/runner/work/pyfixest/pyfixest/.pixi/envs/dev/lib/R/library",
            "/usr/local/lib/R/site-library",
            "/usr/lib/R/site-library",
            "/usr/lib/R/library",
        ]
    )

    # Combine current R paths with custom paths (avoiding duplicates)
    new_lib_paths = robjects.StrVector(list(set(custom_paths).union(current_r_paths)))

    # Set the combined library paths in the R environment
    robjects.r[".libPaths"](new_lib_paths)


def _check_update_r_paths():
    update_r_paths()
    try:
        importr("did2s")
        print("did2s package imported successfully.")
    except Exception as e:
        print(f"Error importing did2s: {e}")


_check_update_r_paths()
