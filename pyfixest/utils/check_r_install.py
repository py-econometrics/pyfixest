def _catch_import_issue(name: str, strict: bool) -> bool | None:
    if strict:
        raise ImportError(
            f"R package {name!r} is missing. Install the extended dependencies "
            "with pixi run -e py312-r Rscript r_test_requirements.R."
        )
    else:
        print(
            f"Warning: R package {name!r} is missing. To run the extended tests, "
            "install the dependencies with pixi run -e py312-r Rscript r_test_requirements.R."
        )
        return False


def check_r_install(package_names: str | list[str], strict: bool = False) -> bool:
    "Catch R import issues for package_names and raise ImportError if strict is True, otherwise pass a bool for passing check."
    try:
        from rpy2.robjects.packages import importr
    except ModuleNotFoundError as exc:
        raise ImportError(
            "R dependency checks require rpy2. Run the tests through pixi run -e py312-r."
        ) from exc
    except (OSError, ValueError) as exc:
        raise ImportError(
            "rpy2 could not load the R runtime. Run through pixi run -e py312-r "
            "and check that R and rpy2 are available in that environment."
        ) from exc
    utils = importr("utils")
    package_list = package_names if isinstance(package_names, list) else [package_names]
    installed_packages = utils.installed_packages()

    package_status = []
    for package in package_list:
        if package not in installed_packages:
            package_status.append(_catch_import_issue(package, strict))
        else:
            package_status.append(True)
    return all(package_status)
