from rpy2.robjects.packages import importr


def _catch_import_issue(name: str, strict: bool) -> None:
    if strict:
        raise ImportError(
            f"{name} package not found. Make sure the extended R environment is installed."
        )
    else:
        print(f"Warning: {name} is not installed. Extended R tests will be unable to run.")


def check_r_install(package_names: str | list[str], strict: bool = False) -> None:
    "Catch R import issues for package_names and raise ImportError if strict is True."
    utils = importr("utils")
    package_list = package_names if isinstance(package_names, list) else [package_names]
    installed_packages = utils.installed_packages()

    for package in package_list:
        if package not in installed_packages:
            _catch_import_issue(package, strict)
