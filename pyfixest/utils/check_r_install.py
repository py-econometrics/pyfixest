from rpy2.robjects.packages import importr


def _catch_import_issue(name: str, strict: bool) -> None | bool:
    if strict:
        raise ImportError(
            f"{name} package not found. Make sure the extended R environment is installed."
        )
    else:
        print(
            f"Warning: {name} is not installed. Extended R tests will be unable to run."
        )
        return False


def check_r_install(package_names: str | list[str], strict: bool = False) -> bool:
    "Catch R import issues for package_names and raise ImportError if strict is True, otherwise pass a bool for passing check."
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
