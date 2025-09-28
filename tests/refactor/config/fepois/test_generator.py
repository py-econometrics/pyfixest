"""
Generator for FEPOIS test cases based on original test_single_fit_fepois parameters.
"""
from typing import List
from .test_cases import TestSingleFitFepois


# Original formulas from test_vs_fixest.py (ols_fmls only, not ols_but_not_poisson_fml)
OLS_FMLS = [
    "Y~X1",
    #"Y~X1+X2",
    #"Y~X1|f2",
    #"Y~X1|f2+f3",
    #"Y ~ X1 + exp(X2)",
    #"Y ~ X1 + C(f1)",
    #"Y ~ X1 + i(f1, ref = 1)",
    #"Y ~ X1 + C(f1)",
    #"Y ~ X1 + i(f2, ref = 2.0)",
    #"Y ~ X1 + C(f1) + C(f2)",
    #"Y ~ X1 + C(f1) | f2",
    #"Y ~ X1 + i(f1, ref = 3.0) | f2",
    #"Y ~ X1 + C(f1) | f2 + f3",
    #"Y ~ X1 + i(f1, ref = 1) | f2 + f3",
    #"Y ~ X1 + i(f1) + i(f2)",
    #"Y ~ X1 + i(f1, ref = 1) + i(f2, ref = 2)",
    ## ("Y ~ X1 + C(f1):C(fe2)"),                  # currently does not work as C():C() translation not implemented
    ## ("Y ~ X1 + C(f1):C(fe2) | f3"),             # currently does not work as C():C() translation not implemented
    #"Y ~ X1 + X2:f1",
    #"Y ~ X1 + X2:f1 | f3",
    #"Y ~ X1 + X2:f1 | f3 + f1",
    ## ("log(Y) ~ X1:X2 | f3 + f1"),               # currently, causes big problems for Fepois (takes a long time)
    ## ("log(Y) ~ log(X1):X2 | f3 + f1"),          # currently, causes big problems for Fepois (takes a long time)
    ## ("Y ~  X2 + exp(X1) | f3 + f1"),            # currently, causes big problems for Fepois (takes a long time)
    #"Y ~ X1 + i(f1,X2)",
    #"Y ~ X1 + i(f1,X2) + i(f2, X2)",
    #"Y ~ X1 + i(f1,X2, ref =1) + i(f2)",
    #"Y ~ X1 + i(f1,X2, ref =1) + i(f2, X1, ref =2)",
    #"Y ~ X1 + i(f2,X2)",
    #"Y ~ X1 + i(f1,X2) | f2",
    #"Y ~ X1 + i(f1,X2) | f2 + f3",
    #"Y ~ X1 + i(f1,X2, ref=1.0)",
    #"Y ~ X1 + i(f2,X2, ref=2.0)",
    #"Y ~ X1 + i(f1,X2, ref=3.0) | f2",
    ## ("Y ~ C(f1):C(f2)"),                       # currently does not work
    #"Y ~ X1 + I(X2 ** 2)",
    #"Y ~ X1 + I(X1 ** 2) + I(X2**4)",
    #"Y ~ X1*X2",
    #"Y ~ X1*X2 | f1+f2",
    ## ("Y ~ X1/X2"),                             # currently does not work as X1/X2 translation not implemented
    ## ("Y ~ X1/X2 | f1+f2"),                     # currently does not work as X1/X2 translation not implemented
    #"Y ~ X1 + poly(X2, 2) | f1",
]


def should_skip_f3_checks(fml: str, f3_type: str) -> bool:
    """
    Skip f3 type checks if f3 is not in formula and f3_type is not 'str'.
    Based on _skip_f3_checks from original test.
    """
    return ("f3" not in fml) and (f3_type != "str")


def should_skip_dropna(test_counter: int, dropna: bool) -> bool:
    """
    Skip dropna tests based on counter (reduced frequency).
    Based on _skip_dropna from original test.
    """
    return dropna and (test_counter % 10 != 0)


def generate_fepois_test_cases() -> List[TestSingleFitFepois]:
    """Generate all FEPOIS test cases matching the original test_single_fit_fepois."""
    test_cases = []
    test_counter = 0

    # FEPOIS test parameters (based on original test parametrization)
    formulas = OLS_FMLS
    inference_types = ["iid", "hetero", {"CRV1": "group_id"}]

    # FEPOIS has fixed parameters (from original test)
    dropna = False  # Always False
    f3_type = "str"  # Always "str"
    adj = True  # Always True
    cluster_adj = True  # Always True

    for fml in formulas:
        for inference in inference_types:
            test_counter += 1

            # Apply original skip logic
            if should_skip_f3_checks(fml, f3_type):
                continue
            if should_skip_dropna(test_counter, dropna):
                continue

            test_case = TestSingleFitFepois(
                test_id=f"fepois_{test_counter:05d}",
                formula=fml,
                inference=inference,
                dropna=dropna,
                ssc_adj=adj,
                ssc_cluster_adj=cluster_adj,
                f3_type=f3_type,
                description=f"FEPOIS: {fml} | inf={inference}"
            )

            if test_case.validate_params():
                test_cases.append(test_case)

    return test_cases


def setup_fepois_test_registry():
    """Setup the test registry with all FEPOIS test cases."""
    from refactor.config.test_registry import TEST_REGISTRY

    test_cases = generate_fepois_test_cases()
    for test_case in test_cases:
        TEST_REGISTRY.register_test_case(test_case)
