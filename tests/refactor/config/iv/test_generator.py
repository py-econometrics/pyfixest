"""
Generator for IV test cases based on original test_single_fit_iv parameters.
"""
from typing import List
from .test_cases import TestSingleFitIv


# Original IV formulas from test_vs_fixest.py
IV_FMLS = [
    # IV starts here
    "Y ~ 1 | X1 ~ Z1",
    "Y ~  X2 | X1 ~ Z1",
    "Y ~ X2 + C(f1) | X1 ~ Z1",
    "Y2 ~ 1 | X1 ~ Z1",
    "Y2 ~ X2 | X1 ~ Z1",
    "Y2 ~ X2 + C(f1) | X1 ~ Z1",
    # "log(Y) ~ 1 | X1 ~ Z1",  # Commented out in original
    # "log(Y) ~ X2 | X1 ~ Z1",  # Commented out in original
    # "log(Y) ~ X2 + C(f1) | X1 ~ Z1",  # Commented out in original
    "Y ~ 1 | f1 | X1 ~ Z1",
    "Y ~ 1 | f1 + f3 | X1 ~ Z1",
    "Y ~ 1 | f1^f2 | X1 ~ Z1",
    "Y ~  X2| f3 | X1 ~ Z1",
    # tests of overidentified models
    "Y ~ 1 | X1 ~ Z1 + Z2",
    "Y ~ X2 | X1 ~ Z1 + Z2",
    "Y ~ X2 + C(f3) | X1 ~ Z1 + Z2",
    "Y ~ 1 | f1 | X1 ~ Z1 + Z2",
    "Y2 ~ 1 | f1 + f3 | X1 ~ Z1 + Z2",
    "Y2 ~  X2| f2 | X1 ~ Z1 + Z2",
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


def generate_iv_test_cases() -> List[TestSingleFitIv]:
    """Generate all IV test cases matching the original test_single_fit_iv."""
    test_cases = []
    test_counter = 0

    # IV test parameters (based on original test parametrization)
    formulas = IV_FMLS
    inference_types = ["iid", "hetero", {"CRV1": "group_id"}]
    weight_options = [None, "weights"]

    # IV has fixed parameters (from original test)
    dropna = False  # Always False
    f3_type = "str"  # Always "str"
    adj = True  # Always True
    cluster_adj = True  # Always True

    for fml in formulas:
        for inference in inference_types:
            for weights in weight_options:
                test_counter += 1

                # Apply original skip logic
                if should_skip_f3_checks(fml, f3_type):
                    continue
                if should_skip_dropna(test_counter, dropna):
                    continue

                test_case = TestSingleFitIv(
                    test_id=f"iv_{test_counter:05d}",
                    formula=fml,
                    inference=inference,
                    weights=weights,
                    dropna=dropna,
                    ssc_adj=adj,
                    ssc_cluster_adj=cluster_adj,
                    f3_type=f3_type,
                    description=f"IV: {fml} | inf={inference}, weights={weights}"
                )

                if test_case.validate_params():
                    test_cases.append(test_case)

    return test_cases


def setup_iv_test_registry():
    """Setup the test registry with all IV test cases."""
    from refactor.config.test_registry import TEST_REGISTRY

    test_cases = generate_iv_test_cases()
    for test_case in test_cases:
        TEST_REGISTRY.register_test_case(test_case)
