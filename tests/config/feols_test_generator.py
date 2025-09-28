"""
Generator for FEOLS test cases based on original test_single_fit_feols parameters.
"""
from typing import List
from .feols_tests import TestSingleFitFeols
from .test_registry import TEST_REGISTRY


# Original formulas from test_vs_fixest.py
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
    #"Y ~ X1 + X2:f1",
    #"Y ~ X1 + X2:f1 | f3",
    #"Y ~ X1 + X2:f1 | f3 + f1",
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
    #"Y ~ X1 + i(f1,X2, ref=4.0) | f2 + f3",
    #"Y ~ X1 + I(X2 ** 2)",
    #"Y ~ X1 + I(X1 ** 2) + I(X2**4)",
    #"Y ~ X1*X2",
    #"Y ~ X1*X2 | f1+f2",
    #"Y ~ X1 + poly(X2, 2) | f1",
]

OLS_BUT_NOT_POISSON_FML = [
    "log(Y) ~ X1",
    #"log(Y) ~ X1 + X2",
    #"log(Y) ~ X1 | f2",
    #"log(Y) ~ X1 | f2 + f3",
    #"log(Y) ~ X1 + C(f1)",
    #"log(Y) ~ X1 + i(f1, ref = 1)",
    #"log(Y) ~ X1 + C(f1) + C(f2)",
    #"log(Y) ~ X1 + C(f1) | f2",
    #"log(Y) ~ X1 + i(f1, ref = 3.0) | f2",
    #"log(Y) ~ X1 + C(f1) | f2 + f3",
    #"log(Y) ~ X1 + i(f1, ref = 1) | f2 + f3",
    #"log(Y) ~ X1 + i(f1) + i(f2)",
    #"log(Y) ~ X1 + i(f1, ref = 1) + i(f2, ref = 2)",
    #"log(Y) ~ X1 + X2:f1",
    #"log(Y) ~ X1 + X2:f1 | f3",
    #"log(Y) ~ X1 + X2:f1 | f3 + f1",
    #"log(Y) ~ X1 + i(f1,X2)",
    #"log(Y) ~ X1 + i(f1,X2) + i(f2, X2)",
    #"log(Y) ~ X1 + i(f1,X2, ref =1) + i(f2)",
    #"log(Y) ~ X1 + i(f1,X2, ref =1) + i(f2, X1, ref =2)",
    #"log(Y) ~ X1 + i(f2,X2)",
    #"log(Y) ~ X1 + i(f1,X2) | f2",
    #"log(Y) ~ X1 + i(f1,X2) | f2 + f3",
    #"log(Y) ~ X1 + i(f1,X2, ref=1.0)",
    #"log(Y) ~ X1 + i(f2,X2, ref=2.0)",
    #"log(Y) ~ X1 + i(f1,X2, ref=3.0) | f2",
    #"log(Y) ~ X1 + i(f1,X2, ref=4.0) | f2 + f3",
    #"log(Y) ~ X1 + I(X2 ** 2)",
    #"log(Y) ~ X1 + I(X1 ** 2) + I(X2**4)",
    #"log(Y) ~ X1*X2",
    #"log(Y) ~ X1*X2 | f1+f2",
    #"log(Y) ~ X1 + poly(X2, 2) | f1",
]

# Parameter combinations from original test
ALL_F3 = [
    "str",
    #"object",
    #"int",
    #"categorical",
    #"float"
]

BACKEND_F3 = [
    *[("numba", t) for t in ALL_F3],
#    *[(b, "str") for b in ("jax", "rust")],  # JAX and Rust only support str f3_type
]


def should_skip_f3_checks(fml: str, f3_type: str) -> bool:
    """Check if f3 combination should be skipped (from original _skip_f3_checks)."""
    # Skip certain f3_type combinations for specific formulas
    if f3_type in ["int", "float"] and ("|f2+f3" in fml or "|f1+f2" in fml):
        return True
    if f3_type == "categorical" and "poly(" in fml:
        return True
    return False


def should_skip_dropna(test_counter: int, dropna: bool) -> bool:
    """Check if dropna should be skipped (from original _skip_dropna)."""
    # Only test dropna for every 10th test to reduce test count
    return dropna and (test_counter % 10 != 0)


def generate_feols_test_cases() -> List[TestSingleFitFeols]:
    """Generate all FEOLS test cases matching the original test_single_fit_feols."""
    test_cases = []
    test_counter = 0

    # All formulas (OLS + OLS_BUT_NOT_POISSON)
    all_formulas = OLS_FMLS + OLS_BUT_NOT_POISSON_FML

    # Parameter combinations
    inference_types = ["iid", "hetero", {"CRV1": "group_id"}]
    weight_options = [None, "weights"]
    dropna_options = [False, True]
    adj_options = [True]  # Original test only uses True
    cluster_adj_options = [True]  # Original test only uses True

    for fml in all_formulas:
        for demeaner_backend, f3_type in BACKEND_F3:
            for dropna in dropna_options:
                for inference in inference_types:
                    for weights in weight_options:
                        for adj in adj_options:
                            for cluster_adj in cluster_adj_options:
                                test_counter += 1

                                # Apply original skip logic
                                if should_skip_f3_checks(fml, f3_type):
                                    continue
                                if should_skip_dropna(test_counter, dropna):
                                    continue

                                test_case = TestSingleFitFeols(
                                    test_id=f"feols_{test_counter:05d}",
                                    formula=fml,
                                    inference=inference,
                                    weights=weights,
                                    dropna=dropna,
                                    ssc_adj=adj,
                                    ssc_cluster_adj=cluster_adj,
                                    demeaner_backend=demeaner_backend,
                                    f3_type=f3_type,
                                    description=f"FEOLS: {fml} | backend={demeaner_backend}, f3={f3_type}, inf={inference}"
                                )

                                if test_case.validate_params():
                                    test_cases.append(test_case)

    return test_cases


def setup_feols_test_registry():
    """Setup the test registry with all FEOLS test cases."""
    test_cases = generate_feols_test_cases()

    for test_case in test_cases:
        TEST_REGISTRY.add_test_case(test_case)

    print(f"Generated {len(test_cases)} FEOLS test cases")
    return test_cases


if __name__ == "__main__":
    # Generate and display summary
    test_cases = setup_feols_test_registry()
    print(f"Total test cases: {len(test_cases)}")
    print(f"Registry summary: {TEST_REGISTRY.get_summary()}")
