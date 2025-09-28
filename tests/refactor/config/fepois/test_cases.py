"""
FEPOIS-specific test case implementation.
"""
from dataclasses import dataclass
from typing import Dict, Any, Union, Optional
from ..base.test_cases import BaseTestCase


@dataclass
class TestSingleFitFepois(BaseTestCase):
    """Test case for test_single_fit_fepois function migration."""

    # Tell pytest this is not a test class
    __test__ = False

    # Data generation parameters
    N: int = 1000
    seed: int = 7651  # Different seed from FEOLS
    beta_type: str = "2"
    error_type: str = "2"

    # Estimation parameters
    inference: Union[str, Dict[str, str]] = "iid"  # "iid", "hetero", or {"CRV1": "group_id"}

    # FEPOIS-specific parameters (fixed values based on original test)
    dropna: bool = False  # Always False for FEPOIS
    ssc_adj: bool = True  # Always True for FEPOIS
    ssc_cluster_adj: bool = True  # Always True for FEPOIS
    f3_type: str = "str"  # Always "str" for FEPOIS

    # FEPOIS algorithm parameters
    iwls_tol: float = 1e-10
    iwls_maxiter: int = 100

    @property
    def test_method(self) -> str:
        return "fepois"

    def get_data_params(self) -> Dict[str, Any]:
        """Return parameters for data generation."""
        return {
            "N": self.N,
            "seed": self.seed,
            "beta_type": self.beta_type,
            "error_type": self.error_type,
            "model": "Fepois",  # Different from FEOLS
            "f3_type": self.f3_type
        }

    def get_estimation_params(self) -> Dict[str, Any]:
        """Return parameters for estimation."""
        return {
            "vcov": self.inference,
            "dropna": self.dropna,
            "ssc": {
                "adj": self.ssc_adj,
                "cluster_adj": self.ssc_cluster_adj
            },
            # FEPOIS-specific parameters
            "iwls_tol": self.iwls_tol,
            "iwls_maxiter": self.iwls_maxiter
        }

    def validate_params(self) -> bool:
        """Validate FEPOIS test parameters."""
        if self.N <= 0:
            return False

        # FEPOIS only supports string f3_type
        if self.f3_type != "str":
            return False

        # FEPOIS always uses dropna=False in original tests
        if self.dropna != False:
            return False

        # FEPOIS always uses adj=True and cluster_adj=True in original tests
        if not self.ssc_adj or not self.ssc_cluster_adj:
            return False

        # Validate inference types
        if isinstance(self.inference, dict) and "CRV1" in self.inference:
            # Validate cluster variable exists
            cluster_var = self.inference["CRV1"]
            if cluster_var not in ["group_id", "f1", "f2", "f3"]:
                return False
        elif isinstance(self.inference, str):
            if self.inference not in ["iid", "hetero"]:
                return False
        else:
            return False

        # Validate algorithm parameters
        if self.iwls_tol <= 0 or self.iwls_maxiter <= 0:
            return False

        return True
