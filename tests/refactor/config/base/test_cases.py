"""
Base classes for test case definitions.

This module provides the abstract base classes that all test method implementations
should inherit from to ensure consistency and shared functionality.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any
import json
import hashlib
from enum import Enum


class TestGroup(Enum):
    """Enumeration of available test groups/methods."""
    FEOLS = "feols"
    FEPOIS = "fepois"
    IV = "iv"


class InferenceType(Enum):
    """Enumeration of inference types."""
    IID = "iid"
    HETERO = "hetero"
    CRV1 = "CRV1"


@dataclass
class BaseTestCase(ABC):
    """
    Abstract base class for all test cases.

    This class provides the common interface and functionality that all
    test method implementations should follow.
    """
    test_id: str
    formula: str
    description: str = ""

    # Tell pytest this is not a test class
    __test__ = False

    @property
    @abstractmethod
    def test_method(self) -> str:
        """Return the test method name (e.g., 'feols', 'fepois', 'iv')."""
        pass

    @abstractmethod
    def get_data_params(self) -> Dict[str, Any]:
        """Return parameters for data generation."""
        pass

    @abstractmethod
    def get_estimation_params(self) -> Dict[str, Any]:
        """Return parameters for estimation."""
        pass

    @abstractmethod
    def validate_params(self) -> bool:
        """Validate test parameters."""
        pass

    def to_dict(self) -> Dict[str, Any]:
        """Convert test case to dictionary for serialization."""
        return {
            'test_id': self.test_id,
            'test_method': self.test_method,
            'formula': self.formula,
            'description': self.description,
            'data_params': self.get_data_params(),
            'estimation_params': self.get_estimation_params(),
            'class_name': self.__class__.__name__
        }

    def get_hash(self) -> str:
        """
        Generate a hash for this test case based on its parameters.

        This hash is used for cache validation to ensure cached results
        match the current test parameters.
        """
        hash_dict = self.to_dict().copy()
        # Remove non-parameter fields from hash calculation
        hash_dict.pop('test_id', None)
        hash_dict.pop('description', None)

        # Create deterministic hash
        content = json.dumps(hash_dict, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()

    @property
    def original_function(self) -> str:
        """Return the name of the original test function this migrates."""
        return f"test_single_fit_{self.test_method}"

    @property
    def original_file(self) -> str:
        """Return the original file containing the test function."""
        return "tests/test_vs_fixest.py"
