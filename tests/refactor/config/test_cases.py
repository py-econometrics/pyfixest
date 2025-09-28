"""
Simple abstract base class for test cases with caching support.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional
import json
import hashlib


@dataclass
class BaseTestCase(ABC):
    """Abstract base class for all test cases."""

    test_id: str
    formula: str
    description: str = ""

    @property
    @abstractmethod
    def test_group(self) -> str:
        """Return the test group this case belongs to (e.g., 'feols', 'fepois')."""
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
        """Validate that parameters are consistent."""
        pass

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'test_id': self.test_id,
            'test_group': self.test_group,
            'formula': self.formula,
            'description': self.description,
            'data_params': self.get_data_params(),
            'estimation_params': self.get_estimation_params(),
            'class_name': self.__class__.__name__
        }

    def get_hash(self) -> str:
        """Generate unique hash for this test case."""
        # Exclude test_id and description from hash to focus on actual test parameters
        hash_dict = self.to_dict().copy()
        hash_dict.pop('test_id', None)
        hash_dict.pop('description', None)
        content = json.dumps(hash_dict, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()
