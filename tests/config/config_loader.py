"""
Configuration loader for pyfixest test specifications.

This module provides utilities to load and work with the shared test configuration
that's used by both Python tests and R result generation.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Union
import pandas as pd


class TestConfigLoader:
    """Loads and provides access to test specifications."""

    def __init__(self, config_path: str = None):
        """Initialize the config loader.

        Args:
            config_path: Path to the configuration file. If None, uses default location.
        """
        if config_path is None:
            # Default path relative to this file
            config_dir = Path(__file__).parent
            config_path = config_dir / "test_specifications.json"

        self.config_path = Path(config_path)
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load the configuration from JSON file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(self.config_path, 'r') as f:
            return json.load(f)

    def get_data_params(self, test_type: str = "feols") -> Dict[str, Any]:
        """Get data generation parameters for a specific test type."""
        return self.config["data_generation"].get(f"{test_type}_params",
                                                 self.config["data_generation"]["default_params"])

    def get_tolerance(self, test_type: str = "default") -> Dict[str, float]:
        """Get tolerance settings for a specific test type."""
        return self.config["tolerance_settings"].get(test_type,
                                                    self.config["tolerance_settings"]["default"])

    def get_formulas(self, test_type: str) -> List[str]:
        """Get formulas for a specific test type."""
        test_config = self.config["test_configurations"].get(test_type, {})
        return test_config.get("formulas", [])

    def get_test_config(self, test_type: str) -> Dict[str, Any]:
        """Get complete test configuration for a specific test type."""
        return self.config["test_configurations"].get(test_type, {})

    def get_inference_types(self, test_type: str) -> List[Union[str, Dict[str, str]]]:
        """Get inference types for a specific test type."""
        test_config = self.get_test_config(test_type)
        return test_config.get("inference_types", ["iid"])

    def get_weights_options(self, test_type: str) -> List[Union[str, None]]:
        """Get weight options for a specific test type."""
        test_config = self.get_test_config(test_type)
        return test_config.get("weights", [None])

    def get_dropna_options(self, test_type: str) -> List[bool]:
        """Get dropna options for a specific test type."""
        test_config = self.get_test_config(test_type)
        return test_config.get("dropna", [False])

    def get_output_fields(self, field_type: str = "core_results") -> List[str]:
        """Get list of output fields for result comparison."""
        return self.config["output_fields"].get(field_type, [])

    def get_estimation_settings(self) -> Dict[str, Any]:
        """Get estimation settings (IWLS, SSC, etc.)."""
        return self.config["estimation_settings"]

    def generate_test_combinations(self, test_type: str) -> List[Dict[str, Any]]:
        """Generate all parameter combinations for a test type.

        Returns a list of dictionaries, each containing one combination of parameters
        for testing.
        """
        test_config = self.get_test_config(test_type)

        if not test_config:
            return []

        # Get all parameter lists
        formulas = test_config.get("formulas", [])
        inference_types = test_config.get("inference_types", ["iid"])
        weights = test_config.get("weights", [None])
        dropna_options = test_config.get("dropna", [False])

        # Special handling for GLM which has families
        families = test_config.get("families", [None])

        # Generate all combinations
        combinations = []
        for formula in formulas:
            for inference in inference_types:
                for weight in weights:
                    for dropna in dropna_options:
                        if families[0] is not None:  # GLM case
                            for family in families:
                                combinations.append({
                                    "formula": formula,
                                    "inference": inference,
                                    "weights": weight,
                                    "dropna": dropna,
                                    "family": family,
                                    "test_type": test_type
                                })
                        else:
                            combinations.append({
                                "formula": formula,
                                "inference": inference,
                                "weights": weight,
                                "dropna": dropna,
                                "test_type": test_type
                            })

        return combinations


class CachedResultsLoader:
    """Loads cached R results for comparison with Python results."""

    def __init__(self, cache_dir: str = None):
        """Initialize the cached results loader.

        Args:
            cache_dir: Directory containing cached CSV results. If None, uses default.
        """
        if cache_dir is None:
            # Default path relative to this file
            test_dir = Path(__file__).parent.parent
            cache_dir = test_dir / "data" / "cached_results"

        self.cache_dir = Path(cache_dir)
        self._cached_results = {}

    def load_results(self, test_type: str) -> pd.DataFrame:
        """Load cached results for a specific test type.

        Args:
            test_type: Type of test (feols, iv, glm, fepois)

        Returns:
            DataFrame with cached R results
        """
        if test_type not in self._cached_results:
            result_file = self.cache_dir / f"{test_type}_results.csv"

            if not result_file.exists():
                raise FileNotFoundError(f"Cached results not found: {result_file}")

            df = pd.read_csv(result_file)
            # Set multi-index for easy lookup
            index_cols = ["formula", "inference", "weights", "dropna"]
            if "family" in df.columns:
                index_cols.append("family")

            df = df.set_index(index_cols)
            self._cached_results[test_type] = df

        return self._cached_results[test_type]

    def get_result(self, test_type: str, formula: str, inference: str,
                   weights: str = "none", dropna: bool = False,
                   family: str = "none") -> pd.Series:
        """Get a specific cached result.

        Args:
            test_type: Type of test (feols, iv, glm, fepois)
            formula: Model formula
            inference: Inference type
            weights: Weights specification
            dropna: Whether dropna was applied
            family: Family for GLM models

        Returns:
            Series with the cached result
        """
        df = self.load_results(test_type)

        # Prepare lookup key
        if "family" in df.index.names:
            key = (formula, inference, weights, dropna, family)
        else:
            key = (formula, inference, weights, dropna)

        try:
            return df.loc[key]
        except KeyError:
            raise KeyError(f"Result not found for {test_type}: {key}")

    def get_metadata(self) -> pd.DataFrame:
        """Load metadata about cached results."""
        metadata_file = self.cache_dir / "metadata.csv"
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

        return pd.read_csv(metadata_file)

    def is_cache_valid(self) -> bool:
        """Check if cached results exist and seem valid."""
        metadata_file = self.cache_dir / "metadata.csv"
        return metadata_file.exists()

    def list_available_test_types(self) -> List[str]:
        """List available test types in cache."""
        if not self.cache_dir.exists():
            return []

        result_files = list(self.cache_dir.glob("*_results.csv"))
        return [f.stem.replace("_results", "") for f in result_files]


def inference_to_string(inference: Union[str, Dict[str, str]]) -> str:
    """Convert inference specification to string for indexing."""
    if isinstance(inference, str):
        return inference
    elif isinstance(inference, dict) and "CRV1" in inference:
        return "CRV1_group_id"
    else:
        return str(inference)


def weights_to_string(weights: Union[str, None]) -> str:
    """Convert weights specification to string for indexing."""
    return "none" if weights is None else weights
