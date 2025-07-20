"""
Multiple estimation strategies for expanding variables.

This module contains the strategy pattern implementations for different
types of multiple estimation syntax (sw, csw, sw0, csw0).
"""

from abc import ABC, abstractmethod
from typing import List

from ..core.types import MultipleEstimationType


class MultipleEstimationStrategy(ABC):
    """
    Abstract base class for different multiple estimation strategies.

    This class defines the interface for expanding variables according to
    different multiple estimation patterns. Concrete implementations handle
    specific syntax like stepwise (sw) or cumulative stepwise (csw).
    """

    @abstractmethod
    def expand_variables(self, variables: List[str], include_zero: bool = False) -> List[List[str]]:
        """
        Expand variables according to the strategy.

        Parameters
        ----------
        variables : List[str]
            Variables to expand
        include_zero : bool
            Whether to include a model with no variables (for sw0/csw0)

        Returns
        -------
        List[List[str]]
            List of variable combinations according to the strategy
        """
        pass


class StepwiseStrategy(MultipleEstimationStrategy):
    """
    Handles sw() and sw0() syntax.

    Creates models with each individual variable separately:
    sw(x1, x2, x3) -> [[x1], [x2], [x3]]
    sw0(x1, x2, x3) -> [[], [x1], [x2], [x3]]

    This strategy is useful when you want to test the effect of each
    variable individually.
    """

    def expand_variables(self, variables: List[str], include_zero: bool = False) -> List[List[str]]:
        """
        Expand variables using stepwise strategy.

        Parameters
        ----------
        variables : List[str]
            Variables to expand
        include_zero : bool
            Whether to include an empty model (for sw0)

        Returns
        -------
        List[List[str]]
            Each variable as a separate model

        Examples
        --------
        >>> strategy = StepwiseStrategy()
        >>> strategy.expand_variables(["x1", "x2", "x3"])
        [['x1'], ['x2'], ['x3']]

        >>> strategy.expand_variables(["x1", "x2"], include_zero=True)
        [[], ['x1'], ['x2']]
        """
        result = []

        if include_zero:
            result.append([])

        for var in variables:
            result.append([var])

        return result


class CumulativeStepwiseStrategy(MultipleEstimationStrategy):
    """
    Handles csw() and csw0() syntax.

    Creates models with cumulative variables:
    csw(x1, x2, x3) -> [[x1], [x1, x2], [x1, x2, x3]]
    csw0(x1, x2, x3) -> [[], [x1], [x1, x2], [x1, x2, x3]]

    This strategy is useful for testing nested models where each subsequent
    model includes all previous variables plus one additional variable.
    """

    def expand_variables(self, variables: List[str], include_zero: bool = False) -> List[List[str]]:
        """
        Expand variables using cumulative stepwise strategy.

        Parameters
        ----------
        variables : List[str]
            Variables to expand in order
        include_zero : bool
            Whether to include an empty model (for csw0)

        Returns
        -------
        List[List[str]]
            Cumulative combinations of variables

        Examples
        --------
        >>> strategy = CumulativeStepwiseStrategy()
        >>> strategy.expand_variables(["x1", "x2", "x3"])
        [['x1'], ['x1', 'x2'], ['x1', 'x2', 'x3']]

        >>> strategy.expand_variables(["x1", "x2"], include_zero=True)
        [[], ['x1'], ['x1', 'x2']]
        """
        result = []

        if include_zero:
            result.append([])

        for i in range(1, len(variables) + 1):
            result.append(variables[:i])

        return result


class StrategyFactory:
    """
    Factory for creating multiple estimation strategies.

    This factory provides a centralized way to create the appropriate
    strategy instance based on the multiple estimation type.
    """

    @staticmethod
    def create_strategy(estimation_type: MultipleEstimationType) -> MultipleEstimationStrategy:
        """
        Create the appropriate strategy for the given estimation type.

        Parameters
        ----------
        estimation_type : MultipleEstimationType
            The type of multiple estimation syntax

        Returns
        -------
        MultipleEstimationStrategy
            The appropriate strategy instance

        Raises
        ------
        ValueError
            If the estimation type is not supported
        """
        if estimation_type in [MultipleEstimationType.STEPWISE, MultipleEstimationType.STEPWISE_ZERO]:
            return StepwiseStrategy()
        elif estimation_type in [MultipleEstimationType.CUMULATIVE_STEPWISE, MultipleEstimationType.CUMULATIVE_STEPWISE_ZERO]:
            return CumulativeStepwiseStrategy()
        else:
            raise ValueError(f"Unsupported multiple estimation type: {estimation_type}")

    @staticmethod
    def get_all_strategies() -> dict[MultipleEstimationType, MultipleEstimationStrategy]:
        """
        Get a dictionary mapping all estimation types to their strategies.

        Returns
        -------
        dict[MultipleEstimationType, MultipleEstimationStrategy]
            Mapping of estimation types to strategy instances
        """
        return {
            MultipleEstimationType.STEPWISE: StepwiseStrategy(),
            MultipleEstimationType.STEPWISE_ZERO: StepwiseStrategy(),
            MultipleEstimationType.CUMULATIVE_STEPWISE: CumulativeStepwiseStrategy(),
            MultipleEstimationType.CUMULATIVE_STEPWISE_ZERO: CumulativeStepwiseStrategy(),
        }