from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from pyfixest.estimation.internals.model_state import (
    GlmWorkingState,
    ObservationWeights,
    WithinIvData,
    WithinLinearData,
)


def test_observation_weights_unweighted_fast_path() -> None:
    weights = ObservationWeights.unweighted(n_rows=4)

    assert weights.values is None
    assert weights.weights_type is None
    assert weights.n_rows == 4
    assert weights.n_effective == 4
    assert isinstance(weights.n_effective, int)
    assert not weights.is_weighted
    assert not hasattr(weights, "__dict__")


@pytest.mark.parametrize(
    ("weights_type", "expected_n"), [("aweights", 3.0), ("fweights", 6.0)]
)
def test_observation_weights_keep_canonical_user_values(
    weights_type, expected_n
) -> None:
    user_weights = np.array([[1.0], [2.0], [3.0]])
    weights = ObservationWeights.from_values(user_weights, weights_type=weights_type)
    np.testing.assert_array_equal(weights.values, user_weights.flatten())
    assert weights.weights_type == weights_type
    assert weights.n_rows == 3
    assert weights.n_effective == expected_n
    assert isinstance(weights.n_effective, int if weights_type == "aweights" else float)
    assert weights.is_weighted


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {
                "values": np.ones(2),
                "weights_type": None,
                "n_rows": 2,
                "n_effective": 2.0,
            },
            "Weighted observations must declare a `weights_type`",
        ),
        (
            {
                "values": np.ones(3),
                "weights_type": "aweights",
                "n_rows": 2,
                "n_effective": 2,
            },
            "Observation weights must contain one value per row",
        ),
    ],
)
def test_observation_weights_reject_inconsistent_state(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        ObservationWeights(**kwargs)


def test_within_linear_data_is_structurally_immutable() -> None:
    response = np.arange(3.0)[:, None]
    design = np.column_stack((np.ones(3), np.arange(3.0)))
    state = WithinLinearData(response=response, design=design)
    assert state.response is response
    assert state.design is design
    assert not hasattr(state, "__dict__")
    assert not hasattr(state, "instruments")
    assert not hasattr(state, "endogenous")
    with pytest.raises(FrozenInstanceError):
        state.response = design  # type: ignore[misc]
    with pytest.raises(TypeError):
        WithinLinearData(response=response, design=design, instruments=design)  # type: ignore[call-arg]


def test_glm_working_state_names_every_persisted_domain() -> None:
    working_response = np.arange(3.0)
    design = np.arange(6.0).reshape(3, 2)
    working_weights = np.array([1.0, 2.0, 3.0])
    eta = np.array([-1.0, 0.0, 1.0])
    mu = np.array([0.2, 0.5, 0.8])
    response_residuals = np.array([-0.2, 0.5, 0.2])
    working_residuals = np.array([-1.0, 1.0, 0.5])

    state = GlmWorkingState(
        working_response_within=working_response,
        design_within=design,
        working_weights=working_weights,
        eta=eta,
        mu=mu,
        response_residuals=response_residuals,
        working_residuals=working_residuals,
    )

    assert state.working_response_within is working_response
    assert state.design_within is design
    assert state.working_weights is working_weights
    assert state.eta is eta
    assert state.mu is mu
    assert state.response_residuals is response_residuals
    assert state.working_residuals is working_residuals
    assert not hasattr(state, "sqrt_weights")
    assert not hasattr(state, "__dict__")


def test_within_iv_data_requires_instrument_roles() -> None:
    response = np.arange(3.0)[:, None]
    design = np.column_stack((np.ones(3), np.arange(3.0)))
    instruments = np.arange(6.0).reshape(3, 2)
    endogenous = np.arange(3.0)[:, None]
    state = WithinIvData(
        response=response,
        design=design,
        instruments=instruments,
        endogenous=endogenous,
    )
    assert isinstance(state, WithinLinearData)
    assert state.instruments is instruments
    assert state.endogenous is endogenous
    assert not hasattr(state, "__dict__")

    with pytest.raises(TypeError):
        WithinIvData(response=response, design=design)  # type: ignore[call-arg]

    reduced = replace(state, design=design[:, :1])
    assert isinstance(reduced, WithinIvData)
    assert reduced.instruments is instruments
    assert reduced.design.shape == (3, 1)
