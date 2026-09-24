from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pandas as pd
import pytest

import pyfixest as pf
from pyfixest.estimation.internals.model_state import (
    DroppedRowCounts,
    EstimationSample,
    ObservationWeights,
    WithinIvData,
    WithinLinearData,
)


def test_observation_weights_unweighted_fast_path() -> None:
    weights = ObservationWeights.unweighted()

    assert weights.values is None
    assert weights.weights_type is None
    assert not weights.is_weighted
    assert not hasattr(weights, "__dict__")
    assert not hasattr(weights, "n_rows")
    assert not hasattr(weights, "n_obs")


@pytest.mark.parametrize("weights_type", ["aweights", "fweights"])
@pytest.mark.parametrize("input_writeable", [False, True])
def test_observation_weights_keep_canonical_user_values(
    weights_type, input_writeable
) -> None:
    user_weights = np.array([[1.0], [2.0], [3.0]])
    user_weights.setflags(write=input_writeable)
    weights = ObservationWeights.from_values(user_weights, weights_type=weights_type)
    np.testing.assert_array_equal(weights.values, user_weights.flatten())
    assert weights.weights_type == weights_type
    assert weights.is_weighted
    assert user_weights.flags.writeable == input_writeable


def test_observation_weights_reject_inconsistent_state() -> None:
    with pytest.raises(
        ValueError, match="Weighted observations must declare a `weights_type`"
    ):
        ObservationWeights(values=np.ones(2), weights_type=None)


def test_estimation_sample_rejects_inconsistent_state() -> None:
    with pytest.raises(ValueError, match="must sum to the size of the dropped"):
        EstimationSample(
            dropped_row_index=frozenset({3, 4}),
            n_rows=3,
            n_obs=3,
            dropped_by_stage=DroppedRowCounts(missing=1),
        )


@pytest.mark.parametrize(
    ("weights_type", "expected_n_obs"),
    [
        (None, 3),
        ("aweights", 3),
        ("fweights", 6.0),
    ],
    ids=["unweighted", "aweights", "fweights"],
)
def test_estimation_sample_counts_frequency_weights(
    weights_type, expected_n_obs
) -> None:
    data = pd.DataFrame(
        {
            "y": [1.0, 2.0, 3.0, 4.0, 4.0],
            "x": [1.0, np.nan, 2.0, np.nan, 3.0],
            "weight": [1.0, 10.0, 2.0, 10.0, 3.0],
        }
    )
    fit = pf.feols(
        "y ~ x",
        data,
        weights="weight" if weights_type else None,
        weights_type=weights_type or "aweights",
    )
    sample_info = fit.sample_info

    assert sample_info.n_rows == 3
    assert sample_info.n_obs == expected_n_obs
    assert type(sample_info.n_obs) is type(expected_n_obs)
    assert sample_info.dropped_row_index == frozenset({1, 3})
    assert sample_info.dropped_by_stage == DroppedRowCounts(missing=2)
    with pytest.raises(FrozenInstanceError):
        sample_info.n_rows = 0  # type: ignore[misc]


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
    assert reduced.instruments is state.instruments
    assert reduced.design.shape == (3, 1)


def _capability_fit(model: str):
    data = pf.get_data()
    if model == "feols":
        return pf.feols("Y ~ X1 | f1", data)
    if model == "feols-iv":
        return pf.feols("Y ~ 1 | f1 | X1 ~ Z1", data)
    if model == "fepois":
        return pf.fepois("Y ~ X1 | f1", pf.get_data(model="Fepois"))
    if model == "feglm-logit":
        data = data.dropna()
        data["Y"] = (data["Y"] > data["Y"].median()).astype(int)
        return pf.feglm("Y ~ X1 | f1", data, family="logit")
    if model == "quantreg":
        with pytest.warns(FutureWarning, match="experimental"):
            return pf.quantreg("Y ~ X1", data)
    if model == "did2s":
        did_data = pd.read_csv("pyfixest/did/data/df_het.csv")
        return pf.did2s(
            did_data,
            yname="dep_var",
            first_stage="~ 0 | state + year",
            second_stage="~ treat",
            treatment="treat",
            cluster="state",
        )
    raise ValueError(model)


@pytest.mark.parametrize(
    "model,expected",
    [
        ("feols", (True, True, True, True)),
        ("feols-iv", (False, False, False, False)),
        ("fepois", (True, True, True, False)),
        ("feglm-logit", (True, True, False, False)),
        ("quantreg", (True, True, False, False)),
        ("did2s", (True, True, False, False)),
    ],
)
def test_capabilities_post_estimation_methods(
    model: str, expected: tuple[bool, bool, bool, bool]
) -> None:
    """Each model class declares predict, fixef, ritest, and update support."""
    capabilities = _capability_fit(model).capabilities
    assert (
        capabilities.prediction,
        capabilities.fixed_effect_recovery,
        capabilities.randomization_inference,
        capabilities.sherman_morrison_update,
    ) == expected
