from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from pyfixest.estimation.internals.model_state import (
    DroppedRowCounts,
    ObservationWeights,
    SampleInfo,
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


def test_dropped_row_counts_sum_over_stages() -> None:
    counts = DroppedRowCounts(missing=2, nonfinite=1, singleton=3, separation=4)
    assert counts.total == 10
    assert DroppedRowCounts().total == 0
    assert not hasattr(counts, "__dict__")
    with pytest.raises(FrozenInstanceError):
        counts.missing = 0  # type: ignore[misc]
    assert replace(counts, separation=0).total == 6


def test_sample_info_is_structurally_immutable() -> None:
    sample = SampleInfo(
        dropped_positions=frozenset({1, 3, 4}),
        n_rows=3,
        n_obs=7.0,
        dropped_by_stage=DroppedRowCounts(missing=2, singleton=1),
    )
    assert sample.n_rows == 3
    assert sample.n_obs == 7.0
    assert sample.dropped_by_stage.total == 3
    assert not hasattr(sample, "__dict__")
    with pytest.raises(FrozenInstanceError):
        sample.n_rows = 4  # type: ignore[misc]
    with pytest.raises(TypeError):
        SampleInfo(frozenset(), 1, 1, DroppedRowCounts())  # type: ignore[misc]


def test_sample_info_rejects_inconsistent_state() -> None:
    with pytest.raises(ValueError, match="must sum to the number of dropped"):
        SampleInfo(
            dropped_positions=frozenset({3, 4}),
            n_rows=3,
            n_obs=3,
            dropped_by_stage=DroppedRowCounts(missing=1),
        )


@pytest.mark.parametrize(
    ("weights", "expected_n_effective"),
    [
        (ObservationWeights.unweighted(), 3),
        (
            ObservationWeights.from_values(
                np.array([1.0, 2.0, 3.0]), weights_type="aweights"
            ),
            3,
        ),
        (
            ObservationWeights.from_values(
                np.array([1.0, 2.0, 3.0]), weights_type="fweights"
            ),
            6.0,
        ),
    ],
    ids=["unweighted", "aweights", "fweights"],
)
def test_sample_info_from_weights_counts_frequency_weights(
    weights, expected_n_effective
) -> None:
    sample = SampleInfo.from_weights(
        n_rows=3,
        dropped_positions=frozenset({1, 3}),
        dropped_by_stage=DroppedRowCounts(missing=2),
        weights=weights,
    )
    assert sample.n_rows == 3
    assert sample.n_obs == expected_n_effective
    assert type(sample.n_obs) is type(expected_n_effective)
    assert sample.dropped_positions == frozenset({1, 3})


def test_sample_info_from_weights_rejects_misaligned_weights() -> None:
    with pytest.raises(ValueError, match="one value per row"):
        SampleInfo.from_weights(
            n_rows=2,
            dropped_positions=frozenset(),
            dropped_by_stage=DroppedRowCounts(),
            weights=ObservationWeights.from_values(np.ones(3), weights_type="aweights"),
        )


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
