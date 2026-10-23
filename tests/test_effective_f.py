"""Overidentified effective-F references: Montiel Olea and Pflueger (2013)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

import pyfixest as pf


@pytest.mark.parametrize(
    "has_fe,weights_type",
    [(False, None), (True, None), (True, "aweights"), (True, "fweights")],
)
def test_effective_f_partials_controls(has_fe, weights_type):
    rng = np.random.default_rng(7)
    n = 240
    data = pd.DataFrame(rng.normal(size=(n, 5)), columns=["y", "x", "z1", "z2", "u"])
    data["g"] = np.repeat(np.arange(12), 20)
    data["w"] = rng.integers(1, 4, n)
    data["z1"] += 2 * data.x
    data["z2"] += 3 * data.x + 5
    data["d"] = data.z1 + 0.5 * data.z2 + data.x + data.u
    fe = " | g" if has_fe else ""
    kwargs = (
        {} if weights_type is None else {"weights": "w", "weights_type": weights_type}
    )
    # Disable SSC so explicit FE dummies and absorbed FE have the same HC0 covariance.
    fit = pf.feols(
        "y ~ x + [d ~ z1 + z2]" + fe,
        data=data,
        vcov="hetero",
        ssc=pf.ssc(k_adj=False),
        **kwargs,
    )
    fit.eff_F()
    controls = sm.add_constant(data[["x"]]).to_numpy()
    if has_fe:
        controls = np.column_stack(
            [controls, pd.get_dummies(data.g, drop_first=True, dtype=float)]
        )
    instruments = data[["z1", "z2"]].to_numpy()
    design = np.column_stack([controls, instruments])
    weights = np.ones(n) if weights_type is None else data.w.to_numpy()
    if weights_type == "fweights":
        design = np.repeat(design, weights, axis=0)
        response = np.repeat(data.d.to_numpy(), weights)
        reference = sm.OLS(response, design).fit(cov_type="HC0")
    else:
        reference = sm.WLS(data.d, design, weights=weights).fit(cov_type="HC0")
    # statsmodels WLS projects each excluded instrument on all included exogenous columns.
    residualized = np.column_stack(
        [
            sm.WLS(instruments[:, j], controls, weights=weights).fit().resid
            for j in range(2)
        ]
    )
    Q_zz = residualized.T @ (weights[:, None] * residualized)
    pi_hat = np.asarray(reference.params)[-2:]
    Sigma = np.asarray(reference.cov_params())[-2:, -2:]
    expected = (pi_hat @ Q_zz @ pi_hat) / np.trace(Sigma @ Q_zz)
    # The absorbed and dummy regressions differ only by floating-point solver error.
    np.testing.assert_allclose(
        fit.first_stage.diagnostics.eff_f,
        expected,
        rtol=1e-8,
        atol=1e-10,
        err_msg="Effective F differs from the statsmodels partialled-out reference",
    )
    changed = data.assign(z2=data.z2 + 4 * data.x + 8)
    shifted = pf.feols(
        "y ~ x + [d ~ z1 + z2]" + fe,
        data=changed,
        vcov="hetero",
        ssc=pf.ssc(k_adj=False),
        **kwargs,
    )
    shifted.eff_F()
    np.testing.assert_allclose(
        shifted.first_stage.diagnostics.eff_f,
        expected,
        rtol=1e-8,
        atol=1e-10,
        err_msg="Effective F changed under an equivalent instrument parameterization",
    )
