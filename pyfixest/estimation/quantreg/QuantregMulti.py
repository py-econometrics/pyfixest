from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
from scipy.stats import norm

from pyfixest.estimation.formula.parse import Formula as FixestFormula
from pyfixest.estimation.internals.demean_ import DemeanedData
from pyfixest.estimation.internals.literals import QuantregMultiOptions
from pyfixest.estimation.internals.model_state import (
    QuantregEstimationOptions,
    SampleSplit,
    VcovSpec,
)
from pyfixest.estimation.quantreg.quantreg_ import Quantreg
from pyfixest.estimation.quantreg.utils import get_hall_sheather_bandwidth
from pyfixest.utils.dev_utils import DataFrameType


class QuantregMulti:
    "Run the quantile regression process efficiently. Wrapper around Quantreg calls."

    def __init__(
        self,
        FixestFormula: FixestFormula,
        data: pd.DataFrame,
        *,
        options: QuantregEstimationOptions,
        quantile: list[float],
        multi_method: QuantregMultiOptions,
        lookup_demeaned_data: dict[frozenset[int], DemeanedData],
        sample_split: SampleSplit | None = None,
    ):
        # `options.quantile` is the first requested quantile; each child fit
        # carries its own quantile and shares every other option.
        self.options = options
        self.quantiles = quantile
        self.all_quantregs = {
            q: Quantreg(
                FixestFormula=FixestFormula,
                data=data,
                options=replace(options, quantile=q),
                lookup_demeaned_data=lookup_demeaned_data,
                sample_split=sample_split,
            )
            for q in self.quantiles
        }
        self.multi_method = multi_method

    def prepare_model_matrix(self) -> None:
        """Prepare the model inputs for every requested quantile."""
        # TODO: prepare once and share immutable state across quantiles.
        for quantreg in self.all_quantregs.values():
            quantreg.prepare_model_matrix()
            quantreg.to_array()
            quantreg.drop_multicol_vars()

        self._X_is_empty = False

    def get_fit(self) -> dict[float, Quantreg]:
        "Fit multiple quantile regressions via either algo 2 or 3 of CFM."
        # sort q increasing
        q = np.sort(self.quantiles)
        n_quantiles = len(q)

        if n_quantiles % 2 == 1:
            q_median_idx = n_quantiles // 2
        else:
            q_median_idx = (n_quantiles // 2) - 1

        q_median = q[q_median_idx]

        # data fixed across qregs, just need take from first one
        X = self.all_quantregs[q[q_median_idx]].within_data.design
        Y = self.all_quantregs[q[q_median_idx]].within_data.response
        hessian = X.T @ X
        N = self.all_quantregs[q[q_median_idx]].sample_info.n_obs

        # Fit the "central" quantile first, on a stream of its own, so the
        # child's generator stays fresh for its "nid" bandwidth refits.
        median_quantreg = self.all_quantregs[q_median]
        median_quantreg._publish_solution(
            median_quantreg._solve(
                X=X,
                Y=Y,
                q=q_median,
                rng=np.random.default_rng(median_quantreg.options.seed),
            )
        )

        def _direction_helper(i, direction):
            if direction == "left":
                i_prev = i + 1
            elif direction == "right":
                i_prev = i - 1
            else:
                raise ValueError(
                    f"Direction must be 'left' or 'right' but is {direction}."
                )

            return i_prev

        if self.multi_method == "cfm1":

            def _cfm1_fun(i, direction):
                i_prev = _direction_helper(i, direction)

                beta_hat_prev = self.all_quantregs[q[i_prev]]._beta_hat
                quantreg = self.all_quantregs[q[i]]
                quantreg._publish_solution(
                    quantreg.fit_qreg_pfn(
                        X=X, Y=Y, q=q[i], beta_init=beta_hat_prev, eta=0.5
                    )
                )

            for i in range(q_median_idx - 1, -1, -1):
                _cfm1_fun(i, "left")

            for i in range(q_median_idx + 1, n_quantiles, 1):
                _cfm1_fun(i, "right")

        elif self.multi_method == "cfm2":

            def _cfm2_fun(i, direction):
                i_prev = _direction_helper(i, direction)

                beta_hat_prev = self.all_quantregs[q[i_prev]]._beta_hat
                u_hat_prev = self.all_quantregs[q[i_prev]]._u_hat

                kappa = np.median(np.abs(u_hat_prev - np.median(u_hat_prev)))
                h_G = get_hall_sheather_bandwidth(q=q[i_prev], N=N)
                delta = kappa * (norm.ppf(q[i_prev] + h_G) - norm.ppf(q[i_prev] - h_G))
                J = (np.sum(np.abs(u_hat_prev) < delta) * hessian) / (2 * N * delta)

                M = X.T @ (q[i] - (u_hat_prev < 0))[:, None]
                beta_new = beta_hat_prev + np.linalg.solve(J, M).flatten()

                self.all_quantregs[q[i]]._publish_coefficients(beta_new)

            for i in range(q_median_idx - 1, -1, -1):
                _cfm2_fun(i, "left")

            for i in range(q_median_idx + 1, n_quantiles, 1):
                _cfm2_fun(i, "right")

        else:
            raise ValueError(
                f"Multi method needs to be of type 'cfm1' or 'cfm2' but is {self.multi_method}."
            )

        # sort self.all_quantregs by q
        self.all_quantregs = dict(
            sorted(self.all_quantregs.items(), key=lambda item: item[0])
        )
        return self.all_quantregs

    def vcov(
        self,
        vcov: str | dict[str, str],
        vcov_kwargs: dict[str, str | int] | None = None,
        data: DataFrameType | None = None,
    ) -> dict[float, Quantreg]:
        "Compute variance-covariance matrices for all models in the quantile regression process."
        for quantreg in self.all_quantregs.values():
            quantreg.vcov(vcov=vcov, vcov_kwargs=vcov_kwargs, data=data)

        return self.all_quantregs

    def _publish_fit_statistics(self) -> None:
        "Publish the goodness-of-fit measures of every quantile."
        for quantreg in self.all_quantregs.values():
            quantreg._publish_fit_statistics()

    def _check_vcov_support(self, spec: VcovSpec) -> None:
        "Reject a covariance estimator the quantile regressions cannot compute."
        for quantreg in self.all_quantregs.values():
            quantreg._check_vcov_support(spec)

    def _vcov_from_spec(self, spec: VcovSpec) -> dict[float, Quantreg]:
        "Compute the covariance of a parsed `spec` for every quantile."
        for quantreg in self.all_quantregs.values():
            quantreg._vcov_from_spec(spec)

        return self.all_quantregs

    def get_inference(self) -> dict[float, Quantreg]:
        "Compute inference for all models of the quantile regression process."
        for quantreg in self.all_quantregs.values():
            quantreg.get_inference()

        return self.all_quantregs

    def _validate_response(self) -> None:
        """Quantile regression has no additional response constraint."""

    def _finalize_fit(self) -> None:
        """Quantile models require no additional post-fit orchestration."""

    def _iter_fitted_models(self) -> tuple[Quantreg, ...]:
        """Yield each fitted quantile to the result container."""
        return tuple(self.all_quantregs.values())
