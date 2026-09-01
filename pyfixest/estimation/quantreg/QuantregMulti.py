from __future__ import annotations

import gc
import inspect
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

from pyfixest.demeaners import AnyDemeaner
from pyfixest.estimation.formula.parse import Formula as FixestFormula
from pyfixest.estimation.internals.demean_ import DemeanedData
from pyfixest.estimation.internals.literals import (
    QuantregMethodOptions,
    QuantregMultiOptions,
    SolverOptions,
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
        quantile: list[float],
        ssc_dict: dict[str, str | bool],
        drop_singletons: bool,
        drop_intercept: bool,
        weights: str | None,
        weights_type: str | None,
        collin_tol: float,
        lookup_demeaned_data: dict[frozenset[int], DemeanedData],
        solver: SolverOptions = "np.linalg.solve",
        demeaner: AnyDemeaner | None = None,
        store_data: bool = True,
        copy_data: bool = True,
        lean: bool = False,
        context: int | Mapping[str, Any] = 0,
        sample_split_var: str | None = None,
        sample_split_value: str | int | None = None,
        method: QuantregMethodOptions = "fn",
        multi_method: QuantregMultiOptions = "cfm1",
        quantile_tol: float = 1e-06,
        quantile_maxiter: int | None = None,
        seed: int | None = None,
    ):
        frame = inspect.currentframe()
        if frame is None:
            raise ValueError("The current frame is None.")
        args, _, _, values = inspect.getargvalues(frame)
        args_dict = {
            arg: values[arg]
            for arg in args
            if arg not in ("self", "quantile", "multi_method")
        }

        # initiate a list of Quantreg objects
        self.quantiles = quantile
        self.all_quantregs = {
            q: Quantreg(**args_dict, quantile=q) for q in self.quantiles
        }
        self.method = method
        self.multi_method = multi_method
        self._is_iv = False

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
        X = self.all_quantregs[q[q_median_idx]]._X
        Y = self.all_quantregs[q[q_median_idx]]._Y
        hessian = X.T @ X
        N = self.all_quantregs[q[q_median_idx]]._N
        rng = np.random.default_rng(self.all_quantregs[q[q_median_idx]]._seed)

        # fit first quantile regression using "pfn"

        fit_kwargs = {
            "X": X,
            "Y": Y,
            "q": q_median,  # first eval at the "central" quantile
        }

        if self.method == "pfn":
            fit_kwargs["rng"] = rng
        beta_hat = self.all_quantregs[q[q_median_idx]]._fit(**fit_kwargs)[0]

        self._publish_fit_state(
            quantreg=self.all_quantregs[q[q_median_idx]],
            beta_hat=beta_hat,
            hessian=hessian,
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
                beta_hat = self.all_quantregs[q[i]].fit_qreg_pfn(
                    X=X, Y=Y, q=q[i], beta_init=beta_hat_prev, eta=0.5
                )[0]
                self._publish_fit_state(
                    quantreg=self.all_quantregs[q[i]],
                    beta_hat=beta_hat,
                    hessian=hessian,
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

                self._publish_fit_state(
                    quantreg=self.all_quantregs[q[i]],
                    beta_hat=beta_new,
                    hessian=hessian,
                )

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

    @staticmethod
    def _publish_fit_state(
        *, quantreg: Quantreg, beta_hat: np.ndarray, hessian: np.ndarray
    ) -> None:
        """Publish canonical child state after a multi-quantile solver step."""
        y_hat = quantreg._X @ beta_hat
        quantreg._beta_hat = beta_hat
        quantreg._Y_hat_link = y_hat
        quantreg._Y_hat_response = y_hat
        quantreg._u_hat = quantreg._Y.flatten() - y_hat
        quantreg._hessian = hessian

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

    def get_inference(self) -> dict[float, Quantreg]:
        "Compute inference for all models of the quantile regression process."
        for quantreg in self.all_quantregs.values():
            quantreg.get_inference()

        return self.all_quantregs

    def _validate_response(self) -> None:
        """Quantile regression has no additional response constraint."""

    def _inference_data(self) -> pd.DataFrame:
        """Return the shared estimation data from the first quantile model."""
        return next(iter(self.all_quantregs.values()))._data

    def _finalize_fit(self) -> None:
        """Quantile models require no additional post-fit orchestration."""

    def _iter_fitted_models(self) -> tuple[Quantreg, ...]:
        """Yield each fitted quantile to the result container."""
        return tuple(self.all_quantregs.values())

    def _clear_attributes(self) -> None:
        "Clear all large non-necessary attributes to free memory."
        for quantreg in self.all_quantregs.values():
            quantreg._clear_attributes()
        gc.collect()
