from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pyfixest.core.demean import Preconditioner
from pyfixest.demeaners import AnyDemeaner


@dataclass(frozen=True, slots=True, kw_only=True)
class DemeanedData:
    """Cache entry for named, demeaned columns.

    ``columns`` records the insertion order of the columns in ``values``.
    ``values`` is marked read-only before publication because cache entries are
    shared across fitted models. The frozen dataclass prevents field rebinding;
    the array flag prevents element assignment through ordinary NumPy APIs.
    """

    values: NDArray[np.float64]
    columns: tuple[str, ...]


class DemeanCache:
    """Model-side helper around the demeaner strategies, with two caches.

    `Compute once, never forget`:

    - `lookup_demeaned_data`: already-demeaned columns from previous fits.
    - `lookup_preconditioner`: the preconditioner from the first fit on a
       data set / na index combination.

    The index for both caches is the frozen set of `na_index` - as all fits
    operate on the same fixed effects / data structure.

    Model classes call :meth:`demean_array` (IWLS) or :meth:`demean_yx`
    (OLS/IV).
    """

    def __init__(
        self,
        lookup_demeaned_data: dict[frozenset[int], DemeanedData] | None = None,
        lookup_preconditioner: dict[frozenset[int], Preconditioner] | None = None,
    ) -> None:
        self.lookup_demeaned_data = (
            {} if lookup_demeaned_data is None else lookup_demeaned_data
        )
        self.lookup_preconditioner = (
            {} if lookup_preconditioner is None else lookup_preconditioner
        )

    def seed_preconditioner(
        self, na_index: frozenset[int], used: Preconditioner | None
    ) -> None:
        """Store the first preconditioner observed for ``na_index``.

        For IWLS (Poisson, GLM) the demeaner is called once per iteration
        and returns a preconditioner each time; we keep the one from the
        first call and ignore later ones.
        """
        if used is not None and na_index not in self.lookup_preconditioner:
            self.lookup_preconditioner[na_index] = used

    def demean_array(
        self,
        x: np.ndarray,
        flist: np.ndarray,
        weights: np.ndarray | None,
        na_index: frozenset[int],
        demeaner: AnyDemeaner,
    ) -> np.ndarray:
        """Demean `x`, reusing and seeding the cached preconditioner for `na_index`.

        Raises `ValueError` if the demeaning algorithm does not converge.
        """
        result, _ = self._run_or_raise(x, flist, weights, na_index, demeaner)
        return result

    def _run_or_raise(
        self,
        x: np.ndarray,
        flist: np.ndarray,
        weights: np.ndarray | None,
        na_index: frozenset[int],
        demeaner: AnyDemeaner,
    ) -> tuple[np.ndarray, Preconditioner | None]:
        cached_preconditioner = self.lookup_preconditioner.get(na_index)
        result, success, used_preconditioner = demeaner.demean(
            x, flist, weights, cached_preconditioner=cached_preconditioner
        )
        self.seed_preconditioner(na_index, used_preconditioner)
        if not success:
            raise ValueError(
                f"Demeaning failed after {demeaner.fixef_maxiter} iterations."
            )
        return result, used_preconditioner

    def demean_yx(
        self,
        Y: NDArray[np.float64],
        X: NDArray[np.float64],
        *,
        y_names: Sequence[str],
        x_names: Sequence[str],
        fe: np.ndarray | None,
        weights: NDArray[np.float64] | None,
        na_index: frozenset[int],
        demeaner: AnyDemeaner,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], Preconditioner | None]:
        """Demean response and design arrays and cache missing named columns.

        New columns are appended to the cache in their requested order. Returned
        arrays always follow ``y_names`` and ``x_names``, independently of the
        cache's insertion order.

        Parameters
        ----------
        Y : NDArray[np.float64]
            Response array, shape ``(n_rows, n_responses)``.
        X : NDArray[np.float64]
            Design array, shape ``(n_rows, n_regressors)``.
        y_names : Sequence[str]
            Ordered response names corresponding to the columns of ``Y``.
        x_names : Sequence[str]
            Ordered regressor names corresponding to the columns of ``X``.
        fe : np.ndarray or None
            Encoded fixed-effect identifiers, or ``None`` for no fixed effects.
        weights : NDArray[np.float64] or None
            Observation weights passed to the within transformation.
        na_index : frozenset[int]
            Row-removal identity used to share cached data between model fits.
        demeaner : AnyDemeaner
            Configured within-transformation strategy.
        """
        Y_array = np.asarray(Y, dtype=np.float64)
        X_array = np.asarray(X, dtype=np.float64)
        if fe is None:
            return Y_array, X_array, None

        y_names_tuple = tuple(y_names)
        x_names_tuple = tuple(x_names)
        requested_names = y_names_tuple + x_names_tuple
        cached = self.lookup_demeaned_data.get(na_index)
        used: Preconditioner | None = None
        if cached is None:
            requested_data = np.concatenate((Y_array, X_array), axis=1)
            requested_values, used = self._run_or_raise(
                requested_data, fe, weights, na_index, demeaner
            )
            requested_values.setflags(write=False)
            cached = DemeanedData(values=requested_values, columns=requested_names)
            self.lookup_demeaned_data[na_index] = cached
        else:
            cached_column_names = cached.columns
            cached_name_set = frozenset(cached_column_names)
            new_positions = tuple(
                index
                for index, name in enumerate(requested_names)
                if name not in cached_name_set
            )
            if new_positions:
                requested_data = np.concatenate((Y_array, X_array), axis=1)
                new_values, used = self._run_or_raise(
                    requested_data[:, new_positions], fe, weights, na_index, demeaner
                )
                new_column_names = tuple(requested_names[i] for i in new_positions)
                cached_values = np.concatenate((cached.values, new_values), axis=1)
                cached_values.setflags(write=False)
                cached = DemeanedData(
                    values=cached_values,
                    columns=cached_column_names + new_column_names,
                )
                self.lookup_demeaned_data[na_index] = cached
            requested_values = self._select_columns(cached, requested_names)

        n_response_columns = len(y_names_tuple)
        return (
            requested_values[:, :n_response_columns],
            requested_values[:, n_response_columns:],
            used,
        )

    @staticmethod
    def _select_columns(
        cached: DemeanedData, requested_names: tuple[str, ...]
    ) -> NDArray[np.float64]:
        positions_by_name = {
            name: position for position, name in enumerate(cached.columns)
        }
        positions = tuple(positions_by_name[name] for name in requested_names)
        selects_cached_prefix = positions == tuple(range(len(positions)))
        if selects_cached_prefix:
            return cached.values[:, : len(positions)]
        return cached.values[:, positions]
