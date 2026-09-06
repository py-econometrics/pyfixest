from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from pyfixest.core.demean import Preconditioner
from pyfixest.demeaners import AnyDemeaner


@dataclass(frozen=True, slots=True, kw_only=True)
class DemeanedData:
    """Cache entry for named, demeaned columns.

    Attributes
    ----------
    values : NDArray[np.float64]
        Demeaned columns, shape ``(n_rows, len(columns))``.
    columns : tuple[str, ...]
        Column names, in the insertion order of the columns in ``values``.

    Notes
    -----
    One entry is shared by every model fitted on the same ``na_index``, and
    callers receive slices of ``values`` rather than copies, so the entry must
    not change after publication. The frozen dataclass prevents field
    rebinding, and ``values`` is flagged read-only before it is stored, so a
    model writing into a returned slice fails loudly instead of silently
    corrupting the demeaned data of the models it shares the cache with.
    """

    values: NDArray[np.float64]
    columns: tuple[str, ...]


class DemeanCache:
    """Model-side helper around the demeaner strategies, with two caches.

    `Compute once, never forget`:

    - `lookup_demeaned_data`: already-demeaned named arrays from previous fits.
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
        self, na_index: frozenset[int], used_preconditioner: Preconditioner | None
    ) -> None:
        """Store the first preconditioner observed for ``na_index``.

        For IWLS (Poisson, GLM) the demeaner is called once per iteration
        and returns a preconditioner each time; we keep the one from the
        first call and ignore later ones.
        """
        if (
            used_preconditioner is not None
            and na_index not in self.lookup_preconditioner
        ):
            self.lookup_preconditioner[na_index] = used_preconditioner

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
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        Preconditioner | None,
    ]:
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

        Returns
        -------
        tuple[NDArray[np.float64], NDArray[np.float64], Preconditioner or None]
            Demeaned response and design arrays, in their requested column order,
            plus the preconditioner used when new columns were transformed.
        """
        Y_array = np.asarray(Y, dtype=np.float64)
        X_array = np.asarray(X, dtype=np.float64)
        if fe is None:
            return Y_array, X_array, None

        y_names_tuple = tuple(y_names)
        x_names_tuple = tuple(x_names)
        yx_names = y_names_tuple + x_names_tuple

        cached = self.lookup_demeaned_data.get(na_index)
        used_preconditioner: Preconditioner | None = None
        if cached is None:
            YX = np.concatenate((Y_array, X_array), axis=1)
            YX_demeaned, used_preconditioner = self._run_or_raise(
                YX, fe, weights, na_index, demeaner
            )
            # Callers get slices of this array; see DemeanedData.
            YX_demeaned.setflags(write=False)
            cached = DemeanedData(
                values=YX_demeaned,
                columns=yx_names,
            )
            self.lookup_demeaned_data[na_index] = cached
        else:
            cached_names = cached.columns
            cached_name_set = frozenset(cached_names)
            uncached_positions = tuple(
                index
                for index, name in enumerate(yx_names)
                if name not in cached_name_set
            )
            if uncached_positions:
                YX = np.concatenate((Y_array, X_array), axis=1)
                uncached_demeaned, used_preconditioner = self._run_or_raise(
                    YX[:, uncached_positions], fe, weights, na_index, demeaner
                )
                uncached_names = tuple(yx_names[index] for index in uncached_positions)
                cached_demeaned = np.concatenate(
                    (cached.values, uncached_demeaned), axis=1
                )
                cached_demeaned.setflags(write=False)
                cached = DemeanedData(
                    values=cached_demeaned,
                    columns=cached_names + uncached_names,
                )
                self.lookup_demeaned_data[na_index] = cached
            # Every requested column is demeaned and cached at this point.
            YX_demeaned = self._select_columns(cached, yx_names)

        # ``yx_names`` lists the responses first, so they lead the columns.
        n_response_columns = len(y_names_tuple)
        response_demeaned = YX_demeaned[:, :n_response_columns]
        design_demeaned = YX_demeaned[:, n_response_columns:]
        return response_demeaned, design_demeaned, used_preconditioner

    def demean_yx_frames(
        self,
        Y: pd.DataFrame,
        X: pd.DataFrame,
        fe: pd.DataFrame | None,
        weights: np.ndarray | None,
        na_index: frozenset[int],
        demeaner: AnyDemeaner,
    ) -> tuple[pd.DataFrame, pd.DataFrame, Preconditioner | None]:
        """Adapt DataFrame model consumers to the named-array cache.

        Linear and IV models still hold formula tables at this stage; they move
        to array-native within state in the next layer, which removes this
        adapter. Column names and row index are preserved so callers see the
        same frames they passed in.
        """
        response, design, used_preconditioner = self.demean_yx(
            Y.to_numpy(dtype=np.float64),
            X.to_numpy(dtype=np.float64),
            y_names=tuple(Y.columns),
            x_names=tuple(X.columns),
            fe=None if fe is None else fe.to_numpy(),
            weights=weights,
            na_index=na_index,
            demeaner=demeaner,
        )
        return (
            pd.DataFrame(response, columns=Y.columns, index=Y.index),
            pd.DataFrame(design, columns=X.columns, index=X.index),
            used_preconditioner,
        )

    @staticmethod
    def _select_columns(
        cached: DemeanedData,
        yx_names: tuple[str, ...],
    ) -> NDArray[np.float64]:
        cached_position_by_name = {
            name: position for position, name in enumerate(cached.columns)
        }
        positions = tuple(cached_position_by_name[name] for name in yx_names)
        selects_cached_prefix = positions == tuple(range(len(positions)))
        if selects_cached_prefix:
            return cached.values[:, : len(positions)]
        return cached.values[:, positions]
