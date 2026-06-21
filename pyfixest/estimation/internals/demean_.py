import numpy as np
import pandas as pd

from pyfixest.core.demean import Preconditioner
from pyfixest.demeaners import AnyDemeaner, LsmrDemeaner, MapDemeaner


class DemeanStrategy:
    """Resolved demeaner plus per-model fixef parameters and cache.

    Extracts the logic that was previously inlined in ``Feols.__init__``
    (default demeaner, fixef tolerance/maxiter, cache wiring) into one
    place so it can be shared by OLS, IV, and GLM model classes.
    """

    def __init__(
        self,
        demeaner: AnyDemeaner | None,
        lookup_demeaned_data: dict[frozenset[int], pd.DataFrame] | None,
        lookup_preconditioner: dict[frozenset[int], Preconditioner] | None,
    ):
        if demeaner is None:
            demeaner = MapDemeaner()
        self.demeaner = demeaner
        if isinstance(demeaner, LsmrDemeaner):
            self.fixef_tol = max(demeaner.fixef_atol, demeaner.fixef_btol)
        else:
            self.fixef_tol = demeaner.fixef_tol
        self.fixef_maxiter = demeaner.fixef_maxiter
        self.cache = DemeanCache(lookup_demeaned_data, lookup_preconditioner)


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
        lookup_demeaned_data: dict[frozenset[int], pd.DataFrame] | None = None,
        lookup_preconditioner: dict[frozenset[int], Preconditioner] | None = None,
    ) -> None:
        self.lookup_demeaned_data: dict[frozenset[int], pd.DataFrame] = (
            {} if lookup_demeaned_data is None else lookup_demeaned_data
        )
        self.lookup_preconditioner: dict[frozenset[int], Preconditioner] = (
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
        Y: pd.DataFrame,
        X: pd.DataFrame,
        fe: pd.DataFrame | None,
        weights: np.ndarray | None,
        na_index: frozenset[int],
        demeaner: AnyDemeaner,
    ) -> tuple[pd.DataFrame, pd.DataFrame, Preconditioner | None]:
        """Demean a regression model: check cache, demean what's missing, update cache.

        Prior to demeaning, checks whether some of the variables have already
        been demeaned and reuses values from `self.lookup_demeaned_data` if
        possible. If the model has no fixed effects, the data is returned
        undemeaned.

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame, Preconditioner | None]
            The demeaned `Y`, the demeaned `X`, and the within
            preconditioner used during the solve (`None` for non-within
            backends, `preconditioner='off'`, the single-FE MAP fallback,
            or when no demeaning happened).
        """
        used: Preconditioner | None = None
        YX = pd.concat([Y, X], axis=1)
        yx_names = YX.columns
        YX_array = YX.to_numpy()
        if YX_array.dtype != np.dtype("float64"):
            YX_array = YX_array.astype(np.float64)

        if fe is None:
            YX_demeaned = pd.DataFrame(YX_array, columns=yx_names)
            return YX_demeaned[Y.columns], YX_demeaned[X.columns], None

        fe_array = fe.to_numpy()
        cached_demeaned = self.lookup_demeaned_data.get(na_index)
        if cached_demeaned is None:
            arr, used = self._run_or_raise(
                YX_array, fe_array, weights, na_index, demeaner
            )
            YX_demeaned = pd.DataFrame(arr, columns=yx_names)
        else:
            # demean only the not-yet-demeaned columns
            new_names = list(set(yx_names) - set(cached_demeaned.columns))
            if new_names:
                yx_names_list = list(yx_names)
                new_index = [yx_names_list.index(name) for name in new_names]
                arr, used = self._run_or_raise(
                    YX_array[:, new_index], fe_array, weights, na_index, demeaner
                )
                YX_demeaned = pd.DataFrame(
                    np.concatenate([cached_demeaned, arr], axis=1),
                    columns=list(cached_demeaned.columns) + new_names,
                )
            else:
                # all variables already demeaned
                YX_demeaned = cached_demeaned[yx_names]

        self.lookup_demeaned_data[na_index] = YX_demeaned
        return YX_demeaned[Y.columns], YX_demeaned[X.columns], used
