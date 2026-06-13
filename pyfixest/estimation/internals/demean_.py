import numpy as np
import pandas as pd

from pyfixest.core.demean import Preconditioner
from pyfixest.demeaners import AnyDemeaner


class DemeanCache:
    """Model-side helper around the demeaner strategies, with two caches.

    The two cached objects have *different lifetimes*:

    - `shared_lookup` is reused *across models*. It is a dictionary
      keyed by `na_index` (the set of dropped rows) whose values are the
      already-demeaned columns for that row sample. `FixestMulti` builds
      it once per formula block and passes the same dict to every model in
      the block (e.g. stepwise `sw`/`csw` siblings), so a column
      demeaned for one model is picked up by any later model that has the
      same `na_index`.
    - `preconditioner` is reused *within a single model*. It is a
      single slot (not a dict): the first within-solve stores its
      preconditioner here, and every subsequent solve on the same model
      reuses it. This matters for IWLS (Poisson, GLM), where `demean` is
      called once per iteration on changing working data but always against
      the same fixed-effect design. It is scoped per model because there is
      no key — a fresh `DemeanCache` (or at least a fresh
      `preconditioner` slot) is needed whenever the FE design or row
      sample changes.

    Model classes hold one `DemeanCache` and call :meth:`demean_array`
    (IWLS paths) or :meth:`demean_yx` (OLS/IV paths) instead of dispatching
    to the demeaner backends directly.
    """

    def __init__(
        self, shared_lookup: dict[frozenset[int], pd.DataFrame] | None = None
    ) -> None:
        self.shared_lookup: dict[frozenset[int], pd.DataFrame] = (
            {} if shared_lookup is None else shared_lookup
        )
        self.preconditioner: Preconditioner | None = None

    def seed_preconditioner(self, used: Preconditioner | None) -> None:
        """Store only the first preconditioner returned by a demean solve.

        For IWLS (Poisson, GLM) the demeaner is called once per iteration
        and returns a preconditioner each time; we keep the one from the
        first call and ignore the rest. ``used`` is ``None`` when no
        preconditioner participated in the solve (MAP fallback,
        ``preconditioner='off'``, non-within backend), in which case there
        is nothing to store.
        """
        if self.preconditioner is None and used is not None:
            self.preconditioner = used

    def demean_array(
        self,
        x: np.ndarray,
        flist: np.ndarray,
        weights: np.ndarray | None,
        demeaner: AnyDemeaner,
    ) -> np.ndarray:
        """Demean ``x``, reusing and seeding the cached preconditioner.

        Raises ``ValueError`` if the demeaning algorithm does not converge.
        """
        result, _ = self._run_or_raise(x, flist, weights, demeaner)
        return result

    def _run_or_raise(
        self,
        x: np.ndarray,
        flist: np.ndarray,
        weights: np.ndarray | None,
        demeaner: AnyDemeaner,
    ) -> tuple[np.ndarray, Preconditioner | None]:
        result, success, used = demeaner.demean(
            x, flist, weights, cached_preconditioner=self.preconditioner
        )
        self.seed_preconditioner(used)
        if not success:
            raise ValueError(
                f"Demeaning failed after {demeaner.fixef_maxiter} iterations."
            )
        return result, used

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
        been demeaned and reuses values from ``self.shared_lookup`` if
        possible. If the model has no fixed effects, the data is returned
        undemeaned.

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame, Preconditioner | None]
            The demeaned ``Y``, the demeaned ``X``, and the within
            preconditioner used during the solve (``None`` for non-within
            backends, ``preconditioner='off'``, the single-FE MAP fallback,
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
        cached = self.shared_lookup.get(na_index)
        if cached is None:
            arr, used = self._run_or_raise(YX_array, fe_array, weights, demeaner)
            YX_demeaned = pd.DataFrame(arr, columns=yx_names)
        else:
            # demean only the not-yet-demeaned columns
            new_names = list(set(yx_names) - set(cached.columns))
            if new_names:
                yx_names_list = list(yx_names)
                new_index = [yx_names_list.index(name) for name in new_names]
                arr, used = self._run_or_raise(
                    YX_array[:, new_index], fe_array, weights, demeaner
                )
                YX_demeaned = pd.DataFrame(
                    np.concatenate([cached, arr], axis=1),
                    columns=list(cached.columns) + new_names,
                )
            else:
                # all variables already demeaned
                YX_demeaned = cached[yx_names]

        self.shared_lookup[na_index] = YX_demeaned
        return YX_demeaned[Y.columns], YX_demeaned[X.columns], used
