import numpy as np
import pandas as pd

from pyfixest.core.demean import Preconditioner
from pyfixest.demeaners import AnyDemeaner


class DemeanCache:
    """Model-side helper around the demeaner strategies, with two caches.

    Both caches are keyed by ``na_index`` (the frozenset of dropped rows)
    and shared across all sibling models of a ``FixestMulti`` formula
    block. The block holds the FE structure and weights fixed, so models
    with the same ``na_index`` see the same within operator and can reuse:

    - ``demeaned_lookup``: already-demeaned columns, reused at the column
      level across siblings.
    - ``preconditioner_lookup``: the LSMR within-preconditioner from the
      first solve at each ``na_index``, reused as a warm start by every
      later solve at the same ``na_index`` (including IWLS iterations and
      sibling IWLS models). Working weights drift across iterations, but
      preconditioner reuse is still safe because it only affects
      convergence speed, not correctness.

    Model classes call :meth:`demean_array` (IWLS) or :meth:`demean_yx`
    (OLS/IV) instead of dispatching to the demeaner backends directly.
    """

    def __init__(
        self,
        demeaned_lookup: dict[frozenset[int], pd.DataFrame] | None = None,
        preconditioner_lookup: dict[frozenset[int], Preconditioner] | None = None,
    ) -> None:
        self.demeaned_lookup: dict[frozenset[int], pd.DataFrame] = (
            {} if demeaned_lookup is None else demeaned_lookup
        )
        self.preconditioner_lookup: dict[frozenset[int], Preconditioner] = (
            {} if preconditioner_lookup is None else preconditioner_lookup
        )

    def seed_preconditioner(
        self, na_index: frozenset[int], used: Preconditioner | None
    ) -> None:
        """Store the first preconditioner observed for ``na_index``.

        For IWLS (Poisson, GLM) the demeaner is called once per iteration
        and returns a preconditioner each time; we keep the one from the
        first call and ignore later ones. ``used`` is ``None`` when no
        preconditioner participated in the solve (MAP fallback,
        ``preconditioner='off'``, non-within backend), in which case there
        is nothing to store.
        """
        if used is not None and na_index not in self.preconditioner_lookup:
            self.preconditioner_lookup[na_index] = used

    def demean_array(
        self,
        x: np.ndarray,
        flist: np.ndarray,
        weights: np.ndarray | None,
        na_index: frozenset[int],
        demeaner: AnyDemeaner,
    ) -> np.ndarray:
        """Demean ``x``, reusing and seeding the cached preconditioner for ``na_index``.

        Raises ``ValueError`` if the demeaning algorithm does not converge.
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
        cached = self.preconditioner_lookup.get(na_index)
        result, success, used = demeaner.demean(
            x, flist, weights, cached_preconditioner=cached
        )
        self.seed_preconditioner(na_index, used)
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
        been demeaned and reuses values from ``self.demeaned_lookup`` if
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
        cached = self.demeaned_lookup.get(na_index)
        if cached is None:
            arr, used = self._run_or_raise(
                YX_array, fe_array, weights, na_index, demeaner
            )
            YX_demeaned = pd.DataFrame(arr, columns=yx_names)
        else:
            # demean only the not-yet-demeaned columns
            new_names = list(set(yx_names) - set(cached.columns))
            if new_names:
                yx_names_list = list(yx_names)
                new_index = [yx_names_list.index(name) for name in new_names]
                arr, used = self._run_or_raise(
                    YX_array[:, new_index], fe_array, weights, na_index, demeaner
                )
                YX_demeaned = pd.DataFrame(
                    np.concatenate([cached, arr], axis=1),
                    columns=list(cached.columns) + new_names,
                )
            else:
                # all variables already demeaned
                YX_demeaned = cached[yx_names]

        self.demeaned_lookup[na_index] = YX_demeaned
        return YX_demeaned[Y.columns], YX_demeaned[X.columns], used
