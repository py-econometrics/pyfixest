from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray

from pyfixest.core.demean import Preconditioner, within_preconditioner_name
from pyfixest.demeaners import AnyDemeaner, LsmrDemeaner


@dataclass(frozen=True, slots=True, kw_only=True)
class DemeanedData:
    """Cache entry for demeaned columns and their keys.

    Attributes
    ----------
    values : NDArray[np.float64]
        Demeaned columns, shape ``(n_rows, len(column_keys))``.
    column_keys : tuple[Hashable, ...]
        Column keys, in the insertion order of the columns in ``values``.
    fixed_effects : NDArray[np.float64] or None
        Within projection coefficients, shape ``(n_dummy_columns, len(column_keys))``,
        in treatment-coded dummy order. Read-only, like ``values``; absent for
        backends that do not return coefficients.

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
    column_keys: tuple[Hashable, ...]
    fixed_effects: NDArray[np.float64] | None = None


class DemeanCache:
    """Model-side helper around the demeaner strategies, with two caches.

    `Compute once, never forget`:

    - `lookup_demeaned_data`: already-demeaned columns from previous fits.
    - `lookup_preconditioner`: the preconditioner from the first fit on a
       data set / na index combination.

    The index for both caches is the frozen set of `na_index` - as all fits
    operate on the same fixed effects / data structure.

    OLS/IV models call :meth:`demean_yx`. GLMs call :meth:`demean_array` to
    recompute each weighted projection while reusing the preconditioner.
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
        """Store the preconditioner to reuse for ``na_index``.

        The first preconditioner observed is kept, unless a later solve
        escalates an adaptive preconditioner. Retain that additive Schwarz
        replacement for subsequent IWLS iterations and sibling models.
        """
        if used_preconditioner is None:
            return
        cached = self.lookup_preconditioner.get(na_index)
        if cached is None or (
            within_preconditioner_name(cached) == "adaptive"
            and within_preconditioner_name(used_preconditioner) == "additive"
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
        """Recompute a projection with current weights and a cached preconditioner.

        Unlike :meth:`demean_yx`, this does not cache demeaned columns. Within
        a cache block, keep using the first handle or its additive replacement
        after adaptive escalation, including for an explicitly supplied handle.
        """
        return self.demean_array_with_fixef(
            x=x, flist=flist, weights=weights, na_index=na_index, demeaner=demeaner
        )[0]

    def demean_array_with_fixef(
        self,
        x: np.ndarray,
        flist: np.ndarray,
        weights: np.ndarray | None,
        na_index: frozenset[int],
        demeaner: AnyDemeaner,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Return a fresh projection and its within fixed-effect coefficients."""
        cached = self.lookup_preconditioner.get(na_index)
        if (
            cached is not None
            and isinstance(demeaner, LsmrDemeaner)
            and isinstance(demeaner.preconditioner, Preconditioner)
        ):
            # An explicit adaptive handle must not restart the ladder after
            # an earlier solve has already built the additive preconditioner.
            demeaner = replace(demeaner, preconditioner=cached)
        result, _, fixed_effects = self._run_or_raise(
            x=x, flist=flist, weights=weights, na_index=na_index, demeaner=demeaner
        )
        return result, fixed_effects

    def _run_or_raise(
        self,
        x: np.ndarray,
        flist: np.ndarray,
        weights: np.ndarray | None,
        na_index: frozenset[int],
        demeaner: AnyDemeaner,
    ) -> tuple[np.ndarray, Preconditioner | None, np.ndarray | None]:
        cached_preconditioner = self.lookup_preconditioner.get(na_index)
        fixed_effects = None
        if isinstance(demeaner, LsmrDemeaner) and demeaner.backend == "within":
            result, success, used_preconditioner, fixed_effects = (
                demeaner._demean_within(
                    x=x,
                    flist=flist,
                    weights=weights,
                    cached_preconditioner=cached_preconditioner,
                )
            )
        else:
            result, success, used_preconditioner = demeaner.demean(
                x=x,
                flist=flist,
                weights=weights,
                cached_preconditioner=cached_preconditioner,
            )
        self.seed_preconditioner(na_index, used_preconditioner)
        if not success:
            raise ValueError(
                f"Demeaning failed after {demeaner.fixef_maxiter} iterations."
            )
        return result, used_preconditioner, fixed_effects

    def demean_yx(
        self,
        Y: NDArray[np.float64],
        X: NDArray[np.float64],
        *,
        y_keys: Sequence[Hashable],
        x_keys: Sequence[Hashable],
        fe: np.ndarray | None,
        weights: NDArray[np.float64] | None,
        na_index: frozenset[int],
        demeaner: AnyDemeaner,
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        Preconditioner | None,
    ]:
        """Demean response and design arrays and cache missing columns.

        New columns are appended to the cache in their requested order. Returned
        arrays always follow ``y_keys`` and ``x_keys``, independently of the
        cache's insertion order.

        Parameters
        ----------
        Y : NDArray[np.float64]
            Response array, shape ``(n_rows, n_responses)``.
        X : NDArray[np.float64]
            Design array, shape ``(n_rows, n_regressors)``.
        y_keys : Sequence[Hashable]
            Ordered response keys corresponding to the columns of ``Y``.
        x_keys : Sequence[Hashable]
            Ordered regressor keys corresponding to the columns of ``X``.
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
            Cached arrays and their selections are read-only. Without fixed
            effects, inputs pass through without changing their writeability.
        """
        Y_array = np.asarray(Y, dtype=np.float64)
        X_array = np.asarray(X, dtype=np.float64)
        if fe is None:
            return Y_array, X_array, None

        y_keys_tuple = tuple(y_keys)
        x_keys_tuple = tuple(x_keys)
        yx_keys = y_keys_tuple + x_keys_tuple

        cached = self.lookup_demeaned_data.get(na_index)
        used_preconditioner: Preconditioner | None = None
        if cached is None:
            YX = np.concatenate((Y_array, X_array), axis=1)
            YX_demeaned, used_preconditioner, fixed_effects = self._run_or_raise(
                YX, fe, weights, na_index, demeaner
            )
            # Callers get slices of this array; see DemeanedData.
            YX_demeaned.setflags(write=False)
            if fixed_effects is not None:
                fixed_effects.setflags(write=False)
            cached = DemeanedData(
                values=YX_demeaned, column_keys=yx_keys, fixed_effects=fixed_effects
            )
            self.lookup_demeaned_data[na_index] = cached
        else:
            cached_keys = cached.column_keys
            cached_key_set = frozenset(cached_keys)
            uncached_positions = tuple(
                index for index, key in enumerate(yx_keys) if key not in cached_key_set
            )
            if uncached_positions:
                YX = np.concatenate((Y_array, X_array), axis=1)
                uncached_demeaned, used_preconditioner, fixed_effects = (
                    self._run_or_raise(
                        YX[:, uncached_positions], fe, weights, na_index, demeaner
                    )
                )
                uncached_keys = tuple(yx_keys[index] for index in uncached_positions)
                cached_demeaned = np.concatenate(
                    (cached.values, uncached_demeaned), axis=1
                )
                cached_demeaned.setflags(write=False)
                if fixed_effects is not None and cached.fixed_effects is not None:
                    fixed_effects = np.concatenate(
                        (cached.fixed_effects, fixed_effects), axis=1
                    )
                    fixed_effects.setflags(write=False)
                else:
                    fixed_effects = None
                cached = DemeanedData(
                    values=cached_demeaned,
                    column_keys=cached_keys + uncached_keys,
                    fixed_effects=fixed_effects,
                )
                self.lookup_demeaned_data[na_index] = cached
            # Every requested column is demeaned and cached at this point.
            YX_demeaned = self._select_columns(cached, yx_keys)

        # ``yx_keys`` lists the responses first, so they lead the columns.
        n_response_columns = len(y_keys_tuple)
        response_demeaned = YX_demeaned[:, :n_response_columns]
        design_demeaned = YX_demeaned[:, n_response_columns:]
        return response_demeaned, design_demeaned, used_preconditioner

    @staticmethod
    def _select_columns(
        cached: DemeanedData,
        yx_keys: tuple[Hashable, ...],
    ) -> NDArray[np.float64]:
        cached_position_by_key = {
            key: position for position, key in enumerate(cached.column_keys)
        }
        positions = tuple(cached_position_by_key[key] for key in yx_keys)
        selects_cached_prefix = positions == tuple(range(len(positions)))
        if selects_cached_prefix:
            # Basic slicing gives a view, which inherits the read-only flag.
            return cached.values[:, : len(positions)]
        # Fancy indexing gives a fresh, writable copy. Flag it too, so that
        # whether a caller may write into its demeaned data never depends on
        # the order the cache happened to fill up in.
        reordered = cached.values[:, positions]
        reordered.setflags(write=False)
        return reordered
