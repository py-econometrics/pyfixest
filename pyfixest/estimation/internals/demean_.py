from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray

from pyfixest.core.demean import Preconditioner
from pyfixest.demeaners import AnyDemeaner


@dataclass(frozen=True, slots=True, kw_only=True)
class DemeanedData:
    """Immutable blocks of demeaned columns for one estimation sample.

    Each block has shape ``(n_rows, n_new_columns)`` and is read-only. Appending
    a block shares all earlier arrays. ``columns`` follows block insertion
    order; ``locations`` maps each name to its block and column position and is
    replaced, never mutated, when the cache grows. ``design`` retains only the
    most recently requested design selection
    (shape ``(n_rows, len(design_names))``), so identical X across responses can
    share an assembled array without retaining every historical combination.
    """

    blocks: tuple[NDArray[np.float64], ...]
    columns: tuple[str, ...]
    locations: Mapping[str, tuple[int, int]]
    design_names: tuple[str, ...] = ()
    design: NDArray[np.float64] | None = None


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

        New blocks are appended to the cache in their requested order. Returned
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
            Cached arrays and their selections are read-only. Without fixed
            effects, inputs pass through without changing their writeability.
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
            YX_demeaned.setflags(write=False)
            cached = DemeanedData(
                blocks=(YX_demeaned,),
                columns=yx_names,
                locations={
                    name: (0, position) for position, name in enumerate(yx_names)
                },
            )
        else:
            cached_names = cached.columns
            uncached_positions = tuple(
                index
                for index, name in enumerate(yx_names)
                if name not in cached.locations
            )
            if uncached_positions:
                # Gather only missing columns; never concatenate the already
                # cached response and controls just to discard them again.
                missing = np.empty(
                    (Y_array.shape[0], len(uncached_positions)), order="F"
                )
                for target, position in enumerate(uncached_positions):
                    if position < len(y_names_tuple):
                        missing[:, target] = Y_array[:, position]
                    else:
                        missing[:, target] = X_array[:, position - len(y_names_tuple)]
                uncached_demeaned, used_preconditioner = self._run_or_raise(
                    x=missing,
                    flist=fe,
                    weights=weights,
                    na_index=na_index,
                    demeaner=demeaner,
                )
                uncached_names = tuple(yx_names[index] for index in uncached_positions)
                uncached_demeaned.setflags(write=False)
                cached = replace(
                    cached,
                    blocks=(*cached.blocks, uncached_demeaned),
                    columns=cached_names + uncached_names,
                    locations={
                        **cached.locations,
                        **{
                            name: (len(cached.blocks), position)
                            for position, name in enumerate(uncached_names)
                        },
                    },
                )

        # Select roles independently: a new response must not force a copy of
        # an unchanged design, and contiguous ranges anywhere in a block view it.
        response_demeaned = self._select_columns(cached=cached, names=y_names_tuple)
        if cached.design is not None and cached.design_names == x_names_tuple:
            design_demeaned = cached.design
        else:
            design_demeaned = self._select_columns(cached=cached, names=x_names_tuple)
            cached = replace(cached, design_names=x_names_tuple, design=design_demeaned)
        self.lookup_demeaned_data[na_index] = cached
        return response_demeaned, design_demeaned, used_preconditioner

    @staticmethod
    def _select_columns(
        cached: DemeanedData,
        names: tuple[str, ...],
    ) -> NDArray[np.float64]:
        if not names:
            return cached.blocks[0][:, :0]
        positions = tuple(cached.locations[name] for name in names)
        first_block, start = positions[0]
        if positions == tuple((first_block, start + j) for j in range(len(names))):
            return cached.blocks[first_block][:, start : start + len(names)]
        selected = np.empty((cached.blocks[0].shape[0], len(names)), order="F")
        for target, (block_index, position) in enumerate(positions):
            selected[:, target] = cached.blocks[block_index][:, position]
        selected.setflags(write=False)
        return selected
