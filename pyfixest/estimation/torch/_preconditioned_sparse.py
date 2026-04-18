from __future__ import annotations

import torch


class _PreconditionedSparse:
    """
    Wrap a sparse matrix ``D`` and diagonal preconditioner ``M_inv``.

    Presents the linear operator ``A_precond = D @ diag(M_inv)`` without
    materializing the preconditioned matrix explicitly.
    """

    def __init__(
        self,
        D: torch.Tensor,
        M_inv: torch.Tensor,
        *,
        _transposed: bool = False,
        _D_t: torch.Tensor | None = None,
    ):
        m, n = D.shape
        self.shape = (n, m) if _transposed else (m, n)
        self._D = D
        self._M_inv = M_inv
        self._transposed = _transposed
        self._T: _PreconditionedSparse | None = None
        self._D_t = _D_t if _D_t is not None else self._materialize_transpose(D)

    @staticmethod
    def _materialize_transpose(D: torch.Tensor) -> torch.Tensor:
        """Pre-compute ``D^T`` in a GPU-friendly sparse layout."""
        D_t = D.t()
        layout = D_t.layout
        if layout == torch.sparse_coo:
            return D_t.coalesce()
        if layout in (torch.sparse_csr, torch.sparse_csc):
            return D_t.to_sparse_csr()
        return D_t

    def mv(self, v: torch.Tensor) -> torch.Tensor:
        """Compute ``A_precond @ v`` or ``A_precond^T @ v``."""
        if self._transposed:
            return self._M_inv * (self._D_t @ v)
        return self._D @ (self._M_inv * v)

    def mm(self, V: torch.Tensor) -> torch.Tensor:
        """Compute batched matvec with ``V`` of shape ``(n, K)`` or ``(m, K)``."""
        if self._transposed:
            return self._M_inv.unsqueeze(1) * (self._D_t @ V)
        return self._D @ (self._M_inv.unsqueeze(1) * V)

    def t(self) -> _PreconditionedSparse:
        """Return a cached transpose view."""
        if self._T is None:
            self._T = _PreconditionedSparse(
                self._D,
                self._M_inv,
                _transposed=not self._transposed,
                _D_t=self._D_t,
            )
            self._T._T = self
        return self._T
