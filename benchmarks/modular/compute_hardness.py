"""Compute graph-hardness diagnostics for benchmark datasets.

For every pair of fixed-effect dimensions (q, r) we compute the two-factor MAP
contraction rate

    rho_qr = sigma_2(H)^2,    H = N_W^(-1/2) C_qr N_F^(-1/2)

on each connected component of the bipartite (q, r) graph, then report the
worst nontrivial component together with its observation share. For datasets
with three or more FE dimensions the largest pairwise rho is a *diagnostic*,
not an upper bound on the joint MAP rate.

Outputs:
- benchmarks/results/hardness.csv : per-(dataset, fe-pair) worst component
- prints a per-dataset "hardest pair" summary table
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import svds

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pyfixest.core.detect_singletons import detect_singletons  # noqa: E402

CORREIA_DIR = PROJECT_ROOT / "benchmarks" / "correia-benchmark-data"
DATA_DIR = PROJECT_ROOT / "benchmarks" / "data"
OUTPUT_CSV = PROJECT_ROOT / "benchmarks" / "results" / "hardness.csv"

CORREIA_DATASETS = [
    "credit2",
    "credit",
    "soccer",
    "synthetic-complete",
    "synthetic-uniform-easy",
    "synthetic-uniform-hard",
    "synthetic-uniform-harder",
    "synthetic-assortative",
    "synthetic-zigzag",
    "enron",
    "github",
    "patents",
    "workers",
    "schools",
    "directors",
]
CORREIA_FE = ("id1", "id2")
AKM_FE = ("indiv_id", "firm_id", "year")
SYNTH_FE = ("indiv_id", "firm_id", "year")


# ---------------------------------------------------------------------------
# Graph hardness
# ---------------------------------------------------------------------------
@dataclass
class PairHardness:
    n_q_levels: int
    n_r_levels: int
    n_components: int
    worst_rho: float
    worst_obs_share: float
    worst_n_obs: int
    worst_n_q_levels: int
    worst_n_r_levels: int


def _factorize(arr: np.ndarray) -> tuple[np.ndarray, int]:
    codes, _ = pd.factorize(arr, sort=False)
    n = int(codes.max()) + 1 if len(codes) else 0
    return codes.astype(np.int64, copy=False), n


def _build_cooccurrence(
    q_codes: np.ndarray, r_codes: np.ndarray, n_q: int, n_r: int
) -> sp.csr_matrix:
    data = np.ones(len(q_codes), dtype=np.float64)
    C = sp.coo_matrix((data, (q_codes, r_codes)), shape=(n_q, n_r)).tocsr()
    C.sum_duplicates()
    return C


def _bipartite_components(C: sp.csr_matrix) -> tuple[int, np.ndarray, np.ndarray]:
    n_q, _ = C.shape
    A = sp.bmat([[None, C], [C.T, None]], format="csr")
    n_comp, labels = connected_components(A, directed=False, return_labels=True)
    return n_comp, labels[:n_q], labels[n_q:]


_DENSE_SVD_THRESHOLD = 64


def _component_rho(C_comp: sp.csr_matrix) -> float:
    """sigma_2(H)^2 for one connected component (0 if trivial)."""
    nq, nr = C_comp.shape
    if nq < 2 or nr < 2:
        return 0.0
    row_sums = np.asarray(C_comp.sum(axis=1)).flatten()
    col_sums = np.asarray(C_comp.sum(axis=0)).flatten()
    inv_sqrt_row = sp.diags(1.0 / np.sqrt(row_sums))
    inv_sqrt_col = sp.diags(1.0 / np.sqrt(col_sums))
    H = (inv_sqrt_row @ C_comp @ inv_sqrt_col).tocsr()
    # svds requires k < min(m, n); use dense SVD for small components
    if min(H.shape) <= _DENSE_SVD_THRESHOLD:
        sv = np.linalg.svd(H.toarray(), compute_uv=False)
    else:
        try:
            sv = svds(H, k=2, return_singular_vectors=False, which="LM")
        except Exception:
            return float("nan")
    sv_sorted = np.sort(sv)[::-1]
    if len(sv_sorted) < 2:
        return 0.0
    sigma_2 = float(sv_sorted[1])
    # numerical noise can push sigma_2 slightly above 1 in degenerate cases
    sigma_2 = min(max(sigma_2, 0.0), 1.0)
    return sigma_2 * sigma_2


def pair_hardness(q: np.ndarray, r: np.ndarray) -> PairHardness:
    q_codes, n_q = _factorize(q)
    r_codes, n_r = _factorize(r)
    C = _build_cooccurrence(q_codes, r_codes, n_q, n_r)
    n_obs = len(q)
    n_comp, q_labels, r_labels = _bipartite_components(C)

    worst = PairHardness(
        n_q_levels=n_q,
        n_r_levels=n_r,
        n_components=n_comp,
        worst_rho=0.0,
        worst_obs_share=0.0,
        worst_n_obs=0,
        worst_n_q_levels=0,
        worst_n_r_levels=0,
    )
    for c in range(n_comp):
        q_mask = q_labels == c
        r_mask = r_labels == c
        if not q_mask.any() or not r_mask.any():
            continue
        C_comp = C[q_mask][:, r_mask]
        comp_n_obs = int(C_comp.sum())
        if comp_n_obs == 0:
            continue
        rho = _component_rho(C_comp)
        if rho > worst.worst_rho:
            worst = PairHardness(
                n_q_levels=n_q,
                n_r_levels=n_r,
                n_components=n_comp,
                worst_rho=rho,
                worst_obs_share=comp_n_obs / n_obs,
                worst_n_obs=comp_n_obs,
                worst_n_q_levels=int(q_mask.sum()),
                worst_n_r_levels=int(r_mask.sum()),
            )
    return worst


# ---------------------------------------------------------------------------
# Dataset enumeration
# ---------------------------------------------------------------------------
@dataclass
class DatasetSpec:
    dataset_id: str
    kind: str
    path: Path
    fe_cols: tuple[str, ...]
    reader: Callable[[Path, list[str]], pd.DataFrame]


def _read_csv_columns(path: Path, columns: list[str]) -> pd.DataFrame:
    return pd.read_csv(path, usecols=columns)


def _read_parquet_columns(path: Path, columns: list[str]) -> pd.DataFrame:
    return pd.read_parquet(path, columns=columns)


def enumerate_datasets() -> list[DatasetSpec]:
    """Discover datasets from what is already cached on disk."""
    specs: list[DatasetSpec] = []

    for name in CORREIA_DATASETS:
        path = CORREIA_DIR / f"{name}.csv"
        if path.exists():
            specs.append(
                DatasetSpec(
                    dataset_id=name,
                    kind="correia",
                    path=path,
                    fe_cols=CORREIA_FE,
                    reader=_read_csv_columns,
                )
            )

    for path in sorted(DATA_DIR.glob("akm_*_iter_1.parquet")):
        specs.append(
            DatasetSpec(
                dataset_id=path.stem,
                kind="akm",
                path=path,
                fe_cols=AKM_FE,
                reader=_read_parquet_columns,
            )
        )

    for prefix in ("simple", "difficult"):
        for path in sorted(DATA_DIR.glob(f"{prefix}_*_iter_1.parquet")):
            specs.append(
                DatasetSpec(
                    dataset_id=path.stem,
                    kind="synthetic-base",
                    path=path,
                    fe_cols=SYNTH_FE,
                    reader=_read_parquet_columns,
                )
            )

    return specs


# ---------------------------------------------------------------------------
# Singleton pruning
# ---------------------------------------------------------------------------
def _factorize_fe_block(df: pd.DataFrame, fe_cols: tuple[str, ...]) -> np.ndarray:
    """Stack FE columns as a C-contiguous (n, m) int64 array of category codes."""
    blocks = [pd.factorize(df[col].to_numpy(), sort=False)[0] for col in fe_cols]
    return np.column_stack(blocks).astype(np.int64, copy=False)


def drop_singleton_rows(
    df: pd.DataFrame, fe_cols: tuple[str, ...]
) -> tuple[pd.DataFrame, int]:
    """Drop singleton rows across all FE columns jointly."""
    if len(fe_cols) < 2:
        return df, 0
    ids = _factorize_fe_block(df, fe_cols)
    mask = detect_singletons(ids)
    n_dropped = int(mask.sum())
    if n_dropped == 0:
        return df, 0
    return df.loc[~mask].reset_index(drop=True), n_dropped


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def compute_hardness_rows(
    specs: list[DatasetSpec], *, drop_singletons: bool = True
) -> pd.DataFrame:
    rows: list[dict] = []
    for spec in specs:
        t0 = time.perf_counter()
        df = spec.reader(spec.path, list(spec.fe_cols))
        n_obs_raw = len(df)
        if drop_singletons:
            df, n_dropped = drop_singleton_rows(df, spec.fe_cols)
        else:
            n_dropped = 0
        n_obs = len(df)
        drop_note = f" (dropped {n_dropped:,} singletons)" if n_dropped else ""
        print(
            f"[hardness] {spec.dataset_id:<45} kind={spec.kind:<14} "
            f"n_obs={n_obs:>10,}{drop_note}",
            flush=True,
        )
        if n_obs == 0:
            print("    all rows were singletons; skipping", flush=True)
            continue
        for fe_a, fe_b in combinations(spec.fe_cols, 2):
            try:
                h = pair_hardness(df[fe_a].to_numpy(), df[fe_b].to_numpy())
            except Exception as exc:
                print(f"  pair ({fe_a}, {fe_b}) failed: {exc}", flush=True)
                continue
            rows.append(
                {
                    "dataset_id": spec.dataset_id,
                    "kind": spec.kind,
                    "n_obs_raw": n_obs_raw,
                    "n_obs": n_obs,
                    "n_singletons_dropped": n_dropped,
                    "fe_a": fe_a,
                    "fe_b": fe_b,
                    "n_levels_a": h.n_q_levels,
                    "n_levels_b": h.n_r_levels,
                    "n_components": h.n_components,
                    "rho_qr": h.worst_rho,
                    "one_minus_rho": 1.0 - h.worst_rho,
                    "worst_component_obs_share": h.worst_obs_share,
                    "worst_component_n_obs": h.worst_n_obs,
                    "worst_component_n_levels_a": h.worst_n_q_levels,
                    "worst_component_n_levels_b": h.worst_n_r_levels,
                }
            )
            print(
                f"    ({fe_a}, {fe_b}): rho={h.worst_rho:.6f} "
                f"1-rho={1.0 - h.worst_rho:.2e} "
                f"share={h.worst_obs_share:.3f} "
                f"components={h.n_components}",
                flush=True,
            )
        dt = time.perf_counter() - t0
        print(f"  [{dt:.2f}s]", flush=True)
    return pd.DataFrame(rows)


def print_hardest_pair_summary(df: pd.DataFrame) -> None:
    if df.empty:
        print("[hardness] no rows produced")
        return
    idx = df.groupby("dataset_id")["rho_qr"].idxmax()
    summary = df.loc[idx].sort_values(["kind", "rho_qr"], ascending=[True, False])

    dataset_w = max(12, int(summary["dataset_id"].str.len().max()))
    header = (
        f"{'dataset':<{dataset_w}} {'kind':<14} {'n_obs':>10} "
        f"{'hardest pair':<24} {'rho':>10} {'1-rho':>12} "
        f"{'share':>8} {'#comp':>6}"
    )
    sep = "-" * len(header)
    print("\n  Graph hardness (hardest pair per dataset)")
    print(f"  {sep}")
    print(f"  {header}")
    print(f"  {sep}")
    for _, r in summary.iterrows():
        pair = f"{r['fe_a']} x {r['fe_b']}"
        print(
            "  "
            f"{r['dataset_id']:<{dataset_w}} {r['kind']:<14} "
            f"{int(r['n_obs']):>10,} {pair:<24} "
            f"{r['rho_qr']:>10.6f} {r['one_minus_rho']:>12.3e} "
            f"{r['worst_component_obs_share']:>8.3f} {int(r['n_components']):>6}"
        )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--keep-singletons",
        action="store_true",
        help="Skip iterative singleton dropping (default: drop singletons "
        "across all FE columns jointly, matching what fixest / pyfixest "
        "actually solve).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_CSV,
        help=f"Output CSV path (default: {OUTPUT_CSV}).",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    specs = enumerate_datasets()
    drop = not args.keep_singletons
    print(
        f"[hardness] {len(specs)} datasets queued "
        f"(singleton dropping: {'on' if drop else 'off'})",
        flush=True,
    )
    results = compute_hardness_rows(specs, drop_singletons=drop)
    results.to_csv(args.output, index=False)
    print(f"\n[hardness] wrote {args.output}")
    print_hardest_pair_summary(results)
