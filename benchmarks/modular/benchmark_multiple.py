"""Reproducible end-to-end multiple-estimation benchmarks (see MULTIPLE.md)."""

from __future__ import annotations

import argparse
import cProfile
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.modular.akm_dgp import (
    AKMConfig,
    simulate_akm_panel,
    summarize_akm_panel,
)
from benchmarks.modular.dgp_functions import base_dgp

THREAD_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "RAYON_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def formulas(pattern, count, controls, fe):
    """Return Python/R multiple formulas and their ordered independent fits."""
    xs = [f"x{i}" for i in range(1, count + 1)]
    common = " + ".join(f"z{i}" for i in range(1, controls + 1))
    suffix = " | " + " + ".join(fe) if fe else ""
    if pattern == "lhs":
        ys = [f"y{i}" for i in range(1, count + 1)]
        return {
            "multi": f"{' + '.join(ys)} ~ {common}{suffix}",
            "r_multi": f"c({', '.join(ys)}) ~ {common}{suffix}",
            "separate": [f"{y} ~ {common}{suffix}" for y in ys],
        }
    if pattern == "shared":
        rhs = [f"{common} + {x}" for x in xs]
        multi = f"{common} + sw({', '.join(xs)})"
    else:
        rhs = [
            " + ".join(xs[: i + 1]) if pattern == "csw" else x for i, x in enumerate(xs)
        ]
        multi = f"{pattern}({', '.join(xs)})"
    return {
        "multi": f"y ~ {multi}{suffix}",
        "r_multi": f"y ~ {multi}{suffix}",
        "separate": [f"y ~ {x}{suffix}" for x in rhs],
    }


def prepare(args):
    args.directory.mkdir(parents=True, exist_ok=True)
    manifest_path = args.directory / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"Use a new dataset directory: {manifest_path}")
    cases = []
    for dgp in args.dgps:
        for n in args.sizes:
            for seed in args.seeds:
                rng = np.random.default_rng(seed)
                if dgp in ("simple", "difficult"):
                    data = base_dgp(n=n, type_=dgp, seed=seed)
                    diagnostics = {}
                else:
                    data = simulate_akm_panel(
                        AKMConfig(
                            n_workers=n // 10,
                            n_firms=max(5, n // 100),
                            delta=0.05 if dgp == "akm_low_mobility" else 0.2,
                        ),
                        seed=seed,
                    )
                    diagnostics = summarize_akm_panel(data)
                for prefix, width in (("x", args.models), ("z", args.controls)):
                    for j in range(1, width + 1):
                        data[f"{prefix}{j}"] = rng.normal(size=len(data))
                for j in range(1, args.models + 1):
                    data[f"y{j}"] = data["y"] + rng.normal(size=len(data))
                data["weight"] = rng.uniform(0.5, 1.5, len(data))
                if args.scenario == "missing":
                    for j in range(1, args.models + 1):
                        rows = rng.choice(
                            len(data), max(1, len(data) // 100), replace=False
                        )
                        data.loc[rows, [f"x{j}", f"y{j}"]] = np.nan
                elif args.scenario == "singletons":
                    data.loc[:4, "firm_id"] = np.arange(5) + data.firm_id.max() + 1
                name = f"{dgp}-{n}-{seed}-{args.scenario}"
                path = args.directory / f"{name}.csv"
                data.to_csv(path, index=False)
                metadata = {
                    "data": path.name,
                    "sha256": file_hash(path),
                    "dgp": dgp,
                    "requested_n": n,
                    "n": len(data),
                    "seed": seed,
                    "fe_levels": {
                        col: int(data[col].nunique())
                        for col in ("indiv_id", "firm_id", "year")
                    },
                    "diagnostics": diagnostics,
                }
                for pattern in args.patterns:
                    for fe_count in args.fe_counts:
                        fe = ["indiv_id", "firm_id", "year"][:fe_count]
                        cases.append(
                            {
                                **metadata,
                                "id": f"{name}-{pattern}-fe{fe_count}",
                                "pattern": pattern,
                                "models": args.models,
                                "controls": args.controls,
                                "scenario": args.scenario,
                                **formulas(pattern, args.models, args.controls, fe),
                            }
                        )
    write_json(manifest_path, {"schema": 1, "cases": cases})


def estimates(models):
    return [
        {
            "names": m.coef().index.tolist(),
            "coef": m.coef().tolist(),
            "vcov": m.variance_covariance.vcov.tolist(),
            "se": m.se().tolist(),
            "nobs": m.sample_info.n_obs,
        }
        for m in models
    ]


def compare_estimates(left, right):
    """Align names; tolerances allow iterative projection stopping error."""
    for a, b in zip(left, right, strict=True):
        a_names = [
            "Intercept" if name == "(Intercept)" else name for name in a["names"]
        ]
        b_names = [
            "Intercept" if name == "(Intercept)" else name for name in b["names"]
        ]
        assert set(a_names) == set(b_names), "retained coefficient names"
        assert a["nobs"] == b["nobs"], "nobs"
        order = [b_names.index(name) for name in a_names]
        np.testing.assert_allclose(
            a["coef"],
            np.asarray(b["coef"])[order],
            rtol=1e-7,
            atol=1e-8,
            err_msg="coefficients",
        )
        np.testing.assert_allclose(
            a["se"],
            np.asarray(b["se"])[order],
            rtol=1e-6,
            atol=1e-8,
            err_msg="standard errors",
        )
        np.testing.assert_allclose(
            a["vcov"],
            np.asarray(b["vcov"])[np.ix_(order, order)],
            rtol=1e-6,
            atol=1e-9,
            err_msg="vcov",
        )


def provenance():
    import pyfixest as pf
    import pyfixest.core._core_impl as core

    checkout = Path(pf.__file__).resolve().parents[1]

    def git(*command):
        return subprocess.check_output(
            ["git", "-C", str(checkout), *command], text=True
        ).strip()

    return {
        "python": sys.version,
        "platform": platform.platform(),
        "checkout": str(checkout),
        "head": git("rev-parse", "HEAD"),
        "diff_sha256": hashlib.sha256(git("diff", "HEAD").encode()).hexdigest(),
        "extension": core.__file__,
        "extension_sha256": file_hash(Path(core.__file__)),
        "versions": {
            p: importlib.metadata.version(p)
            for p in ("pyfixest", "numpy", "pandas", "formulaic")
        },
        "threads": {key: os.environ.get(key) for key in THREAD_VARS},
    }


def worker(args):
    import pyfixest as pf

    case = json.loads(args.case.read_text())
    data = pd.read_csv(args.directory / case["data"], float_precision="round_trip")
    demeaner = (
        pf.MapDemeaner(backend="rust", fixef_tol=1e-8, fixef_maxiter=10000)
        if args.backend == "map"
        else pf.LsmrDemeaner(
            backend="within",
            preconditioner="additive",
            fixef_atol=1e-8,
            fixef_btol=1e-8,
            fixef_maxiter=10000,
        )
    )
    kwargs = {
        "data": data,
        "vcov": "iid",
        "fixef_rm": "singleton" if case["scenario"] == "singletons" else "none",
        "demeaner": demeaner,
        "weights": "weight" if case["scenario"] == "weights" else None,
        "lean": args.lean,
        "store_data": not args.no_store_data,
    }

    def fit(label):
        if label == "separate":
            return [pf.feols(f, **kwargs) for f in case["separate"]]
        return pf.feols(case["multi"], **kwargs).to_list()

    metadata = provenance()
    # Warm both paths and check correctness outside timing. Every call creates
    # a fresh estimation cache; importing/compiling and reading data are excluded.
    reference = estimates(fit("separate"))
    shared = estimates(fit("multi"))
    compare_estimates(reference, shared)
    if args.profile:
        from unittest.mock import patch

        from pyfixest.estimation.internals.demean_ import DemeanCache

        calls = []
        original = DemeanCache._run_or_raise

        def traced(self, x, flist, weights, na_index, demeaner):
            calls.append(
                {
                    "columns": x.shape[1],
                    "cached_preconditioner": na_index in self.lookup_preconditioner,
                }
            )
            return original(self, x, flist, weights, na_index, demeaner)

        with (
            patch.object(DemeanCache, "_run_or_raise", traced),
            cProfile.Profile() as profiler,
        ):
            result = fit("multi")
        del result
        profiler.dump_stats(str(args.output.with_suffix(".prof")))
        record = {"demean_calls": calls}
    else:
        times = {"separate": [], "multi": []}
        for rep in range(args.reps):
            for label in (
                ("separate", "multi") if rep % 2 == 0 else ("multi", "separate")
            ):
                gc.collect()
                start = time.perf_counter()
                result = fit(label)
                times[label].append(time.perf_counter() - start)
                assert len(result) == case["models"]
                del result
        record = {"times": times}
    write_json(
        args.output,
        {
            "case": case,
            "backend": args.backend,
            "status": "passed",
            "estimates": shared,
            "environment": metadata,
            **record,
            "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * (1 if sys.platform == "darwin" else 1024),
        },
    )


def run(args):
    cases = json.loads((args.directory / "manifest.json").read_text())["cases"]
    args.output.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, **{key: str(args.threads) for key in THREAD_VARS}}
    verified = set()
    failed = False
    for case in cases:
        if args.match and args.match not in case["id"]:
            continue
        path = args.directory / case["data"]
        if path not in verified:
            if file_hash(path) != case["sha256"]:
                raise ValueError(f"Dataset hash mismatch: {path}")
            verified.add(path)
        for backend in args.backends:
            output = (
                args.output
                / f"{case['id']}-{backend}{'-profile' if args.profile else ''}.json"
            )
            if output.exists():
                raise FileExistsError(f"Use a new output directory: {output}")
            request = output.with_suffix(".request.json")
            write_json(request, case)
            if backend == "fixest":
                if args.profile:
                    raise ValueError("Profiling is supported for Python backends only")
                command = [
                    "Rscript",
                    str(Path(__file__).with_suffix(".R")),
                    str(args.directory),
                    str(request),
                    str(output),
                    str(args.reps),
                    str(args.threads),
                    str(int(args.lean)),
                    str(int(args.no_store_data)),
                ]
            else:
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "worker",
                    str(args.directory),
                    "--case",
                    str(request),
                    "--output",
                    str(output),
                    "--backend",
                    backend,
                    "--reps",
                    str(args.reps),
                ]
                command += [
                    flag
                    for flag, enabled in (
                        ("--profile", args.profile),
                        ("--lean", args.lean),
                        ("--no-store-data", args.no_store_data),
                    )
                    if enabled
                ]
            with output.with_suffix(".log").open("w") as log:
                try:
                    completed = subprocess.run(
                        command,
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        timeout=args.timeout,
                        check=False,
                    )
                    status = "passed" if completed.returncode == 0 else "failed"
                except subprocess.TimeoutExpired:
                    status = "timeout"
            if status != "passed":
                failed = True
                write_json(output, {"case": case, "backend": backend, "status": status})
            else:
                record = json.loads(output.read_text())
                record["settings"] = {
                    "reps": args.reps,
                    "threads": args.threads,
                    "timeout": args.timeout,
                    "lean": args.lean,
                    "store_data": not args.no_store_data,
                    "profile": args.profile,
                    "tolerance": 1e-8,
                    "maxiter": 10000,
                    "harness_sha256": file_hash(Path(__file__)),
                    "r_harness_sha256": file_hash(Path(__file__).with_suffix(".R")),
                }
                write_json(output, record)
            print(f"{case['id']} {backend}: {status}", flush=True)
    if not verified:
        raise ValueError("No benchmark cases matched")
    if failed:
        raise SystemExit("Some cases failed or timed out; inspect the result logs")


def summarize(args):
    records = []
    for directory in args.results:
        for path in sorted(directory.glob("*.json")):
            if path.name.endswith(".request.json"):
                continue
            record = json.loads(path.read_text())
            if "status" not in record:
                continue
            row = {
                "run": str(directory),
                "case": record["case"]["id"],
                "backend": record["backend"],
                "status": record["status"],
            }
            if "times" in record:
                for kind, times in record["times"].items():
                    row[kind] = statistics.median(times)
                    row[f"{kind}_min"] = min(times)
                    row[f"{kind}_max"] = max(times)
                row["speedup"] = row["separate"] / row["multi"]
            records.append(row)
    pd.DataFrame(records).to_csv(args.output, index=False)
    # Cross-run and cross-package correctness, only for identical data/settings.
    reference = {}
    for directory in args.results:
        for path in sorted(directory.glob("*.json")):
            record = json.loads(path.read_text())
            if "estimates" not in record:
                continue
            key = (record["case"]["id"], record["case"]["sha256"])
            if key in reference:
                compare_estimates(reference[key], record["estimates"])
            else:
                reference[key] = record["estimates"]
    print(f"Wrote {len(records)} rows; compared estimates for {len(reference)} cases.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    p.add_argument("directory", type=Path)
    p.add_argument(
        "--dgps",
        nargs="+",
        choices=("simple", "difficult", "akm", "akm_low_mobility"),
        default=["simple", "difficult"],
    )
    p.add_argument(
        "--sizes", nargs="+", type=int, default=[1000, 10000, 100000, 1000000]
    )
    p.add_argument("--seeds", nargs="+", type=int, default=[20260926])
    p.add_argument(
        "--patterns",
        nargs="+",
        choices=("csw", "shared", "lhs", "sw"),
        default=["csw", "shared", "lhs", "sw"],
    )
    p.add_argument(
        "--fe-counts", nargs="+", type=int, choices=(0, 1, 2, 3), default=[2, 3]
    )
    p.add_argument("--models", type=int, default=10)
    p.add_argument("--controls", type=int, default=10)
    p.add_argument(
        "--scenario",
        choices=("clean", "weights", "missing", "singletons"),
        default="clean",
    )
    for mode in ("run", "worker"):
        p = commands.add_parser(mode)
        p.add_argument("directory", type=Path)
        p.add_argument("--output", type=Path, required=True)
        p.add_argument("--reps", type=int, default=5)
        p.add_argument("--profile", action="store_true")
        p.add_argument("--lean", action="store_true")
        p.add_argument("--no-store-data", action="store_true")
        if mode == "worker":
            p.add_argument("--case", type=Path, required=True)
            p.add_argument("--backend", choices=("map", "within"), required=True)
        else:
            p.add_argument(
                "--backends",
                nargs="+",
                choices=("map", "within", "fixest"),
                default=["map", "within"],
            )
            p.add_argument("--threads", type=int, default=1)
            p.add_argument("--timeout", type=float, default=300)
            p.add_argument("--match", default="")
    p = commands.add_parser("summarize")
    p.add_argument("results", nargs="+", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for name in ("models", "controls", "reps", "threads", "timeout"):
        if hasattr(args, name) and getattr(args, name) < (2 if name == "models" else 1):
            parser.error(f"--{name} is too small")
    if args.command == "prepare" and any(n < 1000 for n in args.sizes):
        parser.error("--sizes must be at least 1000")
    globals()[args.command](args)


if __name__ == "__main__":
    main()
