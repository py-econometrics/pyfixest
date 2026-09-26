"""Smoke the persisted-data benchmark path without asserting machine timings."""

from __future__ import annotations

import argparse
import json

from benchmarks.modular.benchmark_multiple import prepare, worker, write_json


def test_multiple_benchmark_saved_cases(tmp_path):
    """All formula recipes execute and validate separate versus multiple fits."""
    prepare(
        argparse.Namespace(
            directory=tmp_path,
            dgps=["simple"],
            sizes=[1000],
            seeds=[42],
            models=3,
            controls=2,
            scenario="missing",
            patterns=["csw", "shared", "lhs", "sw"],
            fe_counts=[2],
        )
    )
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    for case in manifest["cases"]:
        request = tmp_path / "request.json"
        output = tmp_path / "result.json"
        write_json(request, case)
        worker(
            argparse.Namespace(
                case=request,
                directory=tmp_path,
                backend="map",
                lean=False,
                no_store_data=True,
                profile=False,
                reps=1,
                output=output,
            )
        )
        result = json.loads(output.read_text())
        assert result["status"] == "passed"
        assert len(result["estimates"]) == 3
        assert len(result["times"]["multi"]) == 1
        assert len(result["times"]["separate"]) == 1
