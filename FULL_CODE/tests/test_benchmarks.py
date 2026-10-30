"""benchmarks.collect(): every number comes from a report file; missing or bad files are flagged, never invented."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from pardadb import benchmarks


def _metrics(f1: float, person: int = 10, org: int = 0) -> dict[str, Any]:
    return {
        "person": {"precision": f1, "recall": f1, "f1": f1, "support": person},
        "address": {"precision": 1.0, "recall": 1.0, "f1": 1.0, "support": 2},
        "organization": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": org},
        "micro": {"precision": f1, "recall": f1, "f1": f1},
    }


def _variant(name: str, fmt: str, mb: float, f1: float, notes: str = "", verified: bool = True) -> dict[str, Any]:
    return {
        "variant": name,
        "format": fmt,
        "weights_mb": mb,
        "dir_mb": mb + 3,
        "bank_f1": f1,
        "golden_f1": f1,
        "saffron_name_recall": 0.99,
        "narration_recall": 0.9,
        "cpu_ms_per_text": 10.0,
        "peak_rss_mb": 900.0,
        "threshold": 0.4,
        "verified": verified,
        "notes": notes,
    }


def _write(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj) if not isinstance(obj, str) else obj)


@pytest.fixture
def reports(tmp_path: Path) -> Path:
    """A reports directory with small synthetic files of the real shapes."""
    _write(
        tmp_path / "compression.json",
        {
            "description": "test run",
            "env": {"device": "cpu", "threads": 2},
            "chosen": "small",
            "variants": [
                _variant("big", "fp32", 200.0, 0.90),
                _variant("small", "int8", 50.0, 0.92),
                _variant("broken", "int8-all", 40.0, 0.0, notes="FAILED: predicts nothing", verified=False),
            ],
        },
    )
    model = {"model": "m", "device": "cpu", "threshold": 0.4}
    _write(tmp_path / "base.json", dict(model, bank_test=_metrics(0.5, org=4), golden=_metrics(0.6)))
    _write(tmp_path / "ft.json", dict(model, bank_test=_metrics(0.9, org=4), golden=_metrics(0.8)))
    _write(
        tmp_path / "saffron_full.json",
        {
            "customer_name_cases": 5,
            "narration_cases": 7,
            "rules_only": {"name_recall": 0.6},
            "base": {"name_recall": 0.8},
            "fine_tuned": {"name_recall": 1.0},
        },
    )
    return tmp_path


def test_compression_headline_is_computed_from_the_variants(reports: Path, tmp_path: Path) -> None:
    out = benchmarks.collect(reports, tmp_path / "no-soak.json")
    sec = out["compression"]
    assert sec["status"] == "ok"
    assert sec["source"].endswith("compression.json")
    h = sec["data"]["headline"]
    assert (h["from_mb"], h["to_mb"], h["times_smaller"]) == (200.0, 50.0, 4.0)
    assert (h["from_f1"], h["to_f1"]) == (0.90, 0.92)
    rows = {r["variant"]: r for r in sec["data"]["variants"]}
    assert rows["small"]["chosen"] and not rows["big"]["chosen"]
    assert rows["broken"]["failed"] and not rows["broken"]["verified"]
    assert sec["data"]["threads"] == 2


def test_model_eval_keeps_only_labels_with_cases(reports: Path, tmp_path: Path) -> None:
    out = benchmarks.collect(reports, tmp_path / "no-soak.json")
    ft = out["fine_tuned"]["data"]
    assert ft["bank_f1"] == 0.9 and ft["golden_f1"] == 0.8
    assert ft["bank_labels"] == {"person": 0.9, "address": 1.0, "organization": 0.0}
    assert "organization" not in ft["golden_labels"]
    assert ft["bank_n"] == 16 and ft["golden_n"] == 12
    assert out["base_model"]["data"]["bank_f1"] == 0.5


def test_missing_files_are_marked_not_invented(reports: Path, tmp_path: Path) -> None:
    out = benchmarks.collect(reports, tmp_path / "no-soak.json")
    for key in ("pipeline", "soak"):
        assert out[key]["status"] == "missing"
        assert out[key]["data"] is None
        assert out[key]["detail"] == "not measured yet"
    assert out["saffron"]["data"]["name_cases"] == 5


def test_corrupt_json_is_invalid(reports: Path, tmp_path: Path) -> None:
    _write(reports / "ft.json", "{not json")
    sec = benchmarks.collect(reports, tmp_path / "no-soak.json")["fine_tuned"]
    assert sec["status"] == "invalid"
    assert sec["data"] is None
    assert "unreadable" in sec["detail"]


def test_wrong_shape_is_invalid(reports: Path, tmp_path: Path) -> None:
    _write(reports / "compression.json", {"chosen": "x", "variants": []})
    sec = benchmarks.collect(reports, tmp_path / "no-soak.json")["compression"]
    assert sec["status"] == "invalid"
    assert sec["data"] is None


def test_soak_and_pipeline_are_read_when_present(reports: Path, tmp_path: Path) -> None:
    soak = tmp_path / "soak.json"
    _write(soak, {"duration_s": 3600, "queries": 1000})
    names = {
        "n": 3,
        "model_only": {"name_recall": 0.9},
        "with_bank_name_list": {"rules_only": {"name_recall": 0.5}, "product": {"name_recall": 1.0}},
        "without_name_list": {"rules_only": {"name_recall": 0.3}, "product": {"name_recall": 0.7}},
    }
    _write(
        reports / "pipeline.json",
        {
            "date": "2026-10-31",
            "model_dir": "models/x",
            "device": "cpu",
            "threads": 4,
            "threshold": 0.5,
            "customer_names": names,
            "free_text": {"n_texts": 9},
        },
    )
    out = benchmarks.collect(reports, soak)
    assert out["soak"]["status"] == "ok" and out["soak"]["data"]["queries"] == 1000
    p = out["pipeline"]["data"]
    assert (p["n"], p["threshold"]) == (3, 0.5)
    assert p["with_bank_name_list"]["product"]["name_recall"] == 1.0
    assert p["without_name_list"]["product"]["name_recall"] == 0.7


def test_soak_must_be_an_object(reports: Path, tmp_path: Path) -> None:
    soak = tmp_path / "soak.json"
    _write(soak, [1, 2, 3])
    assert benchmarks.collect(reports, soak)["soak"]["status"] == "invalid"


def test_display_path_outside_repo_is_unchanged(tmp_path: Path) -> None:
    assert benchmarks.display_path(tmp_path / "x.json") == str(tmp_path / "x.json")
