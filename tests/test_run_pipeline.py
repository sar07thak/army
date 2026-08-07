"""Tests for the ``run_pipeline`` CLI (stage dispatch)."""

from __future__ import annotations

import pytest

import run_pipeline

ALL_STAGES = (
    "ingest",
    "features",
    "labels",
    "split",
    "train",
    "compare",
    "explain",
    "visualize",
    "forecast",
)


def test_parse_args_default_stage() -> None:
    assert run_pipeline.parse_args([]).stage == "ingest"


def test_parse_args_accepts_every_stage() -> None:
    for stage in (*ALL_STAGES, "all"):
        assert run_pipeline.parse_args(["--stage", stage]).stage == stage


def test_parse_args_rejects_unknown_stage() -> None:
    with pytest.raises(SystemExit):
        run_pipeline.parse_args(["--stage", "bogus"])


def test_run_all_stages_runs_every_stage_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """run_all_stages calls all 8 stage runners in dependency order."""
    calls: list[str] = []

    def fake(stage: str) -> object:
        def _runner(stage: str = stage) -> dict[str, object]:
            calls.append(stage)
            return {"geo_units": 1}

        return _runner

    for name in ALL_STAGES:
        monkeypatch.setattr(run_pipeline, f"run_{name}_stage", fake(name))

    result = run_pipeline.run_all_stages()
    assert calls == list(ALL_STAGES)
    assert list(result) == list(ALL_STAGES)


def test_stage_runner_functions_exist() -> None:
    for name in ALL_STAGES:
        runner = getattr(run_pipeline, f"run_{name}_stage", None)
        assert callable(runner), f"run_{name}_stage missing"
