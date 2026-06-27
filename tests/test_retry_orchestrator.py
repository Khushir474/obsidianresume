"""Tests for the retry loop orchestrator."""
import json
import pytest
from unittest.mock import MagicMock, call, patch

from obsidianresumeforge.retry_orchestrator import run_with_retry, _increment_run_id


def _make_crew_output(passed: bool, recommendation: str = ""):
    task_out = MagicMock()
    task_out.name = "evaluate_pipeline_output"
    task_out.raw = json.dumps({
        "composite_score": 0.9 if passed else 0.3,
        "passed": passed,
        "retry_recommendation": recommendation,
        "issues": [],
    })
    crew_out = MagicMock()
    crew_out.tasks_output = [task_out]
    return crew_out


# ── _increment_run_id ──────────────────────────────────────────────────────

def test_increment_run_id_basic():
    assert _increment_run_id("run_20260624_001") == "run_20260624_002"


def test_increment_run_id_large():
    assert _increment_run_id("run_20260624_099") == "run_20260624_100"


# ── run_with_retry ──────────────────────────────────────────────────────────

def test_no_retry_on_passing_eval(tmp_path):
    kickoff_count = 0

    def _factory():
        nonlocal kickoff_count
        m = MagicMock()
        m.crew().kickoff.side_effect = lambda inputs: (
            _make_crew_output(True)
        )
        return m

    output, retries = run_with_retry(
        crew_factory=_factory,
        inputs={"run_id": "run_20260624_001"},
        max_retries=2,
        logs_folder=str(tmp_path),
    )
    assert retries == 0


def test_retries_on_failing_eval(tmp_path):
    call_results = [
        _make_crew_output(False, "Add more metrics"),
        _make_crew_output(True),
    ]
    idx = 0

    def _factory():
        nonlocal idx
        result = call_results[idx]
        idx += 1
        m = MagicMock()
        m.crew().kickoff.return_value = result
        return m

    output, retries = run_with_retry(
        crew_factory=_factory,
        inputs={"run_id": "run_20260624_001"},
        max_retries=2,
        logs_folder=str(tmp_path),
    )
    assert retries == 1


def test_retry_context_injected(tmp_path):
    received_inputs = []
    call_results = [
        _make_crew_output(False, "Improve keyword density"),
        _make_crew_output(True),
    ]
    idx = 0

    def _factory():
        nonlocal idx
        result = call_results[idx]
        idx += 1
        m = MagicMock()

        def _kickoff(inputs):
            received_inputs.append(dict(inputs))
            return result

        m.crew().kickoff.side_effect = _kickoff
        return m

    run_with_retry(
        crew_factory=_factory,
        inputs={"run_id": "run_20260624_001"},
        max_retries=2,
        logs_folder=str(tmp_path),
    )
    assert len(received_inputs) == 2
    assert "retry_context" in received_inputs[1]
    assert "Improve keyword density" in received_inputs[1]["retry_context"]


def test_run_id_increments_on_retry(tmp_path):
    received_run_ids = []
    call_results = [
        _make_crew_output(False, "Fix sourcing map"),
        _make_crew_output(True),
    ]
    idx = 0

    def _factory():
        nonlocal idx
        result = call_results[idx]
        idx += 1
        m = MagicMock()

        def _kickoff(inputs):
            received_run_ids.append(inputs["run_id"])
            return result

        m.crew().kickoff.side_effect = _kickoff
        return m

    run_with_retry(
        crew_factory=_factory,
        inputs={"run_id": "run_20260624_001"},
        max_retries=2,
        logs_folder=str(tmp_path),
    )
    assert received_run_ids[0] == "run_20260624_001"
    assert received_run_ids[1] == "run_20260624_002"


def test_stops_after_max_retries(tmp_path):
    call_count = 0

    def _factory():
        nonlocal call_count
        call_count += 1
        m = MagicMock()
        m.crew().kickoff.return_value = _make_crew_output(False, "Still failing")
        return m

    output, retries = run_with_retry(
        crew_factory=_factory,
        inputs={"run_id": "run_20260624_001"},
        max_retries=2,
        logs_folder=str(tmp_path),
    )
    assert call_count == 3  # initial + 2 retries
    assert retries == 2


def test_eval_logs_written_each_attempt(tmp_path):
    call_results = [
        _make_crew_output(False, "Retry"),
        _make_crew_output(True),
    ]
    idx = 0

    def _factory():
        nonlocal idx
        result = call_results[idx]
        idx += 1
        m = MagicMock()
        m.crew().kickoff.return_value = result
        return m

    run_with_retry(
        crew_factory=_factory,
        inputs={"run_id": "run_20260624_001"},
        max_retries=1,
        logs_folder=str(tmp_path),
    )
    eval_files = list((tmp_path / "eval").glob("*.json"))
    assert len(eval_files) == 2, f"Expected 2 eval logs (one per attempt), found: {[f.name for f in eval_files]}"
