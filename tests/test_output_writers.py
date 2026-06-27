"""Tests for output_writers: eval log, optimization report, KG note."""
import json
import pathlib
import pytest
from unittest.mock import MagicMock

from obsidianresumeforge.output_writers import (
    write_eval_log,
    write_optimization_report,
    write_knowledge_graph_note,
    _extract_json,
)

SAMPLE_EVAL_JSON = {
    "composite_score": 0.72,
    "passed": False,
    "judge_scores": {
        "ATSKeywordHitRateJudge": {"score": 0.8, "weight": 0.35, "details": "4/5 matched"},
    },
    "issues": [],
    "retry_recommendation": "Improve metric density",
}


def _mock_crew_output(task_name: str, raw_text: str):
    task_out = MagicMock()
    task_out.name = task_name
    task_out.raw = raw_text
    crew_out = MagicMock()
    crew_out.tasks_output = [task_out]
    return crew_out


# ── _extract_json ──────────────────────────────────────────────────────────

def test_extract_json_plain():
    result = _extract_json(json.dumps(SAMPLE_EVAL_JSON))
    assert result["composite_score"] == 0.72


def test_extract_json_fenced():
    text = f"Here is the eval:\n```json\n{json.dumps(SAMPLE_EVAL_JSON)}\n```"
    result = _extract_json(text)
    assert result["composite_score"] == 0.72


def test_extract_json_returns_none_on_garbage():
    assert _extract_json("not json at all") is None


# ── write_eval_log ─────────────────────────────────────────────────────────

def test_write_eval_log_valid_json(tmp_vault):
    crew_out = _mock_crew_output("evaluate_pipeline_output", json.dumps(SAMPLE_EVAL_JSON))
    result = write_eval_log("run_20260101_001", crew_out, str(tmp_vault))

    assert result is not None
    assert result.suffix == ".json"
    assert result.exists()
    data = json.loads(result.read_text())
    assert data["composite_score"] == 0.72


def test_write_eval_log_creates_eval_dir(tmp_path):
    crew_out = _mock_crew_output("evaluate_pipeline_output", json.dumps(SAMPLE_EVAL_JSON))
    write_eval_log("run_20260101_001", crew_out, str(tmp_path))
    assert (tmp_path / "eval").is_dir()


def test_write_eval_log_fallback_txt_on_bad_json(tmp_vault):
    crew_out = _mock_crew_output("evaluate_pipeline_output", "Evaluation complete. Score was decent.")
    result = write_eval_log("run_20260101_002", crew_out, str(tmp_vault))
    assert result is not None
    assert result.suffix == ".txt"
    assert result.exists()


def test_write_eval_log_returns_none_when_task_missing(tmp_vault):
    crew_out = _mock_crew_output("some_other_task", "{}")
    result = write_eval_log("run_20260101_003", crew_out, str(tmp_vault))
    assert result is None


# ── write_optimization_report ──────────────────────────────────────────────

def test_write_optimization_report_new_file(tmp_vault):
    report_text = "# Optimization Report\n\nIssue 1: low metric density."
    crew_out = _mock_crew_output("log_run_and_generate_optimization_report", report_text)
    result = write_optimization_report("run_20260101_001", crew_out, str(tmp_vault))

    assert result is not None
    assert result.exists()
    assert "Optimization Report" in result.read_text()


def test_write_optimization_report_no_double_slash(tmp_vault):
    crew_out = _mock_crew_output("log_run_and_generate_optimization_report", "# Report")
    result = write_optimization_report("run_20260101_001", crew_out, str(tmp_vault))
    assert "//" not in str(result)


def test_write_optimization_report_idempotent(tmp_vault):
    existing = tmp_vault / "optimization_report_run_20260101_001.md"
    existing.write_text("# Existing Report")
    crew_out = _mock_crew_output("log_run_and_generate_optimization_report", "# New Report")

    result = write_optimization_report("run_20260101_001", crew_out, str(tmp_vault))
    assert result.read_text() == "# Existing Report", "Should not overwrite existing file"


# ── write_knowledge_graph_note ─────────────────────────────────────────────

def test_write_knowledge_graph_note(tmp_vault):
    result = write_knowledge_graph_note(
        run_id="run_20260101_001",
        jd_path="/path/to/JDs/AI Engineer.md",
        role="ai_ml_engineer",
        composite_score=0.85,
        pdf_path="/path/to/Resumes/PDF/Khushi_Resume.pdf",
        vault_path=str(tmp_vault),
    )
    assert result.exists()
    content = result.read_text()
    assert "[[AI Engineer]]" in content
    assert "[[ai_ml_engineer]]" in content
    assert "[[Khushi_Resume]]" in content
    assert "0.85" in content


def test_write_knowledge_graph_note_creates_dir(tmp_path):
    result = write_knowledge_graph_note(
        run_id="run_test",
        jd_path="/JDs/Role.md",
        role="data_scientist",
        composite_score=0.7,
        pdf_path="/out/resume.pdf",
        vault_path=str(tmp_path),
    )
    assert (tmp_path / "KnowledgeGraph").is_dir()
    assert result.name == "run_test.md"
