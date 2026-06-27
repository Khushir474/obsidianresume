"""Tests for write_interview_prep_note: output structure, content, edge cases."""
import pytest

from obsidianresumeforge.knowledge_graph_store import KGStore
from obsidianresumeforge.output_writers import write_interview_prep_note


@pytest.fixture
def vault(tmp_path):
    (tmp_path / "KnowledgeGraph").mkdir()
    (tmp_path / "InterviewPrep").mkdir()
    return tmp_path


def _populated_store(vault, *, run_id="run_20260627_001", keywords=None):
    """Return a KGStore with one run and optional keywords pre-ingested."""
    store = KGStore(str(vault))
    store.ingest_run(
        run_id=run_id,
        date="2026-06-27",
        jd_path="/JDs/ML Engineer @ Stripe.md",
        role="ai_ml_engineer",
        company="Stripe",
        composite_score=0.87,
        passed=True,
        keywords=keywords or [],
        sourcing_map=[],
    )
    store.save()
    return store


# ── file creation ──────────────────────────────────────────────────────────

def test_creates_file_in_interview_prep_dir(vault):
    _populated_store(vault)
    out = write_interview_prep_note("run_20260627_001", str(vault))
    assert out is not None
    assert out.parent.name == "InterviewPrep"
    assert out.name == "run_20260627_001.md"
    assert out.exists()


def test_creates_interview_prep_dir_if_missing(tmp_path):
    (tmp_path / "KnowledgeGraph").mkdir()
    _populated_store(tmp_path)
    out = write_interview_prep_note("run_20260627_001", str(tmp_path))
    assert out is not None
    assert (tmp_path / "InterviewPrep").is_dir()


def test_returns_none_for_unknown_run_id(vault):
    KGStore(str(vault)).save()  # empty store
    out = write_interview_prep_note("run_99990101_999", str(vault))
    assert out is None


# ── note content: metadata ─────────────────────────────────────────────────

def test_contains_run_id(vault):
    _populated_store(vault)
    content = write_interview_prep_note("run_20260627_001", str(vault)).read_text()
    assert "run_20260627_001" in content


def test_contains_role(vault):
    _populated_store(vault)
    content = write_interview_prep_note("run_20260627_001", str(vault)).read_text()
    assert "Ai Ml Engineer" in content


def test_contains_company(vault):
    _populated_store(vault)
    content = write_interview_prep_note("run_20260627_001", str(vault)).read_text()
    assert "Stripe" in content


def test_contains_score_and_pass_indicator(vault):
    _populated_store(vault)
    content = write_interview_prep_note("run_20260627_001", str(vault)).read_text()
    assert "0.87" in content
    assert "✓" in content


def test_contains_jd_wikilink(vault):
    _populated_store(vault)
    content = write_interview_prep_note("run_20260627_001", str(vault)).read_text()
    assert "[[ML Engineer @ Stripe]]" in content


def test_failed_run_shows_cross_indicator(vault):
    store = KGStore(str(vault))
    store.ingest_run(
        run_id="run_20260627_002",
        date="2026-06-27",
        jd_path="/JDs/Role.md",
        role="data_scientist",
        company="Acme",
        composite_score=0.55,
        passed=False,
        keywords=[],
        sourcing_map=[],
    )
    store.save()
    content = write_interview_prep_note("run_20260627_002", str(vault)).read_text()
    assert "✗" in content


# ── note content: keywords ─────────────────────────────────────────────────

def test_critical_keywords_appear_under_correct_section(vault):
    _populated_store(vault, keywords=[
        {"text": "Python", "priority": "Critical", "decision": "REUSE", "domain": "hard"},
        {"text": "PyTorch", "priority": "Critical", "decision": "NEW", "domain": "hard"},
    ])
    content = write_interview_prep_note("run_20260627_001", str(vault)).read_text()
    assert "## Critical Keywords" in content
    assert "- Python" in content
    assert "- PyTorch" in content


def test_high_keywords_appear_under_correct_section(vault):
    _populated_store(vault, keywords=[
        {"text": "stakeholder communication", "priority": "High", "decision": "ADAPT", "domain": "soft"},
    ])
    content = write_interview_prep_note("run_20260627_001", str(vault)).read_text()
    assert "## High Keywords" in content
    assert "- stakeholder communication" in content


def test_medium_and_low_keywords_grouped(vault):
    _populated_store(vault, keywords=[
        {"text": "agility", "priority": "Medium", "decision": "NEW", "domain": "soft"},
        {"text": "ownership", "priority": "Low", "decision": "NEW", "domain": "soft"},
    ])
    content = write_interview_prep_note("run_20260627_001", str(vault)).read_text()
    assert "## Medium Keywords" in content
    assert "## Low Keywords" in content


def test_empty_priority_group_not_rendered(vault):
    _populated_store(vault, keywords=[
        {"text": "Python", "priority": "Critical", "decision": "REUSE", "domain": "hard"},
    ])
    content = write_interview_prep_note("run_20260627_001", str(vault)).read_text()
    assert "## High Keywords" not in content
    assert "## Medium Keywords" not in content
    assert "## Low Keywords" not in content


def test_keywords_sorted_alphabetically_within_group(vault):
    _populated_store(vault, keywords=[
        {"text": "Spark", "priority": "High", "decision": "NEW", "domain": "hard"},
        {"text": "AWS", "priority": "High", "decision": "ADAPT", "domain": "hard"},
        {"text": "dbt", "priority": "High", "decision": "NEW", "domain": "hard"},
    ])
    content = write_interview_prep_note("run_20260627_001", str(vault)).read_text()
    aws_pos = content.index("- AWS")
    dbt_pos = content.index("- dbt")
    spark_pos = content.index("- Spark")
    assert aws_pos < dbt_pos < spark_pos


def test_no_keywords_run_still_produces_note(vault):
    _populated_store(vault, keywords=[])
    out = write_interview_prep_note("run_20260627_001", str(vault))
    assert out is not None
    assert out.exists()
    content = out.read_text()
    assert "## Prep Notes" in content


# ── note content: prep notes placeholder ──────────────────────────────────

def test_contains_prep_notes_section(vault):
    _populated_store(vault)
    content = write_interview_prep_note("run_20260627_001", str(vault)).read_text()
    assert "## Prep Notes" in content


def test_prep_notes_contains_placeholder_text(vault):
    _populated_store(vault)
    content = write_interview_prep_note("run_20260627_001", str(vault)).read_text()
    assert "STAR" in content


# ── idempotency ────────────────────────────────────────────────────────────

def test_overwrites_existing_file(vault):
    _populated_store(vault)
    first = write_interview_prep_note("run_20260627_001", str(vault))
    first.write_text("old content")
    second = write_interview_prep_note("run_20260627_001", str(vault))
    assert second.read_text() != "old content"
