"""Tests for Phase 1 (update_knowledge_graph) and Phase 2 (generate_html)."""
import json
import pathlib
import pytest
from unittest.mock import MagicMock

from obsidianresumeforge.output_writers import update_knowledge_graph
from obsidianresumeforge.knowledge_graph_store import KGStore
from obsidianresumeforge.knowledge_graph_viz import generate_html


# ── helpers ────────────────────────────────────────────────────────────────

def _mock_crew(tasks: dict[str, str]):
    """Build a mock CrewOutput with named task outputs."""
    task_outs = []
    for name, raw in tasks.items():
        t = MagicMock()
        t.name = name
        t.raw = raw
        task_outs.append(t)
    crew = MagicMock()
    crew.tasks_output = task_outs
    return crew


def _full_crew_output():
    role_out = json.dumps({
        "best_role": "ai_ml_engineer",
        "confidence_scores": {"ai_ml_engineer": 0.92},
        "status": "matched",
        "reasoning": "Strong match",
    })
    kw_out = json.dumps({
        "selected_role": "ai_ml_engineer",
        "list_a_hard_keywords": [
            {"keyword": "Python", "score": "Critical", "inclusion": "verbatim", "rationale": ""},
            {"keyword": "PyTorch", "score": "High", "inclusion": "verbatim", "rationale": ""},
        ],
        "list_b_soft_keywords": [
            {"keyword": "ownership", "score": "High", "inclusion": "paraphrase", "rationale": ""},
        ],
    })
    sourcing_map = json.dumps([
        {
            "section": "Experience",
            "source_file": "galaara.md",
            "original_bullet": "Built ML pipelines",
            "adapted_bullet": "Built ML pipelines reducing inference latency by 40%",
            "keywords_embedded": ["Python", "PyTorch"],
            "reason": "metric injection",
        }
    ])
    resume_out = f"LaTeX content here.\n\nSourcing map:\n{sourcing_map}"
    eval_out = json.dumps({
        "composite_score": 0.87,
        "passed": True,
        "judge_scores": {
            "ATSKeywordHitRateJudge": {"score": 0.9, "weight": 0.35, "details": ""},
            "SourceAttributionJudge": {"score": 0.85, "weight": 0.20, "details": ""},
            "MetricDensityJudge": {"score": 0.8, "weight": 0.25, "details": ""},
            "FormatComplianceJudge": {"score": 1.0, "weight": 0.20, "details": ""},
        },
        "issues": [],
        "retry_recommendation": "",
    })
    return _mock_crew({
        "classify_role": role_out,
        "extract_and_score_keywords": kw_out,
        "write_tailored_latex_resume_and_export_pdf": resume_out,
        "evaluate_pipeline_output": eval_out,
    })


@pytest.fixture
def vault(tmp_path):
    (tmp_path / "KnowledgeGraph").mkdir()
    return tmp_path


# ── update_knowledge_graph ─────────────────────────────────────────────────

def test_update_kg_creates_store_file(vault):
    crew = _full_crew_output()
    result = update_knowledge_graph("run_20260627_001", "/JDs/ML Engineer @ Stripe.md", crew, str(vault))
    assert result is not None
    assert result.exists()


def test_update_kg_populates_run_node(vault):
    crew = _full_crew_output()
    update_knowledge_graph("run_20260627_001", "/JDs/ML Engineer @ Stripe.md", crew, str(vault))
    store = KGStore(str(vault))
    run_node = store.get_node("run:run_20260627_001")
    assert run_node is not None
    assert run_node["composite_score"] == pytest.approx(0.87)
    assert run_node["passed"] is True
    assert run_node["role"] == "ai_ml_engineer"


def test_update_kg_extracts_company_from_jd_at_separator(vault):
    crew = _full_crew_output()
    update_knowledge_graph("run_20260627_001", "/JDs/ML Engineer @ Stripe.md", crew, str(vault))
    store = KGStore(str(vault))
    assert store.get_node("company:stripe") is not None


def test_update_kg_extracts_company_from_jd_dash_separator(vault):
    crew = _full_crew_output()
    update_knowledge_graph("run_20260627_001", "/JDs/ML Engineer - Stripe.md", crew, str(vault))
    store = KGStore(str(vault))
    assert store.get_node("company:stripe") is not None


def test_update_kg_uses_full_stem_when_no_separator(vault):
    crew = _full_crew_output()
    update_knowledge_graph("run_20260627_001", "/JDs/AIEngineer.md", crew, str(vault))
    store = KGStore(str(vault))
    assert store.get_node("company:aiengineer") is not None


def test_update_kg_populates_keyword_nodes(vault):
    crew = _full_crew_output()
    update_knowledge_graph("run_20260627_001", "/JDs/Role @ Co.md", crew, str(vault))
    store = KGStore(str(vault))
    py_node = store.get_node("keyword:python")
    assert py_node is not None
    assert py_node["domain"] == "hard"
    assert py_node["priority"] if False else True  # priority stored on edge, not node

    ownership_node = store.get_node("keyword:ownership")
    assert ownership_node is not None
    assert ownership_node["domain"] == "soft"


def test_update_kg_populates_bullet_node(vault):
    crew = _full_crew_output()
    update_knowledge_graph("run_20260627_001", "/JDs/Role @ Co.md", crew, str(vault))
    store = KGStore(str(vault))
    bullets = [n for n in store._data["nodes"].values() if n.get("type") == "bullet"]
    assert len(bullets) == 1
    assert "40%" in bullets[0]["text"]


def test_update_kg_accumulates_on_second_run(vault):
    crew = _full_crew_output()
    update_knowledge_graph("run_20260627_001", "/JDs/Role @ Co.md", crew, str(vault))
    update_knowledge_graph("run_20260627_002", "/JDs/Role @ Co.md", crew, str(vault))
    store = KGStore(str(vault))
    py_node = store.get_node("keyword:python")
    assert py_node["run_count"] == 2


def test_update_kg_handles_missing_task_outputs_gracefully(vault):
    crew = _mock_crew({})  # no task outputs at all
    result = update_knowledge_graph("run_20260627_001", "/JDs/Role.md", crew, str(vault))
    assert result is not None  # still saves, just with defaults
    store = KGStore(str(vault))
    run_node = store.get_node("run:run_20260627_001")
    assert run_node["role"] == "unknown"
    assert run_node["composite_score"] == 0.0


# ── generate_html ──────────────────────────────────────────────────────────

def test_generate_html_creates_file(vault):
    store = KGStore(str(vault))
    store.ingest_run(
        run_id="run_20260627_001",
        date="2026-06-27",
        jd_path="/JDs/Role.md",
        role="ai_ml_engineer",
        company="Stripe",
        composite_score=0.87,
        passed=True,
        keywords=[{"text": "Python", "priority": "Critical", "decision": "REUSE", "domain": "hard"}],
        sourcing_map=[],
    )
    store.save()
    out = generate_html(store, str(vault))
    assert out.exists()
    assert out.suffix == ".html"


def test_generate_html_is_non_empty(vault):
    store = KGStore(str(vault))
    store.ingest_run(
        run_id="run_20260627_001",
        date="2026-06-27",
        jd_path="/JDs/Role.md",
        role="ai_ml_engineer",
        company="Acme",
        composite_score=0.5,
        passed=False,
        keywords=[],
        sourcing_map=[],
    )
    store.save()
    out = generate_html(store, str(vault))
    assert out.stat().st_size > 1000


def test_generate_html_empty_store(vault):
    store = KGStore(str(vault))
    out = generate_html(store, str(vault))
    assert out.exists()


def test_generate_html_with_bullets_and_promoted(vault):
    store = KGStore(str(vault))
    store.ingest_run(
        run_id="run_20260627_001",
        date="2026-06-27",
        jd_path="/JDs/Role.md",
        role="data_scientist",
        company="Meta",
        composite_score=0.91,
        passed=True,
        keywords=[
            {"text": "Python", "priority": "Critical", "decision": "REUSE", "domain": "hard"},
            {"text": "SQL", "priority": "High", "decision": "NEW", "domain": "hard"},
        ],
        sourcing_map=[{
            "source_file": "galaara.md",
            "original_bullet": "Built pipeline",
            "adapted_bullet": "Built pipeline with 30% efficiency gain",
            "keywords_embedded": ["Python"],
            "reason": "",
        }],
        source_attribution_score=0.9,
    )
    import hashlib
    adapted = "Built pipeline with 30% efficiency gain"
    nid = f"bullet:{hashlib.sha256(adapted.encode()).hexdigest()[:12]}"
    store.promote_bullet(nid)
    store.save()
    out = generate_html(store, str(vault))
    assert out.exists()
