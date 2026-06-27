"""Tests for knowledge_graph_analytics and write_kg_insights_note."""
import pytest

from obsidianresumeforge.knowledge_graph_store import KGStore
from obsidianresumeforge.knowledge_graph_analytics import (
    resume_anchors,
    score_trend,
    role_fit,
    interview_priorities,
    skill_demand,
    skill_gaps,
    promoted_bullets_summary,
)
from obsidianresumeforge.output_writers import write_kg_insights_note, _extract_company


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def vault(tmp_path):
    (tmp_path / "KnowledgeGraph").mkdir()
    return tmp_path


def _make_store(vault, runs):
    """Helper: ingest a list of run dicts into a KGStore and save."""
    store = KGStore(str(vault))
    for r in runs:
        store.ingest_run(**r)
    store.save()
    return store


def _run(run_id, role="data_scientist", company="Acme", score=0.85, passed=True, keywords=None, date="2026-06-25"):
    return dict(
        run_id=run_id,
        date=date,
        jd_path=f"/JDs/{run_id}.md",
        role=role,
        company=company,
        composite_score=score,
        passed=passed,
        keywords=keywords or [],
        sourcing_map=[],
    )


def _kw(text, priority="High", decision="REUSE", domain="hard"):
    return {"text": text, "priority": priority, "decision": decision, "domain": domain}


# ── _extract_company ──────────────────────────────────────────────────────────

def test_extract_company_at_separator(tmp_path):
    jd = tmp_path / "AI Engineer @ Stripe.md"
    jd.write_text("")
    assert _extract_company(str(jd)) == "Stripe"


def test_extract_company_at_underscore_separator(tmp_path):
    jd = tmp_path / "Data_Scientist_at_Acme.md"
    jd.write_text("")
    assert _extract_company(str(jd)) == "Acme"


def test_extract_company_from_content_frontmatter(tmp_path):
    jd = tmp_path / "Some Role.md"
    jd.write_text("---\ncompany: OpenAI\ntitle: Some Role\n---\n")
    assert _extract_company(str(jd)) == "OpenAI"


def test_extract_company_from_content_bold_field(tmp_path):
    jd = tmp_path / "Analyst.md"
    jd.write_text("# Analyst\n**Company:** Databricks\n\nJob description here.")
    assert _extract_company(str(jd)) == "Databricks"


def test_extract_company_from_content_h1_at(tmp_path):
    jd = tmp_path / "Engineer.md"
    jd.write_text("# Senior Engineer @ Anthropic\n\nWe are hiring...")
    assert _extract_company(str(jd)) == "Anthropic"


def test_extract_company_falls_back_to_stem(tmp_path):
    jd = tmp_path / "Unstructured Role Title.md"
    jd.write_text("No structured company info here.")
    assert _extract_company(str(jd)) == "Unstructured Role Title"


def test_extract_company_missing_file_falls_back_to_stem():
    assert _extract_company("/nonexistent/path/My Role.md") == "My Role"


# ── resume_anchors ────────────────────────────────────────────────────────────

def test_anchors_empty_store(vault):
    store = _make_store(vault, [])
    assert resume_anchors(store) == []


def test_anchors_requires_min_runs(vault):
    # Keyword used 2 times (below default min_runs=3) → not an anchor
    store = _make_store(vault, [
        _run("r1", keywords=[_kw("Python", decision="REUSE")]),
        _run("r2", keywords=[_kw("Python", decision="REUSE")]),
    ])
    assert resume_anchors(store) == []


def test_anchors_requires_min_reuse_rate(vault):
    # Python: 3 REUSE + 1 NEW = 75% → below 80% threshold
    store = _make_store(vault, [
        _run("r1", keywords=[_kw("Python", decision="REUSE")]),
        _run("r2", keywords=[_kw("Python", decision="REUSE")]),
        _run("r3", keywords=[_kw("Python", decision="REUSE")]),
        _run("r4", keywords=[_kw("Python", decision="NEW")]),
    ])
    assert resume_anchors(store) == []


def test_anchors_qualifies_high_reuse_keyword(vault):
    store = _make_store(vault, [
        _run("r1", keywords=[_kw("SQL", decision="REUSE")]),
        _run("r2", keywords=[_kw("SQL", decision="REUSE")]),
        _run("r3", keywords=[_kw("SQL", decision="REUSE")]),
    ])
    anchors = resume_anchors(store)
    assert len(anchors) == 1
    assert anchors[0]["text"] == "SQL"
    assert anchors[0]["reuse_pct"] == 100


def test_anchors_sorted_by_reuse_pct_desc(vault):
    # SQL: 3/3 = 100%, Python: 4/5 = 80%
    kws3 = [_kw("SQL", decision="REUSE")] * 3
    runs = [_run(f"rs{i}", keywords=[_kw("SQL", decision="REUSE")]) for i in range(3)]
    runs += [_run(f"rp{i}", keywords=[_kw("Python", decision="REUSE")]) for i in range(4)]
    runs += [_run("rp4", keywords=[_kw("Python", decision="NEW")])]
    store = _make_store(vault, runs)
    anchors = resume_anchors(store)
    texts = [a["text"] for a in anchors]
    assert texts[0] == "SQL"


# ── score_trend ───────────────────────────────────────────────────────────────

def test_score_trend_empty(vault):
    store = _make_store(vault, [])
    assert score_trend(store) == []


def test_score_trend_sorted_by_date(vault):
    store = _make_store(vault, [
        _run("r2", date="2026-06-26", score=0.9),
        _run("r1", date="2026-06-25", score=0.7),
        _run("r3", date="2026-06-27", score=0.8),
    ])
    trend = score_trend(store)
    assert [r["run_id"] for r in trend] == ["r1", "r2", "r3"]


def test_score_trend_contains_expected_fields(vault):
    store = _make_store(vault, [_run("r1", company="Stripe", role="ai_ml_engineer", score=0.91)])
    trend = score_trend(store)
    assert trend[0]["company"] == "Stripe"
    assert trend[0]["composite_score"] == 0.91
    assert trend[0]["passed"] is True


# ── role_fit ─────────────────────────────────────────────────────────────────

def test_role_fit_empty(vault):
    store = _make_store(vault, [])
    assert role_fit(store) == []


def test_role_fit_averages_correctly(vault):
    store = _make_store(vault, [
        _run("r1", role="data_scientist", score=0.8),
        _run("r2", role="data_scientist", score=0.9),
    ])
    fit = role_fit(store)
    assert len(fit) == 1
    assert fit[0]["avg_score"] == pytest.approx(0.85, abs=0.001)
    assert fit[0]["run_count"] == 2
    assert fit[0]["best_score"] == 0.9


def test_role_fit_sorted_by_avg_score_desc(vault):
    store = _make_store(vault, [
        _run("r1", role="data_scientist", score=0.7),
        _run("r2", role="ai_ml_engineer", score=0.95),
    ])
    fit = role_fit(store)
    assert fit[0]["role"] == "Ai Ml Engineer"


def test_role_fit_counts_passed(vault):
    store = _make_store(vault, [
        _run("r1", role="data_scientist", score=0.9, passed=True),
        _run("r2", role="data_scientist", score=0.5, passed=False),
    ])
    fit = role_fit(store)
    assert fit[0]["passed_count"] == 1


# ── interview_priorities ──────────────────────────────────────────────────────

def test_interview_priorities_empty(vault):
    store = _make_store(vault, [])
    assert interview_priorities(store) == []


def test_interview_priorities_single_company_excluded(vault):
    store = _make_store(vault, [
        _run("r1", company="Stripe", keywords=[_kw("Python", priority="Critical", decision="REUSE")]),
    ])
    assert interview_priorities(store, min_companies=2) == []


def test_interview_priorities_two_companies_qualifies(vault):
    store = _make_store(vault, [
        _run("r1", company="Stripe", keywords=[_kw("Python", priority="Critical", decision="REUSE")]),
        _run("r2", company="Acme", keywords=[_kw("Python", priority="Critical", decision="REUSE")]),
    ])
    prios = interview_priorities(store, min_companies=2)
    assert len(prios) == 1
    assert prios[0]["text"] == "Python"
    assert set(prios[0]["companies"]) == {"Stripe", "Acme"}


def test_interview_priorities_non_critical_not_counted(vault):
    store = _make_store(vault, [
        _run("r1", company="A", keywords=[_kw("Spark", priority="High", decision="NEW")]),
        _run("r2", company="B", keywords=[_kw("Spark", priority="High", decision="NEW")]),
    ])
    assert interview_priorities(store, min_companies=2) == []


# ── skill_demand ──────────────────────────────────────────────────────────────

def test_skill_demand_empty(vault):
    store = _make_store(vault, [])
    assert skill_demand(store) == []


def test_skill_demand_excludes_pure_reuse(vault):
    store = _make_store(vault, [
        _run("r1", keywords=[_kw("Python", decision="REUSE")]),
    ])
    assert skill_demand(store) == []


def test_skill_demand_sorted_by_gap_rate(vault):
    store = _make_store(vault, [
        _run("r1", keywords=[_kw("LangChain", decision="NEW"), _kw("SQL", decision="NEW")]),
        _run("r2", keywords=[_kw("SQL", decision="REUSE")]),
    ])
    demand = skill_demand(store)
    # LangChain: 1/1 = 100%, SQL: 1/2 = 50%
    assert demand[0]["text"] == "LangChain"
    assert demand[0]["gap_rate"] == 1.0
    assert demand[1]["text"] == "SQL"
    assert demand[1]["gap_rate"] == pytest.approx(0.5)


# ── skill_gaps ────────────────────────────────────────────────────────────────

def test_skill_gaps_empty(vault):
    store = _make_store(vault, [])
    gaps = skill_gaps(store)
    assert gaps["hard"] == []
    assert gaps["soft"] == []


def test_skill_gaps_separates_domains(vault):
    store = _make_store(vault, [
        _run("r1", keywords=[
            _kw("Docker", decision="NEW", domain="hard"),
            _kw("leadership", decision="NEW", domain="soft"),
        ]),
    ])
    gaps = skill_gaps(store)
    assert any(g["text"] == "Docker" for g in gaps["hard"])
    assert any(g["text"] == "leadership" for g in gaps["soft"])


def test_skill_gaps_sorted_by_priority_then_new_count(vault):
    store = _make_store(vault, [
        _run("r1", keywords=[
            _kw("Kafka", priority="High", decision="NEW"),
            _kw("Spark", priority="Critical", decision="NEW"),
        ]),
    ])
    gaps = skill_gaps(store)
    hard_texts = [g["text"] for g in gaps["hard"]]
    assert hard_texts.index("Spark") < hard_texts.index("Kafka")


def test_skill_gaps_excludes_pure_reuse(vault):
    store = _make_store(vault, [
        _run("r1", keywords=[_kw("Python", decision="REUSE", domain="hard")]),
    ])
    assert skill_gaps(store)["hard"] == []


# ── promoted_bullets_summary ──────────────────────────────────────────────────

def test_promoted_bullets_summary_empty(vault):
    store = _make_store(vault, [])
    assert promoted_bullets_summary(store) == []


def test_promoted_bullets_summary_only_promoted(vault):
    store = KGStore(str(vault))
    run_data = _run("r1")
    run_data["sourcing_map"] = [{
        "source_file": "exp.md",
        "original_bullet": "orig",
        "adapted_bullet": "Adapted bullet with metric 50%",
        "keywords_embedded": [],
        "reason": "",
    }]
    store.ingest_run(**run_data)
    import hashlib
    nid = f"bullet:{hashlib.sha256('Adapted bullet with metric 50%'.encode()).hexdigest()[:12]}"
    store.promote_bullet(nid)
    store.save()

    bullets = promoted_bullets_summary(store)
    assert len(bullets) == 1
    assert "Adapted bullet" in bullets[0]["text"]
    assert bullets[0]["source_file"] == "exp.md"


# ── write_kg_insights_note ────────────────────────────────────────────────────

def test_insights_note_creates_file(vault):
    _make_store(vault, [_run("r1")])
    out = write_kg_insights_note(str(vault))
    assert out.exists()
    assert out.name == "insights.md"
    assert out.parent.name == "KnowledgeGraph"


def test_insights_note_non_empty(vault):
    _make_store(vault, [_run("r1")])
    out = write_kg_insights_note(str(vault))
    assert out.stat().st_size > 500


def test_insights_note_empty_store(vault):
    _make_store(vault, [])
    out = write_kg_insights_note(str(vault))
    assert out.exists()


def test_insights_note_contains_all_sections(vault):
    _make_store(vault, [_run("r1")])
    content = write_kg_insights_note(str(vault)).read_text()
    for section in ("Resume Anchors", "Score Trend", "Role Fit",
                    "Interview Prep Priorities", "Skill Demand Heatmap",
                    "Skill Gap Report", "Promoted Bullets"):
        assert section in content, f"Missing section: {section}"


def test_insights_note_overwrites_on_second_call(vault):
    _make_store(vault, [_run("r1")])
    out1 = write_kg_insights_note(str(vault))
    out1.write_text("old content")
    out2 = write_kg_insights_note(str(vault))
    assert out2.read_text() != "old content"


def test_insights_note_shows_anchor_data(vault):
    runs = [_run(f"r{i}", keywords=[_kw("SQL", decision="REUSE")]) for i in range(3)]
    _make_store(vault, runs)
    content = write_kg_insights_note(str(vault)).read_text()
    assert "SQL" in content
    assert "100%" in content


def test_insights_note_shows_skill_gap(vault):
    store = _make_store(vault, [
        _run("r1", keywords=[_kw("LangChain", priority="Critical", decision="NEW", domain="hard")]),
    ])
    content = write_kg_insights_note(str(vault)).read_text()
    assert "LangChain" in content
