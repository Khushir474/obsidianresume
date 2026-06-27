"""Tests for KGStore: data model, ingestion, queries, persistence."""
import json
import pathlib
import pytest

from obsidianresumeforge.knowledge_graph_store import KGStore


# ── fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture
def vault(tmp_path):
    (tmp_path / "KnowledgeGraph").mkdir()
    return tmp_path


def _store(vault) -> KGStore:
    return KGStore(str(vault))


def _minimal_run(**overrides) -> dict:
    base = dict(
        run_id="run_20260626_001",
        date="2026-06-26",
        jd_path="/JDs/AI Engineer.md",
        role="ai_ml_engineer",
        company="Stripe",
        composite_score=0.87,
        passed=True,
        keywords=[],
        sourcing_map=[],
    )
    base.update(overrides)
    return base


# ── init and persistence ───────────────────────────────────────────────────

def test_init_empty_when_no_file(vault):
    store = _store(vault)
    assert store.run_nodes() == []
    assert store.keyword_nodes() == []


def test_init_loads_existing_json(vault):
    existing = {"nodes": {"run:r1": {"type": "run", "run_id": "r1"}}, "edges": []}
    (vault / "KnowledgeGraph" / "kg_store.json").write_text(json.dumps(existing))
    store = _store(vault)
    assert store.get_node("run:r1") is not None


def test_init_falls_back_on_corrupt_json(vault):
    (vault / "KnowledgeGraph" / "kg_store.json").write_text("not json {{{")
    store = _store(vault)
    assert store.run_nodes() == []


def test_save_writes_valid_json(vault):
    store = _store(vault)
    store.upsert_node("run:r1", type="run", run_id="r1")
    path = store.save()
    assert path.exists()
    data = json.loads(path.read_text())
    assert "run:r1" in data["nodes"]


def test_save_creates_kg_dir(tmp_path):
    store = KGStore(str(tmp_path))
    store.upsert_node("run:r1", type="run")
    store.save()
    assert (tmp_path / "KnowledgeGraph" / "kg_store.json").exists()


# ── low-level ops ──────────────────────────────────────────────────────────

def test_upsert_node_creates_then_merges(vault):
    store = _store(vault)
    store.upsert_node("keyword:python", type="keyword", text="Python", run_count=1)
    store.upsert_node("keyword:python", run_count=2, last_decision="REUSE")
    node = store.get_node("keyword:python")
    assert node["text"] == "Python"
    assert node["run_count"] == 2
    assert node["last_decision"] == "REUSE"


def test_get_node_returns_none_for_missing(vault):
    store = _store(vault)
    assert store.get_node("run:nonexistent") is None


def test_add_edge_and_get_edges_by_rel(vault):
    store = _store(vault)
    store.add_edge("run:r1", "role:ml", "for_role")
    store.add_edge("run:r1", "keyword:python", "used_keyword", decision="REUSE")
    assert len(store.get_edges(rel="for_role")) == 1
    assert len(store.get_edges(rel="used_keyword")) == 1


def test_get_edges_filter_src(vault):
    store = _store(vault)
    store.add_edge("run:r1", "role:ml", "for_role")
    store.add_edge("run:r2", "role:ml", "for_role")
    assert len(store.get_edges(src="run:r1")) == 1


def test_get_edges_filter_dst(vault):
    store = _store(vault)
    store.add_edge("run:r1", "keyword:python", "used_keyword")
    store.add_edge("run:r1", "keyword:sql", "used_keyword")
    assert len(store.get_edges(dst="keyword:python")) == 1


# ── ingest_run: nodes created ──────────────────────────────────────────────

def test_ingest_run_creates_run_node(vault):
    store = _store(vault)
    store.ingest_run(**_minimal_run())
    node = store.get_node("run:run_20260626_001")
    assert node["composite_score"] == 0.87
    assert node["passed"] is True
    assert node["company"] == "Stripe"


def test_ingest_run_creates_role_node(vault):
    store = _store(vault)
    store.ingest_run(**_minimal_run())
    node = store.get_node("role:ai_ml_engineer")
    assert node["type"] == "role"
    assert node["run_count"] == 1
    assert node["display"] == "Ai Ml Engineer"


def test_ingest_run_creates_company_node(vault):
    store = _store(vault)
    store.ingest_run(**_minimal_run())
    node = store.get_node("company:stripe")
    assert node["name"] == "Stripe"
    assert node["run_count"] == 1


def test_ingest_run_creates_keyword_nodes(vault):
    store = _store(vault)
    store.ingest_run(**_minimal_run(keywords=[
        {"text": "Python", "priority": "Critical", "decision": "REUSE", "domain": "hard"},
        {"text": "stakeholder communication", "priority": "High", "decision": "NEW", "domain": "soft"},
    ]))
    py_node = store.get_node("keyword:python")
    assert py_node["reuse_count"] == 1
    assert py_node["new_count"] == 0
    assert py_node["domain"] == "hard"

    sk_node = store.get_node("keyword:stakeholder communication")
    assert sk_node["new_count"] == 1
    assert sk_node["domain"] == "soft"


def test_ingest_run_creates_bullet_node(vault):
    store = _store(vault)
    store.ingest_run(**_minimal_run(
        sourcing_map=[{
            "source_file": "galaara.md",
            "original_bullet": "Built ML pipelines",
            "adapted_bullet": "Built ML pipelines reducing latency by 40%",
            "keywords_embedded": ["Python"],
            "reason": "metric injection",
        }],
        source_attribution_score=0.9,
    ))
    bullets = [n for n in store._data["nodes"].values() if n.get("type") == "bullet"]
    assert len(bullets) == 1
    assert bullets[0]["source_file"] == "galaara.md"
    assert bullets[0]["avg_source_score"] == 0.9
    assert bullets[0]["promoted"] is False


# ── ingest_run: edges created ──────────────────────────────────────────────

def test_ingest_run_creates_for_role_edge(vault):
    store = _store(vault)
    store.ingest_run(**_minimal_run())
    edges = store.get_edges(src="run:run_20260626_001", rel="for_role")
    assert len(edges) == 1
    assert edges[0]["dst"] == "role:ai_ml_engineer"


def test_ingest_run_creates_used_keyword_edge_with_attrs(vault):
    store = _store(vault)
    store.ingest_run(**_minimal_run(keywords=[
        {"text": "Python", "priority": "Critical", "decision": "REUSE"},
    ]))
    edges = store.get_edges(src="run:run_20260626_001", rel="used_keyword")
    assert len(edges) == 1
    assert edges[0]["attrs"]["priority"] == "Critical"
    assert edges[0]["attrs"]["decision"] == "REUSE"


def test_ingest_run_creates_backed_by_edge(vault):
    store = _store(vault)
    store.ingest_run(**_minimal_run(
        keywords=[{"text": "Python", "priority": "Critical", "decision": "REUSE"}],
        sourcing_map=[{
            "source_file": "galaara.md",
            "original_bullet": "Built ML pipelines",
            "adapted_bullet": "Built ML pipelines reducing latency by 40%",
            "keywords_embedded": ["Python"],
            "reason": "",
        }],
    ))
    edges = store.get_edges(src="keyword:python", rel="backed_by")
    assert len(edges) == 1


# ── cross-run accumulation ─────────────────────────────────────────────────

def test_keyword_counters_accumulate_across_runs(vault):
    store = _store(vault)
    store.ingest_run(**_minimal_run(
        run_id="run_20260626_001",
        keywords=[{"text": "Python", "priority": "Critical", "decision": "REUSE"}],
    ))
    store.ingest_run(**_minimal_run(
        run_id="run_20260626_002",
        keywords=[{"text": "Python", "priority": "Critical", "decision": "ADAPT"}],
    ))
    node = store.get_node("keyword:python")
    assert node["run_count"] == 2
    assert node["reuse_count"] == 1
    assert node["adapt_count"] == 1
    assert node["last_decision"] == "ADAPT"


def test_role_run_count_increments(vault):
    store = _store(vault)
    store.ingest_run(**_minimal_run(run_id="run_20260626_001"))
    store.ingest_run(**_minimal_run(run_id="run_20260626_002"))
    assert store.get_node("role:ai_ml_engineer")["run_count"] == 2


def test_bullet_use_count_and_avg_score_accumulate(vault):
    adapted = "Built ML pipelines reducing latency by 40%"
    entry = {
        "source_file": "galaara.md",
        "original_bullet": "Built ML pipelines",
        "adapted_bullet": adapted,
        "keywords_embedded": [],
        "reason": "",
    }
    store = _store(vault)
    store.ingest_run(**_minimal_run(
        run_id="run_20260626_001", sourcing_map=[entry], source_attribution_score=0.8,
    ))
    store.ingest_run(**_minimal_run(
        run_id="run_20260626_002", sourcing_map=[entry], source_attribution_score=1.0,
    ))
    import hashlib
    bullet_hash = hashlib.sha256(adapted.encode()).hexdigest()[:12]
    node = store.get_node(f"bullet:{bullet_hash}")
    assert node["use_count"] == 2
    assert node["avg_source_score"] == pytest.approx(0.9, abs=1e-4)
    assert node["origin_run"] == "run_20260626_001"


# ── high-level queries ─────────────────────────────────────────────────────

def test_keyword_texts_returns_all(vault):
    store = _store(vault)
    store.ingest_run(**_minimal_run(keywords=[
        {"text": "Python", "priority": "Critical", "decision": "REUSE"},
        {"text": "AWS", "priority": "High", "decision": "NEW"},
    ]))
    texts = store.keyword_texts()
    assert "Python" in texts
    assert "AWS" in texts


def test_keyword_decision_history(vault):
    store = _store(vault)
    store.ingest_run(**_minimal_run(
        run_id="run_20260626_001",
        keywords=[{"text": "Python", "priority": "Critical", "decision": "NEW"}],
    ))
    store.ingest_run(**_minimal_run(
        run_id="run_20260626_002",
        keywords=[{"text": "Python", "priority": "Critical", "decision": "REUSE"}],
    ))
    history = store.keyword_decision_history("Python")
    assert history == ["NEW", "REUSE"]


def test_keyword_decision_history_case_insensitive(vault):
    store = _store(vault)
    store.ingest_run(**_minimal_run(keywords=[
        {"text": "PyTorch", "priority": "High", "decision": "ADAPT"},
    ]))
    assert store.keyword_decision_history("pytorch") == ["ADAPT"]
    assert store.keyword_decision_history("PyTorch") == ["ADAPT"]


def test_promoted_bullets_empty_initially(vault):
    store = _store(vault)
    store.ingest_run(**_minimal_run(sourcing_map=[{
        "source_file": "x.md",
        "original_bullet": "Did a thing",
        "adapted_bullet": "Did a thing with 50% impact",
        "keywords_embedded": [],
        "reason": "",
    }]))
    assert store.promoted_bullets() == []


# ── bullet promotion ───────────────────────────────────────────────────────

def test_promote_bullet_marks_node(vault):
    store = _store(vault)
    adapted = "Did a thing with 50% impact"
    store.ingest_run(**_minimal_run(sourcing_map=[{
        "source_file": "x.md",
        "original_bullet": "Did a thing",
        "adapted_bullet": adapted,
        "keywords_embedded": [],
        "reason": "",
    }]))
    import hashlib
    bullet_nid = f"bullet:{hashlib.sha256(adapted.encode()).hexdigest()[:12]}"
    result = store.promote_bullet(bullet_nid)
    assert result is True
    assert store.get_node(bullet_nid)["promoted"] is True
    assert len(store.promoted_bullets()) == 1


def test_promote_bullet_returns_false_when_already_promoted(vault):
    store = _store(vault)
    adapted = "Did a thing with 50% impact"
    store.ingest_run(**_minimal_run(sourcing_map=[{
        "source_file": "x.md",
        "original_bullet": "Did a thing",
        "adapted_bullet": adapted,
        "keywords_embedded": [],
        "reason": "",
    }]))
    import hashlib
    bullet_nid = f"bullet:{hashlib.sha256(adapted.encode()).hexdigest()[:12]}"
    store.promote_bullet(bullet_nid)
    assert store.promote_bullet(bullet_nid) is False


def test_promote_bullet_returns_false_for_missing(vault):
    store = _store(vault)
    assert store.promote_bullet("bullet:doesnotexist") is False


# ── edge case: skip empty keyword/bullet text ──────────────────────────────

def test_ingest_skips_empty_keyword_text(vault):
    store = _store(vault)
    store.ingest_run(**_minimal_run(keywords=[
        {"text": "", "priority": "Critical", "decision": "NEW"},
        {"text": "   ", "priority": "High", "decision": "REUSE"},
    ]))
    assert store.keyword_nodes() == []


def test_ingest_skips_empty_adapted_bullet(vault):
    store = _store(vault)
    store.ingest_run(**_minimal_run(sourcing_map=[
        {"source_file": "x.md", "original_bullet": "x", "adapted_bullet": "", "keywords_embedded": [], "reason": ""},
    ]))
    assert store.promoted_bullets() == []
    bullet_nodes = [n for n in store._data["nodes"].values() if n.get("type") == "bullet"]
    assert bullet_nodes == []
