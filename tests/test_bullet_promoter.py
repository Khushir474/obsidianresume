"""Tests for auto_promote_bullets: promotion criteria and edge cases."""
import hashlib
import pytest

from obsidianresumeforge.knowledge_graph_store import KGStore
from obsidianresumeforge.bullet_promoter import auto_promote_bullets


@pytest.fixture
def vault(tmp_path):
    (tmp_path / "KnowledgeGraph").mkdir()
    return tmp_path


def _bullet_nid(text: str) -> str:
    return f"bullet:{hashlib.sha256(text.encode()).hexdigest()[:12]}"


def _store_with_run(vault, *, run_id="run_20260627_001", keywords, bullets, sa_score=0.9):
    """Populate a KGStore with one run and return (store, bullet_nid_list)."""
    store = KGStore(str(vault))
    store.ingest_run(
        run_id=run_id,
        date="2026-06-27",
        jd_path="/JDs/Role.md",
        role="ai_ml_engineer",
        company="Acme",
        composite_score=0.87,
        passed=True,
        keywords=keywords,
        sourcing_map=bullets,
        source_attribution_score=sa_score,
    )
    bullet_nids = [_bullet_nid(b["adapted_bullet"]) for b in bullets if b.get("adapted_bullet")]
    return store, bullet_nids


def _bullet(adapted, keywords_embedded):
    return {
        "source_file": "galaara.md",
        "original_bullet": "original",
        "adapted_bullet": adapted,
        "keywords_embedded": keywords_embedded,
        "reason": "",
    }


# ── happy-path promotion ───────────────────────────────────────────────────

def test_promotes_bullet_backed_by_new_keyword(vault):
    kws = [{"text": "Kubernetes", "priority": "Critical", "decision": "NEW", "domain": "hard"}]
    bullets = [_bullet("Deployed on Kubernetes cutting rollout time by 30%", ["Kubernetes"])]
    store, nids = _store_with_run(vault, keywords=kws, bullets=bullets, sa_score=0.9)

    promoted = auto_promote_bullets(store, "run_20260627_001", composite_score=0.87, source_attribution_score=0.9)

    assert nids[0] in promoted
    assert store.get_node(nids[0])["promoted"] is True


def test_promotes_bullet_backed_by_adapt_keyword(vault):
    kws = [{"text": "cross-functional collaboration", "priority": "High", "decision": "ADAPT", "domain": "soft"}]
    bullets = [_bullet("Led cross-functional collaboration across 4 teams", ["cross-functional collaboration"])]
    store, nids = _store_with_run(vault, keywords=kws, bullets=bullets, sa_score=0.9)

    promoted = auto_promote_bullets(store, "run_20260627_001", composite_score=0.87, source_attribution_score=0.9)

    assert nids[0] in promoted


def test_does_not_promote_bullet_backed_only_by_reuse_keyword(vault):
    kws = [{"text": "Python", "priority": "Critical", "decision": "REUSE", "domain": "hard"}]
    bullets = [_bullet("Built Python pipeline with 40% latency reduction", ["Python"])]
    store, nids = _store_with_run(vault, keywords=kws, bullets=bullets, sa_score=0.95)

    promoted = auto_promote_bullets(store, "run_20260627_001", composite_score=0.87, source_attribution_score=0.95)

    assert promoted == []
    assert store.get_node(nids[0])["promoted"] is False


# ── threshold gates ────────────────────────────────────────────────────────

def test_skips_when_composite_below_threshold(vault):
    kws = [{"text": "Spark", "priority": "High", "decision": "NEW", "domain": "hard"}]
    bullets = [_bullet("Processed 10TB daily with Spark", ["Spark"])]
    store, nids = _store_with_run(vault, keywords=kws, bullets=bullets, sa_score=0.9)

    promoted = auto_promote_bullets(
        store, "run_20260627_001",
        composite_score=0.60,  # below default 0.75
        source_attribution_score=0.9,
    )

    assert promoted == []
    assert store.get_node(nids[0])["promoted"] is False


def test_skips_when_sa_score_below_threshold(vault):
    kws = [{"text": "dbt", "priority": "High", "decision": "NEW", "domain": "hard"}]
    bullets = [_bullet("Built dbt models for analytics pipelines", ["dbt"])]
    store, nids = _store_with_run(vault, keywords=kws, bullets=bullets, sa_score=0.5)

    promoted = auto_promote_bullets(
        store, "run_20260627_001",
        composite_score=0.87,
        source_attribution_score=0.50,  # below default 0.85
    )

    assert promoted == []


def test_custom_thresholds_respected(vault):
    kws = [{"text": "Airflow", "priority": "High", "decision": "NEW", "domain": "hard"}]
    bullets = [_bullet("Orchestrated pipelines via Airflow", ["Airflow"])]
    store, nids = _store_with_run(vault, keywords=kws, bullets=bullets, sa_score=0.7)

    # with strict defaults this would be skipped (SA 0.7 < 0.85)
    promoted_strict = auto_promote_bullets(
        store, "run_20260627_001",
        composite_score=0.87, source_attribution_score=0.7,
    )
    assert promoted_strict == []

    # with relaxed SA threshold it should promote
    promoted_relaxed = auto_promote_bullets(
        store, "run_20260627_001",
        composite_score=0.87, source_attribution_score=0.7,
        sa_threshold=0.65,
    )
    assert nids[0] in promoted_relaxed


# ── mixed bullets in one run ───────────────────────────────────────────────

def test_only_new_adapt_backed_bullets_promoted_in_mixed_run(vault):
    kws = [
        {"text": "Python", "priority": "Critical", "decision": "REUSE", "domain": "hard"},
        {"text": "Triton", "priority": "High", "decision": "NEW", "domain": "hard"},
    ]
    b_reuse = _bullet("Built Python ML pipeline with 50% throughput gain", ["Python"])
    b_new = _bullet("Optimised inference kernels using Triton cutting latency by 35%", ["Triton"])
    store, nids = _store_with_run(vault, keywords=kws, bullets=[b_reuse, b_new], sa_score=0.9)
    reuse_nid, new_nid = nids

    promoted = auto_promote_bullets(store, "run_20260627_001", composite_score=0.87, source_attribution_score=0.9)

    assert new_nid in promoted
    assert reuse_nid not in promoted


def test_bullet_with_mixed_keywords_reuse_and_new_is_promoted(vault):
    """A bullet embedding both REUSE and NEW keywords should be promoted."""
    kws = [
        {"text": "Python", "priority": "Critical", "decision": "REUSE", "domain": "hard"},
        {"text": "Triton", "priority": "High", "decision": "NEW", "domain": "hard"},
    ]
    b = _bullet("Wrote Triton kernels in Python reducing GPU latency by 40%", ["Python", "Triton"])
    store, nids = _store_with_run(vault, keywords=kws, bullets=[b], sa_score=0.9)

    promoted = auto_promote_bullets(store, "run_20260627_001", composite_score=0.87, source_attribution_score=0.9)

    assert nids[0] in promoted


# ── edge cases ─────────────────────────────────────────────────────────────

def test_returns_empty_when_no_bullets(vault):
    kws = [{"text": "AWS", "priority": "High", "decision": "NEW", "domain": "hard"}]
    store, _ = _store_with_run(vault, keywords=kws, bullets=[], sa_score=0.9)

    promoted = auto_promote_bullets(store, "run_20260627_001", composite_score=0.87, source_attribution_score=0.9)

    assert promoted == []


def test_returns_empty_when_no_keywords(vault):
    bullets = [_bullet("Built a pipeline", [])]
    store, _ = _store_with_run(vault, keywords=[], bullets=bullets, sa_score=0.9)

    promoted = auto_promote_bullets(store, "run_20260627_001", composite_score=0.87, source_attribution_score=0.9)

    assert promoted == []


def test_already_promoted_bullet_not_double_counted(vault):
    kws = [{"text": "Kubernetes", "priority": "High", "decision": "NEW", "domain": "hard"}]
    bullets = [_bullet("Deployed on Kubernetes", ["Kubernetes"])]
    store, nids = _store_with_run(vault, keywords=kws, bullets=bullets, sa_score=0.9)

    first = auto_promote_bullets(store, "run_20260627_001", composite_score=0.87, source_attribution_score=0.9)
    second = auto_promote_bullets(store, "run_20260627_001", composite_score=0.87, source_attribution_score=0.9)

    assert len(first) == 1
    # promote_bullet returns False when already promoted → not re-added
    assert len(second) == 0


def test_promoted_bullets_visible_via_store_query(vault):
    kws = [{"text": "Ray", "priority": "High", "decision": "NEW", "domain": "hard"}]
    bullets = [_bullet("Scaled training with Ray Tune across 8 GPUs", ["Ray"])]
    store, _ = _store_with_run(vault, keywords=kws, bullets=bullets, sa_score=0.9)

    assert store.promoted_bullets() == []
    auto_promote_bullets(store, "run_20260627_001", composite_score=0.87, source_attribution_score=0.9)
    assert len(store.promoted_bullets()) == 1


# ── integration: promotion persists after save/reload ─────────────────────

def test_promotion_persists_after_save_reload(vault):
    kws = [{"text": "Terraform", "priority": "High", "decision": "NEW", "domain": "hard"}]
    bullets = [_bullet("Provisioned infra with Terraform cutting deploy time by 50%", ["Terraform"])]
    store, nids = _store_with_run(vault, keywords=kws, bullets=bullets, sa_score=0.9)

    auto_promote_bullets(store, "run_20260627_001", composite_score=0.87, source_attribution_score=0.9)
    store.save()

    reloaded = KGStore(str(vault))
    assert reloaded.get_node(nids[0])["promoted"] is True
    assert len(reloaded.promoted_bullets()) == 1
