"""Tests for semantic_filter hybrid similarity scoring (local model, no network)."""
import pathlib
import pytest

from obsidianresumeforge.semantic_filter import (
    compute_hybrid_score,
    classify_keywords,
    rank_experience_files,
    _jaccard,
    _compute_history_score,
    REUSE_THRESHOLD,
    ADAPT_THRESHOLD,
    _DEFAULT_MIN_RUNS,
    _DEFAULT_STABILITY_THRESHOLD,
    _HISTORY_BLEND_WEIGHT,
    _HISTORY_DECISION_SCORES,
)

# Load model once for the whole test session — avoids repeated slow loads
@pytest.fixture(scope="session")
def model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


# ── _jaccard ───────────────────────────────────────────────────────────────

def test_jaccard_identical():
    assert _jaccard("python machine learning", "python machine learning") == 1.0


def test_jaccard_disjoint():
    assert _jaccard("python", "java") == 0.0


def test_jaccard_partial():
    score = _jaccard("python machine learning", "machine learning tensorflow")
    assert 0.0 < score < 1.0


# ── compute_hybrid_score ───────────────────────────────────────────────────

def test_identical_strings_score_near_one(model):
    score = compute_hybrid_score("Python", "Python", model)
    assert score >= REUSE_THRESHOLD, f"Expected ≥{REUSE_THRESHOLD}, got {score}"


def test_related_terms_in_adapt_range(model):
    # "Python developer" vs "Python engineer" — share the "Python" token and are
    # semantically close, so hybrid score should be well above ADAPT_THRESHOLD
    # but below perfect (not identical strings).
    score = compute_hybrid_score("Python developer", "Python engineer", model)
    assert score >= ADAPT_THRESHOLD, (
        f"Expected ≥{ADAPT_THRESHOLD} for close synonyms, got {score}"
    )


def test_unrelated_terms_in_new_range(model):
    score = compute_hybrid_score("basket weaving", "Python machine learning AWS", model)
    assert score < ADAPT_THRESHOLD, f"Expected NEW range (<{ADAPT_THRESHOLD}), got {score}"


def test_score_bounded_zero_to_one(model):
    for a, b in [("Python", "JavaScript"), ("AI", "artificial intelligence"), ("x", "y")]:
        score = compute_hybrid_score(a, b, model)
        assert 0.0 <= score <= 1.0, f"Score out of bounds for ({a!r}, {b!r}): {score}"


# ── classify_keywords ──────────────────────────────────────────────────────

def test_classify_identical_keyword_as_reuse(model):
    result = classify_keywords(["Python"], ["Python", "Java", "SQL"], model)
    assert result["Python"] == "REUSE"


def test_classify_unrelated_keyword_as_new(model):
    result = classify_keywords(["basket weaving"], ["Python", "machine learning", "AWS"], model)
    assert result["basket weaving"] == "NEW"


def test_classify_returns_all_keywords(model):
    keywords = ["Python", "leadership", "AWS"]
    kg_nodes = ["Python", "cloud infrastructure"]
    result = classify_keywords(keywords, kg_nodes, model)
    assert set(result.keys()) == set(keywords)


def test_classify_all_new_when_no_kg_nodes(model):
    keywords = ["Python", "PyTorch"]
    result = classify_keywords(keywords, [], model)
    assert all(v == "NEW" for v in result.values())


# ── rank_experience_files ──────────────────────────────────────────────────

def test_rank_returns_most_relevant_first(tmp_path, model):
    ml_file = tmp_path / "ml_engineer.md"
    ml_file.write_text(
        "# ML Engineer | AI Corp\nBuilt PyTorch models, deployed to AWS SageMaker, "
        "ran inference pipelines, machine learning, deep learning, Python."
    )
    unrelated_file = tmp_path / "accountant.md"
    unrelated_file.write_text(
        "# Accountant | Finance Inc\nManaged accounts payable, reconciled ledgers, "
        "prepared tax returns, financial auditing, bookkeeping."
    )

    ranked = rank_experience_files(
        jd_keywords=["Python", "PyTorch", "machine learning", "AWS"],
        experience_folder=str(tmp_path),
        top_n=5,
        model=model,
    )
    assert len(ranked) == 2
    assert str(ml_file) == ranked[0], f"ML file should rank first, got: {ranked[0]}"


def test_rank_returns_at_most_top_n(tmp_path, model):
    for i in range(6):
        (tmp_path / f"exp_{i}.md").write_text(f"# Role {i}\nContent {i}")
    ranked = rank_experience_files(["Python"], str(tmp_path), top_n=3, model=model)
    assert len(ranked) == 3


def test_rank_empty_folder(tmp_path, model):
    ranked = rank_experience_files(["Python"], str(tmp_path), top_n=5, model=model)
    assert ranked == []


# ── _compute_history_score ─────────────────────────────────────────────────

def test_history_score_all_reuse():
    node = {"run_count": 5, "reuse_count": 5, "adapt_count": 0, "new_count": 0}
    assert _compute_history_score(node) == pytest.approx(_HISTORY_DECISION_SCORES["REUSE"])


def test_history_score_all_new():
    node = {"run_count": 3, "reuse_count": 0, "adapt_count": 0, "new_count": 3}
    assert _compute_history_score(node) == pytest.approx(_HISTORY_DECISION_SCORES["NEW"])


def test_history_score_mixed():
    # 2 REUSE, 1 NEW out of 3 runs
    node = {"run_count": 3, "reuse_count": 2, "adapt_count": 0, "new_count": 1}
    expected = (2 / 3) * _HISTORY_DECISION_SCORES["REUSE"] + (1 / 3) * _HISTORY_DECISION_SCORES["NEW"]
    assert _compute_history_score(node) == pytest.approx(expected, rel=1e-4)


def test_history_score_zero_runs():
    assert _compute_history_score({"run_count": 0}) == 0.0


# ── warm-start shortcut ────────────────────────────────────────────────────

def test_stable_reuse_keyword_skips_embedding(model):
    # Python appears 5 times, always REUSE → shortcut fires
    kg_history = {
        "python": {"text": "Python", "run_count": 5, "reuse_count": 5, "adapt_count": 0, "new_count": 0}
    }
    result = classify_keywords(["Python"], [], model=model, kg_history=kg_history)
    assert result["Python"] == "REUSE"


def test_stable_reuse_requires_min_runs(model):
    # run_count=2 < default min_runs=3 → shortcut should NOT fire, falls through to embedding
    kg_history = {
        "python": {"text": "Python", "run_count": 2, "reuse_count": 2, "adapt_count": 0, "new_count": 0}
    }
    # With no kg_nodes the embedding pass classifies as NEW (no KG content to match against)
    result = classify_keywords(["Python"], [], model=model, kg_history=kg_history)
    assert result["Python"] == "NEW"  # shortcut did not fire; embedding sees no kg_nodes


def test_stable_reuse_requires_stability_threshold(model):
    # run_count=4, reuse_rate=0.5 < default 0.8 → shortcut does NOT fire
    kg_history = {
        "python": {"text": "Python", "run_count": 4, "reuse_count": 2, "adapt_count": 0, "new_count": 2}
    }
    result = classify_keywords(["Python"], [], model=model, kg_history=kg_history)
    assert result["Python"] == "NEW"  # falls through to embedding, no kg_nodes


def test_custom_min_runs_shortcut(model):
    # run_count=1, but caller relaxes min_runs to 1 → shortcut fires
    kg_history = {
        "python": {"text": "Python", "run_count": 1, "reuse_count": 1, "adapt_count": 0, "new_count": 0}
    }
    result = classify_keywords(
        ["Python"], [], model=model, kg_history=kg_history,
        min_runs_for_shortcut=1, stability_threshold=0.8,
    )
    assert result["Python"] == "REUSE"


def test_custom_stability_threshold(model):
    # reuse_rate=0.6 normally fails, but caller lowers threshold to 0.5
    kg_history = {
        "python": {"text": "Python", "run_count": 5, "reuse_count": 3, "adapt_count": 0, "new_count": 2}
    }
    result = classify_keywords(
        ["Python"], [], model=model, kg_history=kg_history,
        min_runs_for_shortcut=3, stability_threshold=0.5,
    )
    assert result["Python"] == "REUSE"


def test_shortcut_is_case_insensitive(model):
    # kg_history keyed on lowercase; keyword comes in mixed case
    kg_history = {
        "pytorch": {"text": "PyTorch", "run_count": 4, "reuse_count": 4, "adapt_count": 0, "new_count": 0}
    }
    result = classify_keywords(["PyTorch"], [], model=model, kg_history=kg_history)
    assert result["PyTorch"] == "REUSE"


# ── history blending ───────────────────────────────────────────────────────

def test_history_blending_nudges_score_toward_reuse(model):
    # "basket weaving" vs ML nodes would normally be NEW without history.
    # With heavy REUSE history the blended score should rise into ADAPT or REUSE.
    kg_nodes = ["Python", "machine learning", "AWS"]
    kg_history = {
        "basket weaving": {
            "text": "basket weaving",
            "run_count": 10, "reuse_count": 10, "adapt_count": 0, "new_count": 0,
        }
    }
    result_no_history = classify_keywords(["basket weaving"], kg_nodes, model=model)
    result_with_history = classify_keywords(["basket weaving"], kg_nodes, model=model, kg_history=kg_history)

    # History nudges the score upward; result should be >= no-history decision
    order = {"NEW": 0, "ADAPT": 1, "REUSE": 2}
    assert order[result_with_history["basket weaving"]] >= order[result_no_history["basket weaving"]]


def test_no_history_is_backward_compatible(model):
    # Without kg_history the function behaves exactly as before
    result = classify_keywords(["Python"], ["Python", "Java", "SQL"], model=model)
    assert result["Python"] == "REUSE"


def test_empty_kg_history_dict_is_backward_compatible(model):
    result = classify_keywords(["Python"], ["Python"], model=model, kg_history={})
    assert result["Python"] == "REUSE"


# ── multiple keywords: mix of shortcut and embedding ──────────────────────

def test_mixed_shortcut_and_embedding_keywords(model):
    kg_history = {
        "python": {"text": "Python", "run_count": 5, "reuse_count": 5, "adapt_count": 0, "new_count": 0},
    }
    # "Python" → shortcut (REUSE); "Kubernetes" → no history, runs embedding
    result = classify_keywords(
        ["Python", "Kubernetes"],
        ["Python", "container orchestration"],
        model=model,
        kg_history=kg_history,
    )
    assert result["Python"] == "REUSE"
    assert result["Kubernetes"] in ("REUSE", "ADAPT", "NEW")  # determined by embedding
