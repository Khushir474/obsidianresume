"""
Hybrid semantic similarity for keyword-to-KG matching and experience file ranking.

Base score formula: hybrid_score = 0.7 × cosine_sim + 0.3 × jaccard_sim
  - cosine_sim: SentenceTransformer embedding cosine similarity
  - jaccard_sim: token-level Jaccard on lowercased words

Warm-start (kg_history):
  Stable keywords (run_count ≥ min_runs AND reuse_rate ≥ stability_threshold) are
  classified REUSE without running embeddings — saving compute on repeat keywords.
  Keywords with any prior history blend their embedding score with a history signal:
    final_score = 0.85 × hybrid_score + 0.15 × history_score
  where history_score = weighted average of past REUSE/ADAPT/NEW outcomes.

Thresholds:
  ≥ 0.90 → REUSE  (pull existing bullet/metric from KG)
  ≥ 0.70 → ADAPT  (rephrase KG node to embed keyword)
  < 0.70 → NEW    (add keyword as new KG node, source from experience files)
"""
from __future__ import annotations

import pathlib
import re
from functools import lru_cache
from typing import Sequence

import numpy as np

MODEL_NAME = "all-MiniLM-L6-v2"

REUSE_THRESHOLD = 0.90
ADAPT_THRESHOLD = 0.70

# Warm-start constants
_HISTORY_BLEND_WEIGHT = 0.15          # fraction of final score from KG history
_HISTORY_DECISION_SCORES = {"REUSE": 1.0, "ADAPT": 0.75, "NEW": 0.3}
_DEFAULT_MIN_RUNS = 3                  # minimum runs before stable-REUSE shortcut fires
_DEFAULT_STABILITY_THRESHOLD = 0.80   # reuse_rate required for shortcut


@lru_cache(maxsize=1)
def _get_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MODEL_NAME)


def _jaccard(a: str, b: str) -> float:
    tokens_a = set(re.findall(r"\b[a-z0-9]+\b", a.lower()))
    tokens_b = set(re.findall(r"\b[a-z0-9]+\b", b.lower()))
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def _cosine(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    denom = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    if denom == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / denom)


def compute_hybrid_score(kw: str, node_label: str, model=None) -> float:
    """Compute 0.7 × cosine_sim + 0.3 × jaccard_sim between kw and node_label."""
    if model is None:
        model = _get_model()
    vecs = model.encode([kw, node_label])
    cos = _cosine(vecs[0], vecs[1])
    jac = _jaccard(kw, node_label)
    return round(0.7 * cos + 0.3 * jac, 4)


def _bucket(score: float) -> str:
    if score >= REUSE_THRESHOLD:
        return "REUSE"
    if score >= ADAPT_THRESHOLD:
        return "ADAPT"
    return "NEW"


def _compute_history_score(node_attrs: dict) -> float:
    """Convert a keyword node's historical decision distribution into a 0–1 score."""
    run_count = node_attrs.get("run_count", 0)
    if run_count == 0:
        return 0.0
    return sum(
        node_attrs.get(f"{dec.lower()}_count", 0) / run_count * weight
        for dec, weight in _HISTORY_DECISION_SCORES.items()
    )


def classify_keywords(
    keyword_list: Sequence[str],
    kg_nodes: Sequence[str],
    model=None,
    kg_history: dict[str, dict] | None = None,
    min_runs_for_shortcut: int = _DEFAULT_MIN_RUNS,
    stability_threshold: float = _DEFAULT_STABILITY_THRESHOLD,
) -> dict[str, str]:
    """For each keyword, classify as REUSE/ADAPT/NEW.

    kg_history: optional {keyword_text_lower: node_attrs} from KGStore.
      Stable keywords (run_count >= min_runs_for_shortcut and reuse_rate >=
      stability_threshold) are returned as REUSE immediately, skipping embeddings.
      All other keywords with any history blend their embedding score with a
      history signal weighted at _HISTORY_BLEND_WEIGHT.

    Returns {keyword: "REUSE"|"ADAPT"|"NEW"}.
    """
    if model is None:
        model = _get_model()

    result: dict[str, str] = {}
    needs_embedding: list[str] = []

    # ── warm-start pass ───────────────────────────────────────────────────
    if kg_history:
        for kw in keyword_list:
            node = kg_history.get(kw) or kg_history.get(kw.lower())
            if node is None:
                needs_embedding.append(kw)
                continue
            run_count = node.get("run_count", 0)
            reuse_rate = node.get("reuse_count", 0) / run_count if run_count else 0.0
            if run_count >= min_runs_for_shortcut and reuse_rate >= stability_threshold:
                result[kw] = "REUSE"
            else:
                needs_embedding.append(kw)
    else:
        needs_embedding = list(keyword_list)

    if not needs_embedding:
        return result

    # ── embedding pass for remaining keywords ─────────────────────────────
    if not kg_nodes:
        for kw in needs_embedding:
            result[kw] = "NEW"
        return result

    all_texts = needs_embedding + list(kg_nodes)
    embeddings = model.encode(all_texts)
    kw_vecs = embeddings[: len(needs_embedding)]
    node_vecs = embeddings[len(needs_embedding) :]

    for i, kw in enumerate(needs_embedding):
        best_score = 0.0
        for j, node_text in enumerate(kg_nodes):
            cos = _cosine(kw_vecs[i], node_vecs[j])
            jac = _jaccard(kw, node_text)
            score = 0.7 * cos + 0.3 * jac
            if score > best_score:
                best_score = score

        if kg_history:
            node = kg_history.get(kw) or kg_history.get(kw.lower())
            if node and node.get("run_count", 0) > 0:
                hist = _compute_history_score(node)
                best_score = (1 - _HISTORY_BLEND_WEIGHT) * best_score + _HISTORY_BLEND_WEIGHT * hist

        result[kw] = _bucket(best_score)

    return result


def rank_experience_files(
    jd_keywords: Sequence[str],
    experience_folder: str,
    top_n: int = 5,
    model=None,
) -> list[str]:
    """Return top_n experience file paths ranked by hybrid similarity to JD keywords.

    Falls back to scoring raw file text when KG nodes are unavailable.
    """
    if model is None:
        model = _get_model()

    folder = pathlib.Path(experience_folder)
    files = sorted(folder.glob("*.md"))
    if not files:
        return []

    jd_text = " ".join(jd_keywords)
    jd_vec = model.encode([jd_text])[0]

    scored = []
    for f in files:
        try:
            content = f.read_text(errors="replace")
        except OSError:
            continue
        file_vec = model.encode([content[:2000]])[0]  # cap at 2000 chars for speed
        cos = _cosine(jd_vec, file_vec)
        jac = _jaccard(jd_text, content[:2000])
        hybrid = 0.7 * cos + 0.3 * jac
        scored.append((hybrid, str(f)))

    scored.sort(reverse=True)
    return [path for _, path in scored[:top_n]]
