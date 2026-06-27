"""
Cross-run bullet promotion: mark adapted bullets for REUSE in future runs.

A bullet is promoted when the run it came from passes two quality bars AND
at least one keyword it embeds was classified as NEW or ADAPT (meaning the
pipeline had to generate fresh content — worth locking in for next time).

Thresholds (overridable via kwargs):
  composite_threshold  0.75  — overall run quality floor
  sa_threshold         0.85  — SourceAttributionJudge floor (grounding check)

Promoted bullets appear gold in graph.html and are surfaced to the Resume
Writer agent in future runs (Phase 4 warm-start).
"""
from __future__ import annotations

import logging
from obsidianresumeforge.knowledge_graph_store import KGStore

logger = logging.getLogger(__name__)

_DEFAULT_COMPOSITE_THRESHOLD = 0.75
_DEFAULT_SA_THRESHOLD = 0.85


def auto_promote_bullets(
    store: KGStore,
    run_id: str,
    composite_score: float,
    source_attribution_score: float,
    composite_threshold: float = _DEFAULT_COMPOSITE_THRESHOLD,
    sa_threshold: float = _DEFAULT_SA_THRESHOLD,
) -> list[str]:
    """Promote qualifying bullets from run_id in-place on store (does not save).

    Returns list of bullet node IDs that were promoted this call.
    """
    if composite_score < composite_threshold:
        logger.debug(
            "auto_promote_bullets: skipping %s — composite %.2f < threshold %.2f",
            run_id, composite_score, composite_threshold,
        )
        return []
    if source_attribution_score < sa_threshold:
        logger.debug(
            "auto_promote_bullets: skipping %s — SA score %.2f < threshold %.2f",
            run_id, source_attribution_score, sa_threshold,
        )
        return []

    run_nid = f"run:{run_id}"

    # keyword node IDs whose decision for this run was NEW or ADAPT
    new_adapt_kw_nids: set[str] = {
        e["dst"]
        for e in store.get_edges(src=run_nid, rel="used_keyword")
        if e["attrs"].get("decision") in ("NEW", "ADAPT")
    }

    if not new_adapt_kw_nids:
        logger.debug("auto_promote_bullets: %s has no NEW/ADAPT keywords — nothing to promote", run_id)
        return []

    # bullet node IDs used in this run
    bullet_nids = [e["dst"] for e in store.get_edges(src=run_nid, rel="used_bullet")]

    promoted: list[str] = []
    for bullet_nid in bullet_nids:
        # keywords that back this specific bullet (scoped to this run)
        backing_kw_nids = {
            e["src"]
            for e in store.get_edges(dst=bullet_nid, rel="backed_by")
            if e["attrs"].get("run_id") == run_id
        }
        if backing_kw_nids & new_adapt_kw_nids:
            if store.promote_bullet(bullet_nid):
                promoted.append(bullet_nid)
                logger.info("Promoted bullet %s (run %s)", bullet_nid, run_id)

    if promoted:
        logger.info("auto_promote_bullets: %d bullet(s) promoted from %s", len(promoted), run_id)

    return promoted
