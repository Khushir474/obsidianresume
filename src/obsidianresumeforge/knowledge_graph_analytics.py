"""
Cross-run analytics derived from KGStore data.

All functions are pure (read-only) and return structured dicts.
output_writers.write_kg_insights_note() renders these as Obsidian markdown.
"""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from obsidianresumeforge.knowledge_graph_store import KGStore

_PRIORITY_RANK = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}


def resume_anchors(
    store: "KGStore",
    min_runs: int = 3,
    min_reuse_rate: float = 0.80,
) -> list[dict]:
    """Keywords stable enough to always include without re-evaluation.

    A keyword qualifies when it has appeared in >= min_runs runs AND was
    classified REUSE at least min_reuse_rate fraction of those times.

    Returns list sorted by reuse_pct desc:
      {text, domain, reuse_pct, reuse_count, total_uses}
    """
    results = []
    for node in store.keyword_nodes():
        total = node.get("run_count", 0)
        reuse = node.get("reuse_count", 0)
        if total < min_runs:
            continue
        rate = reuse / total
        if rate < min_reuse_rate:
            continue
        results.append({
            "text": node.get("text", ""),
            "domain": node.get("domain", "hard"),
            "reuse_pct": round(rate * 100),
            "reuse_count": reuse,
            "total_uses": total,
        })
    return sorted(results, key=lambda x: (-x["reuse_pct"], -x["total_uses"]))


def score_trend(store: "KGStore") -> list[dict]:
    """All run nodes sorted by date ascending.

    Returns list of dicts:
      {run_id, date, role, company, composite_score, passed}
    """
    runs = []
    for node in store.run_nodes():
        runs.append({
            "run_id": node.get("run_id", ""),
            "date": node.get("date", ""),
            "role": node.get("role", "").replace("_", " ").title(),
            "company": node.get("company", ""),
            "composite_score": round(node.get("composite_score", 0.0), 3),
            "passed": node.get("passed", False),
        })
    return sorted(runs, key=lambda x: x["date"])


def role_fit(store: "KGStore") -> list[dict]:
    """Average composite score per role, sorted by avg_score desc.

    Returns list of dicts:
      {role, avg_score, run_count, best_score, passed_count}
    """
    groups: dict[str, list[float]] = {}
    passed_counts: dict[str, int] = {}
    for node in store.run_nodes():
        role = node.get("role", "unknown")
        score = node.get("composite_score", 0.0)
        groups.setdefault(role, []).append(score)
        if node.get("passed", False):
            passed_counts[role] = passed_counts.get(role, 0) + 1

    results = []
    for role, scores in groups.items():
        results.append({
            "role": role.replace("_", " ").title(),
            "avg_score": round(sum(scores) / len(scores), 3),
            "run_count": len(scores),
            "best_score": round(max(scores), 3),
            "passed_count": passed_counts.get(role, 0),
        })
    return sorted(results, key=lambda x: -x["avg_score"])


def interview_priorities(store: "KGStore", min_companies: int = 2) -> list[dict]:
    """Keywords marked Critical in runs across >= min_companies distinct companies.

    Returns list sorted by company_count desc then critical_count desc:
      {text, domain, companies, critical_count}
    """
    run_to_company: dict[str, str] = {}
    for node in store.run_nodes():
        run_to_company[f"run:{node['run_id']}"] = node.get("company", "")

    kw_companies: dict[str, set[str]] = {}
    kw_critical_count: dict[str, int] = {}

    for edge in store.get_edges(rel="used_keyword"):
        if edge["attrs"].get("priority") != "Critical":
            continue
        company = run_to_company.get(edge["src"], "")
        if not company:
            continue
        kw_nid = edge["dst"]
        kw_companies.setdefault(kw_nid, set()).add(company)
        kw_critical_count[kw_nid] = kw_critical_count.get(kw_nid, 0) + 1

    results = []
    for kw_nid, companies in kw_companies.items():
        if len(companies) < min_companies:
            continue
        node = store.get_node(kw_nid)
        if node is None:
            continue
        results.append({
            "text": node.get("text", kw_nid),
            "domain": node.get("domain", "hard"),
            "companies": sorted(companies),
            "critical_count": kw_critical_count.get(kw_nid, 0),
        })
    return sorted(results, key=lambda x: (-len(x["companies"]), -x["critical_count"]))


def skill_demand(store: "KGStore") -> list[dict]:
    """All keywords that appeared as NEW at least once, ranked by gap_rate.

    gap_rate = new_count / total_uses.
    1.0 = you've never successfully covered this skill.
    0.0 = always covered (these should be anchors).

    Returns list sorted by gap_rate desc, then new_count desc:
      {text, domain, new_count, adapt_count, reuse_count, total_uses, gap_rate, last_decision}
    """
    results = []
    for node in store.keyword_nodes():
        new_c = node.get("new_count", 0)
        if new_c == 0:
            continue
        total = node.get("run_count", 0)
        if total == 0:
            continue
        results.append({
            "text": node.get("text", ""),
            "domain": node.get("domain", "hard"),
            "new_count": new_c,
            "adapt_count": node.get("adapt_count", 0),
            "reuse_count": node.get("reuse_count", 0),
            "total_uses": total,
            "gap_rate": round(new_c / total, 2),
            "last_decision": node.get("last_decision", ""),
        })
    return sorted(results, key=lambda x: (-x["gap_rate"], -x["new_count"]))


def skill_gaps(store: "KGStore", min_new_count: int = 1) -> dict[str, list[dict]]:
    """Skill gaps grouped by domain (hard / soft), sorted by top priority seen.

    A keyword is a gap when new_count >= min_new_count (i.e. the market asked
    for it and you had no strong coverage at least once).

    Returns:
      {
        "hard": [{text, new_count, total_uses, gap_rate, top_priority}, ...],
        "soft": [{text, new_count, total_uses, gap_rate, top_priority}, ...],
      }
    """
    kw_top_priority: dict[str, str] = {}
    for edge in store.get_edges(rel="used_keyword"):
        kw_nid = edge["dst"]
        priority = edge["attrs"].get("priority", "Medium")
        current = kw_top_priority.get(kw_nid)
        if current is None or _PRIORITY_RANK.get(priority, 3) < _PRIORITY_RANK.get(current, 3):
            kw_top_priority[kw_nid] = priority

    hard: list[dict] = []
    soft: list[dict] = []

    for node in store.keyword_nodes():
        new_c = node.get("new_count", 0)
        if new_c < min_new_count:
            continue
        total = node.get("run_count", 0)
        if total == 0:
            continue
        kw_nid = f"keyword:{node['text'].lower()}"
        entry = {
            "text": node.get("text", ""),
            "new_count": new_c,
            "total_uses": total,
            "gap_rate": round(new_c / total, 2),
            "top_priority": kw_top_priority.get(kw_nid, "Medium"),
        }
        if node.get("domain", "hard") == "soft":
            soft.append(entry)
        else:
            hard.append(entry)

    def _sort(x: dict) -> tuple:
        return (_PRIORITY_RANK.get(x["top_priority"], 3), -x["new_count"], -x["gap_rate"])

    return {"hard": sorted(hard, key=_sort), "soft": sorted(soft, key=_sort)}


def promoted_bullets_summary(store: "KGStore") -> list[dict]:
    """Promoted bullets sorted by use_count desc.

    Returns list of dicts:
      {text, source_file, use_count, avg_source_score, origin_run}
    """
    results = []
    for node in store.promoted_bullets():
        results.append({
            "text": node.get("text", ""),
            "source_file": node.get("source_file", ""),
            "use_count": node.get("use_count", 1),
            "avg_source_score": round(node.get("avg_source_score", 0.0), 3),
            "origin_run": node.get("origin_run", ""),
        })
    return sorted(results, key=lambda x: -x["use_count"])
