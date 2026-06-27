"""
Python-layer output writers: intercept crew task outputs and persist them to disk.
These run after crew.kickoff() returns so agents don't need to reliably write files.
"""
import json
import logging
import pathlib
import re
import datetime

logger = logging.getLogger(__name__)


def _extract_json(text: str) -> dict | None:
    """Try to parse a JSON object from an agent's text output."""
    text = text.strip()
    # 1. Direct parse (agent returned pure JSON)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # 2. Fenced code block ```json ... ```
    match = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    # 3. raw_decode from first '{' — stops exactly at the end of the first valid JSON
    #    object, ignoring any trailing prose the agent appended.
    idx = text.find("{")
    if idx != -1:
        try:
            obj, _ = json.JSONDecoder().raw_decode(text, idx)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    return None


def _find_task_output(crew_output, task_name: str) -> str | None:
    """Return the raw output string for a named task from CrewOutput.tasks_output."""
    if crew_output is None:
        return None
    tasks_output = getattr(crew_output, "tasks_output", None) or []
    for task_out in tasks_output:
        name = getattr(task_out, "name", None) or getattr(task_out, "task_name", None) or ""
        if task_name in name:
            return getattr(task_out, "raw", None) or str(task_out)
    return None


def write_eval_log(run_id: str, crew_output, logs_folder: str) -> pathlib.Path | None:
    """Parse the evaluate_pipeline_output task result and write JSON to {logs_folder}/eval/{run_id}.json."""
    eval_dir = pathlib.Path(logs_folder) / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    out_path = eval_dir / f"{run_id}.json"

    raw = _find_task_output(crew_output, "evaluate_pipeline_output")
    if raw is None:
        logger.warning("write_eval_log: could not find evaluate_pipeline_output in crew output")
        return None

    parsed = _extract_json(raw)
    if parsed is not None:
        out_path.write_text(json.dumps(parsed, indent=2))
        logger.info("Eval log written: %s", out_path)
    else:
        fallback = out_path.with_suffix(".txt")
        fallback.write_text(raw)
        logger.warning("Eval log JSON parse failed — wrote raw text to %s", fallback)
        return fallback

    return out_path


def write_optimization_report(run_id: str, crew_output, logs_folder: str) -> pathlib.Path | None:
    """Write the optimization report to {logs_folder}/optimization_report_{run_id}.md.

    Acts as a safety net: if the agent already wrote the file, this is a no-op.
    Normalizes the path to avoid double-slash issues.
    """
    folder = pathlib.Path(logs_folder)
    out_path = (folder / f"optimization_report_{run_id}.md").resolve()

    if out_path.exists():
        logger.info("Optimization report already exists at %s — skipping write", out_path)
        return out_path

    raw = _find_task_output(crew_output, "log_run_and_generate_optimization_report")
    if raw is None:
        logger.warning("write_optimization_report: could not find optimization task in crew output")
        return None

    out_path.write_text(raw)
    logger.info("Optimization report written: %s", out_path)
    return out_path


def confirm_new_role(role_name: str, scaffold_path: str) -> bool:
    """Prompt the user to confirm, edit, or reject a new role scaffold before the pipeline continues.

    Returns True to continue, False to abort.
    """
    path = pathlib.Path(scaffold_path)
    if path.exists():
        print(f"\n--- New role scaffold: {scaffold_path} ---")
        print(path.read_text()[:1000])
        print("---")

    while True:
        answer = input(f"\nConfirm new role '{role_name}'? (y/edit/n): ").strip().lower()
        if answer == "y":
            return True
        if answer == "n":
            print("Aborting pipeline — new role scaffold not confirmed.")
            return False
        if answer == "edit":
            import subprocess
            editor = __import__("os").environ.get("EDITOR", "nano")
            subprocess.call([editor, scaffold_path])
        else:
            print("Please enter y, edit, or n.")


def _extract_company(jd_path: str) -> str:
    """Extract company name from JD filename, then JD content, then fall back to stem.

    Tries in order:
    1. Filename separators: ' @ ', ' at ' (case-insensitive), ' - ', '_at_', '__'
    2. JD file content: frontmatter `company:`, bold `**Company:**`, or first H1 after '@'
    3. Full filename stem
    """
    stem = pathlib.Path(jd_path).stem

    # 1. Filename separators
    for sep in (" @ ", "_at_", "__"):
        if sep in stem:
            return stem.split(sep)[-1].strip()
    # Case-insensitive " at " (e.g. "Data Scientist at Stripe")
    lower = stem.lower()
    for marker in (" at ", " - "):
        idx = lower.rfind(marker)
        if idx != -1:
            candidate = stem[idx + len(marker):].strip()
            # Reject if the candidate looks like a role word rather than a company
            if candidate and not candidate.lower().startswith(("data", "software", "product", "senior", "jr", "junior")):
                return candidate

    # 2. JD file content
    try:
        content = pathlib.Path(jd_path).read_text(errors="replace")[:1500]
        # frontmatter: company: XYZ
        m = re.search(r"^company:\s*(.+)$", content, re.MULTILINE | re.IGNORECASE)
        if m:
            return m.group(1).strip().strip("\"'")
        # bold field: **Company:** XYZ  or  **Company Name:** XYZ
        m = re.search(r"\*\*Company(?:\s+Name)?:\*\*\s*([^\n]+)", content, re.IGNORECASE)
        if m:
            return m.group(1).strip()
        # H1 with @: # Role @ Company
        m = re.search(r"^#\s+.+?\s+@\s+(.+)$", content, re.MULTILINE)
        if m:
            return m.group(1).strip()
    except OSError:
        pass

    return stem


def update_knowledge_graph(
    run_id: str,
    jd_path: str,
    crew_output,
    vault_path: str,
) -> pathlib.Path | None:
    """Parse crew output and persist run data into KGStore, then return the store path."""
    from obsidianresumeforge.knowledge_graph_store import KGStore
    from obsidianresumeforge.semantic_filter import classify_keywords

    store = KGStore(vault_path)

    # ── classify_role ──────────────────────────────────────────────────────
    role_raw = _find_task_output(crew_output, "classify_role")
    role_parsed = _extract_json(role_raw or "") or {}
    role = role_parsed.get("best_role", "unknown")

    # ── extract_and_score_keywords ─────────────────────────────────────────
    kw_raw = _find_task_output(crew_output, "extract_and_score_keywords")
    kw_parsed = _extract_json(kw_raw or "") or {}
    list_a = kw_parsed.get("list_a_hard_keywords", [])
    list_b = kw_parsed.get("list_b_soft_keywords", [])

    # Classify each keyword against current KG nodes (REUSE/ADAPT/NEW)
    # kg_history enables warm-start: stable keywords skip embedding entirely
    all_kw_texts = [k["keyword"] for k in list_a + list_b if k.get("keyword")]
    kg_node_texts = store.keyword_texts()
    kg_history = {n["text"].lower(): n for n in store.keyword_nodes() if "text" in n}
    decisions = classify_keywords(all_kw_texts, kg_node_texts, kg_history=kg_history) if all_kw_texts else {}

    keywords = []
    for entry in list_a:
        text = entry.get("keyword", "").strip()
        if text:
            keywords.append({
                "text": text,
                "priority": entry.get("score", "Medium"),
                "decision": decisions.get(text, "NEW"),
                "domain": "hard",
            })
    for entry in list_b:
        text = entry.get("keyword", "").strip()
        if text:
            keywords.append({
                "text": text,
                "priority": entry.get("score", "Medium"),
                "decision": decisions.get(text, "NEW"),
                "domain": "soft",
            })

    # ── write_tailored_latex_resume_and_export_pdf ─────────────────────────
    resume_raw = _find_task_output(crew_output, "write_tailored_latex_resume_and_export_pdf")
    sourcing_map: list[dict] = []
    if resume_raw:
        # sourcing map may be embedded as a JSON array anywhere in the output
        match = re.search(r"(\[\s*\{[\s\S]*\}\s*\])", resume_raw)
        if match:
            try:
                sourcing_map = json.loads(match.group(1))
                if not isinstance(sourcing_map, list):
                    sourcing_map = []
            except json.JSONDecodeError:
                sourcing_map = []

    # ── evaluate_pipeline_output ───────────────────────────────────────────
    eval_raw = _find_task_output(crew_output, "evaluate_pipeline_output")
    eval_parsed = _extract_json(eval_raw or "") or {}
    composite_score = float(eval_parsed.get("composite_score", 0.0))
    passed = bool(eval_parsed.get("passed", False))
    sa_score = None
    judge_scores = eval_parsed.get("judge_scores", {})
    if "SourceAttributionJudge" in judge_scores:
        sa_score = float(judge_scores["SourceAttributionJudge"].get("score", 0.0))

    # ── company from JD filename / content ────────────────────────────────
    company = _extract_company(jd_path)

    store.ingest_run(
        run_id=run_id,
        date=datetime.date.today().isoformat(),
        jd_path=jd_path,
        role=role,
        company=company,
        composite_score=composite_score,
        passed=passed,
        keywords=keywords,
        sourcing_map=sourcing_map,
        source_attribution_score=sa_score,
    )

    from obsidianresumeforge.bullet_promoter import auto_promote_bullets
    promoted = auto_promote_bullets(
        store=store,
        run_id=run_id,
        composite_score=composite_score,
        source_attribution_score=sa_score or 0.0,
    )
    if promoted:
        logger.info("%d bullet(s) promoted for future REUSE", len(promoted))

    path = store.save()
    logger.info("KGStore updated: %s", path)
    return path


def write_knowledge_graph_note(
    run_id: str,
    jd_path: str,
    role: str,
    composite_score: float,
    pdf_path: str,
    vault_path: str,
) -> pathlib.Path:
    """Write an Obsidian-compatible KnowledgeGraph note linking the run's artifacts."""
    kg_dir = pathlib.Path(vault_path) / "KnowledgeGraph"
    kg_dir.mkdir(parents=True, exist_ok=True)
    out_path = kg_dir / f"{run_id}.md"

    jd_stem = pathlib.Path(jd_path).stem
    pdf_stem = pathlib.Path(pdf_path).stem if pdf_path else "unknown"
    date_str = datetime.date.today().isoformat()

    content = (
        f"# Resume Run: {run_id}\n"
        f"- JD: [[{jd_stem}]]\n"
        f"- Role: [[{role}]]\n"
        f"- Score: {composite_score}\n"
        f"- Output: [[{pdf_stem}]]\n"
        f"- Date: {date_str}\n"
    )
    out_path.write_text(content)
    logger.info("KnowledgeGraph note written: %s", out_path)
    return out_path


def write_interview_prep_note(run_id: str, vault_path: str) -> pathlib.Path | None:
    """Write a lightweight interview prep stub for a completed run.

    Reads all data from the KGStore — no crew output needed.
    Output: {vault_path}/InterviewPrep/{run_id}.md

    The note lists Critical/High/Medium/Low keywords from the run so you have
    a starting point for building STAR stories. Add your own notes below the
    generated sections. A future `prep_interview --detailed` command will expand
    this into full prep content once Phase 3 promoted bullets are available.
    """
    from obsidianresumeforge.knowledge_graph_store import KGStore

    store = KGStore(vault_path)
    run_node = store.get_node(f"run:{run_id}")
    if run_node is None:
        logger.warning("write_interview_prep_note: run %s not found in KGStore", run_id)
        return None

    role = run_node.get("role", "unknown").replace("_", " ").title()
    company = run_node.get("company", "Unknown")
    date = run_node.get("date", "")
    score = run_node.get("composite_score", 0.0)
    passed = run_node.get("passed", False)
    jd_path = run_node.get("jd_path", "")
    jd_stem = pathlib.Path(jd_path).stem if jd_path else ""

    # Collect keywords grouped by priority from used_keyword edges
    priority_groups: dict[str, list[str]] = {
        "Critical": [], "High": [], "Medium": [], "Low": []
    }
    for edge in store.get_edges(src=f"run:{run_id}", rel="used_keyword"):
        kw_node = store.get_node(edge["dst"])
        if kw_node is None:
            continue
        priority = edge["attrs"].get("priority", "Medium")
        if priority not in priority_groups:
            priority_groups[priority] = []
        priority_groups[priority].append(kw_node.get("text", edge["dst"]))

    # Sort each group alphabetically (case-insensitive) for readability
    for group in priority_groups.values():
        group.sort(key=str.lower)

    score_indicator = "✓" if passed else "✗"
    lines = [
        f"# Interview Prep: {run_id}",
        "",
        f"**Role:** {role}  ",
        f"**Company:** {company}  ",
        f"**Date:** {date}  ",
        f"**Score:** {score:.2f} {score_indicator}  ",
    ]
    if jd_stem:
        lines.append(f"**JD:** [[{jd_stem}]]  ")
    lines += ["", "---", ""]

    for priority in ("Critical", "High", "Medium", "Low"):
        kws = priority_groups.get(priority, [])
        if kws:
            lines.append(f"## {priority} Keywords")
            lines += [f"- {kw}" for kw in kws]
            lines.append("")

    lines += [
        "---",
        "",
        "## Prep Notes",
        "",
        "> Add STAR stories, talking points, and questions here.",
        "",
    ]

    prep_dir = pathlib.Path(vault_path) / "InterviewPrep"
    prep_dir.mkdir(parents=True, exist_ok=True)
    out_path = prep_dir / f"{run_id}.md"
    out_path.write_text("\n".join(lines))
    logger.info("Interview prep note written: %s", out_path)
    return out_path


def write_kg_insights_note(vault_path: str) -> pathlib.Path:
    """Generate KnowledgeGraph/insights.md from KGStore analytics.

    Contains: resume anchors, score trend, role fit, interview priorities,
    skill demand heatmap, skill gap report, and promoted bullets.

    Overwrites on every call — always reflects the current KGStore state.
    """
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

    store = KGStore(vault_path)
    today = datetime.date.today().isoformat()

    run_count = len(store.run_nodes())
    kw_count = len(store.keyword_nodes())
    bullet_count = len(store.promoted_bullets())

    anchors = resume_anchors(store)
    trend = score_trend(store)
    fit = role_fit(store)
    priorities = interview_priorities(store, min_companies=2)
    demand = skill_demand(store)
    gaps = skill_gaps(store)
    bullets = promoted_bullets_summary(store)

    def _row(*cols: str) -> str:
        return "| " + " | ".join(str(c) for c in cols) + " |"

    def _sep(*widths: int) -> str:
        return "| " + " | ".join("-" * max(w, 3) for w in widths) + " |"

    lines: list[str] = [
        f"# KG Insights — {today}",
        "",
        f"> {run_count} run{'s' if run_count != 1 else ''} · "
        f"{kw_count} keywords · "
        f"{bullet_count} promoted bullet{'s' if bullet_count != 1 else ''} · "
        f"auto-updated after each run",
        "",
        "---",
        "",
    ]

    # ── Resume Anchors ────────────────────────────────────────────────────
    lines += [
        "## Resume Anchors",
        "",
        "> Keywords with REUSE rate ≥ 80% across ≥ 3 runs — always include, no re-evaluation.",
        "",
    ]
    if anchors:
        lines += [
            _row("Keyword", "Domain", "REUSE %", "Used in"),
            _sep(20, 6, 8, 8),
        ]
        for a in anchors:
            lines.append(_row(a["text"], a["domain"], f"{a['reuse_pct']}%", f"{a['total_uses']} runs"))
    else:
        lines.append("_Not enough data yet — needs ≥ 3 runs per keyword._")
    lines += ["", "---", ""]

    # ── Score Trend ────────────────────────────────────────────────────────
    lines += [
        "## Score Trend",
        "",
        _row("Run", "Date", "Role", "Company", "Score", ""),
        _sep(20, 10, 20, 15, 6, 2),
    ]
    for r in trend:
        indicator = "✓" if r["passed"] else "✗"
        lines.append(_row(r["run_id"], r["date"], r["role"], r["company"],
                          f"{r['composite_score']:.3f}", indicator))
    lines += ["", "---", ""]

    # ── Role Fit ───────────────────────────────────────────────────────────
    lines += [
        "## Role Fit",
        "",
        "> Average composite score per role type.",
        "",
        _row("Role", "Avg Score", "Best", "Runs", "Passed"),
        _sep(20, 10, 6, 5, 7),
    ]
    for f in fit:
        lines.append(_row(f["role"], f"{f['avg_score']:.3f}", f"{f['best_score']:.3f}",
                          f["run_count"], f"{f['passed_count']}/{f['run_count']}"))
    lines += ["", "---", ""]

    # ── Interview Priorities ───────────────────────────────────────────────
    lines += [
        "## Interview Prep Priorities",
        "",
        "> Keywords marked Critical across 2+ distinct companies — highest-priority prep topics.",
        "",
    ]
    if priorities:
        lines += [
            _row("Keyword", "Domain", "Companies", "# Critical"),
            _sep(20, 6, 35, 10),
        ]
        for p in priorities:
            companies_str = ", ".join(p["companies"])
            lines.append(_row(p["text"], p["domain"], companies_str, p["critical_count"]))
    else:
        lines.append("_Need runs against 2+ companies to compute priorities._")
    lines += ["", "---", ""]

    # ── Skill Demand Heatmap ──────────────────────────────────────────────
    lines += [
        "## Skill Demand Heatmap",
        "",
        "> Keywords that appeared as NEW — market asked for it, pipeline had no strong coverage.",
        "> Gap % = fraction of appearances as NEW. 100% = never covered.",
        "",
    ]
    if demand:
        lines += [
            _row("Keyword", "Domain", "NEW", "ADAPT", "REUSE", "Total", "Gap %", "Last"),
            _sep(20, 6, 4, 5, 5, 5, 6, 6),
        ]
        for d in demand[:30]:  # cap at 30 rows for readability
            lines.append(_row(
                d["text"], d["domain"],
                d["new_count"], d["adapt_count"], d["reuse_count"], d["total_uses"],
                f"{int(d['gap_rate'] * 100)}%", d["last_decision"],
            ))
        if len(demand) > 30:
            lines.append(f"_… {len(demand) - 30} more — see kg_store.json for full data_")
    else:
        lines.append("_No NEW keywords yet._")
    lines += ["", "---", ""]

    # ── Skill Gap Report ──────────────────────────────────────────────────
    lines += [
        "## Skill Gap Report",
        "",
        "> Skills the market asks for that you haven't yet covered well. Prioritise by column.",
        "",
        "### Hard Skills",
        "",
    ]
    if gaps["hard"]:
        lines += [
            _row("Keyword", "Top Priority", "NEW / Total", "Gap %"),
            _sep(25, 12, 12, 6),
        ]
        for g in gaps["hard"]:
            lines.append(_row(g["text"], g["top_priority"],
                              f"{g['new_count']}/{g['total_uses']}", f"{int(g['gap_rate'] * 100)}%"))
    else:
        lines.append("_No hard skill gaps._")

    lines += ["", "### Soft Skills", ""]
    if gaps["soft"]:
        lines += [
            _row("Keyword", "Top Priority", "NEW / Total", "Gap %"),
            _sep(25, 12, 12, 6),
        ]
        for g in gaps["soft"]:
            lines.append(_row(g["text"], g["top_priority"],
                              f"{g['new_count']}/{g['total_uses']}", f"{int(g['gap_rate'] * 100)}%"))
    else:
        lines.append("_No soft skill gaps._")
    lines += ["", "---", ""]

    # ── Promoted Bullets ───────────────────────────────────────────────────
    lines += [
        "## Promoted Bullets",
        "",
        "> Gold-standard adapted bullets promoted for REUSE in future runs.",
        "",
    ]
    if bullets:
        for b in bullets:
            preview = b["text"][:120] + ("…" if len(b["text"]) > 120 else "")
            lines.append(
                f"- **{b['source_file']}** (used {b['use_count']}×, "
                f"avg score {b['avg_source_score']:.2f}): _{preview}_"
            )
    else:
        lines.append("_No promoted bullets yet. Bullets are promoted when composite_score ≥ 0.75 and SA score ≥ 0.85._")
    lines.append("")

    kg_dir = pathlib.Path(vault_path) / "KnowledgeGraph"
    kg_dir.mkdir(parents=True, exist_ok=True)
    out_path = kg_dir / "insights.md"
    out_path.write_text("\n".join(lines))
    logger.info("KG insights note written: %s", out_path)
    return out_path
