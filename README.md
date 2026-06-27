# ObsidianResumeForge

A CrewAI pipeline that tailors a resume to a job description, compiles it to PDF, evaluates it with 4 programmatic judges, and writes structured output back to an Obsidian vault — accumulating cross-run learning in a persistent knowledge graph.

---

## What it does

1. **Classifies** the incoming JD against known role profiles (`knowledge/roles/`)
2. **Extracts** hard keywords (tools, frameworks) and soft keywords (traits, competencies) with ATS priority scores
3. **Writes** a tailored LaTeX resume by matching experience bullets to Critical/High keywords, then compiles to PDF via `pdflatex`
4. **Evaluates** the output with 4 code-based judges (no gold answer needed):
   - `ATSKeywordHitRateJudge` — Critical+High keyword coverage (weight 35%)
   - `MetricDensityJudge` — fraction of bullets with numeric metrics (weight 25%)
   - `SourceAttributionJudge` — anti-hallucination, bullets traceable to experience files (weight 20%)
   - `FormatComplianceJudge` — pdflatex exit code + required section headers (weight 20%)
5. **Retries** up to `EVAL_MAX_RETRIES` times if the composite score is below threshold, injecting judge feedback into the next attempt
6. **Learns** across runs via a persistent knowledge graph — promotes high-quality adapted bullets for reuse, tracks keyword demand, and generates an insights report with resume anchors, skill gaps, role fit scores, and interview prep priorities

---

## Architecture

```
main.py  ──► run_with_retry()
               │
               ▼
         ObsidianresumeforgeCrew (5 agents, sequential)
           classify_role
           extract_and_score_keywords
           write_tailored_latex_resume_and_export_pdf
           evaluate_pipeline_output
           log_run_and_generate_optimization_report
               │
               ▼
         output_writers.py  ──► eval/{run_id}.json
                            ──► optimization_report_{run_id}.md
                            ──► KnowledgeGraph/{run_id}.md
                            ──► KGStore.ingest_run()
                                  │
                                  ▼
                            bullet_promoter.auto_promote_bullets()
                                  │
                                  ▼
                            KnowledgeGraph/kg_store.json
               │
               ├──► knowledge_graph_viz.py   ──► KnowledgeGraph/graph.html
               └──► knowledge_graph_analytics.py ──► KnowledgeGraph/insights.md
```

**Key source files:**

| File | Purpose |
|------|---------|
| `src/obsidianresumeforge/crew.py` | Agent + task definitions |
| `src/obsidianresumeforge/main.py` | Entry points: `run`, `run_with_trigger`, `kg_report`, `prep_interview`, `train`, `replay`, `test` |
| `src/obsidianresumeforge/tools/custom_tool.py` | `CachedFileReadTool`, `CachedFileWriterTool`, `LatexToPdfTool` |
| `src/obsidianresumeforge/tools/judgeval_local_evaluator_runner.py` | 4-judge evaluation tool |
| `src/obsidianresumeforge/run_id.py` | Auto-generates `run_YYYYMMDD_NNN` run IDs |
| `src/obsidianresumeforge/watcher.py` | Watchdog-based JD file watcher |
| `src/obsidianresumeforge/retry_orchestrator.py` | Retry loop wrapping `crew.kickoff()` |
| `src/obsidianresumeforge/cognee_lifecycle.py` | Cognee health check + optional auto-start |
| `src/obsidianresumeforge/knowledge_graph_store.py` | Persistent cross-run KG: nodes (run/role/company/keyword/bullet), edges, bullet promotion |
| `src/obsidianresumeforge/knowledge_graph_viz.py` | vis.js interactive HTML visualization, regenerated after each run |
| `src/obsidianresumeforge/knowledge_graph_analytics.py` | Cross-run analytics: resume anchors, score trend, role fit, interview priorities, skill gap report |
| `src/obsidianresumeforge/bullet_promoter.py` | Auto-promotes high-quality adapted bullets for REUSE in future runs |
| `src/obsidianresumeforge/semantic_filter.py` | Hybrid cosine+Jaccard matcher with KG warm-start: stable-REUSE shortcut + history blending |
| `src/obsidianresumeforge/output_writers.py` | All post-run writers: eval log, optimization report, KG note, KGStore ingestion, insights note, interview prep stub |

---

## Setup

Requires Python 3.10–3.13 and [uv](https://docs.astral.sh/uv/).

```bash
pip install uv
uv sync
```

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

Required env vars:

```
OPENAI_API_KEY=sk-...
OPENROUTER_API_KEY=sk-or-v1-...

JD_FILE_PATH=/path/to/JobSearch/JDs/YourRole.md
VAULT_PATH=/path/to/JobSearch/
LOGS_FOLDER=/path/to/JobSearch/

ROLE_INSTRUCTIONS_FOLDER=/path/to/repo/knowledge/roles/
EXPERIENCE_FILES_FOLDER=/path/to/repo/knowledge/experience/
LATEX_TEMPLATE_PATH=/path/to/repo/knowledge/template.tex

CREWAI_TOOLS_SAFE_DIRS=/path/to/JobSearch:/path/to/repo/knowledge
CREWAI_TOOLS_ALLOW_UNSAFE_PATHS=true
```

Optional:

```
EVAL_MAX_RETRIES=1      # retry limit if composite score < threshold
```

---

## Usage

### Run once for a specific JD

```bash
JD_FILE_PATH="/path/to/JobSearch/JDs/AI Engineer.md" uv run run_crew
```

After the crew finishes, the following are written automatically:

| Output | Path |
|--------|------|
| PDF resume | `{VAULT_PATH}/Resumes/PDF/Khushi_Ranganatha_Resume_<Company>_<Role>_<Date>.pdf` |
| Eval log | `{LOGS_FOLDER}/eval/{run_id}.json` |
| Optimization report | `{LOGS_FOLDER}/optimization_report_{run_id}.md` |
| KnowledgeGraph run note | `{VAULT_PATH}/KnowledgeGraph/{run_id}.md` |
| KnowledgeGraph store | `{VAULT_PATH}/KnowledgeGraph/kg_store.json` (persists across runs) |
| KnowledgeGraph visualization | `{VAULT_PATH}/KnowledgeGraph/graph.html` (interactive, regenerated each run) |
| KG insights report | `{VAULT_PATH}/KnowledgeGraph/insights.md` (regenerated each run) |

### Watch the JDs folder (Obsidian Web Clipper trigger)

```bash
uv run run_with_trigger
```

Monitors `{VAULT_PATH}/JDs/` for new `.md` files. When Obsidian Web Clipper drops a JD, the crew kicks off automatically after a 2-second debounce.

### Generate the KG insights report

Regenerate `insights.md` on demand without running the crew:

```bash
uv run kg_report
```

The report is also auto-written after every `run_crew`. It contains:

| Section | What it tells you |
|---------|------------------|
| **Resume Anchors** | Keywords with REUSE rate ≥ 80% across ≥ 3 runs — always include, no re-evaluation needed |
| **Score Trend** | All runs in date order with composite scores and pass/fail |
| **Role Fit** | Average and best score per role type — where you naturally perform best |
| **Interview Prep Priorities** | Keywords marked Critical across 2+ distinct companies — highest-priority prep topics |
| **Skill Demand Heatmap** | Keywords that appeared as NEW, ranked by gap rate (100% = market asks for it, never covered) |
| **Skill Gap Report** | Hard and soft skills split by domain, sorted by top priority seen — your learning roadmap |
| **Promoted Bullets** | Gold-standard adapted bullets ready for REUSE in future runs |

No API calls — reads only from `kg_store.json`.

### Generate an interview prep stub

After a run completes, generate a lightweight per-run prep note:

```bash
uv run prep_interview --run-id run_20260627_001
```

Output: `{VAULT_PATH}/InterviewPrep/run_20260627_001.md`

The note contains:
- Run metadata (role, company, date, score)
- Keywords grouped by priority (Critical / High / Medium / Low)
- A `## Prep Notes` section for your STAR stories and talking points

No API calls — reads only from `kg_store.json`. Add your own notes below the generated sections.

### Other commands

```bash
uv run train 5 training_output.pkl   # train for N iterations
uv run replay <task_id>              # replay from a specific task
uv run test 3 gpt-4o                 # test run with evaluation
```

---

## Tests

```bash
uv run pytest tests/ -v
```

191 tests covering all infrastructure modules. Tests run without API calls — the judge tool, file writers, retry orchestrator, watcher, semantic filter, KG store, bullet promoter, interview prep writer, and analytics module are all unit-tested with mocks or isolated fixtures.

---

## Knowledge base

**Role instruction files** (`knowledge/roles/`): one `.md` file per role. The filename stem is the role name. Each file contains keyword extraction rules (STEP 1) and resume writing rules (STEP 2). New roles are scaffolded automatically when the JD confidence score is below 70%.

**Experience files** (`knowledge/experience/`): one `.md` file per company/role. Format:

```
# Job Title | Company Name
**Duration:** YYYY–YYYY

## Consolidated Bullets
- ...

## Metrics Reference
| Metric | Value |
```

**LaTeX template** (`knowledge/template.tex`): Garamond font, 0.5in margins. Populated by the Resume Writer agent. Every run starts from this base template — previously generated PDFs are never read back.

---

## Semantic keyword strategy

After each run, `semantic_filter.classify_keywords()` scores every keyword from List A and List B against the KnowledgeGraph using a hybrid score:

```
hybrid_score = 0.7 × cosine_sim + 0.3 × jaccard_sim
```

| Score | Decision | Action |
|-------|----------|--------|
| ≥ 0.90 | REUSE | Pull existing bullet/metric directly from KG |
| 0.70–0.89 | ADAPT | Use KG node as base, rephrase to embed keyword |
| < 0.70 | NEW | Add keyword as new node, source from experience files |

**KG warm-start:** `classify_keywords` accepts a `kg_history` dict built from the KGStore and applies two optimisations before running any embeddings:

1. **Stable-REUSE shortcut** — if a keyword has appeared in ≥ 3 prior runs with a REUSE rate ≥ 80%, it is classified `REUSE` immediately and skipped from the embedding loop. This saves compute on anchors like "Python" or "SQL" that consistently match.

2. **History blending** — for keywords with prior history below the shortcut threshold, the embedding score is blended with a history signal:
   ```
   final_score = 0.85 × hybrid_score + 0.15 × history_score
   ```
   where `history_score` is the weighted average of past decisions (REUSE=1.0, ADAPT=0.75, NEW=0.3). This nudges borderline keywords toward their historical treatment.

Both thresholds are configurable per call (`min_runs_for_shortcut`, `stability_threshold`).

---

## Knowledge graph

After each run, three files are written to `{VAULT_PATH}/KnowledgeGraph/`:

### `kg_store.json` — persistent graph store

Accumulates data across all runs. Five node types:

| Node | Key | Notable fields |
|------|-----|----------------|
| `run` | `run:{run_id}` | date, composite_score, passed, role, company |
| `role` | `role:{name}` | run_count |
| `company` | `company:{name}` | run_count |
| `keyword` | `keyword:{text}` | domain (hard/soft), reuse_count, adapt_count, new_count, last_decision |
| `bullet` | `bullet:{hash12}` | text, source_file, use_count, avg_source_score, promoted |

Five edge relations: `for_role`, `at_company`, `used_keyword` (attrs: priority, decision), `used_bullet`, `backed_by`.

**Company extraction** resolves from: JD filename separators (` @ `, ` at `, ` - `) → JD file frontmatter (`company: XYZ`) → bold field (`**Company:** XYZ`) → H1 pattern (`# Role @ Company`) → full filename stem.

**Bullet promotion** runs automatically after every successful run. A bullet is promoted when:

1. `composite_score ≥ 0.75` — the run passed a quality floor
2. `SourceAttributionJudge score ≥ 0.85` — the bullet is grounded in real experience
3. At least one keyword the bullet embeds was classified as `NEW` or `ADAPT`

Promoted bullets are the foundation for warm-start: the Resume Writer is seeded with promoted bullets for new runs with similar keywords, compounding quality across runs rather than starting cold each time.

### `graph.html` — interactive vis.js visualization

Self-contained HTML (vis.js from CDN). Open in any browser or Obsidian's built-in browser. Dark `#0d1117` theme. Nodes are pre-positioned at load time — no physics lag.

| Feature | How to use |
|---------|-----------|
| Hover detail panel | Hover any node — right panel shows type-specific fields + degree centrality and PageRank bars |
| Focus / neighbourhood | Click a node to highlight its neighbours; click again to clear |
| Search | Type in the search bar — matching nodes highlight and camera zooms to fit |
| Node type filters | Left sidebar checkboxes hide/show runs, roles, companies, keywords, bullets |
| Decision filters | Filter keywords by REUSE / ADAPT / NEW |
| BFS pathfinding | Click two nodes (or type IDs in the bottom bar) then **Find** — path highlighted in blue |
| Physics toggle | Enable for interactive re-layout; starts OFF for performance |
| Reset View | Fits all visible nodes back into view |

Node colours: run (red→green by score), role (teal), company (purple), keyword REUSE (green), keyword ADAPT (yellow), keyword NEW (red), promoted bullet (gold), bullet (gray).

**Node size** = `12 + degree_centrality × 28 + pagerank × 400` (clamped 10–40px). High-frequency anchors like Python/SQL grow large over multiple runs.

**Edge styles:** solid (REUSE / structural), dashed `[8,4]` (ADAPT), dotted `[3,3]` (NEW). Edge width reflects keyword priority (Critical = 3px).

### `insights.md` — cross-run analytics report

See [Generate the KG insights report](#generate-the-kg-insights-report) above. Overwritten each run.

---

## Eval judge scores: what 0.0 means

If all four judges score 0.0, the most likely cause is the agent calling `JudgevalLocalEvaluatorRunner` without passing the required arguments. Check `/tmp/judge_args_<run_id>.log` to see exactly what was received. The fix is usually a context-passing issue in the `evaluate_pipeline_output` task — ensure both `extract_and_score_keywords` and `write_tailored_latex_resume_and_export_pdf` are in the task's context list.
