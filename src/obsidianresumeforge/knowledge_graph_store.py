"""
Persistent knowledge graph store for cross-run learning.

Stored as {vault_path}/KnowledgeGraph/kg_store.json.

Node ID conventions:
  run:{run_id}          e.g.  run:run_20260626_001
  role:{name}           e.g.  role:ai_ml_engineer
  company:{name_lower}  e.g.  company:stripe
  keyword:{text_lower}  e.g.  keyword:python
  bullet:{hash12}       e.g.  bullet:3f9a2c1b8d4e

Edge relations:
  for_role       run  → role
  at_company     run  → company
  used_keyword   run  → keyword   attrs: priority, decision, score_delta
  used_bullet    run  → bullet    attrs: source_attribution_score
  backed_by      keyword → bullet attrs: run_id
"""
from __future__ import annotations

import hashlib
import json
import logging
import pathlib
from typing import Any

logger = logging.getLogger(__name__)

_STORE_FILENAME = "kg_store.json"


class KGStore:
    def __init__(self, vault_path: str) -> None:
        self._path = pathlib.Path(vault_path) / "KnowledgeGraph" / _STORE_FILENAME
        self._data: dict[str, Any] = {"nodes": {}, "edges": []}
        if self._path.exists():
            try:
                self._data = json.loads(self._path.read_text())
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("KGStore: could not load %s: %s — starting fresh", self._path, exc)

    # ── persistence ────────────────────────────────────────────────────────

    def save(self) -> pathlib.Path:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._data, indent=2))
        logger.info("KGStore saved: %s", self._path)
        return self._path

    # ── low-level graph ops ────────────────────────────────────────────────

    def upsert_node(self, node_id: str, **attrs: Any) -> None:
        existing = self._data["nodes"].get(node_id, {})
        existing.update(attrs)
        self._data["nodes"][node_id] = existing

    def add_edge(self, src: str, dst: str, rel: str, **attrs: Any) -> None:
        self._data["edges"].append({"src": src, "dst": dst, "rel": rel, "attrs": attrs})

    def get_node(self, node_id: str) -> dict | None:
        return self._data["nodes"].get(node_id)

    def get_edges(
        self,
        *,
        src: str | None = None,
        dst: str | None = None,
        rel: str | None = None,
    ) -> list[dict]:
        result = []
        for e in self._data["edges"]:
            if src is not None and e["src"] != src:
                continue
            if dst is not None and e["dst"] != dst:
                continue
            if rel is not None and e["rel"] != rel:
                continue
            result.append(e)
        return result

    # ── high-level queries ─────────────────────────────────────────────────

    def keyword_nodes(self) -> list[dict]:
        return [n for n in self._data["nodes"].values() if n.get("type") == "keyword"]

    def keyword_texts(self) -> list[str]:
        return [n["text"] for n in self.keyword_nodes() if "text" in n]

    def run_nodes(self) -> list[dict]:
        return [n for n in self._data["nodes"].values() if n.get("type") == "run"]

    def promoted_bullets(self) -> list[dict]:
        return [
            n for n in self._data["nodes"].values()
            if n.get("type") == "bullet" and n.get("promoted", False)
        ]

    def keyword_decision_history(self, keyword_text: str) -> list[str]:
        """REUSE/ADAPT/NEW decisions for this keyword across all runs, oldest first."""
        node_id = f"keyword:{keyword_text.lower()}"
        edges = self.get_edges(dst=node_id, rel="used_keyword")
        return [e["attrs"].get("decision", "") for e in edges if e["attrs"].get("decision")]

    # ── run ingestion ──────────────────────────────────────────────────────

    def ingest_run(
        self,
        *,
        run_id: str,
        date: str,
        jd_path: str,
        role: str,
        company: str,
        composite_score: float,
        passed: bool,
        keywords: list[dict],
        sourcing_map: list[dict],
        source_attribution_score: float | None = None,
    ) -> None:
        """Add one run's structured data to the graph.

        keywords entries: {text, priority, decision, domain?, score_delta?}
        sourcing_map entries: {source_file, original_bullet, adapted_bullet,
                               keywords_embedded, reason}
        source_attribution_score: SourceAttributionJudge score for the whole run
        """
        run_nid = f"run:{run_id}"
        role_nid = f"role:{role}"
        company_nid = f"company:{company.lower()}"

        self.upsert_node(
            run_nid,
            type="run",
            run_id=run_id,
            date=date,
            jd_path=jd_path,
            role=role,
            company=company,
            composite_score=composite_score,
            passed=passed,
        )

        existing_role = self.get_node(role_nid) or {}
        self.upsert_node(
            role_nid,
            type="role",
            name=role,
            display=role.replace("_", " ").title(),
            run_count=existing_role.get("run_count", 0) + 1,
        )
        self.add_edge(run_nid, role_nid, "for_role")

        existing_co = self.get_node(company_nid) or {}
        self.upsert_node(
            company_nid,
            type="company",
            name=company,
            run_count=existing_co.get("run_count", 0) + 1,
        )
        self.add_edge(run_nid, company_nid, "at_company")

        for kw in keywords:
            text = kw.get("text", "").strip()
            if not text:
                continue
            kw_nid = f"keyword:{text.lower()}"
            existing_kw = self.get_node(kw_nid) or {}
            decision = kw.get("decision", "NEW")
            self.upsert_node(
                kw_nid,
                type="keyword",
                text=text,
                domain=kw.get("domain", "hard"),
                run_count=existing_kw.get("run_count", 0) + 1,
                reuse_count=existing_kw.get("reuse_count", 0) + (1 if decision == "REUSE" else 0),
                adapt_count=existing_kw.get("adapt_count", 0) + (1 if decision == "ADAPT" else 0),
                new_count=existing_kw.get("new_count", 0) + (1 if decision == "NEW" else 0),
                last_decision=decision,
            )
            self.add_edge(
                run_nid, kw_nid, "used_keyword",
                priority=kw.get("priority", "Medium"),
                decision=decision,
                score_delta=kw.get("score_delta"),
            )

        sa_score = source_attribution_score or 0.0
        for entry in sourcing_map:
            adapted = entry.get("adapted_bullet", "").strip()
            if not adapted:
                continue
            bullet_hash = hashlib.sha256(adapted.encode()).hexdigest()[:12]
            bullet_nid = f"bullet:{bullet_hash}"

            existing_b = self.get_node(bullet_nid) or {}
            prev_count = existing_b.get("use_count", 0)
            prev_avg = existing_b.get("avg_source_score", 0.0)
            new_count = prev_count + 1
            new_avg = round((prev_avg * prev_count + sa_score) / new_count, 4)

            self.upsert_node(
                bullet_nid,
                type="bullet",
                text=adapted,
                source_file=entry.get("source_file", ""),
                original_bullet=entry.get("original_bullet", ""),
                origin_run=existing_b.get("origin_run", run_id),
                use_count=new_count,
                avg_source_score=new_avg,
                promoted=existing_b.get("promoted", False),
            )
            self.add_edge(run_nid, bullet_nid, "used_bullet", source_attribution_score=sa_score)

            for kw_text in entry.get("keywords_embedded", []):
                if kw_text:
                    self.add_edge(
                        f"keyword:{kw_text.lower()}", bullet_nid, "backed_by", run_id=run_id
                    )

    # ── bullet promotion ───────────────────────────────────────────────────

    def promote_bullet(self, bullet_nid: str) -> bool:
        """Mark a bullet node as promoted for REUSE consideration in future runs.

        Returns False if the node does not exist or is already promoted.
        """
        node = self.get_node(bullet_nid)
        if node is None or node.get("promoted", False):
            return False
        node["promoted"] = True
        self._data["nodes"][bullet_nid] = node
        return True
