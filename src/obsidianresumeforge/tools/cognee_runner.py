#!/usr/bin/env python3
"""
Cognee subprocess runner for obsidianresumeforge.

Reads a JSON request from stdin, executes a cognee operation,
and writes a JSON response to stdout. Run via the project's
dedicated .cognee-venv Python interpreter.

Supported operations:
  add    — remember(text) — store text in long-term memory
  search — recall(query)  — retrieve relevant memories
"""
import asyncio
import json
import os
import sys


def _configure() -> None:
    openrouter_key = os.environ.get("OPENROUTER_API_KEY", "")
    if openrouter_key:
        os.environ.setdefault("LLM_PROVIDER", "custom")
        os.environ.setdefault("LLM_ENDPOINT", "https://openrouter.ai/api/v1")
        os.environ.setdefault("LLM_API_KEY", openrouter_key)
        os.environ.setdefault("LLM_MODEL", "google/gemini-2.0-flash-001")
    os.environ.setdefault("EMBEDDING_PROVIDER", "fastembed")
    os.environ.setdefault("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    os.environ.setdefault("COGNEE_SKIP_CONNECTION_TEST", "true")


async def _remember(text: str) -> dict:
    import cognee
    await cognee.remember(text)
    return {"status": "added", "content_preview": text[:120]}


async def _recall(query: str, top_k: int) -> dict:
    import cognee
    from cognee import SearchType
    results = await cognee.recall(
        query_text=query,
        query_type=SearchType.GRAPH_COMPLETION,
        top_k=top_k,
    )

    normalized = []
    for r in results:
        if hasattr(r, "model_dump"):
            normalized.append(r.model_dump())
        elif isinstance(r, dict):
            normalized.append(r)
        else:
            normalized.append({"text": str(r)})

    return {"status": "ok", "query": query, "results_count": len(normalized), "results": normalized}


async def main(request: dict) -> dict:
    op = request.get("op")
    if op == "add":
        return await _remember(request["content"])
    elif op == "search":
        return await _recall(request["query"], request.get("top_k", 5))
    else:
        return {"status": "error", "message": f"Unknown op: {op!r}"}


if __name__ == "__main__":
    try:
        _configure()
        raw = sys.stdin.read()
        request = json.loads(raw)
        result = asyncio.run(main(request))
        print(json.dumps(result, default=str))
    except Exception as exc:
        print(json.dumps({"status": "error", "message": str(exc)}))
        sys.exit(1)
