import json
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type


class CogneeMemoryToolInput(BaseModel):
    """Input schema for CogneeMemoryTool."""

    operation: str = Field(
        ...,
        description="Either 'add' to store a memory or 'search' to retrieve semantically relevant memories.",
    )
    content: str = Field(
        ...,
        description="For 'add': the text content to store. For 'search': the semantic query string.",
    )
    tags: str = Field(
        default="",
        description="Optional. Comma-separated tags for categorizing the memory.",
    )
    top_k: int = Field(
        default=5,
        description="For 'search' only — number of top results to return.",
    )


class CogneeMemoryTool(BaseTool):
    """Stub for the Cognee memory layer — deferred to next sprint.

    Returns empty results for 'search' and a no-op confirmation for 'add'
    so agents can proceed without blocking on memory unavailability.
    """

    name: str = "CogneeMemoryTool"
    description: str = (
        "Add memories to or search memories from the Cognee agentic memory layer. "
        "operation='add' stores a text memory; operation='search' retrieves semantically "
        "relevant past memories by query string."
    )
    args_schema: Type[BaseModel] = CogneeMemoryToolInput

    def _run(
        self,
        operation: str,
        content: str,
        tags: str = "",
        top_k: int = 5,
    ) -> str:
        if operation == "search":
            return json.dumps({
                "status": "ok",
                "query": content,
                "results_count": 0,
                "results": [],
                "note": "Cognee memory not yet active — no historical patterns available.",
            }, indent=2)
        elif operation == "add":
            return json.dumps({
                "status": "skipped",
                "note": "Cognee memory not yet active — memory not stored this run.",
            }, indent=2)
        else:
            return json.dumps({
                "status": "error",
                "message": f"Invalid operation '{operation}'. Must be 'add' or 'search'.",
            }, indent=2)
