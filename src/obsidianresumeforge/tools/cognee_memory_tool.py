import json
import requests
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
        description="Optional. Comma-separated tags for categorizing the memory (e.g. 'eval,phase4,passed'). Used with 'add' only.",
    )
    top_k: int = Field(
        default=5,
        description="For 'search' only — number of top results to return.",
    )
    base_url: str = Field(
        default="http://localhost:8000",
        description="Cognee server base URL. Defaults to 'http://localhost:8000'.",
    )


class CogneeMemoryTool(BaseTool):
    """Tool for interfacing with a local Cognee agentic memory layer via REST API.

    Supports two operations:
    - 'add': stores a text memory with optional metadata tags.
    - 'search': retrieves semantically relevant past memories by query string.
    """

    name: str = "CogneeMemoryTool"
    description: str = (
        "Add memories to or search memories from a Cognee agentic memory layer running locally. "
        "Supports two operations: 'add' stores a text memory with optional metadata tags, "
        "'search' retrieves semantically relevant past memories by query string."
    )
    args_schema: Type[BaseModel] = CogneeMemoryToolInput

    def _run(
        self,
        operation: str,
        content: str,
        tags: str = "",
        top_k: int = 5,
        base_url: str = "http://localhost:8000",
    ) -> str:
        """Execute the CogneeMemoryTool logic for 'add' or 'search' operations."""
        try:
            if operation == "add":
                return self._add_memory(content=content, tags=tags, base_url=base_url)
            elif operation == "search":
                return self._search_memory(content=content, top_k=top_k, base_url=base_url)
            else:
                return json.dumps(
                    {
                        "status": "error",
                        "message": f"Invalid operation '{operation}'. Must be 'add' or 'search'.",
                    },
                    indent=2,
                )
        except Exception as e:
            return json.dumps(
                {
                    "status": "error",
                    "message": f"Unexpected error: {str(e)}",
                },
                indent=2,
            )

    def _add_memory(self, content: str, tags: str, base_url: str) -> str:
        """Store a memory in Cognee via POST /api/v1/add."""
        # Parse comma-separated tags: split, strip whitespace, filter empty strings
        parsed_tags = [t.strip() for t in tags.split(",") if t.strip()] if tags else []

        payload = {
            "content": content,
            "tags": parsed_tags,
        }

        try:
            response = requests.post(
                f"{base_url}/api/v1/add",
                json=payload,
                timeout=30,
            )

            if response.status_code in (200, 201):
                return json.dumps(
                    {
                        "status": "added",
                        "message": "Memory stored successfully",
                        "content_preview": content[:100],
                    },
                    indent=2,
                )
            else:
                return json.dumps(
                    {
                        "status": "error",
                        "http_status": response.status_code,
                        "message": response.text[:300],
                    },
                    indent=2,
                )

        except requests.exceptions.ConnectionError:
            return json.dumps(
                {
                    "status": "error",
                    "message": (
                        f"Cognee server not reachable at {base_url}. "
                        "Ensure Cognee is running: pip install cognee && cognee serve"
                    ),
                },
                indent=2,
            )

    def _search_memory(self, content: str, top_k: int, base_url: str) -> str:
        """Search memories in Cognee via POST /api/v1/search."""
        payload = {
            "query": content,
            "top_k": top_k,
        }

        try:
            response = requests.post(
                f"{base_url}/api/v1/search",
                json=payload,
                timeout=30,
            )

            if response.status_code == 200:
                try:
                    raw_results = response.json()

                    # Normalize each result: extract text/content, score, tags
                    normalized_results = []
                    for item in raw_results:
                        result = {}

                        # Extract text content — prefer 'text', fallback to 'content'
                        if "text" in item:
                            result["text"] = item["text"]
                        elif "content" in item:
                            result["text"] = item["content"]

                        if "score" in item:
                            result["score"] = item["score"]

                        if "tags" in item:
                            result["tags"] = item["tags"]

                        normalized_results.append(result)

                    return json.dumps(
                        {
                            "status": "ok",
                            "query": content,
                            "results_count": len(normalized_results),
                            "results": normalized_results,
                        },
                        indent=2,
                    )

                except (ValueError, TypeError) as parse_err:
                    return json.dumps(
                        {
                            "status": "error",
                            "http_status": response.status_code,
                            "message": f"Failed to parse response JSON: {str(parse_err)}",
                        },
                        indent=2,
                    )
            else:
                return json.dumps(
                    {
                        "status": "error",
                        "http_status": response.status_code,
                        "message": response.text[:300],
                    },
                    indent=2,
                )

        except requests.exceptions.ConnectionError:
            return json.dumps(
                {
                    "status": "error",
                    "message": (
                        f"Cognee server not reachable at {base_url}. "
                        "Ensure Cognee is running: pip install cognee && cognee serve"
                    ),
                },
                indent=2,
            )
