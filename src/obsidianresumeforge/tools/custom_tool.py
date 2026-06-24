import logging
import os
from pathlib import Path
from typing import Any

from crewai.tools import BaseTool
from crewai_tools.tools.file_read_tool.file_read_tool import FileReadToolSchema
from crewai_tools.tools.file_writer_tool.file_writer_tool import FileWriterToolInput, strtobool
from crewai_tools.security.safe_path import validate_file_path
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Persistent cross-run cache. Entries are invalidated whenever a file is written.
_FILE_CACHE: dict[str, str] = {}


def _invalidate(path: str) -> None:
    """Remove all cache entries whose key starts with the resolved path."""
    resolved = os.path.realpath(path)
    stale = [k for k in _FILE_CACHE if k.startswith(resolved + ":")]
    for k in stale:
        del _FILE_CACHE[k]
    if stale:
        logger.info("Cache invalidated (%d entries): %s", len(stale), resolved)


class CachedFileReadTool(BaseTool):
    """FileReadTool with persistent caching and clear file-path logging.

    Logs which file is being read on every call. Returns cached content on
    repeated reads so agents don't re-fetch identical files within or across runs.
    Cache entries are invalidated automatically when CachedFileWriterTool writes
    to the same path.
    """

    name: str = "Read a file's content"
    description: str = (
        "Reads the full content of a file. Provide 'file_path' with the absolute "
        "path to the file. Optionally provide 'start_line' and 'line_count' to read "
        "a specific range. Results are cached — repeated reads of the same path are "
        "free and the cache is automatically invalidated when the file is written."
    )
    args_schema: type[BaseModel] = FileReadToolSchema
    file_path: str | None = None

    def __init__(self, file_path: str | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.file_path = file_path

    def _run(
        self,
        file_path: str | None = None,
        start_line: int | None = 1,
        line_count: int | None = None,
    ) -> str:
        path = file_path or self.file_path
        if path is None:
            return "Error: No file path provided."

        path = validate_file_path(path)
        start_line = start_line or 1
        cache_key = f"{path}:{start_line}:{line_count}"

        if cache_key in _FILE_CACHE:
            logger.info("Reading (cached): %s", path)
            return _FILE_CACHE[cache_key]

        logger.info("Reading: %s", path)
        try:
            with open(path, "r") as f:
                if start_line == 1 and line_count is None:
                    content = f.read()
                else:
                    start_idx = max(start_line - 1, 0)
                    lines = [
                        line
                        for i, line in enumerate(f)
                        if i >= start_idx
                        and (line_count is None or i < start_idx + line_count)
                    ]
                    if not lines and start_idx > 0:
                        return f"Error: Start line {start_line} exceeds file length."
                    content = "".join(lines)
        except FileNotFoundError:
            return f"Error: File not found: {path}"
        except PermissionError:
            return f"Error: Permission denied: {path}"
        except Exception as e:
            return f"Error: Failed to read {path}. {e!s}"

        result = f"[File: {path}]\n{content}"
        _FILE_CACHE[cache_key] = result
        return result


class CachedFileWriterTool(BaseTool):
    """FileWriterTool that invalidates the read cache for any file it writes.

    Keeps the file read cache consistent — if classify_role appends keywords
    to a role file, the next agent to read that file gets the updated content.
    """

    name: str = "File Writer Tool"
    description: str = (
        "Writes content to a file. Accepts 'filename', 'content', and optionally "
        "'directory' and 'overwrite' (default false). Automatically invalidates the "
        "read cache for the written path so subsequent reads see fresh content."
    )
    args_schema: type[BaseModel] = FileWriterToolInput

    def _run(self, **kwargs: Any) -> str:
        try:
            directory = kwargs.get("directory") or "./"
            filename = kwargs["filename"]
            filepath = os.path.join(directory, filename)

            real_directory = Path(directory).resolve()
            real_filepath = Path(filepath).resolve()
            if (
                not real_filepath.is_relative_to(real_directory)
                or real_filepath == real_directory
            ):
                return "Error: Invalid file path — filename must not escape the target directory."

            if kwargs.get("directory"):
                os.makedirs(real_directory, exist_ok=True)

            kwargs["overwrite"] = strtobool(kwargs["overwrite"])

            if os.path.exists(real_filepath) and not kwargs["overwrite"]:
                return f"File {real_filepath} already exists and overwrite option was not passed."

            mode = "w" if kwargs["overwrite"] else "x"
            with open(real_filepath, mode) as f:
                f.write(kwargs["content"])

            _invalidate(str(real_filepath))
            logger.info("Written: %s", real_filepath)
            return f"Content successfully written to {real_filepath}"
        except FileExistsError:
            return f"File {real_filepath} already exists and overwrite option was not passed."
        except KeyError as e:
            return f"An error occurred while accessing key: {e!s}"
        except Exception as e:
            return f"An error occurred while writing to the file: {e!s}"
