import logging
import os
import threading
import time
from pathlib import Path
from typing import Callable

from watchdog.events import FileSystemEventHandler, FileCreatedEvent
from watchdog.observers import Observer

logger = logging.getLogger(__name__)

_DEBOUNCE_SECONDS = 2.0


class JDWatcher(FileSystemEventHandler):
    """Watch {vault_path}/JDs/ for new .md files and trigger crew kickoff."""

    def __init__(self, jds_dir: str, on_new_jd: Callable[[str], None]):
        super().__init__()
        self._jds_dir = jds_dir
        self._on_new_jd = on_new_jd
        self._pending: dict[str, threading.Timer] = {}
        self._lock = threading.Lock()

    def on_created(self, event: FileCreatedEvent) -> None:
        if event.is_directory:
            return
        path = event.src_path
        if not path.endswith(".md"):
            return

        with self._lock:
            existing = self._pending.pop(path, None)
            if existing:
                existing.cancel()
            timer = threading.Timer(_DEBOUNCE_SECONDS, self._fire, args=(path,))
            self._pending[path] = timer
            timer.start()

    def _fire(self, path: str) -> None:
        with self._lock:
            self._pending.pop(path, None)

        if not Path(path).exists() or Path(path).stat().st_size == 0:
            logger.warning("JD file gone or empty, skipping: %s", path)
            return

        logger.info("New JD detected: %s — triggering crew kickoff", path)
        os.environ["CREWAI_TOOLS_ALLOW_UNSAFE_PATHS"] = "true"
        os.environ["JD_FILE_PATH"] = path
        try:
            self._on_new_jd(path)
        except Exception:
            logger.exception("Crew kickoff failed for %s", path)


def start_watcher(vault_path: str, kickoff_fn: Callable[[str], None]) -> Observer:
    """Start watching {vault_path}/JDs/ and call kickoff_fn(jd_path) on new .md files."""
    jds_dir = str(Path(vault_path) / "JDs")
    Path(jds_dir).mkdir(parents=True, exist_ok=True)

    handler = JDWatcher(jds_dir=jds_dir, on_new_jd=kickoff_fn)
    observer = Observer()
    observer.schedule(handler, jds_dir, recursive=False)
    observer.start()
    logger.info("Watching %s for new JD files …", jds_dir)
    return observer
