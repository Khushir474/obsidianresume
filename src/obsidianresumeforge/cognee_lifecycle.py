"""
Cognee server lifecycle manager.
Checks whether Cognee is reachable before the crew starts and optionally
spawns it if not. Sets a module-level flag so CogneeMemoryTool can gate ops.
"""
import logging
import subprocess
import time

logger = logging.getLogger(__name__)

_HEALTH_URL = "http://localhost:8000/health"
_STARTUP_WAIT = 5  # seconds to wait after spawning cognee

_cognee_available: bool = False


def cognee_available() -> bool:
    """Return True if Cognee was confirmed reachable at startup."""
    return _cognee_available


def ensure_cognee_running(health_url: str = _HEALTH_URL) -> bool:
    """Check Cognee health, optionally spawn it, and set the module flag.

    Returns True if Cognee is reachable, False otherwise.
    Does NOT raise — callers degrade gracefully on False.
    """
    global _cognee_available

    if _check_health(health_url):
        _cognee_available = True
        logger.info("Cognee is reachable at %s", health_url)
        return True

    logger.info("Cognee not reachable — attempting to start 'cognee serve'")
    try:
        subprocess.Popen(["cognee", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(_STARTUP_WAIT)
    except FileNotFoundError:
        logger.warning("'cognee' binary not found — memory features disabled")
        _cognee_available = False
        return False
    except Exception as exc:
        logger.warning("Failed to start Cognee: %s — memory features disabled", exc)
        _cognee_available = False
        return False

    if _check_health(health_url):
        _cognee_available = True
        logger.info("Cognee started successfully")
        return True

    logger.warning("Cognee still unreachable after startup — memory features disabled")
    _cognee_available = False
    return False


def _check_health(url: str) -> bool:
    try:
        import requests
        resp = requests.get(url, timeout=2)
        return resp.status_code == 200
    except Exception:
        return False
