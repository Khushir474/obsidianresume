"""
Retry loop orchestrator: wraps crew.kickoff() and retries if eval fails.
"""
import json
import logging
import re
from typing import Callable

from obsidianresumeforge.output_writers import write_eval_log

logger = logging.getLogger(__name__)


def _parse_passed(crew_output) -> tuple[bool, str]:
    """Return (passed, retry_recommendation) from crew output eval task."""
    from obsidianresumeforge.output_writers import _find_task_output, _extract_json

    raw = _find_task_output(crew_output, "evaluate_pipeline_output")
    if not raw:
        return True, ""  # if we can't parse, don't loop forever

    parsed = _extract_json(raw)
    if parsed and isinstance(parsed, dict):
        return bool(parsed.get("passed", True)), parsed.get("retry_recommendation", "")

    return True, ""


def _increment_run_id(run_id: str) -> str:
    """Increment the NNN counter in run_YYYYMMDD_NNN."""
    m = re.match(r"^(run_\d{8}_)(\d{3})$", run_id)
    if m:
        return f"{m.group(1)}{int(m.group(2)) + 1:03d}"
    return run_id + "_retry"


def run_with_retry(
    crew_factory: Callable,
    inputs: dict,
    max_retries: int,
    logs_folder: str,
) -> tuple:
    """Run crew with up to max_retries retries if eval fails.

    Returns (final_crew_output, retry_count).
    """
    current_inputs = dict(inputs)
    last_output = None

    for attempt in range(max_retries + 1):
        logger.info("Crew run attempt %d/%d — run_id=%s", attempt + 1, max_retries + 1, current_inputs.get("run_id"))
        crew_output = crew_factory().crew().kickoff(inputs=current_inputs)
        last_output = crew_output

        write_eval_log(current_inputs["run_id"], crew_output, logs_folder)

        passed, retry_recommendation = _parse_passed(crew_output)
        if passed:
            logger.info("Eval passed on attempt %d", attempt + 1)
            return crew_output, attempt

        if attempt < max_retries:
            new_run_id = _increment_run_id(current_inputs["run_id"])
            logger.info(
                "Eval failed (attempt %d). Retrying as %s. Recommendation: %s",
                attempt + 1, new_run_id, retry_recommendation,
            )
            current_inputs = dict(current_inputs)
            current_inputs["run_id"] = new_run_id
            current_inputs["retry_context"] = retry_recommendation
            import os
            os.environ["RUN_ID"] = new_run_id
        else:
            logger.warning("Eval still failing after %d attempts — returning final output", max_retries + 1)

    return last_output, max_retries
