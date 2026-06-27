"""Tests for run ID generator."""
import datetime
import re
import pathlib
import pytest
from unittest.mock import patch

from obsidianresumeforge.run_id import generate_run_id

RUN_ID_PATTERN = re.compile(r"^run_\d{8}_\d{3}$")


def test_format(tmp_path):
    run_id = generate_run_id(str(tmp_path))
    assert RUN_ID_PATTERN.match(run_id), f"Bad format: {run_id}"


def test_starts_at_001_when_no_logs(tmp_path):
    run_id = generate_run_id(str(tmp_path))
    assert run_id.endswith("_001")


def test_increments_when_log_exists(tmp_path):
    today = datetime.date.today().strftime("%Y%m%d")
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    (eval_dir / f"run_{today}_001.json").write_text("{}")
    run_id = generate_run_id(str(tmp_path))
    assert run_id.endswith("_002")


def test_increments_past_existing_max(tmp_path):
    today = datetime.date.today().strftime("%Y%m%d")
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    for n in [1, 3, 5]:
        (eval_dir / f"run_{today}_{n:03d}.json").write_text("{}")
    run_id = generate_run_id(str(tmp_path))
    assert run_id.endswith("_006")


def test_ignores_other_day_logs(tmp_path):
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    (eval_dir / "run_20200101_099.json").write_text("{}")
    run_id = generate_run_id(str(tmp_path))
    assert run_id.endswith("_001"), "Should not count logs from a different date"


def test_date_in_run_id_matches_today(tmp_path):
    today = datetime.date.today().strftime("%Y%m%d")
    run_id = generate_run_id(str(tmp_path))
    assert today in run_id
