"""Tests for the human confirmation gate."""
import subprocess
import pytest
from unittest.mock import patch, MagicMock

from obsidianresumeforge.output_writers import confirm_new_role


def test_y_returns_true(tmp_path):
    scaffold = tmp_path / "new_role.md"
    scaffold.write_text("# New Role Scaffold")
    with patch("builtins.input", return_value="y"):
        assert confirm_new_role("new_role", str(scaffold)) is True


def test_n_returns_false(tmp_path):
    scaffold = tmp_path / "new_role.md"
    scaffold.write_text("# New Role Scaffold")
    with patch("builtins.input", return_value="n"):
        assert confirm_new_role("new_role", str(scaffold)) is False


def test_edit_opens_editor_then_y(tmp_path):
    scaffold = tmp_path / "new_role.md"
    scaffold.write_text("# New Role Scaffold")
    responses = iter(["edit", "y"])
    with patch("builtins.input", side_effect=lambda _: next(responses)):
        with patch("subprocess.call") as mock_call:
            result = confirm_new_role("new_role", str(scaffold))
    assert result is True
    assert mock_call.called


def test_invalid_input_loops_until_valid(tmp_path):
    scaffold = tmp_path / "role.md"
    scaffold.write_text("scaffold")
    responses = iter(["maybe", "dunno", "y"])
    with patch("builtins.input", side_effect=lambda _: next(responses)):
        result = confirm_new_role("role", str(scaffold))
    assert result is True


def test_works_without_scaffold_file(tmp_path):
    with patch("builtins.input", return_value="y"):
        result = confirm_new_role("ghost_role", str(tmp_path / "nonexistent.md"))
    assert result is True
