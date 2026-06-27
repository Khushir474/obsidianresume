"""Verify the CachedFileWriterTool writes new role scaffolds correctly."""
import pathlib
import pytest
from obsidianresumeforge.tools.custom_tool import CachedFileWriterTool

SCAFFOLD_CONTENT = """\
STEP 1 — BEFORE WRITING ANYTHING:
Extract hard and soft keywords from the JD.

STEP 2 — WRITE THE RESUME:
Follow ATS rules for this role.

ATS RULES:
- Use action verbs
- Quantify achievements

ROLE-SPECIFIC ATS SIGNALS:
- Contract management
- Legal review

SUPPRESS THIS LANGUAGE:
- analyst
- coordinator

EXTRAPOLATION:
Flag any non-verifiable claims.

WRITING STYLE:
Paul Graham, direct, first-person voice.

FORMAT:
Standard single-page LaTeX resume.
"""


def test_scaffold_written_to_role_folder(tmp_path):
    tool = CachedFileWriterTool()
    role_name = "contract_specialist"
    result = tool._run(
        filename=f"{role_name}.md",
        content=SCAFFOLD_CONTENT,
        directory=str(tmp_path),
        overwrite=False,
    )
    scaffold_path = tmp_path / f"{role_name}.md"
    assert scaffold_path.exists(), f"Scaffold not written. Tool returned: {result}"
    assert "STEP 1" in scaffold_path.read_text()
    assert "ATS RULES" in scaffold_path.read_text()


def test_scaffold_not_overwritten_without_flag(tmp_path):
    tool = CachedFileWriterTool()
    existing = tmp_path / "existing_role.md"
    existing.write_text("original content")

    result = tool._run(
        filename="existing_role.md",
        content="new content",
        directory=str(tmp_path),
        overwrite=False,
    )
    assert existing.read_text() == "original content", (
        f"File was overwritten unexpectedly. Tool returned: {result}"
    )


def test_scaffold_overwritten_with_flag(tmp_path):
    tool = CachedFileWriterTool()
    existing = tmp_path / "existing_role.md"
    existing.write_text("old scaffold")

    tool._run(
        filename="existing_role.md",
        content="updated scaffold",
        directory=str(tmp_path),
        overwrite=True,
    )
    assert existing.read_text() == "updated scaffold"
