import pytest


@pytest.fixture
def tmp_vault(tmp_path):
    (tmp_path / "JDs").mkdir()
    (tmp_path / "Resumes" / "PDF").mkdir(parents=True)
    (tmp_path / "Resumes" / "LaTeX").mkdir(parents=True)
    (tmp_path / "eval").mkdir()
    (tmp_path / "KnowledgeGraph").mkdir()
    (tmp_path / "InterviewPrep").mkdir()
    return tmp_path
