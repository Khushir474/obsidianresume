"""
Tests for JudgevalLocalEvaluatorRunnerTool — all 4 judges called with real mock data.
"""
import json
import pytest
from obsidianresumeforge.tools.judgeval_local_evaluator_runner import (
    JudgevalLocalEvaluatorRunnerTool,
)

MOCK_LATEX = r"""
\documentclass{article}
\begin{document}
\section{Summary}
Experienced ML engineer with focus on production systems.

\section{Experience}
\subsection{Software Engineer | Example Corp}
\begin{itemize}
  \item Built Python-based inference pipeline reducing latency by 40\%
  \item Deployed 15 ML models to AWS SageMaker serving 2M daily requests
  \item Led cross-functional team of 8 engineers to deliver on-time
\end{itemize}

\section{Skills}
Python, PyTorch, AWS, LangChain, Docker, Kubernetes

\section{Education}
B.S. Computer Science, State University
\end{document}
"""

MOCK_KEYWORD_LIST_A = json.dumps([
    {"keyword": "Python", "score": "Critical", "inclusion": "verbatim", "rationale": "Core language"},
    {"keyword": "AWS", "score": "Critical", "inclusion": "verbatim", "rationale": "Cloud platform"},
    {"keyword": "PyTorch", "score": "High", "inclusion": "verbatim", "rationale": "ML framework"},
    {"keyword": "LangChain", "score": "High", "inclusion": "verbatim", "rationale": "LLM framework"},
    {"keyword": "inference pipeline", "score": "Critical", "inclusion": "verbatim", "rationale": "Core product"},
])

MOCK_KEYWORD_LIST_B = json.dumps([
    {"keyword": "cross-functional", "score": "High", "inclusion": "verbatim", "rationale": "Collaboration"},
    {"keyword": "production", "score": "Critical", "inclusion": "contextual embedding", "rationale": "Reliability focus"},
])

MOCK_SOURCING_MAP = json.dumps([
    {
        "section": "Experience",
        "source_file": "example_company.md",
        "original_bullet": "Built inference pipeline reducing latency",
        "adapted_bullet": "Built Python-based inference pipeline reducing latency by 40%",
        "keywords_embedded": ["Python", "inference pipeline"],
        "reason": "Critical keyword verbatim embed",
    },
    {
        "section": "Experience",
        "source_file": "example_company.md",
        "original_bullet": "Deployed ML models to cloud platform serving millions of requests",
        "adapted_bullet": "Deployed 15 ML models to AWS SageMaker serving 2M daily requests",
        "keywords_embedded": ["AWS"],
        "reason": "Verbatim AWS embed with metric injection",
    },
    {
        "section": "Experience",
        "source_file": "example_company.md",
        "original_bullet": "Led engineering team to deliver project on schedule",
        "adapted_bullet": "Led engineering team to deliver project on schedule and within budget",
        "keywords_embedded": ["cross-functional"],
        "reason": "Soft keyword embed with experience-grounded phrasing",
    },
])

MOCK_EXPERIENCE = """
# Software Engineer | Example Corp
**Duration:** 2022–2024

## Consolidated Bullets
- Built inference pipeline reducing latency
- Deployed ML models to cloud platform serving millions of daily requests
- Led engineering team to deliver project on schedule and within budget
- Implemented monitoring dashboards for production systems

## Metrics Reference
| Metric | Value |
|--------|-------|
| Latency reduction | 40% |
| Models deployed | 15 |
| Daily requests | 2M |
| Team size | 8 |
"""


@pytest.fixture
def tool():
    return JudgevalLocalEvaluatorRunnerTool()


def test_all_judges_return_nonzero(tool):
    result = tool._run(
        phase_output=MOCK_LATEX,
        keyword_list_a=MOCK_KEYWORD_LIST_A,
        keyword_list_b=MOCK_KEYWORD_LIST_B,
        sourcing_map=MOCK_SOURCING_MAP,
        experience_files_content=MOCK_EXPERIENCE,
        pdflatex_exit_code=0,
    )
    data = json.loads(result)
    assert "error" not in data, f"Tool returned error: {data.get('message')}"
    scores = data["judge_scores"]
    for judge_name, judge_result in scores.items():
        assert judge_result["score"] > 0.0, (
            f"{judge_name} scored 0.0 — details: {judge_result['details']}"
        )


def test_composite_score_structure(tool):
    result = tool._run(
        phase_output=MOCK_LATEX,
        keyword_list_a=MOCK_KEYWORD_LIST_A,
        keyword_list_b=MOCK_KEYWORD_LIST_B,
        sourcing_map=MOCK_SOURCING_MAP,
        experience_files_content=MOCK_EXPERIENCE,
        pdflatex_exit_code=0,
    )
    data = json.loads(result)
    assert "composite_score" in data
    assert "passed" in data
    assert "judge_scores" in data
    assert "issues" in data
    assert "retry_recommendation" in data
    assert 0.0 <= data["composite_score"] <= 1.0


def test_failed_pdflatex_scores_zero_format(tool):
    result = tool._run(
        phase_output=MOCK_LATEX,
        keyword_list_a=MOCK_KEYWORD_LIST_A,
        keyword_list_b=MOCK_KEYWORD_LIST_B,
        sourcing_map=MOCK_SOURCING_MAP,
        experience_files_content=MOCK_EXPERIENCE,
        pdflatex_exit_code=1,
    )
    data = json.loads(result)
    assert data["judge_scores"]["FormatComplianceJudge"]["score"] == 0.0


def test_empty_keyword_lists_handled(tool):
    result = tool._run(
        phase_output=MOCK_LATEX,
        keyword_list_a="[]",
        keyword_list_b="[]",
        sourcing_map=MOCK_SOURCING_MAP,
        experience_files_content=MOCK_EXPERIENCE,
        pdflatex_exit_code=0,
    )
    data = json.loads(result)
    assert "error" not in data
    # Empty keyword lists → ATS judge returns 1.0 (no critical/high keywords to miss)
    assert data["judge_scores"]["ATSKeywordHitRateJudge"]["score"] == 1.0


def test_empty_sourcing_map_gives_zero_attribution(tool):
    result = tool._run(
        phase_output=MOCK_LATEX,
        keyword_list_a=MOCK_KEYWORD_LIST_A,
        keyword_list_b=MOCK_KEYWORD_LIST_B,
        sourcing_map="",
        experience_files_content=MOCK_EXPERIENCE,
        pdflatex_exit_code=0,
    )
    data = json.loads(result)
    assert data["judge_scores"]["SourceAttributionJudge"]["score"] == 0.0
