import re
import json
import logging
import os
import datetime
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type

logger = logging.getLogger(__name__)


class JudgevalLocalEvaluatorRunnerInput(BaseModel):
    """Input schema for JudgevalLocalEvaluatorRunner Tool."""

    phase_output: str = Field(
        description="Full resume .tex content + sourcing map from the Resume Writer"
    )
    keyword_list_a: str = Field(
        description='JSON string — List A hard keywords. Each item has at least "keyword" and "score" fields where score is "Critical", "High", "Medium", or "Low"'
    )
    keyword_list_b: str = Field(
        description='JSON string — List B soft keywords. Same structure as keyword_list_a'
    )
    sourcing_map: str = Field(
        description='JSON string — array of bullets mapped to source experience files. Each item has at least "adapted_bullet" field'
    )
    experience_files_content: str = Field(
        description="Concatenated content of all source experience files"
    )
    pdflatex_exit_code: int = Field(
        default=0,
        description="Exit code from pdflatex run (0 = success)"
    )
    role_instruction_content: str = Field(
        default="",
        description="Full text of the selected role instruction file"
    )


class JudgevalLocalEvaluatorRunnerTool(BaseTool):
    """Tool for running judgeval-based evaluation using 4 programmatic Judge subclasses."""

    name: str = "JudgevalLocalEvaluatorRunner"
    description: str = (
        "Run judgeval-based evaluation using 4 programmatic Judge subclasses (code-based, "
        "no gold answer required). Returns a composite weighted score and per-judge breakdown "
        "as a JSON string. Judges cover ATS keyword hit rate, metric density, source attribution, "
        "and format compliance."
    )
    args_schema: Type[BaseModel] = JudgevalLocalEvaluatorRunnerInput

    # ------------------------------------------------------------------ #
    # Inner judge functions                                                #
    # ------------------------------------------------------------------ #

    def _judge_ats_keyword_hit_rate(
        self, phase_output: str, keyword_list_a: str, keyword_list_b: str
    ) -> dict:
        """Judge 1: ATS Keyword Hit Rate (weight 0.35)."""
        try:
            list_a = json.loads(keyword_list_a) if keyword_list_a.strip() else []
        except Exception as e:
            return {"score": 0.0, "weight": 0.35, "details": f"keyword_list_a parse error: {e}"}

        try:
            list_b = json.loads(keyword_list_b) if keyword_list_b.strip() else []
        except Exception as e:
            return {"score": 0.0, "weight": 0.35, "details": f"keyword_list_b parse error: {e}"}

        combined = list_a + list_b
        priority_levels = {"critical", "high"}
        filtered = [
            item for item in combined
            if str(item.get("score", "")).lower() in priority_levels
        ]

        total = len(filtered)
        if total == 0:
            return {"score": 1.0, "weight": 0.35, "details": "0/0 Critical+High keywords matched (no critical/high keywords defined)"}

        phase_lower = phase_output.lower()
        matched = sum(
            1 for item in filtered
            if str(item.get("keyword", "")).lower() in phase_lower
        )

        score = matched / total
        return {
            "score": round(score, 4),
            "weight": 0.35,
            "details": f"{matched}/{total} Critical+High keywords matched"
        }

    def _judge_metric_density(self, phase_output: str) -> dict:
        """Judge 2: Metric Density (weight 0.25)."""
        # Normalize double-escaped LaTeX (\\item → \item) to handle JSON-encoded strings
        normalized = phase_output.replace('\\\\item', '\\item')
        lines = normalized.splitlines()
        bullet_pattern = re.compile(r'^\s*\\item\s+.+')
        dash_pattern = re.compile(r'^- .{3,}')

        bullets = [
            line for line in lines
            if bullet_pattern.match(line) or dash_pattern.match(line)
        ]

        total_bullets = len(bullets)
        if total_bullets == 0:
            return {"score": 0.0, "weight": 0.25, "details": "0 bullets found"}

        metric_pattern = re.compile(r'\d+\s*%|\d+x|\b\d{2,}\b')
        metric_bullets = sum(1 for b in bullets if metric_pattern.search(b))

        score = metric_bullets / total_bullets
        return {
            "score": round(score, 4),
            "weight": 0.25,
            "details": f"{metric_bullets}/{total_bullets} bullets contain numeric metrics"
        }

    def _judge_source_attribution(
        self, sourcing_map: str, experience_files_content: str
    ) -> dict:
        """Judge 3: Source Attribution (weight 0.20)."""
        if not sourcing_map.strip():
            return {"score": 0.0, "weight": 0.20, "details": "sourcing_map is empty or not provided"}

        try:
            parsed_map = json.loads(sourcing_map)
        except Exception as e:
            return {"score": 0.0, "weight": 0.20, "details": f"sourcing_map parse error: {e}"}

        if not parsed_map:
            return {"score": 0.0, "weight": 0.20, "details": "Empty sourcing map"}

        total = len(parsed_map)
        traceable = 0

        def _word_tokens(text):
            return set(re.findall(r'\b[a-z]{4,}\b', text.lower()))

        for item in parsed_map:
            bullet = str(item.get("adapted_bullet", "")).strip()
            if not bullet:
                continue
            bullet_words = _word_tokens(bullet)
            if len(bullet_words) == 0:
                continue
            # Count how many meaningful words from the bullet appear in experience content
            overlap = len(bullet_words & _word_tokens(experience_files_content))
            # Traceable if at least 40% of bullet's meaningful words appear in source content
            if overlap / len(bullet_words) >= 0.40:
                traceable += 1

        attribution_rate = traceable / total
        score = 1.0 if attribution_rate >= 0.85 else 0.0

        return {
            "score": score,
            "weight": 0.20,
            "details": f"{traceable}/{total} bullets traceable to source files (rate: {attribution_rate:.2f})"
        }

    def _judge_format_compliance(
        self, phase_output: str, pdflatex_exit_code: int
    ) -> dict:
        """Judge 4: Format Compliance (weight 0.20)."""
        if pdflatex_exit_code != 0:
            return {
                "score": 0.0,
                "weight": 0.20,
                "details": f"pdflatex failed with exit code {pdflatex_exit_code}"
            }

        phase_lower = phase_output.lower()
        required_sections = ["experience", "skills", "education", "summary"]
        found = sum(1 for section in required_sections if section in phase_lower)

        score = 1.0 if found >= 3 else 0.0
        return {
            "score": score,
            "weight": 0.20,
            "details": f"pdflatex_exit_code={pdflatex_exit_code}, {found}/4 sections found"
        }

    # ------------------------------------------------------------------ #
    # Main _run                                                            #
    # ------------------------------------------------------------------ #

    def _run(
        self,
        phase_output: str,
        keyword_list_a: str,
        keyword_list_b: str,
        sourcing_map: str,
        experience_files_content: str,
        pdflatex_exit_code: int = 0,
        role_instruction_content: str = ""
    ) -> str:
        """Execute all 4 judges and return composite evaluation as a JSON string."""
        run_id = os.getenv("RUN_ID", "unknown")
        log_path = f"/tmp/judge_args_{run_id}.log"
        arg_summary = {
            "phase_output_len": len(phase_output),
            "keyword_list_a_len": len(keyword_list_a),
            "keyword_list_b_len": len(keyword_list_b),
            "sourcing_map_len": len(sourcing_map),
            "experience_files_content_len": len(experience_files_content),
            "pdflatex_exit_code": pdflatex_exit_code,
            "phase_output_preview": phase_output[:200],
            "keyword_list_a_preview": keyword_list_a[:200],
            "sourcing_map_preview": sourcing_map[:200],
        }
        try:
            with open(log_path, "w") as f:
                f.write(f"[{datetime.datetime.utcnow().isoformat()}] judge args\n")
                json.dump(arg_summary, f, indent=2)
        except Exception:
            pass
        logger.info("JudgevalLocalEvaluatorRunner args: %s", arg_summary)

        try:
            ats = self._judge_ats_keyword_hit_rate(phase_output, keyword_list_a, keyword_list_b)
            metric = self._judge_metric_density(phase_output)
            attribution = self._judge_source_attribution(sourcing_map, experience_files_content)
            format_ = self._judge_format_compliance(phase_output, pdflatex_exit_code)

            composite_score = (
                ats["score"] * 0.35
                + metric["score"] * 0.25
                + attribution["score"] * 0.20
                + format_["score"] * 0.20
            )
            composite_score = round(composite_score, 4)

            PASS_THRESHOLD = 0.80
            passed = composite_score >= PASS_THRESHOLD

            judge_map = {
                "ATSKeywordHitRateJudge": ats,
                "MetricDensityJudge": metric,
                "SourceAttributionJudge": attribution,
                "FormatComplianceJudge": format_,
            }

            issues = []
            for judge_name, result in judge_map.items():
                if result["score"] < PASS_THRESHOLD:
                    issues.append({
                        "judge": judge_name,
                        "score": result["score"],
                        "severity": "Critical" if result["score"] < 0.5 else "High",
                        "description": result["details"]
                    })

            if issues:
                descriptions = "; ".join(issue["description"] for issue in issues)
                retry_recommendation = (
                    f"Address the following issues and retry: {descriptions}"
                )
            else:
                retry_recommendation = "All judges passed. No retry needed."

            output = {
                "composite_score": composite_score,
                "passed": passed,
                "pass_threshold": PASS_THRESHOLD,
                "judge_scores": judge_map,
                "issues": issues,
                "retry_recommendation": retry_recommendation
            }

            return json.dumps(output, indent=2)

        except Exception as e:
            error_output = {
                "error": True,
                "message": f"Unexpected error in JudgevalLocalEvaluatorRunner: {str(e)}"
            }
            return json.dumps(error_output, indent=2)
