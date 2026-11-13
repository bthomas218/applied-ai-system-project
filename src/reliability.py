"""Deterministic reliability checks for the hybrid recommender."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List


@dataclass
class GroundingCheck:
    passed: bool
    reasons: List[str]


@dataclass
class ReliabilityReport:
    extraction_confidence: float
    explanation_confidence: float
    overall_confidence: float
    level: str
    reasons: List[str]
    grounding_checks: List[GroundingCheck]


def confidence_level(value: float) -> str:
    if value >= 0.80:
        return "high"
    if value >= 0.60:
        return "medium"
    return "low"


def extraction_confidence(preferences: Any, used_fallback: bool) -> tuple[float, List[str]]:
    """Score completeness and source quality without trusting model self-ratings."""
    fields = ("genre", "mood", "energy", "acousticness")
    present = [field for field in fields if getattr(preferences, field, None) is not None]
    completeness = len(present) / len(fields)
    source_quality = 0.55 if used_fallback else 1.0
    score = round((completeness * 0.70) + (source_quality * 0.30), 2)
    reasons = [f"{len(present)}/{len(fields)} core preference fields were extracted"]
    reasons.append("deterministic fallback was used" if used_fallback else "LLM extraction passed local schema checks")
    return score, reasons


def check_grounding(explanation: str, song: dict[str, Any]) -> GroundingCheck:
    """Check that an explanation identifies the retrieved song and is non-empty."""
    reasons: List[str] = []
    text = explanation.strip().lower() if isinstance(explanation, str) else ""
    if not text:
        reasons.append("explanation is empty")
    if text and song["title"].lower() not in text and song["artist"].lower() not in text:
        reasons.append("explanation does not identify the retrieved song or artist")
    return GroundingCheck(passed=not reasons, reasons=reasons or ["song identity is present"])


def build_reliability_report(
    preferences: Any,
    recommendations: Iterable[Any],
    extraction_fallback: bool,
    explanation_fallback: bool,
) -> ReliabilityReport:
    recommendations = list(recommendations)
    checks = [check_grounding(item.explanation, item.song) for item in recommendations]
    passed_checks = sum(check.passed for check in checks)
    structural_score = 1.0 if recommendations else 0.0
    grounding_score = passed_checks / len(checks) if checks else 0.0
    explanation_source = 0.70 if explanation_fallback else 1.0
    explanation_score = round(
        (structural_score * 0.25) + (grounding_score * 0.55) + (explanation_source * 0.20),
        2,
    )
    extraction_score, reasons = extraction_confidence(preferences, extraction_fallback)
    overall = round((extraction_score + explanation_score) / 2, 2)
    if explanation_fallback:
        reasons.append("template explanations were used")
    if checks and passed_checks < len(checks):
        reasons.append(f"{len(checks) - passed_checks} explanation grounding check(s) failed")
    return ReliabilityReport(
        extraction_confidence=extraction_score,
        explanation_confidence=explanation_score,
        overall_confidence=overall,
        level=confidence_level(overall),
        reasons=reasons,
        grounding_checks=checks,
    )
