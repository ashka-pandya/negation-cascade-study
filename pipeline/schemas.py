"""Dataclass schemas for the multi-step clinical NLP pipeline."""

from dataclasses import dataclass
from typing import Any


@dataclass
class ClinicalNote:
    """Represents a synthetic clinical note and its ground truth finding label."""

    id: str
    text: str
    finding: str
    negation_type: str
    ground_truth_present: bool


@dataclass
class Step1Output:
    """Schema for step 1 extraction outputs."""

    note_id: str
    extracted_findings: list[dict[str, Any]]
    raw_response: str
    error: str | None


@dataclass
class Step2Output:
    """Schema for step 2 evidence form outputs."""

    note_id: str
    evidence_form: dict[str, Any]
    raw_response: str
    error: str | None


@dataclass
class Step3Output:
    """Schema for step 3 criteria matching outputs."""

    note_id: str
    criteria_scores: dict[str, bool]
    criteria_met: int
    raw_response: str
    error: str | None


@dataclass
class Step4Output:
    """Schema for step 4 verdict outputs."""

    note_id: str
    verdict: str
    reasoning: str
    raw_response: str
    error: str | None


@dataclass
class PipelineResult:
    """Aggregated output from all four pipeline steps for one note."""

    note_id: str
    negation_type: str
    finding: str
    ground_truth_present: bool
    step1: Step1Output
    step2: Step2Output
    step3: Step3Output
    step4: Step4Output
    cascade_reached_step4: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a flattened dictionary suitable for CSV row serialization."""

        row: dict[str, Any] = {
            "note_id": self.note_id,
            "negation_type": self.negation_type,
            "finding": self.finding,
            "ground_truth_present": self.ground_truth_present,
            "cascade_reached_step4": self.cascade_reached_step4,
            "step1_note_id": self.step1.note_id,
            "step1_extracted_findings": self.step1.extracted_findings,
            "step1_raw_response": self.step1.raw_response,
            "step1_error": self.step1.error,
            "step2_note_id": self.step2.note_id,
            "step2_evidence_form": self.step2.evidence_form,
            "step2_raw_response": self.step2.raw_response,
            "step2_error": self.step2.error,
            "step3_note_id": self.step3.note_id,
            "step3_criteria_scores": self.step3.criteria_scores,
            "step3_criteria_met": self.step3.criteria_met,
            "step3_raw_response": self.step3.raw_response,
            "step3_error": self.step3.error,
            "step4_note_id": self.step4.note_id,
            "step4_verdict": self.step4.verdict,
            "step4_reasoning": self.step4.reasoning,
            "step4_raw_response": self.step4.raw_response,
            "step4_error": self.step4.error,
        }
        return row
