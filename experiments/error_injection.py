"""Utilities for injecting synthetic step-1 negation extraction errors."""

from __future__ import annotations

from copy import deepcopy

from pipeline.schemas import ClinicalNote, Step1Output


def inject_negation_error(step1_output: Step1Output, target_finding: str) -> Step1Output:
    """Return a copied Step1Output with one target finding flipped to a worst-case false positive."""

    updated_output = deepcopy(step1_output)
    target_normalized = target_finding.strip().lower()

    for finding in updated_output.extracted_findings:
        if not isinstance(finding, dict):
            continue

        finding_name = str(finding.get("finding", "")).strip().lower()
        if finding_name != target_normalized:
            continue

        finding["present"] = True
        finding["confidence"] = "high"
        return updated_output

    return step1_output


def create_corrupted_pipeline_input(note: ClinicalNote, clean_step1: Step1Output) -> Step1Output:
    """Create a corrupted step-1 output by injecting a negation-flip error for the note target finding."""

    return inject_negation_error(step1_output=clean_step1, target_finding=note.finding)
