"""Helpers to run the full 4-step clinical NLP pipeline."""

from __future__ import annotations

from openai import OpenAI
from pipeline.schemas import (
    ClinicalNote, PipelineResult
)
from pipeline.step1_extraction import (
    extract_findings,
    run_negation_aware_extraction
)
from pipeline.step2_evidence import populate_evidence_form
from pipeline.step3_criteria import match_criteria
from pipeline.step4_verdict import generate_verdict
from tqdm import tqdm


def run_full_pipeline(
    note: ClinicalNote,
    client: OpenAI,
    use_negation_aware: bool = False,
    model: str = "gpt-4o",
) -> PipelineResult:
    """Run all four pipeline steps for one note, retaining partial results on error."""

    if use_negation_aware:
        step1 = run_negation_aware_extraction([note], client, model)[0]
    else:
        step1 = extract_findings(note, client, model)

    if step1.error:
        print(f"[PIPELINE][STEP1][ERROR] note_id={note.id} error={step1.error}")

    step2 = populate_evidence_form(step1, note, client, model)
    if step2.error:
        print(f"[PIPELINE][STEP2][ERROR] note_id={note.id} error={step2.error}")

    step3 = match_criteria(step2, client, model)
    if step3.error:
        print(f"[PIPELINE][STEP3][ERROR] note_id={note.id} error={step3.error}")

    step4 = generate_verdict(step3, client, model)
    if step4.error:
        print(f"[PIPELINE][STEP4][ERROR] note_id={note.id} error={step4.error}")

    return PipelineResult(
        note_id=note.id,
        negation_type=note.negation_type,
        finding=note.finding,
        ground_truth_present=note.ground_truth_present,
        step1=step1,
        step2=step2,
        step3=step3,
        step4=step4,
        cascade_reached_step4=False,
    )


def run_pipeline_batch(
    notes: list[ClinicalNote],
    client: OpenAI,
    use_negation_aware: bool = False,
    model: str = "gpt-4o",
) -> list[PipelineResult]:
    """Run the full pipeline for each note, preserving input order."""

    results: list[PipelineResult] = []
    for note in tqdm(notes, desc="Running pipeline"):
        try:
            result = run_full_pipeline(
                note=note,
                client=client,
                use_negation_aware=use_negation_aware,
                model=model,
            )
            results.append(result)
        except Exception as exc:  # noqa: BLE001
            print(f"[PIPELINE][BATCH][ERROR] note_id={note.id} error={exc}")
            continue
    return results
