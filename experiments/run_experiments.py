"""Experiment runners for clean, cascade, and mitigation conditions."""

from __future__ import annotations

import json
import os
import time

from openai import OpenAI
from tqdm import tqdm

from eval import save_results
from experiments.error_injection import create_corrupted_pipeline_input
from pipeline.schemas import ClinicalNote, PipelineResult, Step1Output
from pipeline.step1_extraction import extract_findings, run_negation_aware_extraction
from pipeline.step2_evidence import populate_evidence_form
from pipeline.step3_criteria import match_criteria
from pipeline.step4_verdict import generate_verdict


def _ensure_parent_dir(save_path: str) -> None:
    """Ensure the parent directory for a file output path exists."""

    os.makedirs(os.path.dirname(save_path), exist_ok=True)


def _run_self_correction_step1(
    note: ClinicalNote,
    baseline_step1: Step1Output,
    client: OpenAI,
    model: str = "gpt-4o",
) -> Step1Output:
    """Run mitigation-3 self-correction and return corrected step-1 findings with retries."""

    system_prompt = (
        "Review these extracted findings. "
        "Flag any that may have been incorrectly extracted due to negation in the source text. "
        "Return corrected findings in same JSON format."
    )

    user_payload = {
        "clinical_note_text": note.text,
        "initial_extracted_findings": baseline_step1.extracted_findings,
        "schema": {
            "findings": [
                {"finding": "str", "present": "bool", "confidence": "high|medium|low"}
            ]
        },
    }

    last_raw_response = baseline_step1.raw_response
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=0,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": json.dumps(user_payload, ensure_ascii=False, indent=2),
                    },
                ],
            )
            raw_response = response.choices[0].message.content or ""
            last_raw_response = raw_response
            parsed = json.loads(raw_response)
            findings = parsed.get("findings", []) if isinstance(parsed, dict) else []
            if not isinstance(findings, list):
                findings = []

            return Step1Output(
                note_id=note.id,
                extracted_findings=findings,
                raw_response=raw_response,
                error=None,
            )
        except Exception as exc:  # noqa: BLE001
            if attempt < 2:
                time.sleep(2)
                continue
            return Step1Output(
                note_id=note.id,
                extracted_findings=baseline_step1.extracted_findings,
                raw_response=last_raw_response,
                error=f"self_correction_failed: {exc}",
            )

    return Step1Output(
        note_id=note.id,
        extracted_findings=baseline_step1.extracted_findings,
        raw_response=last_raw_response,
        error="self_correction_failed",
    )


def run_baseline_experiment(
    notes: list[ClinicalNote],
    client: OpenAI,
    save_path: str = "experiments/baseline/results.csv",
) -> list[PipelineResult]:
    """Run the clean pipeline for all notes, save CSV outputs, and return per-note results."""

    results: list[PipelineResult] = []

    for note in tqdm(notes, desc="Baseline experiment"):
        step1 = extract_findings(note=note, client=client)
        step2 = populate_evidence_form(step1=step1, note=note, client=client)
        step3 = match_criteria(step2=step2, client=client)
        step4 = generate_verdict(step3=step3, client=client)

        results.append(
            PipelineResult(
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
        )

    _ensure_parent_dir(save_path)
    save_results(results=results, filepath=save_path)
    return results


def run_cascade_experiment(
    notes: list[ClinicalNote],
    client: OpenAI,
    clean_results: list[PipelineResult],
    save_path: str = "experiments/baseline/cascade_results.csv",
) -> list[PipelineResult]:
    """Run steps 2-4 from corrupted step-1 outputs and mark step-4 verdict cascades."""

    clean_by_id = {result.note_id: result for result in clean_results}
    results: list[PipelineResult] = []

    for note in tqdm(notes, desc="Cascade experiment"):
        clean = clean_by_id.get(note.id)
        if clean is None:
            continue

        corrupted_step1 = create_corrupted_pipeline_input(note=note, clean_step1=clean.step1)
        step2 = populate_evidence_form(step1=corrupted_step1, note=note, client=client)
        step3 = match_criteria(step2=step2, client=client)
        step4 = generate_verdict(step3=step3, client=client)

        cascade_changed = step4.verdict != clean.step4.verdict

        results.append(
            PipelineResult(
                note_id=note.id,
                negation_type=note.negation_type,
                finding=note.finding,
                ground_truth_present=note.ground_truth_present,
                step1=corrupted_step1,
                step2=step2,
                step3=step3,
                step4=step4,
                cascade_reached_step4=cascade_changed,
            )
        )

    _ensure_parent_dir(save_path)
    save_results(results=results, filepath=save_path)
    return results


def run_mitigation_experiment(
    notes: list[ClinicalNote],
    client: OpenAI,
    mitigation: str,
    save_path: str,
) -> list[PipelineResult]:
    """Run the selected mitigation strategy and return persisted results."""

    allowed = {"negation_aware_prompting", "bert_gate", "self_correction"}
    if mitigation not in allowed:
        raise ValueError(f"mitigation must be one of {sorted(allowed)}")

    if mitigation == "bert_gate":
        print("BERT gate not yet implemented")
        return []

    results: list[PipelineResult] = []

    if mitigation == "negation_aware_prompting":
        step1_outputs = run_negation_aware_extraction(notes=notes, client=client)

        for note, step1 in tqdm(
            list(zip(notes, step1_outputs, strict=False)),
            desc="Mitigation 1 (negation-aware prompting)",
        ):
            step2 = populate_evidence_form(step1=step1, note=note, client=client)
            step3 = match_criteria(step2=step2, client=client)
            step4 = generate_verdict(step3=step3, client=client)
            results.append(
                PipelineResult(
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
            )

    if mitigation == "self_correction":
        for note in tqdm(notes, desc="Mitigation 3 (self-correction)"):
            initial_step1 = extract_findings(note=note, client=client)
            corrected_step1 = _run_self_correction_step1(
                note=note,
                baseline_step1=initial_step1,
                client=client,
            )
            step2 = populate_evidence_form(step1=corrected_step1, note=note, client=client)
            step3 = match_criteria(step2=step2, client=client)
            step4 = generate_verdict(step3=step3, client=client)
            results.append(
                PipelineResult(
                    note_id=note.id,
                    negation_type=note.negation_type,
                    finding=note.finding,
                    ground_truth_present=note.ground_truth_present,
                    step1=corrected_step1,
                    step2=step2,
                    step3=step3,
                    step4=step4,
                    cascade_reached_step4=False,
                )
            )

    _ensure_parent_dir(save_path)
    save_results(results=results, filepath=save_path)
    return results
