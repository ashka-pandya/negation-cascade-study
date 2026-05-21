"""Step 1 extraction for clinical findings with negation handling."""

from __future__ import annotations

import json
import time
from typing import Any

from openai import OpenAI
from tqdm import tqdm

from pipeline.schemas import ClinicalNote, Step1Output

BASE_SYSTEM_PROMPT = (
    "You are a clinical NLP extraction assistant. "
    "Extract all clinical findings mentioned in the text. "
    "For each finding, determine if it is PRESENT or ABSENT. "
    "Pay close attention to negation language including: "
    "'no evidence of', 'ruled out', 'negative for', 'cannot exclude', "
    "'not without', 'possible', 'no signs of', 'without'. "
    "Return ONLY valid JSON, no preamble or explanation. "
    "Use this schema exactly: "
    "{\"findings\": [{\"finding\": str, \"present\": bool, "
    "\"confidence\": \"high\" | \"medium\" | \"low\"}]}"
)

NEGATION_AWARE_FEW_SHOT = (
    "Few-shot negation examples:\n"
    "Example 1:\n"
    "Text: \"Chest X-ray shows no evidence of pneumonia.\"\n"
    "Correct extraction: pneumonia present=false\n\n"
    "Example 2:\n"
    "Text: \"DVT was ruled out following ultrasound.\"\n"
    "Correct extraction: DVT present=false\n\n"
    "Example 3:\n"
    "Text: \"We cannot exclude pulmonary embolism at this time.\"\n"
    "Correct extraction: pulmonary embolism present=false (uncertain)\n\n"
    "Example 4:\n"
    "Text: \"Patient presents with fever, not without signs of early consolidation.\"\n"
    "Correct extraction: consolidation present=true (double negation)"
)


def _extract_findings_with_prompt(
    note: ClinicalNote,
    client: OpenAI,
    model: str,
    system_prompt: str,
) -> Step1Output:
    """Call the LLM for finding extraction with retries and JSON validation."""

    last_raw_response = ""

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=0,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": note.text},
                ],
            )
            usage = getattr(response, "usage", None)
            token_usage = getattr(usage, "total_tokens", "unknown")
            print(f"[STEP1] note_id={note.id} model={model} tokens={token_usage}")

            raw_response = response.choices[0].message.content or ""
            last_raw_response = raw_response

            try:
                parsed = json.loads(raw_response)
            except json.JSONDecodeError:
                if attempt < 2:
                    time.sleep(2)
                    continue
                return Step1Output(
                    note_id=note.id,
                    extracted_findings=[],
                    raw_response=last_raw_response,
                    error="json_parse_failed",
                )

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
                extracted_findings=[],
                raw_response=last_raw_response,
                error=f"api_call_failed: {exc}",
            )

    return Step1Output(
        note_id=note.id,
        extracted_findings=[],
        raw_response=last_raw_response,
        error="json_parse_failed",
    )


def extract_findings(
    note: ClinicalNote,
    client: OpenAI,
    model: str = "gpt-4o",
) -> Step1Output:
    """Extract findings for a single note using the baseline prompt."""

    return _extract_findings_with_prompt(
        note=note,
        client=client,
        model=model,
        system_prompt=BASE_SYSTEM_PROMPT,
    )


def run_baseline_extraction(
    notes: list[ClinicalNote],
    client: OpenAI,
    model: str = "gpt-4o",
) -> list[Step1Output]:
    """Run baseline extraction across a list of notes with a progress bar."""

    outputs: list[Step1Output] = []
    for note in tqdm(notes, desc="Step 1 extraction"):
        try:
            outputs.append(extract_findings(note=note, client=client, model=model))
        except Exception as exc:  # noqa: BLE001
            print(f"[STEP1][ERROR] note_id={note.id} error={exc}")
            outputs.append(
                Step1Output(
                    note_id=note.id,
                    extracted_findings=[],
                    raw_response="",
                    error=f"unexpected_exception: {exc}",
                )
            )
    return outputs


# MITIGATION 1

def run_negation_aware_extraction(
    notes: list[ClinicalNote],
    client: OpenAI,
    model: str = "gpt-4o",
) -> list[Step1Output]:
    """Run extraction with negation-aware few-shot prompting."""

    outputs: list[Step1Output] = []
    negation_prompt = f"{BASE_SYSTEM_PROMPT}\n\n{NEGATION_AWARE_FEW_SHOT}"

    for note in tqdm(notes, desc="Step 1 extraction"):
        try:
            outputs.append(
                _extract_findings_with_prompt(
                    note=note,
                    client=client,
                    model=model,
                    system_prompt=negation_prompt,
                )
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[STEP1][ERROR] note_id={note.id} error={exc}")
            outputs.append(
                Step1Output(
                    note_id=note.id,
                    extracted_findings=[],
                    raw_response="",
                    error=f"unexpected_exception: {exc}",
                )
            )
    return outputs
