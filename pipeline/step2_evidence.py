"""Step 2 evidence form population from step 1 extractions."""

from __future__ import annotations

import json
import time

from openai import OpenAI

from pipeline.schemas import ClinicalNote, Step1Output, Step2Output

SYSTEM_PROMPT = (
    "You are a clinical prior-authorization evidence assistant. "
    "Review the extracted findings from step 1 and fill a structured prior authorization evidence form. "
    "Be precise and only include findings explicitly supported by the extraction. "
    "Return ONLY valid JSON, no preamble. "
    "Use this schema exactly: "
    '{"diagnosis": str, "supporting_findings": list[str], "contraindications": list[str], '
    '"clinical_urgency": "routine" | "urgent" | "emergent", "documentation_complete": bool}'
)


def populate_evidence_form(
    step1: Step1Output,
    note: ClinicalNote,
    client: OpenAI,
    model: str = "gpt-4o",
) -> Step2Output:
    """Populate a structured evidence form from step 1 extraction and source note."""

    last_raw_response = ""

    user_payload = {
        "clinical_note_text": note.text,
        "step1_extracted_findings": step1.extracted_findings,
    }

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=0,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": json.dumps(user_payload, ensure_ascii=False, indent=2),
                    },
                ],
            )
            usage = getattr(response, "usage", None)
            token_usage = getattr(usage, "total_tokens", "unknown")
            print(f"[STEP2] note_id={note.id} model={model} tokens={token_usage}")

            raw_response = response.choices[0].message.content or ""
            last_raw_response = raw_response

            try:
                parsed = json.loads(raw_response)
            except json.JSONDecodeError:
                if attempt < 2:
                    time.sleep(2)
                    continue
                return Step2Output(
                    note_id=note.id,
                    evidence_form={},
                    raw_response=last_raw_response,
                    error="json_parse_failed",
                )

            evidence_form = parsed if isinstance(parsed, dict) else {}
            return Step2Output(
                note_id=note.id,
                evidence_form=evidence_form,
                raw_response=raw_response,
                error=None,
            )
        except Exception as exc:  # noqa: BLE001
            if attempt < 2:
                time.sleep(2)
                continue
            return Step2Output(
                note_id=note.id,
                evidence_form={},
                raw_response=last_raw_response,
                error=f"api_call_failed: {exc}",
            )

    return Step2Output(
        note_id=note.id,
        evidence_form={},
        raw_response=last_raw_response,
        error="json_parse_failed",
    )
