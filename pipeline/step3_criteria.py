"""Step 3 criteria matching from step 2 evidence form."""

from __future__ import annotations

import json
import time

from openai import OpenAI

from pipeline.schemas import Step2Output, Step3Output

CRITERIA = {
    "criterion_1": "Diagnosis is clearly documented",
    "criterion_2": "At least one supporting finding present",
    "criterion_3": "No absolute contraindications documented",
    "criterion_4": "Clinical urgency is documented",
    "criterion_5": "Documentation is complete",
}

SYSTEM_PROMPT = (
    "You are a clinical prior-authorization criteria evaluator. "
    "Review the evidence form from step 2 and evaluate it against 5 prior authorization criteria. "
    "Return ONLY valid JSON, no preamble. "
    "Use this schema exactly: "
    '{"criterion_1": bool, "criterion_2": bool, "criterion_3": bool, '
    '"criterion_4": bool, "criterion_5": bool, "criteria_met": int}'
)


def match_criteria(
    step2: Step2Output,
    client: OpenAI,
    model: str = "gpt-4o",
) -> Step3Output:
    """Evaluate step 2 evidence form against hardcoded prior-auth criteria."""

    last_raw_response = ""
    note_id = step2.note_id

    user_payload = {
        "evidence_form": step2.evidence_form,
        "criteria": CRITERIA,
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
            print(f"[STEP3] note_id={note_id} model={model} tokens={token_usage}")

            raw_response = response.choices[0].message.content or ""
            last_raw_response = raw_response

            try:
                parsed = json.loads(raw_response)
            except json.JSONDecodeError:
                if attempt < 2:
                    time.sleep(2)
                    continue
                return Step3Output(
                    note_id=note_id,
                    criteria_scores={},
                    criteria_met=0,
                    raw_response=last_raw_response,
                    error="json_parse_failed",
                )

            if not isinstance(parsed, dict):
                parsed = {}

            criteria_scores = {
                key: bool(parsed.get(key, False))
                for key in ["criterion_1", "criterion_2", "criterion_3", "criterion_4", "criterion_5"]
            }
            criteria_met = int(parsed.get("criteria_met", sum(criteria_scores.values())))

            return Step3Output(
                note_id=note_id,
                criteria_scores=criteria_scores,
                criteria_met=criteria_met,
                raw_response=raw_response,
                error=None,
            )
        except Exception as exc:  # noqa: BLE001
            if attempt < 2:
                time.sleep(2)
                continue
            return Step3Output(
                note_id=note_id,
                criteria_scores={},
                criteria_met=0,
                raw_response=last_raw_response,
                error=f"api_call_failed: {exc}",
            )

    return Step3Output(
        note_id=note_id,
        criteria_scores={},
        criteria_met=0,
        raw_response=last_raw_response,
        error="json_parse_failed",
    )
