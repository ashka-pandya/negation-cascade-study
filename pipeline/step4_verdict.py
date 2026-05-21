"""Step 4 prior-authorization verdict generation from criteria scores."""

from __future__ import annotations

import json
import time

from openai import OpenAI

from pipeline.schemas import Step3Output, Step4Output

SYSTEM_PROMPT = (
    "You are a clinical prior-authorization decision assistant. "
    "Review criteria scores from step 3 and generate a prior authorization verdict. "
    "Apply these rules exactly: 5/5 criteria met => approved; "
    "3-4/5 criteria met => needs_clarification; "
    "0-2/5 criteria met => insufficient_evidence. "
    "Return ONLY valid JSON, no preamble. "
    "Use this schema exactly: "
    '{"verdict": "approved" | "insufficient_evidence" | "needs_clarification", "reasoning": str}'
)


def generate_verdict(
    step3: Step3Output,
    client: OpenAI,
    model: str = "gpt-4o",
) -> Step4Output:
    """Generate a verdict label and reasoning from step 3 criteria outcomes."""

    last_raw_response = ""
    note_id = step3.note_id

    user_payload = {
        "criteria_scores": step3.criteria_scores,
        "criteria_met": step3.criteria_met,
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
            print(f"[STEP4] note_id={note_id} model={model} tokens={token_usage}")

            raw_response = response.choices[0].message.content or ""
            last_raw_response = raw_response

            try:
                parsed = json.loads(raw_response)
            except json.JSONDecodeError:
                if attempt < 2:
                    time.sleep(2)
                    continue
                return Step4Output(
                    note_id=note_id,
                    verdict="insufficient_evidence",
                    reasoning="",
                    raw_response=last_raw_response,
                    error="json_parse_failed",
                )

            if not isinstance(parsed, dict):
                parsed = {}

            verdict = str(parsed.get("verdict", "insufficient_evidence"))
            reasoning = str(parsed.get("reasoning", ""))

            return Step4Output(
                note_id=note_id,
                verdict=verdict,
                reasoning=reasoning,
                raw_response=raw_response,
                error=None,
            )
        except Exception as exc:  # noqa: BLE001
            if attempt < 2:
                time.sleep(2)
                continue
            return Step4Output(
                note_id=note_id,
                verdict="insufficient_evidence",
                reasoning="",
                raw_response=last_raw_response,
                error=f"api_call_failed: {exc}",
            )

    return Step4Output(
        note_id=note_id,
        verdict="insufficient_evidence",
        reasoning="",
        raw_response=last_raw_response,
        error="json_parse_failed",
    )
