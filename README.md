# Negation Error Propagation in Multi-Step Agentic Clinical Pipelines

> **Research project** — studying how negation errors at step 1 of a clinical prior authorization pipeline cascade through to a final verdict, and which mitigation strategies most effectively reduce downstream decision errors.

**Author:** Ashka Pandya  
**Program:** MS in Artificial Intelligence, University of Texas at Austin  
**Status:** Active — data collection phase  
**Target venues:** ACL BioNLP · EMNLP Clinical NLP · AMIA · ML4H

---

## The Research Question

> *"How do negation errors in LLM-based clinical entity extraction propagate through a multi-step prior authorization pipeline, and which mitigation strategies most effectively reduce downstream decision errors — measured at the final verdict level, not just step-1 extraction accuracy?"*

Clinical text is full of negations. A radiology report might say **"no evidence of pneumonia"** — but an LLM may extract *pneumonia* as a positive finding anyway. In a single-step system, this is a known problem. What nobody has studied is what happens when that error feeds into step 2, which feeds into step 3, which produces a final clinical decision. Does the error survive? Does it amplify? Can it change a prior authorization verdict from "insufficient evidence" to "approved"?

This project measures that cascade and tests three strategies to stop it.

---

## Why This Is Novel

Prior work on clinical negation detection studies extractors **in isolation** — a BERT model on i2b2, or an LLM on MedNLI, measured by F1 score on the extraction task alone. No prior work has:

1. Measured how a step-1 negation error **propagates through a multi-step agentic pipeline**
2. Measured the **decision-level impact** (verdict change rate), not just extraction F1
3. Compared mitigations on **cascade reduction rate** specifically
4. Studied this in the context of **prior authorization workflows**

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    CLINICAL NOTE (input text)                        │
│  "Chest X-ray shows no evidence of pneumonia. DVT ruled out."        │
└─────────────────────────┬───────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 1 — Entity Extraction                           [LLM call]    │
│                                                                      │
│  Input:  raw clinical text                                           │
│  Output: structured JSON                                             │
│          {findings: [{finding: "pneumonia",                          │
│                       present: false,   ← correct                   │
│                       confidence: "high"}]}                          │
│                                                                      │
│  ⚠ NEGATION ERROR injected here for cascade experiment:             │
│     present: false  →  present: true                                │
└─────────────────────────┬───────────────────────────────────────────┘
                          │ extracted findings JSON
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 2 — Evidence Form Population                    [LLM call]    │
│                                                                      │
│  Input:  step 1 JSON + original note text                            │
│  Output: structured prior-auth evidence form                         │
│          {diagnosis: "...",                                          │
│           supporting_findings: [...],                                │
│           contraindications: [...],    ← error may propagate here   │
│           clinical_urgency: "routine",                               │
│           documentation_complete: true}                              │
└─────────────────────────┬───────────────────────────────────────────┘
                          │ evidence form JSON
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 3 — Guideline Criteria Matching                 [LLM call]    │
│                                                                      │
│  Input:  evidence form                                               │
│  Criteria evaluated (5 binary checks):                               │
│    ✓ Diagnosis clearly documented                                    │
│    ✓ At least one supporting finding                                 │
│    ✓ No absolute contraindications     ← error may flip this        │
│    ✓ Clinical urgency documented                                     │
│    ✓ Documentation complete                                          │
│  Output: {criterion_1: bool, ..., criteria_met: int}                │
└─────────────────────────┬───────────────────────────────────────────┘
                          │ criteria scores
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 4 — Readiness Verdict                           [LLM call]    │
│                                                                      │
│  Input:  criteria scores                                             │
│  Rules:  5/5 met → "approved"                                        │
│          3-4/5   → "needs_clarification"                             │
│          0-2/5   → "insufficient_evidence"                           │
│                                                                      │
│  Output: {verdict: "approved" | "insufficient_evidence"             │
│                   | "needs_clarification",                           │
│           reasoning: "..."}                                          │
│                                                                      │
│  ← CASCADE MEASURED HERE: did verdict change vs clean run?          │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Experiment Design

### Condition 1 — Baseline (clean run)
Run all 100 synthetic notes through the full pipeline with no injected errors. Record step 4 verdicts as ground truth.

### Condition 2 — Cascade measurement
For each note, inject a negation error at step 1 (flip `present: false` → `present: true` for the target finding). Run steps 2→3→4 with the corrupted step 1 output. Measure what % of injected errors reach step 4 and change the verdict.

### Condition 3 — Mitigation 1: Negation-aware prompting
Rewrite step 1 system prompt with explicit negation handling instructions and few-shot examples. Measure cascade rate reduction vs Condition 2.

### Condition 4 — Mitigation 2: ClinicalBERT verification gate
Insert a fine-tuned ClinicalBERT classifier between steps 1 and 2. If BERT classifies a finding as "absent", override step 1's extracted value. Measure cascade rate reduction.

### Condition 5 — Mitigation 3: LLM self-correction
After step 1, add a second LLM call that reviews the extraction and flags potential negation errors. Use corrected output for steps 2→3→4. Measure cascade rate reduction.

---

## Dataset

100 synthetic clinical note excerpts generated via LLM, covering:

| Negation type | Count | Example |
|---|---|---|
| Simple negation | 25 | "no evidence of pneumonia" |
| Ruled-out | 25 | "DVT was ruled out" |
| Hedge/uncertainty | 25 | "cannot exclude pulmonary embolism" |
| Double negation | 25 | "not without signs of consolidation" |

10 clinical findings × 10 examples each:
pneumonia, pulmonary embolism, DVT, pleural effusion, pneumothorax, consolidation, atelectasis, cardiomegaly, fracture, malignancy

All notes have `ground_truth_present: false` — every finding is negated.

---

## Three Contributions

| # | Contribution | Description |
|---|---|---|
| C1 | Cascade measurement framework | A reusable methodology for injecting controlled negation errors and measuring decision drift across pipeline steps |
| C2 | Empirical cascade rate findings | First measurement of negation error survival rate across a 4-step clinical agentic pipeline |
| C3 | Mitigation comparison | Head-to-head comparison of 3 mitigation strategies on cascade reduction rate with statistical significance testing |

---

## Repo Structure

```
negation-cascade-study/
│
├── data/
│   ├── synthetic/          # 100 LLM-generated clinical notes (JSON)
│   └── processed/          # tokenized data for BERT training
│
├── pipeline/
│   ├── schemas.py          # dataclasses: ClinicalNote, PipelineResult, etc.
│   ├── step1_extraction.py # LLM entity extraction (baseline + mitigation 1)
│   ├── step2_evidence.py   # evidence form population
│   ├── step3_criteria.py   # guideline criteria matching
│   ├── step4_verdict.py    # readiness verdict generation
│   └── run_pipeline.py     # orchestrator: runs all 4 steps
│
├── models/
│   └── bert_classifier/    # ClinicalBERT fine-tuning code (written in Colab)
│
├── experiments/
│   ├── error_injection.py  # injects controlled negation errors at step 1
│   ├── run_experiments.py  # runs all 5 conditions
│   ├── baseline/           # results from clean + cascade runs
│   ├── mitigation1/        # negation-aware prompting results
│   ├── mitigation2/        # BERT gate results
│   └── mitigation3/        # self-correction results
│
├── eval/
│   ├── cascade_metrics.py  # cascade rate, verdict shift, McNemar test
│   └── __init__.py
│
├── notebooks/              # Colab notebooks for training + experiments
├── figures/                # publication-ready plots
│
├── AGENTS.md               # Codex instructions
├── requirements.txt
├── .env.example
└── README.md
```

---

## Technical Stack

| Component | Tool |
|---|---|
| Pipeline LLM | GPT-4o via OpenAI API |
| Negation classifier | ClinicalBERT (emilyalsentzer/Bio_ClinicalBERT) |
| Training framework | PyTorch — manual training loop |
| Compute | Google Colab Pro (T4/V100) |
| Code generation | OpenAI Codex (scaffolding only) |
| Experiment tracking | CSV logs + pandas |
| Statistical testing | McNemar's test via scipy.stats |
| Visualization | matplotlib + seaborn |


---

## Key Results (to be filled)

| Condition | Cascade rate | vs baseline |
|---|---|---|
| Baseline cascade | TBD | — |
| Mitigation 1 — negation-aware prompting | TBD | TBD |
| Mitigation 2 — ClinicalBERT gate | TBD | TBD |
| Mitigation 3 — LLM self-correction | TBD | TBD |

---

## Reproducibility

All experiments use `temperature=0` for deterministic LLM outputs.  
All results saved as CSV in `experiments/` folders.  
BERT training seeds set explicitly in the training notebook.  
To reproduce: clone repo, add `OPENAI_API_KEY` to environment, run notebooks in order.
