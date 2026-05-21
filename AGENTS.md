# AGENTS.md — Negation Cascade Research Project

## Project
Research project studying negation error propagation in multi-step 
agentic clinical NLP pipelines. Target: academic publication (arXiv, 
ACL BioNLP, ML4H).

## Stack
- Python 3.10
- PyTorch 2.x
- HuggingFace Transformers (ClinicalBERT fine-tuning)
- OpenAI API (GPT-4o for pipeline steps)
- pandas, scikit-learn, matplotlib, seaborn
- Google Colab Pro (T4/V100) for training runs

## Folder Structure
negation-cascade-study/
├── data/
│   ├── synthetic/          # ChatGPT-generated clinical notes
│   └── processed/          # tokenized, ready for training
├── pipeline/
│   ├── step1_extraction.py # LLM entity extraction
│   ├── step2_evidence.py   # evidence field population
│   ├── step3_criteria.py   # guideline criteria matching
│   └── step4_verdict.py    # readiness verdict
├── models/
│   └── bert_classifier/    # ClinicalBERT fine-tuning code
├── experiments/
│   ├── baseline/           # no mitigation
│   ├── mitigation1/        # negation-aware prompting
│   ├── mitigation2/        # BERT verification gate
│   └── mitigation3/        # LLM self-correction
├── eval/
│   ├── cascade_metrics.py  # cascade rate measurement
│   └── stats.py            # McNemar test, CIs
├── notebooks/
│   └── *.ipynb             # Colab notebooks
├── figures/                # paper-ready plots
├── AGENTS.md
└── README.md

## Coding Conventions
- All functions must have docstrings
- All LLM calls must be wrapped in try/except with retry logic
- Save intermediate results to data/processed/ — never recompute
- Every experiment must log: model version, prompt version, timestamp
- Use dataclasses for pipeline step inputs/outputs
- No hardcoded API keys — use os.environ

## What Codex Should NOT Do
- Do not modify notebooks/ — those are edited in Colab directly
- Do not change requirements.txt without flagging it
- Do not use async — keep all LLM calls synchronous for clarity
- Do not generate fake results or mock data silently
