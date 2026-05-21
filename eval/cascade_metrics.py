"""Utilities for measuring negation error cascades in pipeline outputs."""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from scipy import stats

from pipeline.schemas import PipelineResult


def compute_cascade_rate(results: list[PipelineResult]) -> dict:
    """Compute aggregate and stratified cascade statistics for one run.

    Args:
        results: Pipeline results for notes where a step-1 negation error may have
            been injected.

    Returns:
        Dictionary containing overall and grouped cascade rates, step survival
        rates, and corpus-level counts.
    """

    total_notes = len(results)
    total_errors_injected = total_notes

    cascade_count = sum(1 for result in results if result.cascade_reached_step4)

    by_negation_totals: dict[str, int] = Counter(result.negation_type for result in results)
    by_negation_cascades: dict[str, int] = Counter(
        result.negation_type for result in results if result.cascade_reached_step4
    )

    by_finding_totals: dict[str, int] = Counter(result.finding for result in results)
    by_finding_cascades: dict[str, int] = Counter(
        result.finding for result in results if result.cascade_reached_step4
    )

    cascade_by_negation_type = {
        negation_type: (
            by_negation_cascades.get(negation_type, 0) / count if count else 0.0
        )
        for negation_type, count in by_negation_totals.items()
    }

    cascade_by_finding = {
        finding: (by_finding_cascades.get(finding, 0) / count if count else 0.0)
        for finding, count in by_finding_totals.items()
    }

    step_survival_counts: dict[str, int] = defaultdict(int)
    for result in results:
        if result.step1.error:
            step_survival_counts["step1"] += 1
        if result.step2.error:
            step_survival_counts["step2"] += 1
        if result.step3.error:
            step_survival_counts["step3"] += 1
        if result.cascade_reached_step4:
            step_survival_counts["step4"] += 1

    step_survival_rate = {
        step: (step_survival_counts[step] / total_errors_injected if total_errors_injected else 0.0)
        for step in ("step1", "step2", "step3", "step4")
    }

    return {
        "overall_cascade_rate": cascade_count / total_notes if total_notes else 0.0,
        "cascade_by_negation_type": cascade_by_negation_type,
        "cascade_by_finding": cascade_by_finding,
        "step_survival_rate": step_survival_rate,
        "total_notes": total_notes,
        "total_errors_injected": total_errors_injected,
    }


def compute_verdict_shift(
    clean_results: list[PipelineResult],
    corrupted_results: list[PipelineResult],
) -> dict:
    """Measure step-4 verdict changes between clean and corrupted runs.

    Args:
        clean_results: Pipeline results from the clean run.
        corrupted_results: Pipeline results from the run with injected errors.

    Returns:
        Dictionary with verdict change rate, detailed transition counts, and
        changed/unchanged totals.
    """

    clean_by_id = {result.note_id: result for result in clean_results}
    corrupted_by_id = {result.note_id: result for result in corrupted_results}
    shared_ids = sorted(set(clean_by_id) & set(corrupted_by_id))

    change_breakdown: dict[str, int] = Counter()
    changed = 0
    unchanged = 0

    for note_id in shared_ids:
        from_verdict = clean_by_id[note_id].step4.verdict
        to_verdict = corrupted_by_id[note_id].step4.verdict
        if from_verdict == to_verdict:
            unchanged += 1
            continue
        changed += 1
        change_breakdown[f"{from_verdict}_to_{to_verdict}"] += 1

    total = len(shared_ids)
    return {
        "verdict_change_rate": changed / total if total else 0.0,
        "change_breakdown": dict(change_breakdown),
        "unchanged": unchanged,
        "changed": changed,
        "total": total,
    }


def run_mcnemar_test(baseline_errors: list[bool], mitigation_errors: list[bool]) -> dict:
    """Run McNemar's test comparing paired baseline and mitigation errors.

    Args:
        baseline_errors: Boolean error flags from the baseline condition.
        mitigation_errors: Boolean error flags from the mitigation condition.

    Returns:
        Dictionary containing McNemar statistic, p-value, significance flag, and
        an interpretation string.

    Raises:
        ValueError: If paired lists have mismatched lengths.
    """

    if len(baseline_errors) != len(mitigation_errors):
        raise ValueError("baseline_errors and mitigation_errors must be the same length.")

    baseline = np.asarray(baseline_errors, dtype=bool)
    mitigation = np.asarray(mitigation_errors, dtype=bool)

    both_correct = int(np.sum(~baseline & ~mitigation))
    baseline_only_error = int(np.sum(baseline & ~mitigation))
    mitigation_only_error = int(np.sum(~baseline & mitigation))
    both_error = int(np.sum(baseline & mitigation))

    contingency = np.array(
        [
            [both_correct, mitigation_only_error],
            [baseline_only_error, both_error],
        ],
        dtype=int,
    )

    if hasattr(stats, "mcnemar"):
        test_result = stats.mcnemar(contingency, correction=True)
        statistic = float(test_result.statistic)
        p_value = float(test_result.pvalue)
    else:
        discordant = baseline_only_error + mitigation_only_error
        if discordant == 0:
            statistic = 0.0
            p_value = 1.0
        else:
            statistic = float((abs(baseline_only_error - mitigation_only_error) - 1) ** 2 / discordant)
            p_value = float(stats.chi2.sf(statistic, df=1))

    significant = p_value < 0.05
    if significant and baseline_only_error > mitigation_only_error:
        interpretation = f"Mitigation significantly reduces cascade rate (p={p_value:.3g})."
    elif significant and baseline_only_error < mitigation_only_error:
        interpretation = f"Mitigation significantly increases cascade rate (p={p_value:.3g})."
    else:
        interpretation = f"No statistically significant cascade-rate difference (p={p_value:.3g})."

    return {
        "statistic": statistic,
        "p_value": p_value,
        "significant": significant,
        "interpretation": interpretation,
    }


def results_to_dataframe(results: list[PipelineResult]) -> pd.DataFrame:
    """Convert a sequence of pipeline results into a tabular dataframe."""

    return pd.DataFrame([result.to_dict() for result in results])


def save_results(results: list[PipelineResult], filepath: str) -> None:
    """Save pipeline results as a CSV file."""

    df = results_to_dataframe(results)
    df.to_csv(filepath, index=False)
    print(f"Saved {len(results)} results to {filepath}")


def load_results(filepath: str) -> pd.DataFrame:
    """Load pipeline results from a CSV file."""

    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} results from {filepath}")
    return df
