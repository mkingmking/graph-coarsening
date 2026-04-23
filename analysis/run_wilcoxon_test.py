"""
Wilcoxon Signed-Rank Test for paired distance observations.

Tests whether coarsening significantly improves (Greedy) or degrades (Savings)
solution distance compared to the same solver on the uncoarsened graph.

Usage:
    python -m graph_coarsening.run_wilcoxon_test

Output:
    - Per-solver test results (statistic, p-value, interpretation)
    - Paired distance table (uncoarsened vs coarsened, per instance)
"""

import json
from pathlib import Path
from scipy.stats import wilcoxon

RESULTS_PATH = Path(__file__).resolve().parent.parent / "outputs" / "results_classical_final.json"

SOLVER_PAIRS = [
    ("Uncoarsened Greedy",   "Inflated Greedy",   "Greedy"),
    ("Uncoarsened Savings",  "Inflated Savings",  "Savings"),
]


def load_paired_distances(results: dict, unc_key: str, coars_key: str):
    pairs = []
    for instance_path, res in results.items():
        if unc_key not in res or coars_key not in res:
            continue
        d_unc   = res[unc_key].get("total_distance")
        d_coars = res[coars_key].get("total_distance")
        if d_unc is None or d_coars is None:
            continue
        instance_name = Path(instance_path).stem
        pairs.append((instance_name, d_unc, d_coars))
    return pairs


def run_test(pairs, solver_name):
    print(f"\n{'='*60}")
    print(f"  {solver_name} — Wilcoxon Signed-Rank Test (distance)")
    print(f"{'='*60}")
    print(f"  {'Instance':<12} {'Uncoarsened':>14} {'Coarsened':>12} {'Diff':>10} {'Better':>8}")
    print(f"  {'-'*12} {'-'*14} {'-'*12} {'-'*10} {'-'*8}")

    uncoarsened = []
    coarsened   = []

    for name, d_unc, d_coars in sorted(pairs):
        diff = d_unc - d_coars          # positive = coarsening improved
        better = "✓" if diff > 0 else ("=" if diff == 0 else "✗")
        print(f"  {name:<12} {d_unc:>14.2f} {d_coars:>12.2f} {diff:>10.2f} {better:>8}")
        uncoarsened.append(d_unc)
        coarsened.append(d_coars)

    n = len(pairs)
    improvements = sum(1 for u, c in zip(uncoarsened, coarsened) if u > c)
    degradations = sum(1 for u, c in zip(uncoarsened, coarsened) if u < c)
    ties         = n - improvements - degradations

    print(f"\n  N = {n}  |  Improvements: {improvements}  |  Degradations: {degradations}  |  Ties: {ties}")

    diffs = [u - c for u, c in zip(uncoarsened, coarsened)]

    # Wilcoxon requires at least one non-zero difference
    non_zero = [d for d in diffs if d != 0]
    if len(non_zero) == 0:
        print("  All differences are zero — test cannot be run.")
        return

    stat, p = wilcoxon(uncoarsened, coarsened, alternative='two-sided')

    print(f"\n  Wilcoxon statistic : {stat:.4f}")
    print(f"  p-value (2-sided)  : {p:.6f}")

    if p < 0.001:
        sig = "*** (p < 0.001)"
    elif p < 0.01:
        sig = "**  (p < 0.01)"
    elif p < 0.05:
        sig = "*   (p < 0.05)"
    else:
        sig = "    (not significant at α=0.05)"

    print(f"  Significance       : {sig}")

    median_unc   = sorted(uncoarsened)[n // 2]
    median_coars = sorted(coarsened)[n // 2]
    median_pct   = (median_unc - median_coars) / median_unc * 100

    print(f"\n  Median uncoarsened : {median_unc:.2f}")
    print(f"  Median coarsened   : {median_coars:.2f}")
    print(f"  Median improvement : {median_pct:+.2f}%")

    direction = "improvement" if median_pct > 0 else "degradation"
    print(f"\n  Interpretation: Coarsening produces a statistically significant {direction}")
    print(f"  in {solver_name} solution distance ({sig.strip()}).")


def main():
    with open(RESULTS_PATH) as f:
        results = json.load(f)

    print(f"\nLoaded {len(results)} instances from {RESULTS_PATH.name}")

    for unc_key, coars_key, label in SOLVER_PAIRS:
        pairs = load_paired_distances(results, unc_key, coars_key)
        if not pairs:
            print(f"\nNo paired data found for {label} — check key names in JSON.")
            continue
        run_test(pairs, label)

    print(f"\n{'='*60}")
    print("  Done.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
