"""Replaying prompt experiments against recorded answers.

For every (model, task) in HELM Classic's prompt ablations we know how each prompt format
scored on each question. A replay sends requests (questions drawn uniformly, with
replacement), lets the experiment assign each one a variant, and scores it with that
variant's recorded result. The experiment code is the production code; only the traffic is
recorded.

"Shortfall" is the score lost against serving the best prompt all along, in answers for a
0/1 score: during the experiment, then on every remaining request of the horizon while the
chosen prompt serves.
"""
import json
from pathlib import Path

import numpy as np

from . import data
from .experiment import Experiment

HORIZON = 20_000
RESULTS = Path(__file__).resolve().parent.parent / "results"


def matrix(e):
    """(variant names, scores as variants x questions) over the questions every variant answered."""
    names = list(e["scores"])
    qs = sorted(set.intersection(*(set(s) for s in e["scores"].values())))
    return names, np.array([[e["scores"][v][q] for q in qs] for v in names])


def replay(names, scores, allocation, seed, tolerance=0.01, horizon=HORIZON):
    ex = Experiment("replay", names, allocation=allocation, seed=seed, tolerance=tolerance, max_requests=horizon)
    questions = np.random.default_rng([seed, 1])
    mu = scores.mean(1)
    shortfall = 0.0
    while ex.state == "running":
        arms = ex.draw(ex.check_every)
        shortfall += (mu.max() - mu[arms]).sum()
        ex.record_many(arms, scores[arms, questions.integers(scores.shape[1], size=len(arms))])
    chosen = names.index(ex.winner)
    shortfall += (horizon - ex.seen) * (mu.max() - mu[chosen])
    return {"state": ex.state, "requests": ex.seen, "winner": ex.winner, "gap": float(mu.max() - mu[chosen]),
            "shortfall": float(shortfall), "dropped": int((~ex.alive).sum()), "log": ex.log}


def bench(runs=20):
    experiments = data.load()
    pct = lambda x: f"{100 * x:.1f}%"
    lines, spread = ["## How much the prompt matters", "",
                     "Score of the best prompt format against HELM's default format, same model and questions.", "",
                     "| Task | Experiments | Best beats default by 2+ points | Median gain | Largest gain | Worst format vs best (median) |",
                     "|---|---|---|---|---|---|"], {}
    for task in data.TASKS:
        gains, worst = [], []
        for e in experiments.values():
            if e["task"] == task:
                names, s = matrix(e)
                mu = dict(zip(names, s.mean(1)))
                gains.append(max(mu.values()) - mu.get("default", mu[names[0]]))
                worst.append(max(mu.values()) - min(mu.values()))
        spread[task] = {"gains": gains, "worst": worst}
        g = np.array(gains)
        lines.append(f"| {task} | {len(g)} | {(g >= 0.02).sum()} | {100 * np.median(g):.1f} | {100 * g.max():.1f} | "
                     f"{100 * np.median(worst):.1f} |")

    rows, results = {}, {}
    for name, e in sorted(experiments.items()):
        names, s = matrix(e)
        mu = s.mean(1)
        default = names.index("default") if "default" in names else 0
        results[name] = {a: [replay(names, s, a, seed) for seed in range(runs)] for a in ("uniform", "thompson")}
        rows.setdefault("keep the default", []).append({"gap": mu.max() - mu[default], "shortfall": HORIZON * (mu.max() - mu[default]),
                                                        "requests": 0})
        for a, rs in results[name].items():
            rows.setdefault(a, []).extend(rs)
    n_exp = len(experiments)
    lines += ["", "## Running the experiment", "",
              f"{n_exp} experiments x {runs} replays, {HORIZON:,} requests each (the chosen prompt serves once the test "
              "ends). Shortfall: score lost against the best prompt, per 1,000 requests.", "",
              "| Strategy | Chose a prompt within 1 point of the best | Decided before the horizon | Median requests to decide | "
              "Mean shortfall per 1,000 requests |", "|---|---|---|---|---|"]
    table = []
    for label, rs in rows.items():
        decided = [r["requests"] for r in rs if r.get("state") == "decided"]
        row = {"strategy": label, "within_1_point": float(np.mean([r["gap"] <= 0.01 for r in rs])),
               "decided": float(np.mean([r.get("state") == "decided" for r in rs])) if label != "keep the default" else None,
               "median_requests": float(np.median(decided)) if decided else None,
               "shortfall_per_1000": float(np.mean([r["shortfall"] for r in rs]) / HORIZON * 1000)}
        table.append(row)
        decided_pct = pct(row["decided"]) if row["decided"] is not None else "—"
        median = f"{row['median_requests']:,.0f}" if row["median_requests"] else "—"
        lines.append(f"| {label} | {pct(row['within_1_point'])} | {decided_pct} | {median} | {row['shortfall_per_1000']:.1f} |")

    aa = []
    lines += ["", "## Identical variants (A/A): false drops", "",
              "Every variant replaced by the default's recorded answers, tolerance 0: any drop is a false alarm. Target: under 5%.", "",
              "| Allocation | Runs with a false drop | 95% interval |", "|---|---|---|"]
    for a in ("uniform", "thompson"):
        k = n = 0
        for e in experiments.values():
            names, s = matrix(e)
            same = np.repeat(s[[0]], len(names), 0)
            for seed in range(runs):
                k += replay(names, same, a, 10_000 + seed, tolerance=0.0)["dropped"] > 0
                n += 1
        lo, hi = wilson(k, n)
        aa.append({"allocation": a, "false": int(k), "runs": n})
        lines.append(f"| {a} | {k}/{n} ({pct(k / n)}) | {pct(lo)}–{pct(hi)} |")
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "bench.md").write_text("\n".join(lines) + "\n")
    (RESULTS / "summary.json").write_text(json.dumps({"horizon": HORIZON, "runs": runs, "spread": spread, "strategies": table,
                                                      "identical": aa}, indent=1))
    return "\n".join(lines)


def wilson(k, n, z=1.96):
    p, d = k / n, 1 + z * z / n
    mid, half = (p + z * z / (2 * n)) / d, z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, mid - half), min(1.0, mid + half)
