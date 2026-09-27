"""python -m promptab.demo   write the live demo's data (docs/data.json): results/summary.json, plus every
experiment's prompt variants with their score distributions over the questions all of them answered.
A replay draws a question per request and scores the variant it serves, so that distribution is all
the page's in-browser experiment needs (F1 scores are binned to 0.05; accuracies are exact)."""
import json
from collections import Counter
from pathlib import Path

from . import data
from .replay import matrix

ROOT = Path(__file__).resolve().parent.parent


def build(out=ROOT / "docs"):
    summary = json.loads((ROOT / "results" / "summary.json").read_text())
    experiments = []
    for name, e in sorted(data.load().items()):
        names, m = matrix(e)
        variants = []
        for v, row in zip(names, m):
            hist = Counter(round(round(float(x) / 0.05) * 0.05, 2) for x in row)
            variants.append({"name": v, "mean": round(float(row.mean()), 4), "prompt": e["prompts"][v],
                             "hist": sorted([k, n] for k, n in hist.items())})
        experiments.append({"name": name, "model": e["model"], "task": e["task"], "questions": int(m.shape[1]), "variants": variants})
    out.mkdir(exist_ok=True)
    (out / "data.json").write_text(json.dumps({"summary": summary, "experiments": experiments}, indent=1))
    print(f"wrote {out / 'data.json'}: {len(experiments)} experiments")


if __name__ == "__main__":
    build()
