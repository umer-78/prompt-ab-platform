"""Recorded answers under different prompt formats, from HELM Classic's prompt ablations
(v0.3.0): the same model on the same questions, with only the prompt's wording changed.

An experiment is one (model, task). Its variants are the formats HELM ran: the default,
an added "expert" instruction, no instruction, and three ways of labelling the input and
output (I:/O:, Input:/Output:, <input>/<output> tags). Variants whose prompt settings turn out
identical to another's are dropped. Each answer's score is from the first few-shot trial:
quasi exact match (0 or 1) for imdb and civil_comments, F1 (0 to 1) for natural_qa. Downloaded on first use into PROMPTAB_DATA (default ~/.cache/promptab).
"""
import json
import os
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

RELEASE = "v0.3.0"
BUCKET = f"https://storage.googleapis.com/crfm-helm-public/classic/benchmark_output/runs/{RELEASE}"
LISTING = "https://storage.googleapis.com/storage/v1/b/crfm-helm-public/o"
TASKS = ("imdb", "civil_comments", "natural_qa")
MODELS = ("together_bloom", "together_glm", "together_gpt-j-6b", "together_gpt-neox-20b", "together_opt-175b",
          "together_opt-66b", "together_t0pp", "together_t5-11b", "together_ul2", "together_yalm")
VARIANTS = {"": "default", "instructions=expert": "expert", "instructions=none": "none", "prompt=i_o": "i_o",
            "prompt=input_output": "input_output", "prompt=input_output_html": "html"}
METRIC = {"imdb": "quasi_exact_match", "civil_comments": "quasi_exact_match", "natural_qa": "f1_score"}
SPEC_KEYS = ("instructions", "input_prefix", "input_suffix", "output_prefix", "output_suffix")


def cache_dir():
    path = Path(os.environ.get("PROMPTAB_DATA", Path.home() / ".cache" / "promptab"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_json(url, path, tries=4):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(tries):
            try:
                with urllib.request.urlopen(url, timeout=120) as r:
                    body = r.read()
                break
            except OSError:
                if attempt == tries - 1:
                    raise
                time.sleep(2 ** attempt)
        path.write_bytes(body)
    return json.loads(path.read_text())


def runs():
    names, token = [], None
    while True:
        q = {"prefix": f"classic/benchmark_output/runs/{RELEASE}/", "delimiter": "/", "maxResults": 1000}
        if token:
            q["pageToken"] = token
        page = get_json(f"{LISTING}?{urllib.parse.urlencode(q)}", cache_dir() / "listing" / f"{token or 0}.json")
        names += [p.rstrip("/").rsplit("/", 1)[1] for p in page.get("prefixes", [])]
        token = page.get("nextPageToken")
        if not token:
            return [n for n in names if n.endswith("groups=ablation_prompts") and n.split(":")[0] in TASKS]


def parse(run):
    task, rest = run.split(":", 1)
    args = dict(a.split("=", 1) for a in rest.split(","))
    variant = next((f"{k}={args[k]}" for k in ("instructions", "prompt") if k in args), "")
    return task, args["model"], VARIANTS[variant]


def load():
    """{experiment: {"model", "task", "prompts": {variant: prompt settings}, "scores": {variant: {question: score}}}}"""
    wanted = [r for r in runs() if parse(r)[1] in MODELS]

    def fetch(run):
        base, local = f"{BUCKET}/{urllib.parse.quote(run, safe='')}", cache_dir() / urllib.parse.quote(run, safe="")
        return run, get_json(f"{base}/run_spec.json", local / "spec.json"), get_json(f"{base}/display_predictions.json", local / "p.json")

    out = {}
    order = list(VARIANTS.values())
    with ThreadPoolExecutor(8) as pool:          # the default first, so a duplicate of it is what gets dropped
        for run, spec, preds in sorted(pool.map(fetch, wanted), key=lambda r: (r[0].split(",")[0], order.index(parse(r[0])[2]))):
            task, model, variant = parse(run)
            e = out.setdefault(f"{model.removeprefix('together_')}/{task}", {"model": model.removeprefix("together_"), "task": task,
                                                                             "prompts": {}, "scores": {}})
            prompt = {k: spec["adapter_spec"].get(k, "") for k in SPEC_KEYS}
            if prompt in e["prompts"].values():          # the same prompt as a variant already loaded
                continue
            e["prompts"][variant] = prompt
            metric = METRIC[task]
            e["scores"][variant] = {p["instance_id"]: float(p["stats"][metric])
                                    for p in preds if p.get("train_trial_index", 0) == 0 and metric in p["stats"]}
    return out
