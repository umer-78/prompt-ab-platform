import json
import threading
import urllib.error
import urllib.request

import numpy as np
import pytest

from promptab.experiment import Experiment, Prompt
from promptab.replay import replay
from promptab.server import load, make_server


def synthetic(means, questions=1000, seed=0):
    rng = np.random.default_rng(seed)
    return [f"v{i}" for i in range(len(means))], (rng.random((len(means), questions)) < np.array(means)[:, None]).astype(float)


def test_prompt_versions_follow_the_text():
    a, b = Prompt("a", "Review: $text\nSentiment:"), Prompt("b", "Review: $text\nSentiment:")
    assert a.version == b.version != Prompt("c", "Input: $text\nOutput:").version
    assert a.render(text="Great film") == "Review: Great film\nSentiment:"
    with pytest.raises(KeyError):
        a.render()


def test_finds_the_best_prompt_and_drops_the_rest():
    names, s = synthetic([0.6, 0.7, 0.85])
    outs = [replay(names, s, "uniform", seed) for seed in range(30)]
    assert all(o["winner"] == "v2" and o["state"] == "decided" for o in outs)
    assert np.median([o["requests"] for o in outs]) < 3000


def test_thompson_serves_the_better_prompt_more_and_loses_less():
    names, s = synthetic([0.6, 0.7, 0.85])
    ex = Experiment("t", names, allocation="thompson", max_requests=2000)
    rng = np.random.default_rng(1)
    while ex.state == "running":
        arms = ex.draw(ex.check_every)
        ex.record_many(arms, s[arms, rng.integers(s.shape[1], size=len(arms))])
    assert ex.counts[2, 0] > ex.counts[0, 0] + ex.counts[1, 0]
    loss = lambda a: np.mean([replay(names, s, a, seed)["shortfall"] for seed in range(10)])
    assert loss("thompson") < loss("uniform")


def test_identical_prompts_are_rarely_dropped():
    names, s = synthetic([0.7])
    same = np.repeat(s, 4, 0)
    for allocation in ("uniform", "thompson"):
        drops = np.mean([replay([f"v{i}" for i in range(4)], same, allocation, seed, tolerance=0.0)["dropped"] > 0
                         for seed in range(60)])
        assert drops <= 0.08


def test_scores_are_checked_and_the_winner_serves_after():
    ex = Experiment("t", ["a", "b"], max_requests=100, check_every=50)
    with pytest.raises(ValueError):
        ex.record("a", 2)
    for i in range(100):
        ex.record(ex.assign(), 1.0 if i % 2 else 0.0)
    assert ex.state in ("decided", "ended") and ex.assign() == ex.winner


def test_http_api(tmp_path):
    config = tmp_path / "e.yaml"
    config.write_text('experiments:\n  s:\n    variants:\n      a: "Review: $text"\n      b: "Input: $text"\n')
    server = make_server(load(config), port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base, opener = f"http://127.0.0.1:{server.server_address[1]}", urllib.request.build_opener(urllib.request.ProxyHandler({}))
    get = lambda path: json.loads(opener.open(base + path).read())
    try:
        got = get("/experiments/s/assign")
        assert got["variant"] in ("a", "b") and "$text" in got["template"] and len(got["version"]) == 10
        req = urllib.request.Request(base + "/experiments/s/score", json.dumps({"variant": got["variant"], "score": 1}).encode())
        assert json.loads(opener.open(req).read())["scored"] == 1
        with pytest.raises(urllib.error.HTTPError):
            get("/experiments/nope")
    finally:
        server.shutdown()
