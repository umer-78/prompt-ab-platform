"""An experiment: several prompts on live traffic, until one wins.

Each request is assigned a variant still in the running:

- uniform: equal shares, the classic A/B/n test.
- thompson: in proportion to each variant's chance of being the best under a Beta posterior on
  its mean score (Thompson sampling). Traffic shifts to the better prompts while the test runs.

The posterior, like everything else, is updated every `check_every` scores; assignments are
drawn in batches of that size.

A variant is dropped once an always-valid test finds it worse than another variant by more
than `tolerance`. The test is the one-sided mixture likelihood ratio of Johari et al., "Peeking
at A/B Tests" (2017), checked after every batch for as long as the experiment runs; Bonferroni
across the ordered pairs keeps the chance of ever dropping a variant on a difference that isn't
there under alpha. The experiment is decided when one variant is left. At `max_requests` it
ends with the survivors tied as far as the data can tell, and the best of them serves.
"""
import hashlib
import math
import string
from dataclasses import dataclass, field

import numpy as np
from scipy.special import log_ndtr

TAU = 0.05     # spread of the prior on true differences: the test is quickest near 5 points
MIN_N = 20     # scores each variant needs before it can be compared


def evidence(counts, tolerance=0.0, tau=TAU):
    """For every ordered pair (a, b): log likelihood ratio that b is worse than a by more than
    tolerance, and the observed difference b - a. counts[k] = (scores, sum, sum of squares)."""
    c = np.asarray(counts, float)
    a, b = np.meshgrid(np.arange(len(c)), np.arange(len(c)), indexing="ij")
    (n0, s0, q0), (n1, s1, q1) = c[a].transpose(2, 0, 1), c[b].transpose(2, 0, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = (s0 + s1) / (n0 + n1)
        v = ((q0 + q1) / (n0 + n1) - mean ** 2) * (1 / n0 + 1 / n1)   # pooled: right while they are equal
        d = s1 / n1 - s0 / n0
    ok = (n0 >= MIN_N) & (n1 >= MIN_N) & (v > 1e-12) & (a != b)
    d, v = np.where(ok, d, 0.0), np.where(ok, v, 1.0)
    t2, x = tau * tau, d + tolerance
    llr = (0.5 * np.log(v / (v + t2)) + t2 * x * x / (2 * v * (v + t2))
           + math.log(2) + log_ndtr(-x * tau / np.sqrt(v * (v + t2))))
    return np.where(ok, llr, -np.inf), d


@dataclass(frozen=True)
class Prompt:
    """A prompt template; its version is a hash of the text, so results name exactly what ran."""
    name: str
    template: str

    @property
    def version(self):
        return hashlib.sha256(self.template.encode()).hexdigest()[:10]

    def render(self, **values):
        return string.Template(self.template).substitute(values)    # $name placeholders; KeyError if one is missing


@dataclass
class Experiment:
    name: str
    variants: list                  # variant names; the first is the current prompt
    allocation: str = "uniform"     # or "thompson"
    alpha: float = 0.05
    tolerance: float = 0.01         # a variant is only dropped for being worse by more than this
    check_every: int = 50
    max_requests: int = 20_000
    seed: int = 0
    state: str = "running"          # running, decided or ended
    winner: str = None
    seen: int = 0
    log: list = field(default_factory=list)

    def __post_init__(self):
        k = len(self.variants)
        self.counts = np.zeros((k, 3))
        self.alive = np.ones(k, bool)
        self.rng = np.random.default_rng(self.seed)
        self.queue = []

    def draw(self, n):
        """n assignments (variant indices) from the current state."""
        alive = np.flatnonzero(self.alive)
        if self.allocation == "thompson":
            n_, s_ = self.counts[alive, 0], self.counts[alive, 1]
            return alive[self.rng.beta(1 + s_, 1 + n_ - s_, size=(n, len(alive))).argmax(1)]
        return alive[self.rng.integers(len(alive), size=n)]

    def assign(self):
        """The variant for one request: once the experiment is over, the winner."""
        if self.state != "running":
            return self.winner
        if not self.queue:
            self.queue = list(self.draw(self.check_every))
        return self.variants[self.queue.pop(0)]

    def record(self, variant, score):
        if not 0 <= score <= 1:
            raise ValueError("score must be between 0 and 1")
        self.record_many(np.array([self.variants.index(variant)]), np.array([float(score)]))

    def record_many(self, arms, scores):
        if self.state != "running":
            return
        np.add.at(self.counts, arms, np.stack([np.ones_like(scores), scores, scores * scores], 1))
        before, self.seen = self.seen, self.seen + len(arms)
        if self.seen // self.check_every > before // self.check_every:
            self.check()

    def check(self):
        llr, d = evidence(self.counts, self.tolerance)
        k = len(self.variants)
        beaten = llr >= math.log(k * (k - 1) / self.alpha)          # beaten[a, b]: b is worse than a
        for b in np.flatnonzero(beaten.any(0) & self.alive):
            a = int(np.argmax(np.where(beaten[:, b], llr[:, b], -np.inf)))
            self.alive[b] = False
            self.log.append({"at": self.seen, "event": f"dropped {self.variants[b]}: {100 * d[a, b]:+.1f} points vs {self.variants[a]}"})
        means = self.counts[:, 1] / np.maximum(self.counts[:, 0], 1)
        if self.alive.sum() == 1:
            self.state, self.winner = "decided", self.variants[int(np.flatnonzero(self.alive)[0])]
        elif self.seen >= self.max_requests:
            self.state = "ended"
            self.winner = self.variants[int(np.flatnonzero(self.alive)[np.argmax(means[self.alive])])]
        if self.state != "running":
            self.log.append({"at": self.seen, "event": f"{self.state}: {self.winner}"})

    def status(self):
        return {"experiment": self.name, "allocation": self.allocation, "state": self.state, "winner": self.winner,
                "scored": self.seen, "log": self.log[-20:],
                "variants": {v: {"requests": int(c[0]), "mean": round(c[1] / c[0], 4) if c[0] else None, "alive": bool(a)}
                             for v, c, a in zip(self.variants, self.counts, self.alive)}}
