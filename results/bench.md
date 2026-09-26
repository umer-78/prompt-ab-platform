## How much the prompt matters

Score of the best prompt format against HELM's default format, same model and questions.

| Task | Experiments | Best beats default by 2+ points | Median gain | Largest gain | Worst format vs best (median) |
|---|---|---|---|---|---|
| imdb | 10 | 2 | 0.1 | 15.9 | 94.0 |
| civil_comments | 10 | 6 | 2.5 | 63.8 | 62.8 |
| natural_qa | 10 | 2 | 0.1 | 2.6 | 38.7 |

## Running the experiment

30 experiments x 20 replays, 20,000 requests each (the chosen prompt serves once the test ends). Shortfall: score lost against the best prompt, per 1,000 requests.

| Strategy | Chose a prompt within 1 point of the best | Decided before the horizon | Median requests to decide | Mean shortfall per 1,000 requests |
|---|---|---|---|---|
| keep the default | 56.7% | — | — | 51.6 |
| uniform | 99.0% | 30.3% | 1,800 | 12.0 |
| thompson | 99.8% | 0.0% | — | 3.5 |

## Identical variants (A/A): false drops

Every variant replaced by the default's recorded answers, tolerance 0: any drop is a false alarm. Target: under 5%.

| Allocation | Runs with a false drop | 95% interval |
|---|---|---|
| uniform | 8/600 (1.3%) | 0.7%–2.6% |
| thompson | 1/600 (0.2%) | 0.0%–0.9% |
