# AutoJudger

A unified **LLM-as-judge** toolkit. Give it (1) a judgment prompt describing your
criteria and (2) a few LLM API endpoints — it automatically evaluates and ranks,
with built-in safeguards against unreliable judges and position bias.

It merges three research frameworks into one pipeline:

| Source | Contribution | Where it lives |
|---|---|---|
| [PRE](https://github.com/chuzhumin98/PRE) | Peer review + weighted "chair" decision | `judge.py` |
| [Auto-PRE](https://github.com/cjj826/Auto-PRE) | Label-free judge **qualification exam** | `exam.py` |
| [CalibraEval](https://github.com/CSHaitao/CalibraEval) | Logprob **bias calibration** | `calibrate.py` |

## Pipeline

```
collect responses → qualification exam → peer review → chair decision → ranked report
                    (drop unreliable      (text, or 3-prompt    (reliability-
                     judges, no labels)    calibrated review)    weighted vote)
```

Every step checkpoints to disk, so an interrupted run resumes without re-querying any API.

## Install

```bash
pip install -e .
```

Core dependencies are just `openai`, `numpy`, `PyYAML`, `tqdm` (the bias
calibrator is pure numpy — no scikit-learn needed).

## Quickstart

**1. A judgment prompt** (`{{...}}` fields are filled from your task data;
`{{response_A}}`/`{{response_B}}` hold the two responses being compared):

```yaml
# config.yaml
judgment_prompt: |
  Task: {{question}}
  Response A: {{response_A}}
  Response B: {{response_B}}
  Which response is better? Output only 'A' or 'B'.

apis:
  - base_url: "https://api.openai.com/v1"
    api_key: "sk-..."
    model: "gpt-4o-mini"
    role: "evaluator"          # evaluatee | evaluator | both

tasks: "tasks.jsonl"
mode: "pairwise"               # pairwise | pointwise
output_dir: "results"
```

**2. Task data** (`tasks.jsonl`, one JSON object per line). Either supply the
responses to compare directly:

```json
{"question": "What is the capital of France?", "response_A": "Paris.", "response_B": "The capital is Paris, France's largest city."}
```

…or supply only the questions and let evaluatee APIs (`role: evaluatee`/`both`)
generate the responses to be ranked.

**3. Run:**

```bash
python main.py --config config.yaml
# or, programmatically:
python -c "from autojudger import evaluate; import yaml; print(evaluate(yaml.safe_load(open('config.yaml'))))"
```

The report (also written to `results/report.json`) includes the ranking, per-model
scores, which evaluators passed qualification, inter-judge agreement (Fleiss's
kappa), and — if enabled — before/after bias metrics.

## Key options

```yaml
exam:
  enabled: true        # auto-skips with <2 judges
  consistency: true    # symmetric verdict on swapped A/B
  pertinence: true     # prefers on-topic over off-topic answers
  self_confidence: true  # needs logprob-capable endpoints
  max_samples: 20      # caps exam API cost

aggregation:
  strategy: full       # full (win-rate) | elo
  weighted_method: uniform  # uniform | log | exp | poly  (weight reliable judges more)

calibrate:
  enabled: false       # CalibraEval debiasing; needs logprob-capable endpoints
```

## How the safeguards work

- **Qualification exam** (label-free): a judge is dropped if it contradicts itself
  when A/B are swapped (consistency), can't tell an on-topic answer from a fluent
  off-topic one (pertinence), or is no more confident on easy than hard pairs
  (self-confidence). Degrades gracefully when there are too few judges or too
  little data.
- **Bias calibration** (opt-in): for each pair the judge is queried three ways
  (A-first, rephrased, B-first); a monotone map is fit — with no gold labels — to
  enforce order-symmetry and format-consistency, collapsing position bias. In
  testing this cut the order-flip rate from ~0.50 to ~0.003.

## Tests

```bash
python tests/test_integration.py   # pipeline: pairwise/ELO/pointwise/resume
python tests/test_exam.py           # qualification exam filtering
python tests/test_calibrate.py      # NOA calibrator + end-to-end
```

All tests run offline with mocked LLMs — no API key required.
