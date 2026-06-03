<div align="center">

# AutoJudger

**A unified LLM-as-judge toolkit — evaluate and rank model outputs with built-in safeguards against unreliable judges and position bias.**

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-research-orange.svg)](#)

</div>

---

Give AutoJudger (1) a judgment prompt describing your criteria and (2) one or
more LLM API endpoints — it evaluates and ranks the candidates, automatically
disqualifying judges that can't be trusted and calibrating away the order in
which candidates are shown.

It merges three research frameworks into one pipeline:

| Source | Contribution | Where it lives |
|---|---|---|
| [PRE](https://github.com/chuzhumin98/PRE) | Peer review + weighted "chair" decision | [judge.py](autojudger/judge.py) |
| [Auto-PRE](https://github.com/cjj826/Auto-PRE) | Label-free judge **qualification exam** | [exam.py](autojudger/exam.py) |
| [CalibraEval](https://github.com/CSHaitao/CalibraEval) | Logprob **bias calibration** | [calibrate.py](autojudger/calibrate.py) |

## Highlights

- **Drop-in LLM judging** — point it at any OpenAI-compatible endpoint (OpenAI, DeepSeek, Together, vLLM / local servers, …) and rank responses with a single config file.
- **Label-free quality control** — a qualification exam disqualifies self-contradicting, off-topic-blind, or over-confident judges *without any gold labels*.
- **Position-bias calibration** — a logprob-based debiaser collapses the "A vs. B" ordering effect that plagues pairwise LLM judging.
- **Reliability-weighted decisions** — surviving judges vote as a weighted "chair," so trustworthy judges count for more.
- **Crash-safe & resumable** — every step checkpoints to JSONL; an interrupted run resumes without re-querying any API.
- **Pairwise *and* pointwise** modes, with pre-written or model-generated responses.
- **Lightweight** — core install is pure `openai` + `numpy` + `PyYAML` + `tqdm`; no heavyweight ML deps.

## Table of contents

- [Pipeline](#pipeline)
- [Install](#install)
- [Quickstart](#quickstart)
- [Task data](#task-data)
- [Roles & endpoints](#roles--endpoints)
- [Pointwise mode](#pointwise-mode)
- [Config reference](#config-reference)
- [Output / report schema](#output--report-schema)
- [How the safeguards work](#how-the-safeguards-work)
- [Checkpoints & resume](#checkpoints--resume)
- [Module map](#module-map)
- [Tests](#tests)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)
- [License](#license)

## Pipeline

```
collect responses → qualification exam → peer review → chair decision → ranked report
                    (drop unreliable      (text, or 3-prompt    (reliability-
                     judges, no labels)    calibrated review)    weighted vote)
```

Every step checkpoints to disk as JSONL, so an interrupted run resumes without
re-querying any API (see [Checkpoints & resume](#checkpoints--resume)).

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
  Which response is better? Output only the single letter 'A' or 'B'.

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
python -c "from autojudger import evaluate; from autojudger.config import load_config; import json; print(json.dumps(evaluate(load_config('config.yaml')), indent=2))"
```

The report is printed and written to `results/report.json`. A ready-to-edit
example lives at [examples/config_pairwise.yaml](examples/config_pairwise.yaml).

## Task data

`tasks` may be a path to a `.jsonl`, `.json`, or `.csv` file (format is
auto-detected from the extension), or — for programmatic calls — an inline list
of dicts. Each item is a flat dict whose keys the judgment prompt references via
`{{key}}`.

Responses can be **pre-written** or **generated**:

| | Pairwise | Pointwise |
|---|---|---|
| **Pre-written** (no evaluatee API calls) | items contain `response_A` + `response_B` | items contain `response` |
| **Generated** (evaluatee APIs answer each task) | needs ≥2 evaluatees to compare | one or more evaluatees scored individually |

When generating, each evaluatee is prompted with `task_prompt` (default
`"{{question}}"`); set it in the config if your question column has another name.

## Roles & endpoints

Each entry under `apis` is one OpenAI-compatible endpoint with a `role`:

- `evaluatee` — generates answers that get judged.
- `evaluator` — judges answers. At least one is required.
- `both` — does both (the default).

Any provider exposing the OpenAI chat-completions interface works (OpenAI,
DeepSeek, Together, vLLM / local servers, Anthropic-via-proxy, …). The
`model_name` used in checkpoint filenames and result keys defaults to `model`;
add a `name:` field to disambiguate two endpoints that share a model id.

> **Logprob-dependent features.** The self-confidence exam and bias calibration
> read token logprobs. Endpoints that don't return them (e.g. native
> Anthropic-style APIs) are detected automatically — those features skip and the
> judge falls back to plain text review, so the pipeline still runs.

## Pointwise mode

Set `mode: pointwise` to score each source independently instead of comparing
pairs. The judgment prompt then uses a single `{{response}}` field and should ask
for a numeric score; `parser_type` defaults to `int` (use `float` for decimals).

```yaml
mode: pointwise
parser_type: int
judgment_prompt: |
  Rate this answer from 1 to 10.
  Question: {{question}}
  Answer: {{response}}
  Output only the number.
```

## Config reference

Only `judgment_prompt`, `apis`, and `tasks` are required; everything else falls
back to [config/default.yaml](config/default.yaml).

```yaml
mode: pairwise              # pairwise | pointwise
output_dir: results         # checkpoints + report.json land here
parser_type: str            # str (A/B/tie tokens) | int | float
task_prompt: "{{question}}" # used only when generating evaluatee responses

apis:
  - base_url: "..."         # OpenAI-compatible endpoint
    api_key: "..."
    model: "..."
    role: both              # evaluatee | evaluator | both
    name: "judge-1"         # optional; defaults to `model`
    temperature: 0.0
    max_tries: 5            # retries with exponential backoff

exam:                       # Auto-PRE qualification (auto-skips with <2 judges)
  enabled: true
  consistency: true         # symmetric verdict when A/B are swapped
  pertinence: true          # prefers an on-topic answer over an off-topic one
  self_confidence: true     # more certain on easy than hard pairs (needs logprobs)
  threshold: mean           # 'mean' across judges, or an explicit float
  max_samples: 20           # caps exam API cost
  seed: 0

aggregation:                # chair decision
  strategy: full            # full (mean win-rate) | elo
  weighted_method: uniform  # uniform | log | exp | poly (weight reliable judges more)
  alpha: 1.0                # shape parameter for exp / poly

calibrate:                  # CalibraEval debiasing (pairwise only)
  enabled: false            # needs logprob-capable endpoints
  lam: 0.05                 # weight of the format-consistency term
```

**Parsing note.** In pairwise mode responses are matched against the nominal
tokens `one/two/tie/a/b` by earliest position, so keep judge outputs terse
(e.g. "Output only 'A' or 'B'") to avoid a stray letter in prose being picked up.

## Output / report schema

`report.json` (and the return value of `evaluate`) has this shape:

```jsonc
{
  "summary": {
    "mode": "pairwise",
    "sources": ["A", "B"],            // the things being ranked
    "ranking": ["A", "B"],            // best first
    "scores": { "A": 0.7, "B": 0.3 }, // win-rate (full) | rating (elo) | mean score (pointwise)
    "qualified_evaluators": ["gpt-4o-mini"],
    "calibrated": false,
    "inter_judge_agreement": 0.62     // Fleiss's kappa, or null if <2 judges
  },
  "decision": { /* full per-pair detail, e.g. win_rate_A_vs_B */ },
  "evaluator_weights": { "gpt-4o-mini": 1.0 },
  "calibration": { /* present only when calibrate.enabled: order-flip before/after */ }
}
```

## How the safeguards work

- **Qualification exam** (label-free): a judge is dropped if it contradicts
  itself when A/B are swapped (consistency), can't tell an on-topic answer from a
  fluent off-topic one (pertinence), or is no more confident on easy than hard
  pairs (self-confidence). Survivors' scores feed the chair's reliability
  weighting. Degrades gracefully — with too few judges or too little data the
  exam skips rather than disqualifying everyone.
- **Bias calibration** (opt-in, pairwise): each pair is queried three ways
  (A-first, rephrased, B-first); a monotone map is fit — with no gold labels — to
  enforce order-symmetry and format-consistency, collapsing position bias. The
  report records the order-flip rate before vs. after; the bundled synthetic test
  confirms calibration removes the large majority of order-dependent flips.

## Checkpoints & resume

Everything under `output_dir` is re-readable, so a re-run picks up where it left
off (delete a file to force that step to re-query):

```
results/
├── task_responses/<source>.jsonl          # generated evaluatee answers
├── exam_responses/<judge>_*.jsonl          # consistency / pertinence exam calls
├── evaluation_responses/<judge>.jsonl      # peer-review verdicts
├── calibration/<judge>_logits.jsonl        # 3-prompt logprobs (when calibrating)
└── report.json                             # final ranked report
```

## Module map

For navigating or extending the code:

| File | Responsibility |
|---|---|
| [autojudger/__init__.py](autojudger/__init__.py) | `evaluate(config)` — orchestrates the whole pipeline |
| [autojudger/config.py](autojudger/config.py) | load user YAML, deep-merge over defaults, validate |
| [autojudger/api.py](autojudger/api.py) | `LLMClient` — retrying OpenAI-compatible chat + logprobs |
| [autojudger/data.py](autojudger/data.py) | `DataLoader` — jsonl / json / csv / inline task loading |
| [autojudger/judge.py](autojudger/judge.py) | collect responses, peer review, chair decision, agreement |
| [autojudger/exam.py](autojudger/exam.py) | `QualificationExam` — the three label-free judge tests |
| [autojudger/calibrate.py](autojudger/calibrate.py) | `NOACalibrator` + 3-prompt calibrated review |
| [autojudger/utils.py](autojudger/utils.py) | response parsing + `{{key}}` template filling |

## Tests

```bash
python tests/test_integration.py   # pipeline: pairwise/ELO/pointwise/resume
python tests/test_exam.py           # qualification exam filtering
python tests/test_calibrate.py      # NOA calibrator + end-to-end
```

All tests run offline with mocked LLMs — no API key required. To smoke-test
against a real endpoint, fill in credentials and run
[tests/live_api_check.py](tests/live_api_check.py).

## Acknowledgements

AutoJudger stands on the shoulders of three open research frameworks, whose
ideas it unifies into a single pipeline:

- [PRE: Peer Review-based Evaluation](https://github.com/chuzhumin98/PRE)
- [Auto-PRE: Automatic Peer Review Evaluation](https://github.com/cjj826/Auto-PRE)
- [CalibraEval](https://github.com/CSHaitao/CalibraEval)

## Citation

If you use AutoJudger in your research, please cite this repository:

```bibtex
@software{ai_autojudger,
  author  = {Ai, Qingyao},
  title   = {AutoJudger: A Unified LLM-as-Judge Toolkit},
  year    = {2026},
  url     = {https://github.com/QingyaoAi/AutoJudger}
}
```

Please also consider citing the upstream PRE, Auto-PRE, and CalibraEval works.

## License

Released under the [MIT License](LICENSE).
