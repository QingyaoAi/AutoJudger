# AutoJudger — Unified LLM-as-Judge Toolkit

Merging [PRE](https://github.com/chuzhumin98/PRE), [Auto-PRE](https://github.com/cjj826/Auto-PRE), and [CalibraEval](https://github.com/CSHaitao/CalibraEval) into a single toolkit for automatic LLM evaluation.

---

## Source Repo Analysis

### PRE (chuzhumin98/PRE)
3-stage pipeline: Qualification Exam → Peer Review → Chair Decision.

- **Exam**: reference exam (needs gold labels) + inner-consistency exam (two prompt templates)
- **Modes**: pointwise (1–5 score) and pairwise (A/B/tie)
- **Aggregation**: uniform/log/exp/poly weighting; full, ELO, Glicko strategies
- **Strengths**: well-structured, file-based checkpointing (resumable), clean YAML config
- **Weaknesses**: `openai==0.27.4` (ancient), exam needs gold labels, `np.float` deprecation bug, no bias correction

### Auto-PRE (cjj826/Auto-PRE)
AAAI'26. Replaces PRE's manual exam with three automatic qualification criteria (no gold labels):

1. **Consistency exam** — give evaluator A-vs-B then B-vs-A; check verdicts are symmetric
2. **Pertinence exam** — GPT-4 rewrites query Q→Q'; evaluator should prefer strong model on Q' over weak model on Q
3. **Self-confidence exam** — evaluator should have lower token-entropy on easy pairs (strong vs. weak) than hard pairs (similar quality)

- **Strengths**: no gold labels, theoretically sounder qualification, openai v1 SDK
- **Weaknesses**: exam hardcoded to XSum/NFQA datasets and specific model names; pertinence exam hardcodes a GPT-4 proxy endpoint

### CalibraEval (CSHaitao/CalibraEval)
Post-hoc calibration to eliminate position/selection bias.

- **3-prompt protocol**: prompt1 (A vs B), prompt2 (A vs B, rephrased), prompt3 (B vs A reversed)
- **Gets token logprobs** P(A) from each prompt — requires logprob access
- **NOA optimization**: fits monotone mapping `f` such that `f(P_AB) + f(P_BA_reversed) = 1` (symmetry) and `f(P_AB) ≈ f(P_AB2)` (format-consistency); then fits isotonic regression
- **Strengths**: label-free, tackles bias directly at probability level, SOTA results
- **Weaknesses**: pure research code (hardcoded paths), requires logprob-capable API, heavy deps (transformers, trl, accelerate) only needed for local model loading

---

## User Interface

The user provides exactly two things:

```yaml
# config.yaml

judgment_prompt: |
  You are evaluating two AI responses.
  Task: {{question}}
  Response A: {{response_A}}
  Response B: {{response_B}}
  Which response is better? Output only 'A' or 'B'.

apis:
  - base_url: "https://api.openai.com/v1"
    api_key: "sk-..."
    model: "gpt-4o"
    role: "both"         # evaluatee (generates answers) + evaluator (judges)
  - base_url: "https://api.anthropic.com/v1"
    api_key: "sk-ant-..."
    model: "claude-3-5-sonnet-20241022"
    role: "evaluator"    # only judges, doesn't generate

tasks: "tasks.jsonl"     # {"question": "...", "response_A": "...", "response_B": "..."}
                         # OR just {"question": "..."} to auto-generate responses
mode: "pairwise"         # pointwise | pairwise
output_dir: "results/"
# Everything else has sensible defaults
```

---

## Project Structure

```
AutoJudger/
├── autojudger/
│   ├── __init__.py      # Public API: evaluate(config) -> JudgmentResult
│   ├── api.py           # NEW: single OpenAICompatibleAPI class (openai>=1.0)
│   ├── data.py          # FROM PRE: DataLoader, generalized to JSONL/CSV/dict
│   ├── exam.py          # FROM Auto-PRE: 3 automatic qualification criteria
│   ├── judge.py         # FROM PRE: PEER_REVIEW + weighted aggregation + ELO
│   ├── calibrate.py     # FROM CalibraEval: NOA debiasing (opt-in, logprobs only)
│   └── utils.py         # FROM PRE: parse_response(), prompt template filling
├── config/
│   └── default.yaml
├── main.py              # CLI: python main.py --config config.yaml
├── requirements.txt
└── PLAN.md              # this file
```

---

## Data Flow

```
User provides:
  (A) judgment_prompt template
  (B) API list with keys/URLs
  (C) task data (JSONL/CSV)
           │
           ▼
[Step 1] Collect task responses
  For each "evaluatee" API:
    Generate answer for each task prompt → save to outputs/task_responses/
  (Skipped if task data already contains pre-written responses)
           │
           ▼
[Step 2] Auto Qualification Exam          ← AUTO-PRE
  For each "evaluator" API candidate:
    (a) Consistency exam  — symmetry check on reversed A/B pairs
    (b) Self-confidence exam — entropy on easy vs. hard pairs
    (c) Pertinence exam   — distinguish weak vs. rephrased-strong answer
  → Qualified evaluator list + per-evaluator reliability scores
  (Skipped if only 1 API; skipped if exam: none)
           │
           ▼
[Step 3] Peer Review                      ← PRE
  Each qualified evaluator judges all task pairs/responses
  using the user's judgment_prompt (template filled with task fields + responses)
  Results checkpointed to outputs/evaluation_responses/
           │
           ▼
[Step 4] CalibraEval Debiasing            ← CALIBRAEVAL  (optional)
  If calibrate: true AND APIs return logprobs:
    Run 3-prompt protocol per evaluator
    Fit NOA mapping + isotonic regression
    Replace raw logits with calibrated values
  (Off by default; only activates when explicitly enabled)
           │
           ▼
[Step 5] Chair Decision                   ← PRE
  Aggregate evaluator verdicts with reliability-weighted voting
  Output: final rankings/scores + per-task breakdown
  Saved to outputs/evaluation_results/
```

---

## Implementation Phases

### Phase 1 — API Layer + Project Skeleton  ✅ DONE
> Foundation for everything else. Target: project runs end-to-end with a stub judge.

- [x] Create project directory structure
- [x] `autojudger/api.py` — single `LLMClient` + `APIConfig` using `openai>=1.0`
  - [x] `chat(prompt) -> str`
  - [x] `chat_with_logprobs(prompt, tokens) -> dict | None` (returns None if unsupported)
  - [x] Retry logic with exponential backoff
  - [x] Replaces all 6+ vendor-specific classes from PRE and Auto-PRE
  - [x] `build_clients()` factory + role helpers (evaluatee/evaluator/both)
- [x] `autojudger/data.py` — generalized DataLoader
  - [x] Auto-detect JSONL / JSON / CSV format
  - [x] Support inline dict list (for programmatic use)
- [x] `autojudger/utils.py` — ported `parse_response()` + `fill_template()`
  - [x] Parse "A"/"B", "one"/"two", integers, floats; None-safe
  - [x] Default pairwise nominal mapping (A/B/one/two/tie)
- [x] `autojudger/config.py` — deep-merge user YAML over defaults + validation
- [x] `main.py` — CLI entry point (`python main.py --config config.yaml`)
- [x] `config/default.yaml` — defaults for all optional fields
- [x] `requirements.txt` — `openai>=1.0`, `numpy>=1.24`, `PyYAML>=6.0`, `tqdm`

---

### Phase 2 — Peer Review Core  ✅ DONE
> Minimum useful product: given pre-written responses + judge APIs, produces rankings.

- [x] `autojudger/judge.py`
  - [x] `collect_responses()` — pre-written (response columns) OR generated (checkpointed)
  - [x] `PeerReview.run()` with file-based JSONL checkpointing (resumable)
  - [x] `build_judge_prompts()` for pointwise (K×N) and pairwise (K×(K-1)×N, both orderings)
  - [x] `ChairDecision` (ported from PRE's `PRE` class)
    - [x] `compute_weights()` — uniform / log / exp / poly
    - [x] `_collect_pairwise_votes()` — shared helper; normalizes PRE label convention
    - [x] `_aggregate_pointwise()` — weighted mean score per source
    - [x] `_aggregate_pairwise_full()` — win-rate matrix + overall scores
    - [x] `_aggregate_elo()` — ELO rating from pairwise results
    - [x] Fixed `np.float` deprecation; **fixed inverted pairwise label sign bug**
- [x] `autojudger/__init__.py`
  - [x] `evaluate(config: dict) -> dict` report
  - [x] Step 1 (response collection) + Step 3 (peer review) + Step 5 (chair decision)
  - [x] Steps 2 (exam) and 4 (calibration) stubbed for now
- [x] Integration test (`tests/test_integration.py`): pairwise-full, ELO, pointwise,
      and checkpoint-resume — **all passing** with a mocked LLM (no API key needed)

---

### Phase 3 — Auto Qualification Exam  ✅ DONE
> Removes the gold-label requirement. Evaluator pool is filtered automatically.
> NOTE: Auto-PRE's exams were hardcoded to XSum/NFQA + fixed model names. Each is
> **reformulated to run on any task data** without dataset-specific assumptions.

- [x] `autojudger/exam.py` — single `QualificationExam` class with three scorers
  - [x] **Consistency** — symmetry rate on swapped A/B pairs (verdict(X,Y) mirrors
        verdict(Y,X)); threshold = mean of candidates (or explicit float)
  - [x] **Pertinence** — reformulated as on-topic vs off-topic: off-topic answer
        borrowed from a *different task*; judge should prefer the on-topic one.
        Avoids Auto-PRE's GPT-4 query-rewrite + dataset hardcoding entirely.
        Accuracy vs. known label; threshold = mean of candidates.
  - [x] **Self-confidence** — `chat_with_logprobs` confidence higher on easy
        (clearly-different) pairs than hard (similar) pairs; auto-skips if the
        endpoint has no logprobs
  - [x] `run()` — builds shared sample sets (seeded, capped at `max_samples`),
        scores each evaluator, drops failures, returns reliability scores
  - [x] Checkpointed exam responses (resumable, like peer review)
  - [x] Graceful degradation: <2 evaluators / <2 sources / <2 tasks → skip;
        if an exam can't run it's skipped; never disqualifies everyone
- [x] Wired into `autojudger/__init__.py` Step 2 (replaced stub)
- [x] `tests/test_exam.py` — consistency filtering, pertinence filtering,
      single-evaluator degradation — **all passing** offline

---

### Phase 4 — CalibraEval Debiasing (Optional)  ✅ DONE
> Bias mitigation via logprob calibration. Opt-in only.

- [x] `autojudger/calibrate.py`
  - [x] `generate_three_prompts()` — p1 (A first), p2 (rephrased), p3 (reversed)
  - [x] `collect_logits()` — `chat_with_logprobs` per prompt; returns (n,3) P(A)
        array + ok flag; checkpointed; returns ok=False if endpoint lacks logprobs
  - [x] `NOACalibrator` — port of CalibraEval's `iso_regression`
    - [x] NOA objective (symmetry f(p1)+f(p3)=1 + consistency f(p1)=f(p2))
    - [x] Monotone map via cumulative-softmax; **analytic vectorized gradient**
          (verified vs. finite differences to 1e-11)
    - [x] `predict()` via monotone `np.interp` — **no scikit-learn dependency**
    - [x] Checkpoints logits to `output_dir/calibration/`
  - [x] `combined_pA()` / `decisions_from_pA()` — (p1+p2+1−p3)/3 → labels
  - [x] `order_flip_rate()` — assumption-free bias metric (verdict disagreement
        across order swap); reported before/after
  - [x] `run_calibrated_review()` — judges all pairs via 3-prompt + NOA, emits
        pairwise records in ChairDecision format; non-logprob judges fall back to text
- [x] Wired into `autojudger/__init__.py` Steps 3+4 (guarded by `calibrate.enabled`
      + pairwise; logprob judges calibrated, others text — results merged)
- [x] **No extra dependency needed** — `np.interp` replaces `IsotonicRegression`
      (scikit-learn/scipy stay optional, currently unused by core)
- [x] `tests/test_calibrate.py` — NOA reduces order-flips **0.50 → 0.003**,
      improves accuracy 0.935 → 0.983, monotone map, end-to-end pipeline — **all pass**

---

### Phase 5 — Polish  ✅ DONE
> Production-quality output, usability, and documentation.

- [x] Structured JSON report (`results/report.json`)
  - [x] Per-model: final score / ELO rating / win rate; full ranking
  - [x] `qualified_evaluators` + `evaluator_weights`
  - [x] `inter_judge_agreement` — Fleiss's kappa across judges (ported clean)
  - [x] `calibration` — per-judge order-flip before/after (when enabled)
- [x] Progress bars via `tqdm` on every long API loop (responses, review, exam, logits)
- [x] Better error messages / validation in `config.py` (required fields, mode)
- [x] Python API: `autojudger.evaluate(config_dict)` documented in README
- [x] Example configs + task files in `examples/`
- [x] `README.md` with quickstart, pipeline diagram, options table
- [x] `pyproject.toml` — `pip install -e .` works; `autojudger` console script;
      `[calibrate]`/`[dev]` extras; verified installable + importable

---

## Dependency Plan

| Package | Required? | Source |
|---|---|---|
| `openai>=1.0` | Core | All endpoints via OpenAI-compatible client |
| `numpy>=1.24` | Core | PRE / Auto-PRE |
| `pyyaml>=6.0` | Core | PRE config system |
| `tqdm` | Core | Auto-PRE |
| `scikit-learn` | Optional (`calibrate`) | CalibraEval isotonic regression |
| `scipy` | Optional (`calibrate`) | CalibraEval |
| `transformers`, `trl`, `accelerate`, `peft` | **Dropped** | Local model loading — out of scope |
| `zhipuai`, `claude2openai` | **Dropped** | Replaced by generic OpenAI-compatible client |
| `pandas`, `pingouin` | **Dropped** | Metrics reimplemented in numpy |

---

## Non-Obvious Design Decisions

**Evaluatee vs. evaluator roles**: The same API can be both. `role: both` means the model generates task responses first, then participates in judging — mirroring the PRE setup where GPT-4 might be both tested and asked to judge others.

**Exam degrades gracefully**: 1 API → no exam, use it directly. 2 APIs → consistency exam only. 3+ APIs → all three exam types. This prevents the exam from blocking single-API use cases.

**CalibraEval 3-prompt auto-generation**: prompt2 (rephrased A-then-B) and prompt3 (B-then-A) are automatically generated from the user's `judgment_prompt` by swapping `{{response_A}}`/`{{response_B}}` and lightly rephrasing the instruction prefix. No extra user input needed.

**Pertinence exam generalization**: instead of hardcoding GPT-4 + XSum/NFQA, use the first (assumed strongest) API in the evaluator list to generate Q'. If the list is homogeneous or only 1 API exists, this exam is skipped rather than failing.

**Checkpoint format**: inherit PRE's JSONL-per-evaluator-per-task format exactly, so partial runs resume without re-querying APIs. Restarting the process continues from where it left off.

**Glicko strategy**: PRE has a TODO placeholder for Glicko and never implements it. Not porting — ELO is sufficient.

---

## Progress Tracker

| Phase | Status | Notes |
|---|---|---|
| Phase 1: API Layer + Skeleton | ✅ Done | api/data/utils/config/CLI; imports clean |
| Phase 2: Peer Review Core | ✅ Done | full pipeline; 4 integration tests pass offline |
| Phase 3: Auto Qualification Exam | ✅ Done | 3 generalized exams; 3 exam tests pass offline |
| Phase 4: CalibraEval Debiasing | ✅ Done | NOA calibrator; order-flips 0.50→0.003; 4 tests pass |
| Phase 5: Polish | ✅ Done | packaging + README + tqdm + Fleiss kappa; 12 tests pass |

**🎉 All 5 phases complete.** `pip install -e .` → `autojudger --config config.yaml`.
12 offline tests pass (5 pipeline + 3 exam + 4 calibration).

### Current state (2026-06-03)
**All three source frameworks are merged and working.** Full pipeline:
collect → exam (filter unreliable judges) → peer-review **or calibrated 3-prompt
review** → chair-decision, all resumable. 11 tests pass offline (4 pipeline +
3 exam + 4 calibration), no API key needed. Files:
`autojudger/{api,data,utils,config,judge,exam,calibrate,__init__}.py`, `main.py`,
`config/default.yaml`, `examples/`, `tests/{test_integration,test_exam,test_calibrate}.py`.

**Phase 5 (polish) candidates:**
- Structured report extras: Fleiss/ICC agreement metrics; pretty CLI summary.
- `pip install -e .` packaging (`pyproject.toml`), README quickstart.
- Progress bars (`tqdm`) on the long API loops.
- `task_prompt` config key (generated-response path) defaults to `{{question}}`.
- Calibration target-token note: `_target_tokens` keys off `_nominal_list[:2]`
  (default "one"/"two") — judge prompt must ask for those tokens for logprobs to hit.
