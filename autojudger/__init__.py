"""AutoJudger — a unified LLM-as-judge toolkit.

Merges three peer-review / debiasing frameworks into one pipeline:
  * PRE          — peer review + weighted chair decision
  * Auto-PRE     — automatic, label-free evaluator qualification exam
  * CalibraEval  — post-hoc logprob calibration to remove selection bias

Public entry point: ``evaluate(config) -> dict``.
"""

from __future__ import annotations

import os

from .api import build_clients
from .calibrate import run_calibrated_review
from .data import DataLoader
from .exam import QualificationExam
from .judge import (
    ChairDecision,
    PeerReview,
    build_judge_prompts,
    collect_responses,
    inter_judge_agreement,
)
from .utils import PAIRWISE_NOMINAL_LIST, PAIRWISE_NOMINAL_TICKS

__version__ = "0.1.0"
__all__ = ["evaluate"]


def evaluate(config: dict) -> dict:
    """Run the full evaluation pipeline and return a structured report."""
    mode = config["mode"]
    save_dir = config["output_dir"]
    os.makedirs(save_dir, exist_ok=True)

    # Pairwise judging parses nominal A/B/tie tokens unless the user overrides.
    if mode == "pairwise" and config.get("parser_type", "str") == "str":
        config.setdefault("_nominal_list", PAIRWISE_NOMINAL_LIST)
        config.setdefault("_nominal_ticks", PAIRWISE_NOMINAL_TICKS)
        config["parser_type"] = "str"
    elif mode == "pointwise":
        config.setdefault("parser_type", "int")

    # --- clients & roles ---
    clients = build_clients(config["apis"])
    evaluatees = [c for c in clients if c.config.is_evaluatee]
    evaluators = [c for c in clients if c.config.is_evaluator]
    if not evaluators:
        raise ValueError("No evaluator APIs configured (need role 'evaluator' or 'both').")

    # --- load tasks ---
    tasks = DataLoader(config["tasks"]).get_task_items()

    # --- Step 1: collect evaluatee responses ---
    responses_by_source = collect_responses(config, evaluatees, tasks, save_dir)
    source_names = list(responses_by_source)

    # --- Step 2: qualification exam (Auto-PRE) ---
    qualified, scores = _run_exam(config, evaluators, tasks, responses_by_source, save_dir)

    # --- Steps 3 + 4: peer review (with optional CalibraEval debiasing) ---
    results = {}
    calibration_metrics = None
    text_evaluators = qualified
    if config.get("calibrate", {}).get("enabled") and mode == "pairwise":
        # Logprob-capable judges go through the calibrated 3-prompt path;
        # the rest fall back to standard text peer review.
        cal_results, calibration_metrics, text_evaluators = run_calibrated_review(
            config, qualified, tasks, responses_by_source, save_dir
        )
        results.update(cal_results)

    if text_evaluators:
        prompt_records = build_judge_prompts(mode, config["judgment_prompt"], tasks, responses_by_source)
        results.update(PeerReview(config).run(text_evaluators, prompt_records, save_dir))

    # --- Step 5: chair decision ---
    chair = ChairDecision(config)
    weights = chair.compute_weights(scores)
    decision = chair.aggregate(results, weights, source_names)

    report = {
        "summary": {
            "mode": mode,
            "sources": source_names,
            "ranking": decision.get("ranking"),
            "scores": decision.get("scores"),
            "qualified_evaluators": [c.model_name for c in qualified],
            "calibrated": bool(calibration_metrics),
            "inter_judge_agreement": inter_judge_agreement(results),
        },
        "decision": decision,
        "evaluator_weights": {c.model_name: float(w) for c, w in zip(qualified, weights)},
    }
    if calibration_metrics:
        report["calibration"] = calibration_metrics
    return report


def _run_exam(config, evaluators, tasks, responses_by_source, save_dir):
    """Auto-PRE qualification exam: filters unreliable evaluators and returns
    per-evaluator reliability scores for chair weighting. Pure stub-free now —
    QualificationExam handles graceful degradation internally."""
    return QualificationExam(config).run(evaluators, tasks, responses_by_source, save_dir)
