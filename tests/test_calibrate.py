"""Offline tests for CalibraEval debiasing (Phase 4).

Validates the NOA calibrator on synthetic logits with a known position bias:
calibration should cut the bias while preserving ranking accuracy. No API needed.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autojudger.calibrate import (
    NOACalibrator,
    combined_pA,
    decisions_from_pA,
    order_flip_rate,
)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _synthetic_biased(n=400, beta=1.0, beta2=0.7, seed=0):
    """Balanced true preferences + a position bias of `beta` toward whichever
    response is shown first.

    Latent quality gap s (A vs B), balanced around 0. The bias adds +beta to the
    first-shown response. p_k is always P(token "A") = P(first-shown wins):
        p1 = P(first wins | A first)        = sigmoid(s + beta)
        p2 = P(first wins | A first, rephrased) = sigmoid(s + beta2)
        p3 = P(first wins | B first)        = sigmoid(-s + beta)
    Prompt 3 swaps the pair, so its "A" token is original-B; the aggregation uses
    (1 - p3) for A. True label: A better iff s > 0.
    """
    rng = np.random.default_rng(seed)
    s = rng.normal(0, 1.5, size=n)
    p1 = _sigmoid(s + beta)
    p2 = _sigmoid(s + beta2)
    p3 = _sigmoid(-s + beta)
    samples = np.stack([p1, p2, p3], axis=1)
    true_label = np.where(s > 0, -1, 1)  # -1 = A better
    return samples, true_label


def test_calibration_reduces_order_flips():
    samples, true_label = _synthetic_biased()

    flip_before = order_flip_rate(samples)
    cal = NOACalibrator(lam=0.05).fit(samples)
    flip_after = order_flip_rate(samples, cal)

    print(f"order-flip rate: before={flip_before:.3f}  after={flip_after:.3f}")
    # Strong injected bias -> many flips raw; calibration should largely remove them.
    assert flip_before > 0.25, flip_before
    assert flip_after < flip_before * 0.4, (flip_before, flip_after)
    print("PASS: calibration removes order-dependent flips")


def test_calibration_preserves_accuracy():
    samples, true_label = _synthetic_biased()
    cal = NOACalibrator(lam=0.05).fit(samples)

    acc_raw = float(np.mean(decisions_from_pA(combined_pA(samples)) == true_label))
    acc_cal = float(np.mean(decisions_from_pA(combined_pA(samples, cal)) == true_label))
    print(f"accuracy vs. truth: raw={acc_raw:.3f}  calibrated={acc_cal:.3f}")
    # Debiasing should not hurt ranking accuracy; here it should help.
    assert acc_cal >= acc_raw, (acc_raw, acc_cal)
    print("PASS: calibration preserves (improves) accuracy")


def test_monotonicity():
    """The learned mapping must be monotone non-decreasing."""
    samples, _ = _synthetic_biased(n=200)
    cal = NOACalibrator().fit(samples)
    xs = np.linspace(0, 1, 50)
    ys = cal.predict(xs)
    assert np.all(np.diff(ys) >= -1e-9), "calibration map is not monotone"
    assert ys[0] <= 0.05 and ys[-1] >= 0.95, (ys[0], ys[-1])
    print("PASS: calibration map is monotone over [0, 1]")


def test_calibrated_pipeline_end_to_end():
    """evaluate() routes logprob-capable judges through the calibrated path and
    surfaces calibration metrics in the report."""
    import json
    import tempfile

    import autojudger.api as api_mod
    from autojudger import evaluate

    class BiasedLogprobJudge:
        """Prefers the longer response but with a position bias toward slot A."""

        def __init__(self, config):
            self.config = config
            self.model_name = config.model_name

        def _pA(self, prompt):
            a = prompt.split("Response A:", 1)[-1].split("\n", 1)[0].strip()
            b = prompt.split("Response B:", 1)[-1].split("\n", 1)[0].strip()
            s = (len(a) - len(b)) / 20.0
            return _sigmoid(s + 1.2)  # +1.2 position bias toward A

        def chat(self, prompt, system=None):
            return "one" if self._pA(prompt) > 0.5 else "two"

        def chat_with_logprobs(self, prompt, target_tokens=("A", "B"), top_logprobs=5):
            pa = self._pA(prompt)
            return {"text": "one", "probs": {target_tokens[0]: pa, target_tokens[1]: 1 - pa}}

    api_mod.LLMClient = BiasedLogprobJudge
    tasks_path = os.path.join(os.path.dirname(__file__), "..", "examples", "tasks_pairwise.jsonl")
    with tempfile.TemporaryDirectory() as tmp:
        config = {
            "judgment_prompt": (
                "Task: {{question}}\nResponse A: {{response_A}}\nResponse B: {{response_B}}\n"
                "Which is better? Output only 'one' or 'two'."
            ),
            "apis": [{"model": "judge-lp", "api_key": "x", "role": "evaluator"}],
            "tasks": os.path.abspath(tasks_path),
            "mode": "pairwise",
            "output_dir": tmp,
            "exam": {"enabled": False},
            "calibrate": {"enabled": True},
        }
        result = evaluate(config)

    assert result["summary"]["calibrated"] is True, result["summary"]
    assert "calibration" in result and "judge-lp" in result["calibration"], result
    m = result["calibration"]["judge-lp"]
    assert "order_flip_before" in m and "order_flip_after" in m, m
    assert result["summary"]["ranking"], result["summary"]
    print("PASS: calibrated pipeline end-to-end")
    print("  calibration metrics:", m)
    print("  ranking:", result["summary"]["ranking"])


if __name__ == "__main__":
    test_calibration_reduces_order_flips()
    test_calibration_preserves_accuracy()
    test_monotonicity()
    test_calibrated_pipeline_end_to_end()
    print("\nALL CALIBRATION TESTS PASSED")
