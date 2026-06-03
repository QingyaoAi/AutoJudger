"""CalibraEval debiasing (Phase 4) — optional, logprob-based.

LLM judges exhibit *selection bias*: they favor a response by its position
(slot A vs B) or by token identity, independent of quality. CalibraEval removes
this with a label-free calibration:

  * 3-prompt protocol — for each (A, B) pair, query the judge three ways and
    read the probability it assigns to "A":
        p1 = P(A | A shown first)
        p2 = P(A | A shown first, rephrased instruction)
        p3 = P(A | B shown first)         (order reversed)
  * NOA (Non-parametric Order-preserving Algorithm) — learn a monotone map f
    over observed probabilities that enforces, without any gold labels:
        symmetry     f(p1) + f(p3) ≈ 1      (reversing order flips the call)
        consistency  f(p1) ≈ f(p2)          (rephrasing doesn't change it)

This module ports that objective (CSHaitao/CalibraEval, ``iso_regression``) as a
clean vectorized optimizer, and predicts via monotone interpolation (so plain
numpy suffices — scikit-learn is not required).

Only used when ``calibrate.enabled`` is set and the evaluator endpoints return
token logprobs; otherwise the pipeline falls back to text-based peer review.
"""

from __future__ import annotations

import json
import os
from typing import List, Optional

import numpy as np
from tqdm import tqdm

from .utils import fill_template


# ---------------------------------------------------------------------------
# NOA calibrator: learn a monotone f enforcing symmetry + format-consistency
# ---------------------------------------------------------------------------
class NOACalibrator:
    def __init__(self, lam: float = 0.05, epochs: int = 2000, lr: float = 10.0):
        # lam weights the consistency term relative to symmetry (CalibraEval default 0.05).
        # Defaults (lr=10, 2000 epochs of full-batch GD) converge the monotone map
        # without the instability seen at larger lr (the softmax map collapses).
        self.lam = lam
        self.epochs = epochs
        self.lr = lr
        self.values: Optional[np.ndarray] = None   # sorted anchor inputs
        self.f_values: Optional[np.ndarray] = None  # calibrated outputs at anchors

    def fit(self, samples) -> "NOACalibrator":
        """samples: array-like of shape (n, 3) holding (p1, p2, p3) per pair."""
        samples = np.asarray(samples, dtype=float)
        if samples.ndim != 2 or samples.shape[1] != 3 or len(samples) == 0:
            raise ValueError("samples must have shape (n, 3)")

        # Anchor grid = unique observed probabilities, plus 0 and 1 endpoints.
        values = np.unique(np.concatenate([samples.ravel(), [0.0, 1.0]]))
        m = len(values)
        idx_of = {v: i for i, v in enumerate(values)}
        S = np.array([[idx_of[v] for v in row] for row in samples])  # (n, 3) anchor indices

        # Monotone f parameterized via cumulative softmax of `params` (length m-1):
        #   e_j = exp(params_j),  cum_k = sum_{j<=k} e_j,  f_k = cum_k / cum_total
        # so f[0]=0, f[m-1]=1, strictly increasing.
        params = np.zeros(m - 1)
        for _ in range(self.epochs):
            f = self._f_from_params(params)            # (m,)
            f1, f2, f3 = f[S[:, 0]], f[S[:, 1]], f[S[:, 2]]
            sym = f1 + f3 - 1.0
            cons = f1 - f2
            # dL/df at each used anchor
            dL_df1 = 2 * sym + 2 * self.lam * cons
            dL_df2 = -2 * self.lam * cons
            dL_df3 = 2 * sym
            grad = self._accumulate_grad(params, S, dL_df1, dL_df2, dL_df3)
            params -= self.lr * grad / len(samples)

        f = self._f_from_params(params)
        self.values = values
        self.f_values = f
        return self

    def predict(self, p):
        """Map raw probabilities through the learned monotone calibration."""
        if self.values is None:
            raise RuntimeError("NOACalibrator must be fit before predict")
        p = np.asarray(p, dtype=float)
        out = np.interp(p, self.values, self.f_values)
        return np.clip(out, 0.0, 1.0)

    # -- internals ----------------------------------------------------------
    @staticmethod
    def _f_from_params(params):
        e = np.exp(params - params.max())  # stabilized
        cum = np.concatenate([[0.0], np.cumsum(e)])
        total = cum[-1]
        return cum / total if total > 0 else cum

    def _accumulate_grad(self, params, S, dL_df1, dL_df2, dL_df3):
        """Analytic gradient w.r.t. params.

        For anchor index s,  ∂f_s/∂params_k = (e_k/T)·(1[k<=s] − f_s),
        with k indexing the m−1 gap params (1-based over anchors).
        """
        e = np.exp(params - params.max())
        T = e.sum()
        if T <= 0:
            return np.zeros_like(params)
        cum = np.concatenate([[0.0], np.cumsum(e)])
        f = cum / T  # (m,)
        eT = e / T   # (m-1,)
        grad = np.zeros_like(params)
        # Each used anchor contributes dL_df * (eT ⊙ (1[k<=s] − f_s)).
        for col, dLdf in ((0, dL_df1), (1, dL_df2), (2, dL_df3)):
            s_idx = S[:, col]                  # (n,)
            # sum over samples of dLdf_i * (1[k<=s_i] − f_{s_i}) for each k
            # term A: sum dLdf_i * 1[k<=s_i]  -> reverse-cumulative over k
            order = np.zeros(len(params))
            # contribution to indicator: for sample with anchor s, params k=1..s get +dLdf
            counts = np.bincount(np.clip(s_idx, 0, len(params)), weights=dLdf, minlength=len(params) + 1)
            indicator_term = np.cumsum(counts[::-1])[::-1][1:]  # k=1..m-1
            # term B: −sum dLdf_i * f_{s_i}  (same scalar for all k)
            scalar = float(np.sum(dLdf * f[s_idx]))
            grad += eT * (indicator_term - scalar)
        return grad


# ---------------------------------------------------------------------------
# 3-prompt protocol
# ---------------------------------------------------------------------------
def generate_three_prompts(judgment_prompt, task, resp_a, resp_b):
    """Build the (p1, p2, p3) prompt variants for one (A, B) pair.

    p1: A first, original wording.
    p2: A first, lightly rephrased instruction (format perturbation).
    p3: B first (order reversed).
    """
    p1 = fill_template(judgment_prompt, dict(task, response_A=resp_a, response_B=resp_b))
    rephrased = (
        "Carefully and impartially decide which response is better.\n" + judgment_prompt
    )
    p2 = fill_template(rephrased, dict(task, response_A=resp_a, response_B=resp_b))
    p3 = fill_template(judgment_prompt, dict(task, response_A=resp_b, response_B=resp_a))
    return p1, p2, p3


def collect_logits(client, triples, target_tokens=("A", "B"), path=None):
    """For each (task, respA, respB, prompts) triple, read P(A) under each of the
    three prompts. Returns ``(samples, ok)`` where samples is an (n,3) array of
    P(A) values and ok indicates the endpoint produced logprobs.

    Checkpointed to ``path`` when given.
    """
    existing = _read_jsonl(path) if path else []
    n_done = len(existing)
    if n_done < len(triples):
        f = open(path, "w", encoding="utf-8") if path else None
        if f:
            for line in existing:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
        for prompts in tqdm(triples[n_done:], desc="calibration logits"):
            row = []
            for prompt in prompts:
                out = client.chat_with_logprobs(prompt, target_tokens=target_tokens)
                if out is None:
                    if f:
                        f.close()
                    return None, False
                row.append(out["probs"][target_tokens[0]])
            existing.append({"pA": row})
            if f:
                f.write(json.dumps({"pA": row}, ensure_ascii=False) + "\n")
        if f:
            f.close()
    samples = np.array([r["pA"] for r in existing], dtype=float)
    return samples, True


# ---------------------------------------------------------------------------
# Decisions + bias metrics
# ---------------------------------------------------------------------------
def combined_pA(samples, calibrator=None):
    """Aggregate the three prompts into a single P(A) per pair.

    Per CalibraEval: combine (p1, p2, 1−p3); calibrate first if a calibrator is
    supplied. Returns an array of P(A) in [0, 1].
    """
    samples = np.asarray(samples, dtype=float)
    p1, p2, p3 = samples[:, 0], samples[:, 1], samples[:, 2]
    if calibrator is not None:
        p1, p2, p3 = calibrator.predict(p1), calibrator.predict(p2), calibrator.predict(p3)
    return (p1 + p2 + (1.0 - p3)) / 3.0


def decisions_from_pA(p_a):
    """P(A) -> pairwise label using PRE convention (-1 = A better, +1 = B better)."""
    p_a = np.asarray(p_a, dtype=float)
    return np.where(p_a > 0.5, -1, 1)


def order_flip_rate(samples, calibrator=None) -> float:
    """Fraction of pairs whose verdict flips when the response order is swapped.

    A bias-free judge gives the same answer whether A is shown first (prompt 1)
    or second (prompt 3); disagreement is pure selection bias. Unlike a pick-rate
    metric this needs no assumption about the balance of true labels. Reported
    before/after calibration — lower is better.
    """
    samples = np.asarray(samples, dtype=float)
    p1, p3 = samples[:, 0], samples[:, 2]
    if calibrator is not None:
        p1, p3 = calibrator.predict(p1), calibrator.predict(p3)
    verdict1_is_A = p1 > 0.5          # prompt 1 says A better
    verdict3_is_A = p3 < 0.5          # prompt 3 (reversed): 1 - p3 > 0.5 -> A better
    return float(np.mean(verdict1_is_A != verdict3_is_A))


# ---------------------------------------------------------------------------
# Pipeline integration: calibrated pairwise review
# ---------------------------------------------------------------------------
def run_calibrated_review(config, evaluators, tasks, responses_by_source, save_dir):
    """Judge every source pair via the 3-prompt protocol + NOA calibration.

    Returns ``(results, metrics, fallback)`` where:
      * results   — ``{evaluator_name: [pairwise records]}`` for evaluators whose
                    endpoint produced logprobs (one record per unordered pair/task,
                    ``result`` in {-1 (A better), +1 (B better)});
      * metrics   — per-evaluator order-flip rate before/after calibration;
      * fallback  — evaluators lacking logprobs, for the caller to judge as text.
    """
    judgment_prompt = config["judgment_prompt"]
    target = _target_tokens(config)
    sources = list(responses_by_source)
    out_dir = os.path.join(save_dir, "calibration")
    os.makedirs(out_dir, exist_ok=True)

    # Build unordered pair/task list once; same items for every evaluator.
    pair_meta, triples = [], []
    for a in range(len(sources)):
        for b in range(a + 1, len(sources)):
            sa, sb = sources[a], sources[b]
            for j, task in enumerate(tasks):
                pair_meta.append((sa, sb, j))
                triples.append(
                    generate_three_prompts(
                        judgment_prompt, task, responses_by_source[sa][j], responses_by_source[sb][j]
                    )
                )

    results, metrics, fallback = {}, {}, []
    for client in evaluators:
        path = os.path.join(out_dir, f"{client.model_name}_logits.jsonl")
        samples, ok = collect_logits(client, triples, target_tokens=target, path=path)
        if not ok:
            fallback.append(client)
            continue
        cal = NOACalibrator(
            lam=float(config.get("calibrate", {}).get("lam", 0.05))
        ).fit(samples)
        labels = decisions_from_pA(combined_pA(samples, cal))
        results[client.model_name] = [
            {"modelA": sa, "modelB": sb, "task_id": j, "result": int(lbl)}
            for (sa, sb, j), lbl in zip(pair_meta, labels)
        ]
        metrics[client.model_name] = {
            "order_flip_before": order_flip_rate(samples),
            "order_flip_after": order_flip_rate(samples, cal),
        }
    return results, metrics, fallback


def _target_tokens(config):
    nl = config.get("_nominal_list")
    if nl and len(nl) >= 2:
        return (nl[0], nl[1])
    return ("A", "B")


def _read_jsonl(path) -> List[dict]:
    if not path or not os.path.exists(path):
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out
