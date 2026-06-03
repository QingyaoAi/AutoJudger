"""Automatic, label-free qualification exam (generalized from Auto-PRE).

Auto-PRE filters unreliable judges using three criteria. Its original code
hardcodes XSum/NFQA datasets and specific model names; here each exam is
reformulated to run on *any* task data + evaluatee responses:

  * Consistency   — a reliable judge gives symmetric verdicts when the two
                    responses are swapped (A-vs-B should mirror B-vs-A).
  * Pertinence    — a reliable judge prefers the answer that actually addresses
                    the question over a fluent but off-topic one (an off-topic
                    answer is borrowed from a different task).
  * Self-confidence — a reliable judge is more certain (higher token prob) on
                    easy pairs (clearly different quality) than on hard pairs
                    (similar quality). Requires logprob-capable endpoints.

Each exam yields a per-evaluator score; evaluators failing any enabled+runnable
exam are dropped. Surviving scores feed the chair's reliability weighting.
"""

from __future__ import annotations

import json
import os
import random
from typing import Dict, List, Optional

import numpy as np
from tqdm import tqdm

from .utils import fill_template, parse_response


class QualificationExam:
    def __init__(self, config):
        exam_cfg = config.get("exam", {})
        self.enabled = exam_cfg.get("enabled", True)
        self.use_consistency = exam_cfg.get("consistency", True)
        self.use_pertinence = exam_cfg.get("pertinence", True)
        self.use_self_confidence = exam_cfg.get("self_confidence", True)
        self.threshold = exam_cfg.get("threshold", "mean")
        self.max_samples = int(exam_cfg.get("max_samples", 20))

        self.judgment_prompt = config["judgment_prompt"]
        self.parser_type = config.get("parser_type", "str")
        self.nominal_list = config.get("_nominal_list")
        self.nominal_ticks = config.get("_nominal_ticks")
        self._rng = random.Random(exam_cfg.get("seed", 0))

    # ------------------------------------------------------------------ run
    def run(self, evaluators, tasks, responses_by_source, save_dir):
        """Return ``(qualified_clients, scores)`` where ``scores[i]`` is a list
        of reliability values for qualified evaluator ``i`` (empty -> uniform).

        Degrades gracefully: with <2 evaluators, or too little data, exams that
        cannot run are skipped; if none can run, everyone qualifies uniformly.
        """
        sources = list(responses_by_source)
        if not self.enabled or len(evaluators) < 2 or len(sources) < 2 or len(tasks) < 2:
            return evaluators, [[] for _ in evaluators]

        exam_dir = os.path.join(save_dir, "exam_responses")
        os.makedirs(exam_dir, exist_ok=True)

        # Build the shared sample sets once (same items for every evaluator).
        cons_pairs = self._build_consistency_pairs(tasks, responses_by_source)
        pert_items = self._build_pertinence_items(tasks, responses_by_source)
        easy_items, hard_items = self._build_confidence_items(tasks, responses_by_source)

        # Collect per-evaluator scores for each runnable exam.
        names = [c.model_name for c in evaluators]
        cons_scores: Dict[str, float] = {}
        pert_scores: Dict[str, float] = {}
        conf_pass: Dict[str, bool] = {}

        run_cons = self.use_consistency and bool(cons_pairs)
        run_pert = self.use_pertinence and bool(pert_items)
        # self-confidence only runs if the first evaluator exposes logprobs.
        run_conf = (
            self.use_self_confidence
            and bool(easy_items)
            and bool(hard_items)
            and _supports_logprobs(evaluators[0])
        )

        for client in evaluators:
            if run_cons:
                cons_scores[client.model_name] = self._consistency_score(
                    client, cons_pairs, os.path.join(exam_dir, f"{client.model_name}_consistency.jsonl")
                )
            if run_pert:
                pert_scores[client.model_name] = self._pertinence_score(
                    client, pert_items, os.path.join(exam_dir, f"{client.model_name}_pertinence.jsonl")
                )
            if run_conf:
                conf_pass[client.model_name] = self._self_confidence_pass(client, easy_items, hard_items)

        # If nothing ran, fall back to uniform.
        if not (run_cons or run_pert or run_conf):
            return evaluators, [[] for _ in evaluators]

        # Thresholds for accuracy-like exams (mean across candidates by default).
        cons_thr = self._resolve_threshold(list(cons_scores.values())) if run_cons else None
        pert_thr = self._resolve_threshold(list(pert_scores.values())) if run_pert else None

        qualified, scores = [], []
        for client in evaluators:
            name = client.model_name
            reliability = []
            passed = True
            if run_cons:
                if cons_scores[name] < cons_thr:
                    passed = False
                reliability.append(cons_scores[name])
            if run_pert:
                if pert_scores[name] < pert_thr:
                    passed = False
                reliability.append(pert_scores[name])
            if run_conf and not conf_pass[name]:
                passed = False
            if passed:
                qualified.append(client)
                # Primary weight = mean of accuracy-like scores (in (0,1)).
                combined = float(np.mean(reliability)) if reliability else 1.0
                scores.append([combined])

        # Never disqualify everyone — if the exam is too strict, keep all uniform.
        if not qualified:
            return evaluators, [[] for _ in evaluators]
        return qualified, scores

    # -------------------------------------------------------- sample builders
    def _sample_task_ids(self, n_tasks):
        ids = list(range(n_tasks))
        self._rng.shuffle(ids)
        return ids[: self.max_samples]

    def _build_consistency_pairs(self, tasks, responses_by_source):
        """Each item: (task, respX, respY) from two distinct sources."""
        sources = list(responses_by_source)
        pairs = []
        for tid in self._sample_task_ids(len(tasks)):
            sa, sb = self._rng.sample(sources, 2)
            pairs.append((tasks[tid], responses_by_source[sa][tid], responses_by_source[sb][tid]))
        return pairs

    def _build_pertinence_items(self, tasks, responses_by_source):
        """Each item: (task_i, on_topic_resp, off_topic_resp).

        on_topic = a source's answer to task_i; off_topic = the same source's
        answer to a *different* task (fluent but irrelevant to task_i).
        """
        sources = list(responses_by_source)
        n = len(tasks)
        items = []
        for tid in self._sample_task_ids(n):
            src = self._rng.choice(sources)
            other = self._rng.choice([j for j in range(n) if j != tid])
            on_topic = responses_by_source[src][tid]
            off_topic = responses_by_source[src][other]
            items.append((tasks[tid], on_topic, off_topic))
        return items

    def _build_confidence_items(self, tasks, responses_by_source):
        """easy = (task, on_topic, off_topic) — clearly different quality;
        hard = (task, respX, respY) — two on-topic answers, similar quality."""
        easy = self._build_pertinence_items(tasks, responses_by_source)
        hard = self._build_consistency_pairs(tasks, responses_by_source)
        return easy, hard

    # ----------------------------------------------------------- exam scorers
    def _consistency_score(self, client, pairs, path) -> float:
        """Symmetry rate: verdict(X,Y) should mirror verdict(Y,X)."""
        ab = [self._pair_prompt(t, x, y) for (t, x, y) in pairs]
        ba = [self._pair_prompt(t, y, x) for (t, x, y) in pairs]
        res_ab = self._judge(client, ab, path.replace(".jsonl", "_ab.jsonl"))
        res_ba = self._judge(client, ba, path.replace(".jsonl", "_ba.jsonl"))
        ok = tot = 0
        for a, b in zip(res_ab, res_ba):
            if a is None or b is None:
                continue
            tot += 1
            # label -1 => first(A) better. Symmetric if AB and BA agree on winner.
            if np.sign(a) == -np.sign(b):
                ok += 1
        return ok / tot if tot else 0.0

    def _pertinence_score(self, client, items, path) -> float:
        """Accuracy at preferring the on-topic answer (known label)."""
        prompts, correct = [], []
        for (task, on_topic, off_topic) in items:
            if self._rng.random() < 0.5:
                prompts.append(self._pair_prompt(task, on_topic, off_topic))
                correct.append(-1)  # on-topic is A -> A better
            else:
                prompts.append(self._pair_prompt(task, off_topic, on_topic))
                correct.append(1)  # on-topic is B -> B better
        results = self._judge(client, prompts, path)
        ok = tot = 0
        for r, c in zip(results, correct):
            if r is None:
                continue
            tot += 1
            if np.sign(r) == np.sign(c):
                ok += 1
        return ok / tot if tot else 0.0

    def _self_confidence_pass(self, client, easy_items, hard_items) -> bool:
        """True if mean confidence on easy pairs exceeds that on hard pairs."""
        easy_conf = self._mean_confidence(client, easy_items)
        hard_conf = self._mean_confidence(client, hard_items)
        if easy_conf is None or hard_conf is None:
            return True  # can't measure -> don't disqualify
        return easy_conf > hard_conf

    def _mean_confidence(self, client, items) -> Optional[float]:
        target = self._target_tokens()
        confs = []
        for (task, x, y) in items:
            prompt = self._pair_prompt(task, x, y)
            out = client.chat_with_logprobs(prompt, target_tokens=target)
            if out is None:
                return None
            confs.append(max(out["probs"].values()))
        return float(np.mean(confs)) if confs else None

    # --------------------------------------------------------------- helpers
    def _pair_prompt(self, task, resp_a, resp_b) -> str:
        item = dict(task, response_A=resp_a, response_B=resp_b)
        return fill_template(self.judgment_prompt, item)

    def _target_tokens(self):
        # First two nominal labels are the "first/second better" tokens.
        if self.nominal_list and len(self.nominal_list) >= 2:
            return (self.nominal_list[0], self.nominal_list[1])
        return ("A", "B")

    def _judge(self, client, prompts, path) -> List:
        """Judge a list of prompts, checkpointed to ``path``; return parsed
        results (signed labels / None)."""
        existing = _read_jsonl(path)
        if len(existing) < len(prompts):
            with open(path, "w", encoding="utf-8") as f:
                for line in existing:
                    f.write(json.dumps(line, ensure_ascii=False) + "\n")
                for prompt in tqdm(prompts[len(existing):], desc=f"exam[{client.model_name}]", leave=False):
                    resp = client.chat(prompt)
                    result = parse_response(resp, self.parser_type, self.nominal_list, self.nominal_ticks)
                    rec = {"response": resp, "result": result}
                    existing.append(rec)
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return [r["result"] for r in existing]

    def _resolve_threshold(self, scores) -> float:
        if self.threshold == "mean":
            return float(np.mean(scores)) if scores else 0.0
        return float(self.threshold)


def _supports_logprobs(client) -> bool:
    """Probe whether the client's endpoint returns logprobs (cheap 1-token call)."""
    try:
        out = client.chat_with_logprobs("Reply with the single letter A.", target_tokens=("A", "B"))
        return out is not None
    except Exception:
        return False


def _read_jsonl(path) -> List[dict]:
    if not os.path.exists(path):
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out
