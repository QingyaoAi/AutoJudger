"""Peer review + chair decision (ported and modernized from PRE).

Three responsibilities:
  1. collect_responses   — gather the evaluatee answers being judged
  2. PeerReview          — each evaluator judges every prompt (checkpointed)
  3. ChairDecision       — weighted aggregation into scores / rankings

All intermediate outputs are written as JSONL so an interrupted run resumes
without re-querying any API.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import numpy as np
from tqdm import tqdm

from .utils import fill_template, parse_response


# ---------------------------------------------------------------------------
# Step 1: collect the evaluatee responses that judges will compare
# ---------------------------------------------------------------------------
def collect_responses(config, evaluatee_clients, tasks, save_dir) -> Dict[str, List[str]]:
    """Return ``{source_name: [response per task]}``.

    Two paths:
      * Pre-written — task items already contain response columns
        (``response_A``/``response_B`` for pairwise, ``response`` for pointwise).
        Sources come straight from those columns; no API calls.
      * Generated — each evaluatee API produces one answer per task prompt,
        checkpointed to ``save_dir/task_responses/<source>.jsonl``.
    """
    mode = config["mode"]

    # -- pre-written path ---------------------------------------------------
    if mode == "pairwise" and tasks and "response_A" in tasks[0] and "response_B" in tasks[0]:
        return {
            "A": [str(t["response_A"]) for t in tasks],
            "B": [str(t["response_B"]) for t in tasks],
        }
    if mode == "pointwise" and tasks and "response" in tasks[0]:
        return {"response": [str(t["response"]) for t in tasks]}

    # -- generated path -----------------------------------------------------
    if not evaluatee_clients:
        raise ValueError(
            "No pre-written responses found in tasks and no evaluatee APIs "
            "(role 'evaluatee'/'both') available to generate them."
        )
    template = config.get("task_prompt", "{{question}}")
    out_dir = os.path.join(save_dir, "task_responses")
    os.makedirs(out_dir, exist_ok=True)

    responses_by_source = {}
    for client in evaluatee_clients:
        path = os.path.join(out_dir, f"{client.model_name}.jsonl")
        existing = _read_jsonl(path)
        prompts = [fill_template(template, t) for t in tasks]
        if len(existing) < len(prompts):
            with open(path, "w", encoding="utf-8") as f:
                for line in existing:
                    f.write(json.dumps(line, ensure_ascii=False) + "\n")
                for prompt in tqdm(prompts[len(existing):], desc=f"responses[{client.model_name}]"):
                    resp = client.chat(prompt)
                    rec = {"response": resp}
                    existing.append(rec)
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        responses_by_source[client.model_name] = [r["response"] for r in existing]
    return responses_by_source


# ---------------------------------------------------------------------------
# Build the judge prompts
# ---------------------------------------------------------------------------
def build_judge_prompts(mode, template, tasks, responses_by_source) -> List[dict]:
    """Construct judge-prompt records.

    Pointwise: one record per (source, task) -> {source, task_id, prompt}.
    Pairwise:  both orderings per unordered (sourceA, sourceB, task) ->
               {modelA, modelB, task_id, prompt}.
    The template uses ``{{response_A}}`` / ``{{response_B}}`` (pairwise) or
    ``{{response}}`` (pointwise) for response injection.
    """
    sources = list(responses_by_source)
    records = []

    if mode == "pointwise":
        for src in sources:
            for j, task in enumerate(tasks):
                item = dict(task)
                item["response"] = responses_by_source[src][j]
                records.append({"source": src, "task_id": j, "prompt": fill_template(template, item)})
        return records

    # pairwise
    for a in range(len(sources)):
        for b in range(a + 1, len(sources)):
            sa, sb = sources[a], sources[b]
            for j, task in enumerate(tasks):
                base = dict(task)
                ab = dict(base, response_A=responses_by_source[sa][j], response_B=responses_by_source[sb][j])
                ba = dict(base, response_A=responses_by_source[sb][j], response_B=responses_by_source[sa][j])
                records.append({"modelA": sa, "modelB": sb, "task_id": j, "prompt": fill_template(template, ab)})
                records.append({"modelA": sb, "modelB": sa, "task_id": j, "prompt": fill_template(template, ba)})
    return records


# ---------------------------------------------------------------------------
# Step 3: peer review
# ---------------------------------------------------------------------------
class PeerReview:
    def __init__(self, config):
        self.parser_type = config.get("parser_type", "str")
        self.nominal_list = config.get("_nominal_list")
        self.nominal_ticks = config.get("_nominal_ticks")

    def run(self, evaluator_clients, prompt_records, save_dir) -> Dict[str, List[dict]]:
        """Each evaluator judges every prompt record. Returns
        ``{evaluator_name: [record + response + result]}``. Checkpointed."""
        out_dir = os.path.join(save_dir, "evaluation_responses")
        os.makedirs(out_dir, exist_ok=True)

        results = {}
        for client in evaluator_clients:
            path = os.path.join(out_dir, f"{client.model_name}.jsonl")
            existing = _read_jsonl(path)
            if len(existing) < len(prompt_records):
                with open(path, "w", encoding="utf-8") as f:
                    for line in existing:
                        f.write(json.dumps(line, ensure_ascii=False) + "\n")
                    for rec in tqdm(prompt_records[len(existing):], desc=f"review[{client.model_name}]"):
                        resp = client.chat(rec["prompt"])
                        result = parse_response(resp, self.parser_type, self.nominal_list, self.nominal_ticks)
                        out = dict(rec, response=resp, result=result)
                        existing.append(out)
                        f.write(json.dumps(out, ensure_ascii=False) + "\n")
            results[client.model_name] = existing
        return results


# ---------------------------------------------------------------------------
# Step 5: chair decision
# ---------------------------------------------------------------------------
class ChairDecision:
    def __init__(self, config):
        agg = config.get("aggregation", {})
        self.mode = config["mode"]
        self.strategy = agg.get("strategy", "full")
        self.weighted_method = agg.get("weighted_method", "uniform")
        self.alpha = float(agg.get("alpha", 1.0))

    # -- evaluator weighting (from PRE.weighted_function) -------------------
    def compute_weights(self, scores) -> np.ndarray:
        """scores: list (one per evaluator) of reliability score lists from the
        exam. Empty -> uniform. Higher score -> higher weight."""
        n = len(scores)
        if n == 0:
            return np.array([])
        if not scores[0] or self.weighted_method == "uniform":
            return np.full(n, 1.0 / n)
        s = np.array([sc[0] for sc in scores], dtype=float)
        if self.weighted_method == "log":
            s = np.clip(s, 1e-6, 1 - 1e-6)
            w = np.log(s) - np.log(1 - s)
        elif self.weighted_method == "exp":
            w = np.exp(self.alpha * s)
        elif self.weighted_method == "poly":
            w = s ** self.alpha
        else:
            raise ValueError(f"Unknown weighted_method: {self.weighted_method}")
        w = np.clip(w, 0, None)
        total = w.sum()
        return w / total if total > 0 else np.full(n, 1.0 / n)

    def aggregate(self, results, weights, source_names) -> dict:
        if self.mode == "pointwise":
            return self._aggregate_pointwise(results, weights)
        if self.strategy == "elo":
            return self._aggregate_elo(results, weights, source_names)
        return self._aggregate_pairwise_full(results, weights, source_names)

    # -- shared pairwise vote tallying -------------------------------------
    @staticmethod
    def _collect_pairwise_votes(results, weights):
        """Tally weighted votes per source pair.

        Returns ``{"sa%sb" (sa<=sb): {task_id: advantage}}`` where ``advantage``
        is positive when the first source (sa) is judged better, negative when
        the second (sb) is better. This folds together both prompt orderings
        and all evaluators, normalizing PRE's label convention (result -1 =>
        first response better) into a single signed scale.
        """
        votes = {}
        for ei, ev in enumerate(results):
            for item in results[ev]:
                label = item["result"]
                if label is None:
                    continue
                a, b = item["modelA"], item["modelB"]
                # Align to canonical (sa <= sb) ordering; advantage = +1 for sa.
                if a <= b:
                    key, adv = f"{a}%{b}", -np.sign(label)
                else:
                    key, adv = f"{b}%{a}", np.sign(label)
                votes.setdefault(key, {}).setdefault(item["task_id"], 0.0)
                votes[key][item["task_id"]] += weights[ei] * adv
        return votes

    # -- pointwise: weighted mean score per source -------------------------
    def _aggregate_pointwise(self, results, weights):
        evaluators = list(results)
        per_source = {}  # source -> task_id -> [result per evaluator]
        for ei, ev in enumerate(evaluators):
            for item in results[ev]:
                if item["result"] is None:
                    continue
                per_source.setdefault(item["source"], {}).setdefault(item["task_id"], []).append(
                    (ei, item["result"])
                )
        scores = {}
        for src, tasks in per_source.items():
            vals = []
            for _tid, pairs in tasks.items():
                num = sum(weights[ei] * val for ei, val in pairs)
                den = sum(weights[ei] for ei, _ in pairs)
                if den > 0:
                    vals.append(num / den)
            scores[src] = float(np.mean(vals)) if vals else None
        ranking = sorted((s for s in scores if scores[s] is not None), key=lambda s: -scores[s])
        return {"mode": "pointwise", "scores": scores, "ranking": ranking}

    # -- pairwise full: win-rate matrix ------------------------------------
    def _aggregate_pairwise_full(self, results, weights, source_names):
        votes = self._collect_pairwise_votes(results, weights)

        idx = {s: i for i, s in enumerate(source_names)}
        n = len(source_names)
        wins = np.full((n, n), np.nan)
        pair_detail = {}
        for key, tasks in votes.items():
            ma, mb = key.split("%")
            # advantage > 0 -> first source (ma) judged better on that task.
            decisions = np.array([np.sign(v) for v in tasks.values()])
            # ma win-rate = (#ma wins + 0.5 #ties) / total
            a_rate = float(np.mean(decisions > 0) + 0.5 * np.mean(decisions == 0))
            wins[idx[ma], idx[mb]] = a_rate
            wins[idx[mb], idx[ma]] = 1 - a_rate
            pair_detail[key] = a_rate

        # overall score = mean win-rate against all opponents
        scores = {}
        for s in source_names:
            row = wins[idx[s]]
            row = row[~np.isnan(row)]
            scores[s] = float(np.mean(row)) if row.size else None
        ranking = sorted((s for s in scores if scores[s] is not None), key=lambda s: -scores[s])
        return {
            "mode": "pairwise",
            "strategy": "full",
            "win_rate_A_vs_B": pair_detail,
            "scores": scores,
            "ranking": ranking,
        }

    # -- pairwise ELO ------------------------------------------------------
    def _aggregate_elo(self, results, weights, source_names, k=16.0):
        votes = self._collect_pairwise_votes(results, weights)

        idx = {s: i for i, s in enumerate(source_names)}
        games = []
        for key, tasks in votes.items():
            ma, mb = key.split("%")
            for v in tasks.values():
                games.append((idx[ma], idx[mb], int(np.sign(v))))  # +1 -> ma better
        rng = np.random.default_rng(0)
        rng.shuffle(games)

        ratings = np.full(len(source_names), 1000.0)
        for ra, rb, label in games:
            ea = 1.0 / (1.0 + 10 ** ((ratings[rb] - ratings[ra]) / 400.0))
            sa = (1 + label) / 2.0  # +1->1 (A win), 0->0.5, -1->0
            ratings[ra] += k * (sa - ea)
            ratings[rb] += k * ((1 - sa) - (1 - ea))
        scores = {s: float(ratings[idx[s]]) for s in source_names}
        ranking = sorted(source_names, key=lambda s: -scores[s])
        return {"mode": "pairwise", "strategy": "elo", "scores": scores, "ranking": ranking}


# ---------------------------------------------------------------------------
# Inter-judge agreement (Fleiss's kappa) — a reliability diagnostic
# ---------------------------------------------------------------------------
def inter_judge_agreement(results) -> Optional[float]:
    """Fleiss's kappa across evaluators for pairwise verdicts.

    Each (pair, task) is a subject; each evaluator is a rater assigning one of
    three categories (A better / tie / B better, aligned to the canonical pair
    order). Returns kappa in [-1, 1] (1 = perfect agreement, 0 = chance), or
    None if there are too few evaluators/subjects to compute it.
    """
    evaluators = list(results)
    if len(evaluators) < 2:
        return None

    # subject key -> {evaluator: aligned label}
    subjects: Dict[tuple, Dict[str, int]] = {}
    for ev in evaluators:
        for item in results[ev]:
            if "modelA" not in item or item.get("result") is None:
                return None  # not pairwise / incomplete -> skip metric
            a, b = item["modelA"], item["modelB"]
            label = item["result"] if a <= b else -item["result"]
            key = (f"{min(a, b)}%{max(a, b)}", item["task_id"])
            subjects.setdefault(key, {})[ev] = int(np.sign(label))

    # Keep only subjects rated by every evaluator (Fleiss needs equal raters).
    n_raters = len(evaluators)
    cats = [-1, 0, 1]
    table = []
    for key, votes in subjects.items():
        if len(votes) != n_raters:
            continue
        row = [sum(1 for v in votes.values() if v == c) for c in cats]
        table.append(row)
    if len(table) < 2:
        return None
    return _fleiss_kappa(np.array(table, dtype=float))


def _fleiss_kappa(table: np.ndarray) -> float:
    """Fleiss's kappa from a (subjects x categories) count table."""
    n_sub, _ = table.shape
    n_rat = table.sum(axis=1).max()
    if n_rat < 2:
        return 0.0
    p_cat = table.sum(axis=0) / (n_sub * n_rat)
    p_sub = ((table ** 2).sum(axis=1) - n_rat) / (n_rat * (n_rat - 1))
    p_bar = p_sub.mean()
    p_exp = (p_cat ** 2).sum()
    if np.isclose(p_exp, 1.0):
        return 1.0
    return float((p_bar - p_exp) / (1 - p_exp))


# ---------------------------------------------------------------------------
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
