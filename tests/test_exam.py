"""Offline tests for the Auto-PRE qualification exam (Phase 3).

Mocks judges with controllable reliability and checks that the exam keeps the
reliable ones and drops the unreliable ones.
"""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import autojudger.api as api_mod
from autojudger import evaluate


def _extract(prompt):
    """Pull the two response slots out of a filled judge prompt."""
    a = prompt.split("Response A:", 1)[-1].split("\n", 1)[0].strip()
    b = prompt.split("Response B:", 1)[-1].split("\n", 1)[0].strip()
    return a, b


class ConsistentJudge:
    """Deterministic total order over responses -> always symmetric."""

    def __init__(self, config):
        self.config = config
        self.model_name = config.model_name

    def chat(self, prompt, system=None):
        a, b = _extract(prompt)
        return "A" if (len(a), a) <= (len(b), b) else "B"

    def chat_with_logprobs(self, *a, **k):
        return None


class BiasedJudge(ConsistentJudge):
    """Pure position bias: always picks A -> never symmetric."""

    def chat(self, prompt, system=None):
        return "A"


def _digits(s):
    return "".join(ch for ch in s if ch.isdigit())


class SmartJudge(ConsistentJudge):
    """Matches the task id in the question to the response's id -> pertinent.

    Works whichever source the exam samples, since it keys on the numeric id
    shared between a question and its on-topic answer."""

    def chat(self, prompt, system=None):
        question = prompt.split("Task:", 1)[-1].split("\n", 1)[0]
        qid = _digits(question)
        a, b = _extract(prompt)
        return "A" if _digits(a) == qid else "B"


def _make_config(tmp, apis, tasks_path, exam):
    return {
        "judgment_prompt": (
            "Task: {{question}}\nResponse A: {{response_A}}\nResponse B: {{response_B}}\n"
            "Which is better? Output only 'A' or 'B'."
        ),
        "apis": apis,
        "tasks": tasks_path,
        "mode": "pairwise",
        "output_dir": tmp,
        "exam": exam,
    }


def _write_tasks(path, n):
    # Two distinct sources (A/B), each carrying its task id so SmartJudge can
    # match question id -> on-topic answer id.
    with open(path, "w") as f:
        for i in range(n):
            f.write(json.dumps({"question": f"QID{i}", "response_A": f"ANSWER{i}", "response_B": f"OTHER{i}xx"}) + "\n")


def test_consistency_filtering():
    """A consistent judge is kept; a position-biased judge is dropped."""

    def fake(config):
        return ConsistentJudge(config) if "good" in config.model_name else BiasedJudge(config)

    api_mod.LLMClient = fake
    with tempfile.TemporaryDirectory() as tmp:
        tasks_path = os.path.join(tmp, "tasks.jsonl")
        _write_tasks(tasks_path, 6)
        config = _make_config(
            tmp,
            [
                {"model": "good-judge", "api_key": "x", "role": "evaluator"},
                {"model": "biased-judge", "api_key": "x", "role": "evaluator"},
            ],
            tasks_path,
            {"enabled": True, "consistency": True, "pertinence": False, "self_confidence": False},
        )
        result = evaluate(config)
    qualified = result["summary"]["qualified_evaluators"]
    assert qualified == ["good-judge"], qualified
    print("PASS: consistency exam drops the position-biased judge ->", qualified)


def test_pertinence_filtering():
    """A pertinent (topic-matching) judge is kept; a blind judge is dropped."""

    def fake(config):
        return SmartJudge(config) if "smart" in config.model_name else BiasedJudge(config)

    api_mod.LLMClient = fake
    with tempfile.TemporaryDirectory() as tmp:
        tasks_path = os.path.join(tmp, "tasks.jsonl")
        _write_tasks(tasks_path, 12)
        config = _make_config(
            tmp,
            [
                {"model": "smart-judge", "api_key": "x", "role": "evaluator"},
                {"model": "blind-judge", "api_key": "x", "role": "evaluator"},
            ],
            tasks_path,
            {"enabled": True, "consistency": False, "pertinence": True, "self_confidence": False, "max_samples": 12},
        )
        result = evaluate(config)
    qualified = result["summary"]["qualified_evaluators"]
    assert "smart-judge" in qualified, qualified
    assert "blind-judge" not in qualified, qualified
    print("PASS: pertinence exam drops the topic-blind judge ->", qualified)


def test_single_evaluator_skips_exam():
    """With one evaluator there is nothing to filter; it is kept uniformly."""
    api_mod.LLMClient = ConsistentJudge
    with tempfile.TemporaryDirectory() as tmp:
        tasks_path = os.path.join(tmp, "tasks.jsonl")
        _write_tasks(tasks_path, 4)
        config = _make_config(
            tmp,
            [{"model": "solo-judge", "api_key": "x", "role": "evaluator"}],
            tasks_path,
            {"enabled": True},
        )
        result = evaluate(config)
    assert result["summary"]["qualified_evaluators"] == ["solo-judge"], result["summary"]
    print("PASS: single-evaluator skips exam (graceful degradation)")


if __name__ == "__main__":
    test_consistency_filtering()
    test_pertinence_filtering()
    test_single_evaluator_skips_exam()
    print("\nALL EXAM TESTS PASSED")
