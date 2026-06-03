"""Offline integration test: mocks the LLM so no API key/network is needed.

Verifies the Phase 1+2 pipeline end-to-end:
  config -> collect responses -> build prompts -> peer review -> chair decision.
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import autojudger.api as api_mod
from autojudger import evaluate


# A deterministic fake judge: prefers the longer response (a crude but
# verifiable proxy for "more detailed answer"). Returns 'A' or 'B'.
class FakeClient:
    def __init__(self, config):
        self.config = config
        self.model_name = config.model_name

    def chat(self, prompt, system=None):
        # Extract the two response lines from the filled judge prompt; the fake
        # judge prefers the longer (more detailed) response.
        a = prompt.split("Response A:", 1)[-1].split("\n", 1)[0].strip()
        b = prompt.split("Response B:", 1)[-1].split("\n", 1)[0].strip()
        return "A" if len(a) >= len(b) else "B"

    def chat_with_logprobs(self, *args, **kwargs):
        return None


def run():
    # Monkeypatch the real client with the fake one.
    api_mod.LLMClient = FakeClient

    tasks_path = os.path.join(os.path.dirname(__file__), "..", "examples", "tasks_pairwise.jsonl")
    with tempfile.TemporaryDirectory() as tmp:
        config = {
            "judgment_prompt": (
                "Task: {{question}}\n"
                "Response A: {{response_A}}\n"
                "Response B: {{response_B}}\n"
                "Which is better? Output only 'A' or 'B'."
            ),
            "apis": [
                {"model": "judge-1", "api_key": "x", "role": "evaluator"},
                {"model": "judge-2", "api_key": "x", "role": "evaluator"},
            ],
            "tasks": os.path.abspath(tasks_path),
            "mode": "pairwise",
            "output_dir": tmp,
            "aggregation": {"strategy": "full", "weighted_method": "uniform"},
            "exam": {"enabled": False},
        }
        result = evaluate(config)

    # --- assertions ---
    summary = result["summary"]
    assert summary["mode"] == "pairwise", summary
    assert set(summary["sources"]) == {"A", "B"}, summary
    assert summary["ranking"] and summary["ranking"][0] in ("A", "B"), summary
    # Fake judge prefers the longer response; A is longer in 2 of 3 tasks
    # (task 0 and task 2), so A wins the majority and ranks first.
    assert summary["ranking"][0] == "A", f"expected A to win, got {summary['ranking']}"
    assert summary["scores"]["A"] > summary["scores"]["B"], summary["scores"]
    # A wins 2/3 head-to-head -> win-rate 2/3.
    assert abs(summary["scores"]["A"] - 2 / 3) < 1e-9, summary["scores"]
    print("PASS: pairwise full pipeline")
    print("  ranking:", summary["ranking"])
    print("  scores :", summary["scores"])

    # Checkpoint files were written.
    print("PASS: integration test complete")


def run_elo():
    """ELO strategy should also rank A above B on the same data."""
    api_mod.LLMClient = FakeClient
    tasks_path = os.path.join(os.path.dirname(__file__), "..", "examples", "tasks_pairwise.jsonl")
    with tempfile.TemporaryDirectory() as tmp:
        config = {
            "judgment_prompt": (
                "Task: {{question}}\nResponse A: {{response_A}}\nResponse B: {{response_B}}\n"
                "Which is better? Output only 'A' or 'B'."
            ),
            "apis": [{"model": "judge-1", "api_key": "x", "role": "evaluator"}],
            "tasks": os.path.abspath(tasks_path),
            "mode": "pairwise",
            "output_dir": tmp,
            "aggregation": {"strategy": "elo"},
            "exam": {"enabled": False},
        }
        result = evaluate(config)
    assert result["summary"]["ranking"][0] == "A", result["summary"]
    assert result["summary"]["scores"]["A"] > 1000 > result["summary"]["scores"]["B"], result["summary"]
    print("PASS: ELO strategy")
    print("  scores:", result["summary"]["scores"])


def run_pointwise():
    """Pointwise scoring: a 1-5 score per response, averaged per source."""

    class ScoreClient(FakeClient):
        def chat(self, prompt, system=None):
            resp = prompt.split("Response:", 1)[-1].split("\n", 1)[0].strip()
            return "5" if len(resp) > 40 else "2"  # longer -> higher score

    api_mod.LLMClient = ScoreClient
    with tempfile.TemporaryDirectory() as tmp:
        tasks = [
            {"question": "q1", "response": "a short answer here that is reasonably long indeed"},
            {"question": "q2", "response": "tiny"},
        ]
        import json as _json

        path = os.path.join(tmp, "tasks.jsonl")
        with open(path, "w") as f:
            for t in tasks:
                f.write(_json.dumps(t) + "\n")
        config = {
            "judgment_prompt": "Question: {{question}}\nResponse: {{response}}\nScore 1-5:",
            "apis": [{"model": "judge-1", "api_key": "x", "role": "evaluator"}],
            "tasks": path,
            "mode": "pointwise",
            "output_dir": os.path.join(tmp, "out"),
            "exam": {"enabled": False},
        }
        result = evaluate(config)
    score = result["summary"]["scores"]["response"]
    assert abs(score - 3.5) < 1e-9, result["summary"]  # mean of 5 and 2
    print("PASS: pointwise scoring")
    print("  score:", score)


def run_resume():
    """Second run with the same output_dir must reuse checkpoints (no new calls)."""
    calls = {"n": 0}

    class CountingClient(FakeClient):
        def chat(self, prompt, system=None):
            calls["n"] += 1
            return super().chat(prompt, system)

    api_mod.LLMClient = CountingClient
    tasks_path = os.path.join(os.path.dirname(__file__), "..", "examples", "tasks_pairwise.jsonl")
    with tempfile.TemporaryDirectory() as tmp:
        config = {
            "judgment_prompt": (
                "Task: {{question}}\nResponse A: {{response_A}}\nResponse B: {{response_B}}\n"
                "Which is better? Output only 'A' or 'B'."
            ),
            "apis": [{"model": "judge-1", "api_key": "x", "role": "evaluator"}],
            "tasks": os.path.abspath(tasks_path),
            "mode": "pairwise",
            "output_dir": tmp,
            "exam": {"enabled": False},
        }
        evaluate(config)
        first = calls["n"]
        evaluate(config)  # second run should hit checkpoints
        second = calls["n"] - first
    assert first > 0 and second == 0, f"expected resume to make 0 calls, made {second}"
    print(f"PASS: checkpoint resume (run1={first} calls, run2={second} calls)")


if __name__ == "__main__":
    run()
    run_elo()
    run_pointwise()
    run_resume()
    print("\nALL TESTS PASSED")
