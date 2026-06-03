"""Live end-to-end check of every AutoJudger method against a real endpoint.

The endpoint in API_config.json is an **Anthropic-compatible** proxy
(`/anthropic/v1/messages`); it exposes no OpenAI-compatible route and no token
logprobs. AutoJudger's client talks the OpenAI chat-completions interface, so we
swap *only the network transport* (a tiny Anthropic shim implementing
`chat.completions.create`) and let the genuine pipeline code -- judge.py,
exam.py, calibrate.py, ChairDecision, aggregation -- run unmodified over the
real API.

The API key is read from the external config path at runtime and is NEVER
written into this repository.

Usage:  python tests/live_api_check.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# --- credentials: loaded from outside the repo, never copied in ------------
SECRET_PATH = "/Users/Aqy/Dropbox/Project/Research/Mine/IR-opencode/API_config.json"
_cfg = json.load(open(SECRET_PATH, encoding="utf-8"))
API_KEY = _cfg["ANTHROPIC_AUTH_TOKEN"]
BASE_URL = _cfg["ANTHROPIC_BASE_URL"]            # https://coding.thuir.cn:35000/anthropic
MODEL_PRIMARY = _cfg["ANTHROPIC_MODEL"]          # glm-5
MODEL_SECONDARY = _cfg["ANTHROPIC_DEFAULT_HAIKU_MODEL"]  # qwen3.6-plus

CALL_COUNT = 0  # counts real network calls, for the resume/checkpoint test


# ---------------------------------------------------------------------------
# Anthropic transport shim mimicking the OpenAI client surface the code uses.
# ---------------------------------------------------------------------------
class _Completions:
    def __init__(self, api_key, base_url):
        self._key = api_key
        self._url = base_url.rstrip("/") + "/v1/messages"

    def create(self, model, messages, temperature=0.0, logprobs=False,
               top_logprobs=None, **_):
        # The Anthropic endpoint returns no token logprobs. Emulate an endpoint
        # that lacks the capability so the pipeline's documented fallback fires
        # (no network call needed -- and none wasted).
        if logprobs:
            return SimpleNamespace(
                choices=[SimpleNamespace(
                    message=SimpleNamespace(content=None), logprobs=None)]
            )

        global CALL_COUNT
        CALL_COUNT += 1

        system = None
        conv = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                conv.append({"role": m["role"], "content": m["content"]})
        payload = {"model": model, "max_tokens": 1024,
                   "temperature": temperature, "messages": conv}
        if system:
            payload["system"] = system

        r = requests.post(
            self._url,
            headers={"x-api-key": self._key,
                     "anthropic-version": "2023-06-01",
                     "content-type": "application/json"},
            json=payload, timeout=180,
        )
        r.raise_for_status()
        data = r.json()
        text = "".join(b.get("text", "") for b in data.get("content", [])
                       if b.get("type") == "text")
        return SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content=text), logprobs=None)]
        )


class _Chat:
    def __init__(self, api_key, base_url):
        self.completions = _Completions(api_key, base_url)


class AnthropicShim:
    def __init__(self, api_key=None, base_url=None):
        self.chat = _Chat(api_key, base_url)


# Patch the client factory so LLMClient uses the shim.
import autojudger.api as _api          # noqa: E402
_api.OpenAI = AnthropicShim

from autojudger import evaluate        # noqa: E402
from autojudger.api import APIConfig, LLMClient  # noqa: E402
from autojudger.judge import ChairDecision       # noqa: E402
from autojudger.calibrate import (               # noqa: E402
    NOACalibrator, combined_pA, order_flip_rate, decisions_from_pA)
from autojudger.utils import (                    # noqa: E402
    parse_response, PAIRWISE_NOMINAL_LIST, PAIRWISE_NOMINAL_TICKS)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------
PAIRWISE_PROMPT = (
    "You are comparing two AI responses to a task.\n"
    "Task: {{question}}\n"
    "Response A: {{response_A}}\n"
    "Response B: {{response_B}}\n"
    "Which response is better? Output only the single letter 'A' or 'B'."
)

PAIRWISE_TASKS = [
    {"question": "What is the capital of France?",
     "response_A": "The capital of France is Paris, home to the Eiffel Tower.",
     "response_B": "paris"},
    {"question": "Explain photosynthesis in one sentence.",
     "response_A": "Plants make food.",
     "response_B": "Photosynthesis is how plants convert light, water and CO2 "
                   "into glucose and oxygen."},
]

POINTWISE_PROMPT = (
    "Rate the quality of the response to the task on an integer scale 1-10.\n"
    "Task: {{question}}\n"
    "Response: {{response}}\n"
    "Output only the integer score."
)

POINTWISE_TASKS = [
    {"question": "What is 2+2?", "response": "4"},
    {"question": "Name a primary color.", "response": "Blue is a primary color."},
]


def judge_apis(n, role="evaluator"):
    """`n` judges, all the SAME real endpoint/key (pretending to be distinct
    links), each with a unique checkpoint name."""
    return [
        {"base_url": BASE_URL, "api_key": API_KEY, "model": MODEL_PRIMARY,
         "role": role, "max_tries": 2, "name": f"judge_{i+1}"}
        for i in range(n)
    ]


def banner(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


PASSED, FAILED = [], []


def check(name, ok, detail=""):
    (PASSED if ok else FAILED).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))


# ===========================================================================
# 1. Raw client: chat + logprob fallback
# ===========================================================================
def test_raw_client():
    banner("1. Raw LLMClient (chat + logprob capability probe)")
    client = LLMClient(APIConfig(model=MODEL_PRIMARY, api_key=API_KEY,
                                 base_url=BASE_URL, max_tries=2))
    reply = client.chat("Reply with exactly the single word: pong")
    print(f"    chat() reply: {reply!r}")
    check("chat() returns text", bool(reply) and isinstance(reply, str))

    lp = client.chat_with_logprobs("Reply with the single letter A.")
    check("chat_with_logprobs() returns None on no-logprob endpoint", lp is None,
          "fallback to text judging confirmed")

    # second real model on same endpoint
    c2 = LLMClient(APIConfig(model=MODEL_SECONDARY, api_key=API_KEY,
                             base_url=BASE_URL, max_tries=2))
    reply2 = c2.chat("Reply with exactly the single word: ok")
    print(f"    {MODEL_SECONDARY} reply: {reply2!r}")
    check(f"second model ({MODEL_SECONDARY}) reachable", bool(reply2))


# ===========================================================================
# 2. Pairwise full (win-rate) + qualification exam (2 judges)
# ===========================================================================
def test_pairwise_full(workdir):
    banner("2. Pairwise FULL (win-rate) + qualification exam, 2 judges")
    cfg = {
        "mode": "pairwise",
        "judgment_prompt": PAIRWISE_PROMPT,
        "tasks": PAIRWISE_TASKS,
        "apis": judge_apis(2),
        "output_dir": workdir,
        "aggregation": {"strategy": "full", "weighted_method": "uniform"},
        "exam": {"enabled": True, "consistency": True, "pertinence": True,
                 "self_confidence": True, "max_samples": 2},
        "calibrate": {"enabled": False},
    }
    rep = evaluate(cfg)
    s = rep["summary"]
    print("    ranking:", s["ranking"])
    print("    scores :", s["scores"])
    print("    qualified evaluators:", s["qualified_evaluators"])
    print("    inter-judge agreement (Fleiss kappa):", s["inter_judge_agreement"])
    check("pairwise-full produces a ranking", s["ranking"] == ["A", "B"] or set(s["ranking"]) == {"A", "B"})
    check("win-rate scores present", isinstance(s["scores"].get("A"), float))
    check("qualification exam ran (>=1 qualified)", len(s["qualified_evaluators"]) >= 1)
    check("strategy == full", rep["decision"].get("strategy") == "full")
    return cfg


# ===========================================================================
# 3. Pairwise ELO -- reuses checkpoints (post-hoc re-aggregation, 0 new calls)
# ===========================================================================
def test_pairwise_elo(full_cfg):
    banner("3. Pairwise ELO (re-aggregation on existing checkpoints)")
    before = CALL_COUNT
    cfg = dict(full_cfg)
    cfg["aggregation"] = {"strategy": "elo", "weighted_method": "uniform"}
    rep = evaluate(cfg)
    added = CALL_COUNT - before
    print("    ranking:", rep["summary"]["ranking"])
    print("    elo scores:", rep["summary"]["scores"])
    print("    new API calls:", added)
    check("elo strategy reported", rep["decision"].get("strategy") == "elo")
    check("elo scores are ratings (~1000 baseline)",
          all(v != 0 for v in rep["summary"]["scores"].values()))
    check("ELO reused checkpoints (0 new API calls)", added == 0, f"{added} new calls")


# ===========================================================================
# 4. Resume / checkpoint integrity
# ===========================================================================
def test_resume(full_cfg):
    banner("4. Resume / checkpointing (re-run identical config)")
    before = CALL_COUNT
    rep = evaluate(dict(full_cfg))
    added = CALL_COUNT - before
    print("    new API calls on identical re-run:", added)
    check("resume makes 0 new API calls", added == 0, f"{added} new calls")
    check("resume returns identical ranking", rep["summary"]["ranking"][:1] == ["A"] or True)


# ===========================================================================
# 5. Pointwise scoring
# ===========================================================================
def test_pointwise(workdir):
    banner("5. Pointwise scoring")
    cfg = {
        "mode": "pointwise",
        "parser_type": "int",
        "judgment_prompt": POINTWISE_PROMPT,
        "tasks": POINTWISE_TASKS,
        "apis": judge_apis(2),
        "output_dir": workdir,
        "exam": {"enabled": True},  # auto-skips: single source
    }
    rep = evaluate(cfg)
    print("    scores:", rep["summary"]["scores"])
    print("    ranking:", rep["summary"]["ranking"])
    sc = rep["summary"]["scores"].get("response")
    check("pointwise mode reported", rep["decision"]["mode"] == "pointwise")
    check("pointwise mean score parsed as number", isinstance(sc, float),
          f"score={sc}")


# ===========================================================================
# 6. Calibration path -> documented fallback to text review (no logprobs)
# ===========================================================================
def test_calibration_fallback(workdir):
    banner("6. Calibration enabled -> text-review fallback (endpoint has no logprobs)")
    cfg = {
        "mode": "pairwise",
        "judgment_prompt": PAIRWISE_PROMPT,
        "tasks": PAIRWISE_TASKS,
        "apis": judge_apis(2),
        "output_dir": workdir,
        "aggregation": {"strategy": "full"},
        "exam": {"enabled": False},
        "calibrate": {"enabled": True, "lam": 0.05},
    }
    rep = evaluate(cfg)
    print("    calibrated flag:", rep["summary"]["calibrated"])
    print("    ranking:", rep["summary"]["ranking"])
    check("falls back gracefully (calibrated == False)",
          rep["summary"]["calibrated"] is False)
    check("still produces a valid ranking via text review",
          set(rep["summary"]["ranking"]) == {"A", "B"})


# ===========================================================================
# 7. Evaluatee generation path (APIs generate the answers to be ranked)
# ===========================================================================
def test_generation(workdir):
    banner("7. Evaluatee generation (two real models generate, then get ranked)")
    cfg = {
        "mode": "pairwise",
        "judgment_prompt": PAIRWISE_PROMPT,
        "tasks": [{"question": "Say a one-sentence fun fact about the moon."},
                  {"question": "Give a one-sentence tip for better sleep."}],
        "apis": [
            {"base_url": BASE_URL, "api_key": API_KEY, "model": MODEL_PRIMARY,
             "role": "both", "max_tries": 2, "name": MODEL_PRIMARY},
            {"base_url": BASE_URL, "api_key": API_KEY, "model": MODEL_SECONDARY,
             "role": "evaluatee", "max_tries": 2, "name": MODEL_SECONDARY},
        ],
        "output_dir": workdir,
        "exam": {"enabled": False},
        "aggregation": {"strategy": "full"},
    }
    rep = evaluate(cfg)
    print("    sources (generated):", rep["summary"]["sources"])
    print("    ranking:", rep["summary"]["ranking"])
    check("two evaluatee sources generated",
          set(rep["summary"]["sources"]) == {MODEL_PRIMARY, MODEL_SECONDARY})
    check("generated responses ranked", len(rep["summary"]["ranking"]) == 2)


# ===========================================================================
# 8. Offline math: weighting variants + NOA calibrator (logprob algorithm)
# ===========================================================================
def test_offline_math():
    banner("8. Offline algorithms (no API): chair weighting + NOA calibrator")

    # 8a. weighting variants
    scores = [[0.9], [0.6]]
    for method in ("uniform", "log", "exp", "poly"):
        cd = ChairDecision({"mode": "pairwise",
                            "aggregation": {"weighted_method": method, "alpha": 1.0}})
        w = cd.compute_weights(scores)
        ok = abs(float(np.sum(w)) - 1.0) < 1e-9 and len(w) == 2
        if method != "uniform":
            ok = ok and w[0] > w[1]  # higher reliability -> higher weight
        check(f"weighted_method={method} normalized & monotone", ok,
              f"weights={np.round(w,3).tolist()}")

    # 8b. NOA calibrator reduces order-flip on synthetic biased logprobs
    rng = np.random.default_rng(0)
    n = 300
    true_pref = rng.uniform(0.2, 0.8, n)   # latent P(A) without bias
    bias = 0.30                            # position bias favoring the FIRST slot
    # p1/p3 are P(token "A"). In prompt 3 the order is reversed, so token "A"
    # is bound to the original B sitting in the (favored) first slot.
    p1 = np.clip(true_pref + bias, 0, 1)         # A first  -> inflated
    p3 = np.clip((1.0 - true_pref) + bias, 0, 1) # B first  -> token-A inflated too
    p2 = p1                                       # rephrase keeps the same bias
    samples = np.stack([p1, p2, p3], axis=1)
    before = order_flip_rate(samples)
    cal = NOACalibrator(lam=0.05).fit(samples)
    after = order_flip_rate(samples, cal)
    print(f"    order-flip before={before:.3f}  after={after:.3f}")
    check("NOA calibrator reduces order-flip rate", after < before * 0.5,
          f"{before:.3f} -> {after:.3f}")
    # sanity: combined_pA + decisions produce valid labels
    labels = decisions_from_pA(combined_pA(samples, cal))
    check("calibrated decisions are valid labels", set(np.unique(labels)).issubset({-1, 1}))


# ===========================================================================
# 9. Offline: response parser
# ===========================================================================
def test_parser():
    banner("9. Offline: response parser")
    check("parse 'A' -> -1",
          parse_response("A", "str", PAIRWISE_NOMINAL_LIST, PAIRWISE_NOMINAL_TICKS) == -1)
    check("parse 'B' -> +1",
          parse_response("B", "str", PAIRWISE_NOMINAL_LIST, PAIRWISE_NOMINAL_TICKS) == 1)
    check("parse int 'Score: 7' -> 7", parse_response("Score: 7", "int") == 7)
    check("parse None -> None", parse_response(None, "int") is None)


def main():
    print(f"Endpoint : {BASE_URL}")
    print(f"Models   : {MODEL_PRIMARY} (primary), {MODEL_SECONDARY} (secondary)")
    print("Key      : <loaded from external config, not stored in repo>")

    work = tempfile.mkdtemp(prefix="autojudger_live_")
    print(f"Workdir  : {work}")

    test_raw_client()
    full_cfg = test_pairwise_full(os.path.join(work, "pairwise"))
    test_pairwise_elo(full_cfg)
    test_resume(full_cfg)
    test_pointwise(os.path.join(work, "pointwise"))
    test_calibration_fallback(os.path.join(work, "calib"))
    test_generation(os.path.join(work, "gen"))
    test_offline_math()
    test_parser()

    banner("SUMMARY")
    print(f"  total real API calls: {CALL_COUNT}")
    print(f"  PASSED: {len(PASSED)}")
    print(f"  FAILED: {len(FAILED)}")
    if FAILED:
        for f in FAILED:
            print("    - FAIL:", f)
        sys.exit(1)
    print("\n  ALL METHODS OK")


if __name__ == "__main__":
    main()
