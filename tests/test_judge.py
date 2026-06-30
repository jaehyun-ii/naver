"""LLM-as-judge — 점수 파싱 + run_judge_eval(fake client)."""

from __future__ import annotations

from types import SimpleNamespace

from llmops_core.evaluation.judge import parse_score, run_judge_eval


def test_parse_score_variants():
    assert parse_score("0.8") == 0.8
    assert parse_score("점수: 1.0") == 1.0
    assert parse_score("0") == 0.0
    assert parse_score("8") == 0.8        # 10점 척도 방어
    assert parse_score("9/10") == 0.9
    assert parse_score("정답입니다") == 0.0  # 숫자 없음
    assert parse_score("") == 0.0


class _FakeClient:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def complete(self, model, messages, **kw):
        self.calls.append((model, messages))
        return self.replies.pop(0)


def _case(q, ref, ans):
    return SimpleNamespace(question=q, expected=ref, answer=ans)


def test_run_judge_eval_averages():
    client = _FakeClient(["1.0", "0.0"])
    cases = [_case("q1", "a1", "a1"), _case("q2", "a2", "틀림")]
    out = run_judge_eval(client, "judge-x", cases)
    assert out["judge_score"] == 0.5
    assert out["judge_n"] == 2
    assert out["judge_errors"] == 0
    # judge 모델명이 호출에 사용됨
    assert all(c[0] == "judge-x" for c in client.calls)


def test_run_judge_eval_graceful_on_error():
    class Boom:
        def complete(self, *a, **k):
            raise RuntimeError("down")
    out = run_judge_eval(Boom(), "judge-x", [_case("q", "r", "a")])
    assert out["judge_score"] == 0.0
    assert out["judge_errors"] == 1
