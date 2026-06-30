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


def test_run_judge_eval_uses_custom_template():
    client = _FakeClient(["1.0"])
    run_judge_eval(client, "j", [_case("Q", "R", "A")],
                   prompt_template="채점 {q}|{ref}|{ans}")
    sent = client.calls[0][1][0]["content"]
    assert sent == "채점 Q|R|A"  # 커스텀 프롬프트 템플릿 적용


# ── 프롬프트 리졸버: 스토어 등록분 우선, 미등록 시 기본값 ──
def test_resolve_template_store_hit_and_fallback(tmp_path):
    from llmops_core.prompts import resolve_template
    from llmops_core.prompts.store import GitPromptStore

    store = GitPromptStore(repo_path=str(tmp_path))
    pr = store.create_version("judge", "스토어판 {q}")
    store.promote("judge", pr.version, "prod")
    # 같은 repo_path를 설정으로 쓰도록 monkeypatch 대신 직접 store로 확인
    assert store.by_label("judge", "prod").template == "스토어판 {q}"
    # 미등록 이름 → 기본값
    assert resolve_template("___none___", "DEF") == "DEF"
