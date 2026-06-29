"""미설계 연결고리 단위 테스트 — Release Gateway + 평가 추론 단계 (의존성 없음)."""

from __future__ import annotations

import httpx
import pytest

from llmops_core.common.errors import ReleasePending, ReleaseRejected
from llmops_core.common.schemas import EvalCase, EvalResult, ReleaseStatus
from llmops_core.evaluation.predict import PredictConfig, _build_messages, generate_answers
from llmops_core.governance import ReleaseGateway


# ── Release Gateway ──
def _passed_result() -> EvalResult:
    return EvalResult(suite="s", model_ref="m", passed=True, metrics={"faithfulness": 0.9})


def test_release_request_then_approve():
    gw = ReleaseGateway()
    req = gw.request(_passed_result(), requested_by="ci")
    assert req.status == ReleaseStatus.PENDING
    # 승인 전에는 승격 불가
    with pytest.raises(ReleasePending):
        gw.require_approved(req.id)
    approved = gw.approve(req.id, approver="alice")
    assert approved.status == ReleaseStatus.APPROVED
    assert gw.require_approved(req.id).approver == "alice"


def test_release_reject_blocks_promotion():
    gw = ReleaseGateway()
    req = gw.request(_passed_result())
    gw.reject(req.id, approver="bob", reason="회귀 의심")
    with pytest.raises(ReleaseRejected):
        gw.require_approved(req.id)


def test_release_request_blocked_when_auto_gate_failed():
    gw = ReleaseGateway()
    failing = EvalResult(suite="s", model_ref="m", passed=False)
    with pytest.raises(ReleaseRejected):
        gw.request(failing)


def test_approve_is_idempotent():
    gw = ReleaseGateway()
    req = gw.request(_passed_result())
    gw.approve(req.id, "alice")
    # 두 번째 승인 호출은 상태를 바꾸지 않음(멱등)
    again = gw.approve(req.id, "carol")
    assert again.approver == "alice"


# ── 평가 추론 단계 ──
def test_build_messages_injects_contexts():
    case = EvalCase(id="1", question="환불?", contexts=["14일 이내 환불"])
    msgs = _build_messages(case, PredictConfig(base_url="x", api_key="k", model="m"))
    assert msgs[0]["role"] == "system" and "14일" in msgs[0]["content"]
    assert msgs[-1]["content"] == "환불?"


def test_generate_answers_fills_answer_via_gateway():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "생성된 답변"}}]},
        )

    cases = [
        EvalCase(id="1", question="q1"),
        EvalCase(id="2", question="q2", answer="이미 있음"),  # 건너뜀
    ]
    cfg = PredictConfig(base_url="http://gw", api_key="k", model="m")
    client = httpx.Client(transport=httpx.MockTransport(handler))
    out = generate_answers(cases, cfg, client=client)
    assert out[0].answer == "생성된 답변"
    assert out[1].answer == "이미 있음"  # 보존
