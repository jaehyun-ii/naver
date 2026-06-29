"""L2 라벨링 — Argilla SDK 임베드 (서버는 Service).

SFT/선호 데이터 어노테이션을 위해 레코드를 Argilla로 보내고, 라벨링 결과를 회수한다.
라벨링 워크플로우 정의(필드/질문)는 자체 코드가 보유, UI/저장은 Argilla 서버가 담당.
"""

from __future__ import annotations

from llmops_core.common.config import get_settings
from llmops_core.common.errors import OptionalDependencyError
from llmops_core.common.schemas import TextRecord


def _client():
    try:
        import argilla as rg
    except ImportError as exc:  # pragma: no cover
        raise OptionalDependencyError("argilla", "quality") from exc
    s = get_settings().argilla
    return rg, rg.Argilla(api_url=s.url, api_key=s.api_key)


def push_for_labeling(records: list[TextRecord], dataset_name: str) -> int:
    """레코드를 Argilla 데이터셋으로 업로드(없으면 생성). 업로드 건수 반환.

    질문: response(모델/사람 답변) + quality(승인/반려) — SFT 큐레이션 최소 구성.
    """
    rg, client = _client()
    ws = get_settings().argilla.workspace

    settings = rg.Settings(
        fields=[rg.TextField(name="text")],
        questions=[
            rg.TextQuestion(name="response", required=True),
            rg.LabelQuestion(name="quality", labels=["approved", "rejected"]),
        ],
    )
    dataset = client.datasets(name=dataset_name, workspace=ws)
    if dataset is None:
        dataset = rg.Dataset(name=dataset_name, workspace=ws, settings=settings)
        dataset.create()

    dataset.records.log(
        [{"text": r.text, "id": r.id, **r.metadata} for r in records]
    )
    return len(records)


def pull_labeled(dataset_name: str) -> list[dict]:
    """승인(approved)된 라벨링 결과만 회수 → 데이터셋 빌더가 SFT 포맷으로 변환."""
    _, client = _client()
    dataset = client.datasets(name=dataset_name, workspace=get_settings().argilla.workspace)
    if dataset is None:
        return []
    out: list[dict] = []
    for rec in dataset.records(with_responses=True):
        responses = {r.question_name: r.value for r in rec.responses}
        if responses.get("quality") == "approved":
            out.append({"text": rec.fields["text"], "response": responses.get("response")})
    return out
