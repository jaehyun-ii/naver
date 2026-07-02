"""DAG 단계 3b — Release Gateway 승인 요청 기록 (L5→L6 거버넌스).

평가(자동 게이트) 통과 후 실행된다(Argo 의존성). 승인 요청을 거버넌스 스토어에 기록하고
요청 ID를 출력한다. 실제 승인/반려는 이어지는 Argo `suspend` 노드(resume=승인)로 처리되며,
관리 API가 ReleaseGateway.approve/reject로 상태를 갱신한다.

주의: 운영에서는 ReleaseStore를 Postgres 구현으로 교체해야 컨테이너 간 상태가 영속된다
(현재 인메모리는 단일 프로세스용). 실제 메트릭은 MLflow run에서 로드하도록 확장.
"""

from __future__ import annotations

import argparse

from llmops_core.common.schemas import EvalResult
from llmops_core.common.stores import make_release_store
from llmops_core.governance import ReleaseGateway


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model-ref", default="hcx-seed-3b")
    p.add_argument("--suite", default="ci-gate")
    p.add_argument("--data-version", default=None)
    p.add_argument("--requested-by", default="pipeline")
    args = p.parse_args(argv)

    # 자동 게이트는 evaluate 단계에서 이미 통과(미통과 시 DAG가 여기 도달하지 않음)
    result = EvalResult(
        suite=args.suite, model_ref=args.model_ref, data_version=args.data_version, passed=True
    )
    # 설정(store.backend)에 따라 Postgres 스토어로 생성 → 콘솔(별도 프로세스)이 승인 요청을 본다.
    gateway = ReleaseGateway(store=make_release_store())
    req = gateway.request(result, requested_by=args.requested_by)
    print(f"release request 생성: {req.id} (status={req.status.value}) — 승인 대기")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
