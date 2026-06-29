"""플랫폼 공통 예외 — 모듈 경계에서 일관된 에러 계약을 제공한다."""

from __future__ import annotations


class LLMOpsError(Exception):
    """모든 플랫폼 예외의 베이스."""


class ConfigError(LLMOpsError):
    """설정 누락/불일치."""


class OptionalDependencyError(LLMOpsError):
    """선택적 의존성(예: litellm, vllm)이 설치되지 않은 모듈을 사용할 때."""

    def __init__(self, package: str, extra: str) -> None:
        super().__init__(
            f"'{package}' 가 필요합니다. 설치: pip install -e \".[{extra}]\""
        )
        self.package = package
        self.extra = extra


# ── 게이트웨이/멀티테넌시 ──
class AuthError(LLMOpsError):
    """가상키 인증 실패."""


class PolicyViolation(LLMOpsError):
    """예산·RPM·모델 화이트리스트 등 테넌트 정책 위반."""


class BudgetExceeded(PolicyViolation):
    """테넌트 월 예산 초과."""


class RateLimited(PolicyViolation):
    """RPM 초과."""


class ModelNotAllowed(PolicyViolation):
    """테넌트 화이트리스트에 없는 모델 호출."""


# ── 데이터 품질 게이트 (상류 L2) ──
class DataQualityFailed(LLMOpsError):
    """데이터 품질 검증(Great Expectations/네이티브 규칙) 미달 — 학습 진입을 차단한다."""

    def __init__(self, failures: dict[str, str]) -> None:
        # failures: rule -> 위반 설명
        detail = "; ".join(f"{rule}: {msg}" for rule, msg in failures.items())
        super().__init__(f"데이터 품질 게이트 미달: {detail}")
        self.failures = failures


# ── 평가 게이트 ──
class EvalGateFailed(LLMOpsError):
    """평가 임계값 게이트 미달 — 파이프라인을 실패시킨다."""

    def __init__(self, failures: dict[str, tuple[float, float]]) -> None:
        # failures: metric -> (actual, threshold)
        detail = ", ".join(
            f"{m}={a:.4f} < {t:.4f}" for m, (a, t) in failures.items()
        )
        super().__init__(f"평가 게이트 미달: {detail}")
        self.failures = failures


# ── 배포 거버넌스 게이트 (Release Gateway) ──
class ReleasePending(LLMOpsError):
    """승인 대기 중 — 아직 배포 승격 불가(파이프라인 보류)."""


class ReleaseRejected(LLMOpsError):
    """승인 반려 — 배포 차단."""
