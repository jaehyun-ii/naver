"""ETL(MinerU) 설정 — noksan_ax parser 설정을 이식(동일 기본값).

MinerU 2.5를 `hybrid-http-client` 백엔드로 CLI 호출할 때 쓰는 파라미터 묶음.
환경변수 이름·기본값은 noksan_ax `backend/app/core/config.py`와 동일하게 맞춰
파이프라인 동일성을 보장한다. MinerU 자체는 stock(비수정)이므로 같은 버전
(`mineru[pipeline]`, 3.4.x)만 설치하면 된다.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class EtlConfig:
    # MinerU VLM 서버(OpenAI 호환) 엔드포인트. naver에서 새로 띄운 서버 주소.
    endpoint: str = ""
    model: str = "opendatalab/MinerU2.5-Pro-2605-1.2B"  # 서버측 모델(정보용)
    effort: str = "high"          # medium | high
    parse_method: str = "auto"    # auto | ocr | txt
    image_analysis: bool = True
    formula_enabled: bool = True
    table_enabled: bool = True
    timeout_seconds: int = 300
    min_score: float = 0.28       # pdftext 병합 채택 하한(후처리)

    @classmethod
    def from_env(cls) -> "EtlConfig":
        env = os.environ
        return cls(
            endpoint=env.get("PARSER_ENDPOINT", ""),
            model=env.get("PARSER_MODEL", "opendatalab/MinerU2.5-Pro-2605-1.2B"),
            effort=env.get("PARSER_MINERU_EFFORT", "high").strip().lower(),
            parse_method=env.get("PARSER_PARSE_METHOD", "auto").strip().lower(),
            image_analysis=env.get("PARSER_IMAGE_ANALYSIS", "true").lower() == "true",
            formula_enabled=env.get("PARSER_FORMULA_ENABLED", "true").lower() == "true",
            table_enabled=env.get("PARSER_TABLE_ENABLED", "true").lower() == "true",
            timeout_seconds=int(env.get("PARSER_TIMEOUT_SECONDS", "300")),
            min_score=float(env.get("ETL_PDFTEXT_MIN_SCORE", "0.28")),
        )

    def validate(self) -> None:
        if not self.endpoint:
            raise ValueError(
                "MinerU 엔드포인트가 없습니다. PARSER_ENDPOINT(=hybrid VLM 서버 주소)를 "
                "설정하거나 --endpoint 로 넘기세요."
            )
        if self.effort not in {"medium", "high"}:
            raise ValueError("effort must be 'medium' or 'high'")
        if self.parse_method not in {"auto", "ocr", "txt"}:
            raise ValueError("parse_method must be one of: auto, ocr, txt")
