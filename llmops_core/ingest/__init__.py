"""문서 인제스트 서브시스템 — 원본 PDF DB → 자동 e2e(ETL→청킹→벡터DB).

    from llmops_core.ingest import register_document        # 생성/수정 → 큐
    from llmops_core.ingest.worker import run_worker         # 큐 소비 → 처리

CLI: `python -m llmops_core.ingest register doc.pdf [--family kr_rule]`
     `python -m llmops_core.ingest worker`
"""

from .models import Document
from .registry import registry
from .service import register_document

__all__ = ["Document", "register_document", "registry"]
