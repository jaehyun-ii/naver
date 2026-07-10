"""문서 인제스트 관리 — 원본 PDF 등록 → ETL(MinerU) → 청킹 → Qdrant 벡터 적재.

원본 PDF DB(MinIO documents-raw + 레지스트리)를 콘솔에 노출한다. 등록은 서버 경로로,
처리(ETL→청킹→벡터화)는 GPU/외부 파서를 쓰므로 백그라운드 스레드로 실행하고 프런트가
상태(pending→processing→processed|failed)를 폴링한다. (파일 업로드는 python-multipart 필요.)
"""

from __future__ import annotations

import io
import json
import mimetypes
import threading
from collections import Counter
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from llmops_core.common.storage import ObjectStore
from llmops_core.console.security import require_perm
from llmops_core.ingest.models import STATUS_PROCESSED, STATUS_PROCESSING
from llmops_core.ingest.pipeline import PROCESSED, RAW, process_document
from llmops_core.ingest.registry import registry
from llmops_core.ingest.service import register_document

router = APIRouter(
    prefix="/api/documents", tags=["documents"],
    dependencies=[Depends(require_perm("data:write"))],
)


class RegisterBody(BaseModel):
    path: str            # 서버에 있는 원본 PDF 경로
    name: str | None = None
    family: str = "auto"  # kr_rule | abs_guide | … | auto(자동 판별)


def _autostart(doc) -> dict:
    """등록/업로드된 문서가 미처리 상태면 ETL→청킹→벡터화를 백그라운드로 즉시 시작한다.
    (동일 내용·처리완료면 멱등 무동작.)"""
    payload = doc.to_payload()
    if doc.status != STATUS_PROCESSED:
        threading.Thread(target=_process, args=(doc.id,), daemon=True).start()
        payload["status"] = STATUS_PROCESSING
    return payload


@router.get("")
def list_documents() -> list[dict]:
    return [d.to_payload() for d in registry().list()]


@router.get("/{doc_id}")
def get_document(doc_id: str) -> dict:
    d = registry().get(doc_id)
    if d is None:
        raise HTTPException(404, f"문서 없음: {doc_id}")
    return d.to_payload()


@router.post("")
def register(body: RegisterBody) -> dict:
    """서버 경로의 PDF를 원본 DB에 등록(멱등, 큐 없이). 이후 /process로 처리."""
    p = Path(body.path)
    if not p.exists():
        raise HTTPException(422, f"경로를 찾을 수 없습니다: {body.path}")
    try:
        doc = register_document(p, name=body.name, family=body.family, enqueue=False)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"등록 실패: {exc}") from exc
    return _autostart(doc)


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    name: str | None = Form(None),
    family: str = Form("auto"),
) -> dict:
    """브라우저에서 PDF 업로드 → 원본 DB 등록(멱등)."""
    data = await file.read()
    fallback = (file.filename or "document").rsplit(".", 1)[0]
    try:
        doc = register_document(data, name=name or fallback, family=family, enqueue=False)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"업로드 실패: {exc}") from exc
    return _autostart(doc)


@router.get("/{doc_id}/raw")
def raw_pdf(doc_id: str) -> StreamingResponse:
    """원본 PDF 스트림(인라인 뷰어용). documents-raw에서 바이트를 프록시한다."""
    d = registry().get(doc_id)
    if d is None:
        raise HTTPException(404, "문서 없음")
    try:
        data = ObjectStore().get_bytes(RAW, d.raw_key)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"원본 조회 실패: {exc}") from exc
    return StreamingResponse(
        io.BytesIO(data), media_type="application/pdf",
        headers={"Content-Disposition": f'inline; filename="{d.name}.pdf"'},
    )


def _process(doc_id: str) -> None:
    try:
        process_document(doc_id)  # ETL→청킹→벡터화. 상태는 내부에서 processed|failed로 기록
    except Exception:  # noqa: BLE001
        pass


@router.get("/{doc_id}/content-list")
def content_list_preview(doc_id: str, limit: int = 80) -> dict:
    """ETL(MinerU) 산출물 content_list.json 프리뷰 — 블록 유형 통계 + 앞부분 블록."""
    d = registry().get(doc_id)
    if d is None or not d.content_list_key:
        raise HTTPException(404, "MinerU 결과물이 없습니다 (미처리 문서)")
    try:
        items = json.loads(ObjectStore().get_bytes(PROCESSED, d.content_list_key))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"content_list 조회 실패: {exc}") from exc
    types = Counter(b.get("type") for b in items)
    return {"total": len(items), "types": dict(types), "blocks": items[:limit]}


@router.get("/{doc_id}/chunks")
def chunks_preview(doc_id: str, limit: int = 80, chunk_type: str | None = None) -> dict:
    """청킹 산출물 chunks.jsonl 프리뷰 — chunk_type 통계 + 앞부분 청크.

    chunk_type 지정 시 해당 유형만(표·그림은 문서 뒤쪽에 몰려 앞부분 프리뷰에 안 잡힘)."""
    d = registry().get(doc_id)
    if d is None or not d.chunks_key:
        raise HTTPException(404, "청킹 결과가 없습니다 (미처리 문서)")
    try:
        text = ObjectStore().get_bytes(PROCESSED, d.chunks_key).decode("utf-8")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"chunks 조회 실패: {exc}") from exc
    rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    types = Counter(r.get("chunk_type") for r in rows)
    sel = [r for r in rows if r.get("chunk_type") == chunk_type] if chunk_type else rows
    return {"total": len(rows), "types": dict(types), "chunks": sel[:limit]}


@router.get("/{doc_id}/asset")
def asset(doc_id: str, path: str) -> StreamingResponse:
    """청크 그림 원본(images/…) 스트림 — documents-processed 버킷에서 프록시.

    figure 청크의 image_path(ETL 상대경로)를 그대로 받는다."""
    d = registry().get(doc_id)
    if d is None:
        raise HTTPException(404, "문서 없음")
    rel = path.replace("\\", "/")
    if not rel.startswith("images/") or ".." in rel:
        raise HTTPException(422, "images/ 하위 경로만 허용됩니다")
    key = f"{d.name}/v{d.version}/{rel}"
    try:
        data = ObjectStore().get_bytes(PROCESSED, key)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(404, f"이미지 없음: {rel}") from exc
    media = mimetypes.guess_type(rel)[0] or "application/octet-stream"
    return StreamingResponse(io.BytesIO(data), media_type=media)


@router.post("/{doc_id}/process")
def process(doc_id: str) -> dict:
    """ETL→청킹→Qdrant 파이프라인을 백그라운드로 실행."""
    d = registry().get(doc_id)
    if d is None:
        raise HTTPException(404, f"문서 없음: {doc_id}")
    if d.status == STATUS_PROCESSING:
        return d.to_payload()
    threading.Thread(target=_process, args=(doc_id,), daemon=True).start()
    return {**d.to_payload(), "status": STATUS_PROCESSING}
