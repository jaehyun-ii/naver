"""MinIO/S3 통합 뷰 — 버킷·객체 브라우저 + presigned 다운로드(읽기전용 프록시).

MinIO 콘솔을 따로 띄우지 않고 boto3(common.storage)로 콘솔에서 탐색. 미도달 시 graceful.
주의: presigned URL은 설정된 S3 엔드포인트를 가리키므로 브라우저에서 도달 가능한 엔드포인트여야 함.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from llmops_core.common.storage import get_s3_client
from llmops_core.console.security import require_perm

router = APIRouter(prefix="/api/storage", tags=["storage"],
                   dependencies=[Depends(require_perm("read"))])


def _iso(dt) -> str | None:
    return dt.isoformat() if dt is not None else None


@router.get("/buckets")
def buckets() -> dict:
    try:
        cli = get_s3_client()
        bs = cli.list_buckets().get("Buckets", [])
        return {"available": True,
                "buckets": [{"name": b["Name"], "created": _iso(b.get("CreationDate"))}
                            for b in bs]}
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "error": str(exc)}


@router.get("/objects")
def objects(bucket: str, prefix: str = "", limit: int = 200) -> dict:
    try:
        cli = get_s3_client()
        resp = cli.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=limit)
        objs = [{"key": o["Key"], "size": o["Size"], "modified": _iso(o.get("LastModified"))}
                for o in resp.get("Contents", [])]
        return {"available": True, "bucket": bucket, "objects": objs,
                "truncated": resp.get("IsTruncated", False)}
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "error": str(exc)}


@router.get("/presign")
def presign(bucket: str, key: str, expires: int = 600) -> dict:
    """다운로드용 presigned GET URL(기본 10분). 브라우저 도달 가능 엔드포인트 전제."""
    try:
        cli = get_s3_client()
        url = cli.generate_presigned_url(
            "get_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=expires)
        return {"available": True, "url": url}
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "error": str(exc)}


@router.get("/download")
def download(bucket: str, key: str) -> StreamingResponse:
    """객체를 콘솔이 스트리밍 프록시 — 내부 S3 엔드포인트라도 브라우저 다운로드 가능.

    presigned URL이 내부 주소(minio:9000)를 가리켜 브라우저가 못 받는 경우의 대안.
    """
    try:
        obj = get_s3_client().get_object(Bucket=bucket, Key=key)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(404, f"객체 조회 실패: {exc}") from exc
    filename = key.rsplit("/", 1)[-1] or "download"
    return StreamingResponse(
        obj["Body"].iter_chunks(),
        media_type=obj.get("ContentType") or "application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
