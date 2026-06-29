"""S3 백본 접근 — boto3 단일 진입점.

모든 영속 데이터(데이터셋·모델·어댑터·평가 리포트)는 로컬 FS 직접 접근 금지,
반드시 이 래퍼(S3 SDK) 경유. 엔드포인트/키만 바꾸면 MinIO→NCP Object Storage 무손실 전환.
"""

from __future__ import annotations

import io
from functools import lru_cache
from typing import TYPE_CHECKING

import boto3
from botocore.client import Config

from llmops_core.common.config import Settings, get_settings

if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client


@lru_cache
def get_s3_client():  # -> S3Client
    """온프렘 MinIO / NCP Object Storage 공통 S3 클라이언트.

    provider=ncp면 NCP 엔드포인트로, minio면 로컬 엔드포인트로 붙는다(코드 동일).
    하이브리드: 온프렘 학습 산출물을 NCP로 동기화하거나 그 반대도 같은 인터페이스.
    """
    s = get_settings().s3
    return boto3.client(
        "s3",
        endpoint_url=s.effective_endpoint(),
        aws_access_key_id=s.access_key,
        aws_secret_access_key=s.secret_key,
        region_name=s.region,
        config=Config(signature_version="s3v4"),
    )


class ObjectStore:
    """버킷 명명 규칙(`{env}-{domain}-{purpose}`)을 캡슐화한 S3 헬퍼."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.client = get_s3_client()

    def bucket_for(self, purpose: str) -> str:
        return self.settings.bucket(purpose)

    def ensure_bucket(self, purpose: str) -> str:
        bucket = self.bucket_for(purpose)
        existing = {b["Name"] for b in self.client.list_buckets().get("Buckets", [])}
        if bucket not in existing:
            self.client.create_bucket(Bucket=bucket)
        return bucket

    def put_bytes(self, purpose: str, key: str, data: bytes) -> str:
        bucket = self.bucket_for(purpose)
        self.client.put_object(Bucket=bucket, Key=key, Body=io.BytesIO(data))
        return f"s3://{bucket}/{key}"

    def get_bytes(self, purpose: str, key: str) -> bytes:
        bucket = self.bucket_for(purpose)
        return self.client.get_object(Bucket=bucket, Key=key)["Body"].read()

    def upload_file(self, purpose: str, local_path: str, key: str) -> str:
        bucket = self.bucket_for(purpose)
        self.client.upload_file(local_path, bucket, key)
        return f"s3://{bucket}/{key}"
