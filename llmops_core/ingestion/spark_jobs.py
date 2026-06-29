"""대규모 적재/변환 — pyspark 잡 임베드 (L1, 두산 차용).

대용량 원천(문서·로그·DB 덤프)을 분산 정규화·중복제거 후 S3 raw/clean 버킷에 적재.
Spark 클러스터는 Service, 잡 코드만 Embed. s3a 설정은 공통 S3 설정에서 주입한다.

주의: 도메인 SFT 데이터가 소규모(수만~수백만 행)면 Spark는 과한 운영 표면이다.
그 경우 ingestion.text_ops + records(경량 경로)로 충분하며, 본 잡은 진짜 빅데이터일 때만 쓴다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from llmops_core.common.config import get_settings
from llmops_core.common.errors import OptionalDependencyError


@dataclass
class IngestSparkConfig:
    input_path: str  # 예: s3a://dev-demo-raw/source/*.jsonl
    output_path: str  # 예: s3a://dev-demo-datasets/clean/
    text_column: str = "text"
    output_format: str = "parquet"
    repartition: int = 0  # >0이면 출력 파티션 수 지정
    spark_conf: dict = field(default_factory=dict)


def build_spark(app_name: str = "llmops-ingest", extra_conf: dict | None = None):
    """S3 호환 스토리지(s3a)로 구성된 SparkSession 생성 (lazy import)."""
    try:
        from pyspark.sql import SparkSession
    except ImportError as exc:  # pragma: no cover
        raise OptionalDependencyError("pyspark", "ingestion") from exc

    s = get_settings().s3
    builder = (
        SparkSession.builder.appName(app_name)
        .config("spark.hadoop.fs.s3a.endpoint", s.endpoint_url)
        .config("spark.hadoop.fs.s3a.access.key", s.access_key)
        .config("spark.hadoop.fs.s3a.secret.key", s.secret_key)
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
    )
    for k, v in (extra_conf or {}).items():
        builder = builder.config(k, v)
    return builder.getOrCreate()


def normalize_and_dedup(cfg: IngestSparkConfig) -> int:
    """원천 읽기 → 텍스트 정규화 → 내용 해시 기준 중복제거 → 출력. 결과 행수 반환.

    정규화 규칙은 text_ops.normalize_text와 동일(UDF로 래핑)해 경량 경로와 일관.
    """
    from pyspark.sql import functions as F
    from pyspark.sql.types import StringType

    from llmops_core.ingestion.text_ops import normalize_text

    spark = build_spark(extra_conf=cfg.spark_conf)
    try:
        df = spark.read.json(cfg.input_path)
        norm_udf = F.udf(normalize_text, StringType())
        df = df.withColumn(cfg.text_column, norm_udf(F.col(cfg.text_column)))
        df = df.withColumn("_chash", F.sha2(F.col(cfg.text_column), 256))
        df = df.dropDuplicates(["_chash"]).drop("_chash")
        if cfg.repartition > 0:
            df = df.repartition(cfg.repartition)
        df.write.mode("overwrite").format(cfg.output_format).save(cfg.output_path)
        return df.count()
    finally:
        spark.stop()
