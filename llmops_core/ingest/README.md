# llmops_core.ingest — 원본 PDF DB → 자동 e2e 인제스션

원본 PDF를 등록/수정하면 **자동으로** MinerU ETL → 도메인 청킹 → 벡터DB 적재까지
연쇄 실행하는 이벤트 기반 파이프라인.

```
register_document(pdf)                     worker (큐 소비)
┌──────────────────────────┐   redis 큐   ┌─────────────────────────────────────┐
│ MinIO documents-raw (블롭)│──enqueue───▶│ ETL(MinerU@서버 + pdftext 후처리)   │
│ Postgres 레지스트리       │             │  → content_list.json                │
│  id·name·sha256·version·  │◀─processed──│ 청킹(kr_rule|abs_guide) → chunks    │
│  status·keys·n_vectors    │             │ Qdrant 적재(child·표·그림만)         │
└──────────────────────────┘             │ MinIO documents-processed 적재       │
   생성/수정 = sha256 변경 = 새 버전       └─────────────────────────────────────┘
```

## 구성

| 파일 | 역할 |
|---|---|
| `models.py` | `Document` 레지스트리 레코드(id·sha256·version·status·산출물 키) |
| `registry.py` | 문서 메타 DB — `common/stores.make_registry`(backend=postgres 영속) |
| `queue.py` | redis 잡 큐(즉시 dequeue·dead-letter, 미가용 시 로컬 폴백) |
| `service.py` | `register_document()` — 업로드+upsert+enqueue, sha256 dedup·버전 |
| `pipeline.py` | 문서 1건 e2e: raw→ETL→적재→청킹→벡터. family 자동판별 |
| `vectorize.py` | 청크 JSONL → Qdrant(child·표·그림만, 문맥화 임베딩, doc_id 멱등) |
| `worker.py` | 큐 소비 루프 |
| `__main__.py` | CLI(register/worker/list/status) |

## 인프라 (docker)

| 서비스 | 포트 | 용도 |
|---|---|---|
| redis | 6379 | 잡 큐 (`docker run -d --name llmops-redis -p 6379:6379 redis:7-alpine`) |
| MinIO | 19000 | 원본/산출물 블롭 (`llmops-infra-minio-1`) |
| Postgres | 5432 | 레지스트리 (`llmops-infra-postgres-1`, db=mlflow) |
| Qdrant | 6333 | 벡터DB (`llmops-infra-qdrant-1`) |
| MinerU VLM | 8009/원격 | ETL 추론 서버 (`llmops_core/etl` 참고) |

## 환경변수

```bash
export LLMOPS_S3__ENDPOINT_URL=http://localhost:19000        # MinIO
export LLMOPS_STORE__BACKEND=postgres                        # 레지스트리 영속(프로세스 공유 필수)
export LLMOPS_STORE__DSN=postgresql://mlflow:mlflow@localhost:5432/mlflow
export PARSER_ENDPOINT=http://localhost:8009                 # MinerU VLM 서버
export MINERU_MODEL_SOURCE=huggingface   HF_TOKEN=hf_...     # 최초 로컬 모델 다운로드
```

> **주의**: `LLMOPS_STORE__BACKEND=postgres` 필수 — memory 백엔드면 register(프로세스 A)와
> worker(프로세스 B)가 레지스트리를 공유하지 못해 워커가 "unknown document"로 실패한다.
> (큐는 redis라 공유되지만 레지스트리는 프로세스별 인메모리이기 때문.)

## 실행

```bash
# 1) 등록(= 생성/수정 이벤트) → 자동으로 큐에 적재
python -m llmops_core.ingest register 문서.pdf --family kr_rule

# 2) 워커 기동(큐 소비 → e2e). 상시 데몬 또는 --once
python -m llmops_core.ingest worker            # 상시
python -m llmops_core.ingest worker --once     # 큐 빌 때까지

python -m llmops_core.ingest list              # 문서/상태 목록
python -m llmops_core.ingest status <doc_id>   # 상세
```

- **생성/수정 감지**: 같은 name·같은 내용(sha256)이고 processed면 **무동작(멱등)**. 내용이
  바뀌면 **버전++ → pending → 재처리**.
- **family**: `auto`면 content_list에서 KR/ABS 자동 판별.

## e2e 검증 완료 (15p KR PDF)

`register kr15.pdf` → `worker --once` 결과:
```
✓ kr15 → status=processed  chunks=194  vectors=180  coll=dev-demo-docchunks
```
- MinIO documents-raw(원본) / documents-processed(content_list·images·chunks 27객체) 적재
- Postgres 레지스트리 status=processed, 키·카운트 기록
- Qdrant `dev-demo-docchunks` 180 벡터(child·표·그림) 검색 동작
- 동일 재등록 → 큐 depth 0(멱등), 내용 변경 → v2 pending(수정 트리거)

ETL 추론 서버는 로컬 GB10(`localhost:8009`) 또는 원격(`192.168.0.4:8002`). e2e는 원격으로
검증(로컬 cu130-nightly vLLM은 Qwen2-VL 멀티모달 병합 버그로 별도 수정 필요 — `etl` 메모 참고).
