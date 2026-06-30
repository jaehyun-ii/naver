# llmops_core — 자체 LLMOps 플랫폼 코어

> 설계 근거: `design_v2.md` (모듈 조립 방식). v1 설계서의 4·11장을 대체하며, 아키텍처 원칙·워크플로우·Phase 2 연동(1~3·5~10장)은 그대로 유효.

## 핵심 방침

오픈소스 **제품을 통째로 띄우지 않고**, 각 오픈소스에서 **필요한 모듈만 import**해 단일 사내 패키지 `llmops_core`로 조립한다. 멀티테넌시·정책·관측규약·파이프라인 등 **플랫폼의 통제권은 자체 글루 코드가 보유**한다.

| 모드 | 의미 | 대상 |
|---|---|---|
| **Embed** | 핵심 클래스/함수만 import | litellm.Router, vllm engine, trl, unsloth, peft, optuna, mlflow client, ragas/deepeval metrics, OTel SDK |
| **Primitive** | 저수준 프리미티브만, 상위 로직 자체 구현 | langchain-core (Runnable/Prompt/Parser) |
| **Service** | 모듈 분해 불가, 독립 서비스 | Qdrant, MinIO, MLflow 서버, Argo, Harbor, Grafana |

## 패키지 구조 (design_v2 §8)

```
llmops_core/
├─ common/         # 설정·인증·S3(boto3)·공통 스키마·모델클라이언트(가드레일/judge 공용) [핵심 글루]
├─ ingestion/      # pyspark 잡 + 정규화/정확중복 + S3 JSONL io  [L1 Embed]
├─ quality/        # PII(Presidio)·언어(fastText)·근사중복·검증게이트·Argilla [L1·L2]
├─ dataset/        # SFT/Preference 변환·결정적분할·핑거프린트·DVC [L3 Embed]
├─ telemetry/      # OTel SDK + LLM 스팬 시맨틱 규약        [Embed]
├─ gateway/        # litellm.Router + 가상키/예산/정책 + 가드레일 + 응답캐시(litellm) + FastAPI [Embed]
├─ prompts/        # Git-backed 프롬프트 스토어(버전·라벨·롤백) + judge·가드레일 프롬프트 해석
├─ evaluation/     # reference/preference 로컬평가 + LLM-as-judge + 게이트 + 추론(predict) [Embed]
├─ governance/     # Release Gateway(배포 2차 승인 게이트)   [Build]
├─ orchestration/  # langchain-core 프리미티브 기반 체인     [Primitive]
├─ rag/            # 서빙: 임베더(bge-m3/해시폴백)·인메모리스토어·증강 / 헤비: llama_index.core+qdrant [Embed+Service]
├─ serving/        # vllm AsyncLLMEngine + LoRA 매니저       [Embed]
├─ training/       # unsloth + trl + peft 학습기            [Embed]
├─ tuning/         # optuna 탐색                            [Embed]
├─ tracking/       # mlflow client + dvc.api               [Embed+Service]
├─ registry/       # mlflow 재export + LoRA merge/ONNX 변환  [Embed+Build]
└─ pipelines/      # Argo WorkflowTemplate/Events/Cron · 트리거 [Service 선언]

deploy/            # 인프라 서비스 매니페스트 (docker-compose / Helm)
```

## 설치

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .                 # 코어 글루(게이트웨이 API·텔레메트리)
pip install -e ".[gateway]"      # + litellm
pip install -e ".[all,dev]"      # GPU 박스 풀 설치
```

> serving(vllm)·training(unsloth)은 GPU/빌드 의존성이 커서 별도 환경에 설치한다.

## 빠르게 띄우기 (로컬 핵심 슬라이스)

```bash
cp .env.example .env             # 값 채우기
docker compose -f deploy/docker-compose.yaml up -d   # Qdrant·MinIO·MLflow·OTel·Grafana
uvicorn llmops_core.gateway.app:app --reload --port 4000
```

게이트웨이는 OpenAI 호환이므로 앱은 논리 모델명만 호출한다:

```bash
curl http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer <virtual-key>" \
  -d '{"model":"hcx-seed-3b","messages":[{"role":"user","content":"안녕"}]}'
```

## Docker 전체 배포

인프라(MinIO·Qdrant·Postgres·MLflow·OTel·Jaeger) + 코어 앱(게이트웨이·콘솔)을 한 번에 띄운다.

```bash
docker compose -f deploy/docker-compose.yaml up -d --build       # 전체 기동(앱 이미지 빌드 포함)
docker compose -f deploy/docker-compose.yaml --profile gpu up -d # + GPU 서빙(vLLM)
docker compose -f deploy/docker-compose.yaml ps                  # 상태 확인
docker compose -f deploy/docker-compose.yaml down               # 정리
```

| 서비스 | 호스트 포트 | 비고 |
|---|---|---|
| 콘솔 UI | http://localhost:4100 | Master Key: `sk-master-changeme` |
| 게이트웨이 | http://localhost:14000 | OpenAI 호환 (`/v1/...`) |
| MLflow UI | http://localhost:5000 | |
| MinIO 콘솔 | http://localhost:19001 | minioadmin / minioadmin |
| Jaeger UI | http://localhost:16686 | LLM 트레이스 |

- **단일 콘솔 통합 뷰**: MLflow·MinIO·Jaeger UI를 따로 띄우지 않고 메인 콘솔(:4100)의 **관측·자산** 그룹에서 조회 — 실험·모델(`/api/tracking`), 스토리지(`/api/storage`), 트레이스(`/api/traces`). 콘솔이 각 API를 읽기전용 프록시(미도달 시 graceful). 외부 UI 포트는 필요 시 닫아도 됨(백엔드는 유지). 깊은 분석은 외부 UI를 직접 열어 보완.
- 코어 앱은 `deploy/app/Dockerfile`(단일 이미지 `llmops/core:latest`)을 게이트웨이/콘솔 두 서비스로 재사용.
- 호스트 포트 충돌 회피로 게이트웨이=14000, MinIO=19000/19001로 매핑(컨테이너 내부·서비스명 네트워킹은 4000/9000 유지).
- 게이트웨이가 실제 모델로 라우팅하려면 `config/model_list.yaml`의 `api_base`를 컨테이너 네트워크 기준으로(`http://serving:8000/v1` 또는 호스트 서버면 `http://host.docker.internal:8000/v1`) 설정. 미설정 시 콘솔 챗은 echo 백엔드로 동작.

## MLOps PoC 웹 — 10개 역량 ↔ 콘솔 탭

콘솔(`console/static/index.html` + `console/routers/`)에서 아래 역량을 각 탭으로 제공한다.

| # | 역량 | 콘솔 탭 / API |
|---|---|---|
| 1 | 프롬프트 엔지니어링 체계 | **프롬프트** (`/api/prompts` 버전·라벨·승격·변형비교) |
| 2 | Naver AI 모델 학습·튜닝 | **파이프라인** (SEED-0.5B, `/api/pipeline`) |
| 3 | 도메인 특화 학습·튜닝 | **파이프라인** (조선 데이터·형태 변환) |
| 4 | MLOps 자동화(학습·평가·배포) | **파이프라인** + **승인·평가** + 스케줄러 |
| 5 | 온프렘-NCP 하이브리드 | **대시보드** (`/api/infra` provider minio/ncp) |
| 6 | GPU 자원 최적화 | **대시보드** (GPU·최적화 현황) |
| 7 | 공통/테넌트 맞춤 정책 | **테넌트 정책** (`/api/keys` + 멀티-LoRA) |
| 8 | 학습 파라미터 최적화 | **HPO** (`/api/tuning/hpo` Optuna) |
| 9 | 데이터 버전별 성능평가 | **데이터 버전별 성능** (`/api/evaluation/by-version`) |
| 10 | 도메인 데이터셋 파인튜닝 | **데이터셋** + **파이프라인** |

추가 탭(서빙 운영 보강): **RAG 지식베이스**(`/api/rag` 색인·검색·증강) · **가드레일·안전**(`/api/safety` 가드레일 점검·설정·캐시 히트율) · **드리프트 탐지**(`/api/drift`) · **서빙·카나리**(`/api/serving`).

실행: `pip install -e ".[gateway,tracking]"` 후 `uvicorn llmops_core.console.app:app --port 4100`(콘솔은 docker 접근 가능한 호스트에서).

## 콘솔 (운영자/사용자 UI)

게이트웨이(OpenAI 호환)와 별개로, 제어평면 콘솔을 제공한다 — 챗 플레이그라운드 · 가상키/예산/모델 관리 · 릴리스 승인 · 데이터 품질/데이터셋.

```bash
uvicorn llmops_core.console.app:app --reload --port 4100
# 브라우저: http://localhost:4100   (Master Key 입력: 기본 sk-master-changeme)
```

- 백엔드: `llmops_core/console/` (FastAPI, 코어 컴포넌트 재사용). 모델 미연결 시 챗은 **echo 백엔드**로 오프라인 동작.
- 프론트: `console/static/index.html` (React, 빌드 불필요 · CDN). 최초 로드 시 네트워크 필요.
- 프로덕션 Vite/TS 빌드로 이전 시: `static/index.html`의 컴포넌트를 `src/`로 분리하고 `npm create vite@latest` 후 `dist/`를 `static/`에 배치(현재 환경엔 Node 미설치).

## 파이프라인 — 업로드→학습→평가→승인→배포 (콘솔 "파이프라인" 탭)

콘솔의 **파이프라인** 탭에서 데이터 업로드부터 배포까지 한 화면에서 운영 실행한다.
finetune/convert/deploy를 docker로 실제 수행(LoRA 학습→병합→vLLM 서빙 교체까지 e2e).
GPU 학습이 분 단위라 백그라운드로 진행하고 프론트가 `/runs/{id}`를 폴링한다.

### 데이터 유형·형태·학습법 (조선 도메인)

파싱된 문서는 DB에 적재돼 있다고 가정하고, 콘솔/파이프라인은 그 데이터를 소비한다.

| 축 | 지원 |
|---|---|
| **데이터 유형** | Instruction(SFT) · Preference(DPO) |
| **데이터 형태** | QA · RAG(컨텍스트 주입) · 판정(verdict) · 4E Reasoning(CoT) · 도구호출(Tool Calling) — `dataset/formats.py` 변환기 |
| **학습 방법** | SFT · DPO (bf16) / **PET: LoRA · DoRA · QLoRA(4bit, bitsandbytes)** |

- SFT/DPO·LoRA/DoRA는 `training/sft.py`의 `run_sft_peft`/`run_dpo_peft`(+`use_dora`). CLI: `python -m llmops_core.training.run --method {sft,dpo} [--dora] [--qlora]`.
- 파이프라인 실행 시 `method`(sft|dpo)·`pet`(lora|dora) 지정. DPO는 `preference`({prompt,chosen,rejected}), SFT는 `labeled`({text,response}) 입력.

파이프라인 단계 동작:
1. **data-ingest/quality/build** — 코어 실코드(정규화·품질게이트·결정적분할·핑거프린트)
2. **finetune** — `llmops/train` 컨테이너에서 trl+peft LoRA → 어댑터
3. **evaluate** — 자동 게이트(임계값) → **release-gate** 승인 요청 생성
4. **approval** — Release Gateway 실연동(콘솔에서 승인/반려)
5. **register** — MLflow에 params·metrics·어댑터 기록
6. **convert** — `peft merge_and_unload` → 병합 safetensors
7. **deploy** — vLLM 컨테이너로 병합모델 서빙 + `model_list.yaml` 갱신 + 게이트웨이 재기동 → 챗에서 학습된 모델 호출

### 준비 (이미지 빌드)

```bash
docker build -f deploy/serving/Dockerfile.hf    -t llmops/hf-serving:latest .   # 서빙
docker build -f deploy/serving/Dockerfile.train -t llmops/train:latest .        # 학습/병합
```

### 실행 (콘솔을 docker 접근 가능한 호스트에서)

```bash
pip install -e ".[gateway,tracking]"   # 콘솔 프로세스에 litellm·mlflow 필요
uvicorn llmops_core.console.app:app --reload --port 4100
# 브라우저 → 파이프라인 탭 → JSONL 업로드/붙여넣기 → 실행 → 승인
```

> 콘솔을 컨테이너로 띄울 경우 `/var/run/docker.sock` 마운트가 필요하다.
> 실행기 설정은 환경변수 `LLMOPS_EXEC__*`(work_dir·이미지명·서빙포트·게이트웨이 컨테이너명 등)로 주입.
> 단위 테스트는 `LLMOPS_PIPELINE_BACKGROUND=0`(동기) + 실행기 대체(conftest)로 docker 없이 배선만 검증.

## 확장 기능 (운영·최적화 축)

| 기능 | 사용 | 상태 |
|---|---|---|
| **HPO(Optuna)** | `python -m llmops_core.tuning.run --train .. --eval .. --trials N` | ✅ 검증(TPE 탐색→최적 lr/r/alpha) |
| **학습법 SFT/DPO/GRPO** | `training.run --method {sft,dpo,grpo}` | ✅ 검증(trl 0.23) |
| **PET LoRA/DoRA** | `--dora` | ✅ 검증 |
| **QLoRA(4bit)** | `--qlora` | ✅ bitsandbytes 환경(H100/H200)에서 동작 |
| **멀티-LoRA 서빙(테넌트별 어댑터)** | `serving.multi_lora_server --base .. --adapter t1=/p1 --adapter t2=/p2` | ✅ 검증(모델명 라우팅) |
| **프롬프트 엔지니어링 평가** | `prompts.optimize --variants .. --cases .. [--promote]` | ✅ 검증(변형 비교→prod 승격) |
| **재학습 스케줄러(Argo Cron 대체)** | `pipelines.scheduler --console .. --interval N [--auto-approve]` | ✅ 검증(콘솔 트리거→자동승인) |
| **NCP 하이브리드 스토리지** | `LLMOPS_S3__PROVIDER=ncp` (엔드포인트 자동 전환) | ✅ S3 seam 검증(MinIO), NCP는 크레덴셜만 |
| **서빙 마이크로배칭** | `serving.batching.MicroBatcher` | ✅ 유틸 + 단위테스트 (처리량 상한은 vLLM) |

## 서빙 운영 보강 — RAG 서빙·가드레일·캐시·평가정합·LLM-judge

일반 LLMOps 구성요소를 보강하고, **평가가 서빙과 동일 경로로 측정**되도록 정합시켰다(H100 검증, 테스트 120개 통과).

### RAG 서빙 (런타임 검색→컨텍스트 주입)
`rag/`에 임베더(설정 `LLMOPS_RAG__EMBEDDER`: `bge-m3` | 무의존 `hashing` 폴백)·인메모리 벡터스토어(디스크 영속, 또는 `backend=qdrant`)·`RagPipeline`을 추가. 게이트웨이가 요청을 검색·주입한다(설정 `LLMOPS_RAG__SERVING_ENABLED=true` 또는 요청별 `extra.rag`). 콘솔 **RAG 지식베이스** 탭(`/api/rag` 색인·검색·증강 미리보기). 기존 llama_index/Qdrant 헤비 경로와 공존.

### 서빙 가드레일 (`gateway/guardrails.py`)
- **입력**: 프롬프트 인젝션 차단(한/영 휴리스틱) + **선택적 모델 분류**(`LLMOPS_GUARDRAILS__MODEL`=논리명, 예: LlamaGuard). 휴리스틱과 OR 결합, 모델 미가용 시 fail-open.
- **출력**: 금칙어·PII(Presidio) 모더레이션. 콘솔 **가드레일·안전** 탭에서 점검(`/api/safety/guardrail/check`, 분류 모델 즉석 선택).

### 응답 캐싱 (litellm 네이티브)
`litellm.Cache`(local/redis/시맨틱)를 Router에 임베드 — 동일 요청 재사용 시 `cost=0`. 히트율 `/admin/cache/stats`(콘솔 안전 탭에 라이브 표시). 설정 `LLMOPS_CACHE__*`.

### 평가 = 서빙 정합
평가가 서빙과 같은 프롬프트·RAG를 적용하도록 보강. `evaluation.local_eval`:
- `--system-prompt` / `--prompt-name`(GitPromptStore prod 프롬프트) — 서빙 프롬프트로 평가.
- `--rag` — 서빙과 동일 `RagPipeline.compose_system`으로 컨텍스트 주입.
- `--judge-model` — **LLM-as-judge** 채점(`judge_score`). 결정적 reference 메트릭과 병행.

파이프라인 탭/`RunPipelineBody`에 `prompt_name`·`use_rag`·`judge_model` 노출.

### 가드레일·judge 모델 선택
둘 다 **`config/model_list.yaml`의 논리 모델명**으로 선택 → 게이트웨이가 백엔드 라우팅(`common/model_client.py` 공용 클라이언트). 콘솔 드롭다운(파이프라인=judge, 안전=가드레일 분류기).

### judge·가드레일 프롬프트도 버전 자산
하드코딩 대신 **GitPromptStore**로 관리 — `judge`(`{q}{ref}{ans}`)·`guardrail-classifier`(`{text}`)·`rag-system`(`{context}`). prod 라벨 등록 시 그것을, 없으면 내장 기본값(코드 수정 없이 prod 승격으로 동작 변경). 콘솔 **프롬프트** 탭 "시스템 특수 프롬프트" 표(`/api/prompts/catalog`)에서 편집·승격. 설정 `LLMOPS_GUARDRAILS__PROMPT_NAME`·`LLMOPS_EVALUATION__JUDGE_PROMPT_NAME`.

### vLLM 처리량 경로
transformers 서버는 레퍼런스/개발용. 프로덕션 처리량(PagedAttention·연속배칭·Multi-LoRA)은
vLLM 컨테이너(`vllm/vllm-openai`)가 기본 서빙 백엔드(`LLMOPS_EXEC__SERVE_BACKEND=vllm`). transformers(hf_server)는 vLLM 미가용 환경용 대체. seam(`/v1/...`)은 동일.

### 대형/게이트 모델
SEED 1.5B(gated)/3B는 HF 접근 토큰 필요: `--base <model>` + `HF_TOKEN`(또는 `PeftSFTConfig.hf_token`). 0.5B는 공개.

## 모듈별 채용 모듈(구체 import)은 `design_v2.md` 부록 참조.
