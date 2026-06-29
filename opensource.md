# 오픈소스 채택 검토 보고서 — 라이선스 · 모듈 일람
### LLMOps 파이프라인 구성요소별 "우리가 쓸 수 있는가" 검토 (v3 / 파이프라인 설계 기준)

| 항목 | 내용 |
|---|---|
| 문서 목적 | 설계에 채용한 오픈소스의 **라이선스·유지상태·채택 가능 여부** 검토 |
| 검토 기준 | ① 상업적 사용 가능 ② 라이선스 리스크(copyleft/open-core) ③ 유지보수 상태 ④ 채용 모듈 |
| 주의 | 본 문서는 사전 검토 자료이며, 최종 채택 전 **법무 검토 필요**(특히 AGPL 항목) |

---

## 1. 검토 요약 (TL;DR)

대부분의 구성요소는 **Apache 2.0 / MIT 등 허용형 라이선스**로 **그대로 채택 가능**합니다. 단, 아래 **3건만 주의**가 필요합니다.

| 우선순위 | 대상 | 이슈 | 조치 |
|---|---|---|---|
| 🔴 높음 | **MinIO** | AGPLv3 **+ 커뮤니티판이 2026년 초 아카이브/비유지 상태** | **대안 교체 권고** (SeaweedFS 등) |
| 🟠 중간 | **Grafana** | 코어 AGPLv3 (open-core) | 미수정 내부 사용은 가능 / 또는 Enterprise 무료 바이너리·Perses 대안 |
| 🟡 낮음 | **TEI**(임베딩 서버) | 라이선스 이력이 비표준(확인 필요) | vLLM 임베딩 또는 Infinity(MIT)로 대체 |

→ 나머지(vLLM·LiteLLM·LangChain·LlamaIndex·Qdrant·MLflow·DVC·Optuna·TRL·Unsloth·Axolotl·Ragas·DeepEval·Argo·Spark·Argilla·Great Expectations·**Langfuse 코어** 등)는 **채택 안전**입니다.

---

## 2. 라이선스 등급 기준

| 등급 | 의미 | 해당 라이선스 |
|---|---|---|
| ✅ 안전 | 상업적 사용·수정·배포 자유(고지 의무 정도) | Apache 2.0, MIT, BSD |
| ⚠️ 주의 | copyleft(네트워크 제공 시 소스공개) 또는 open-core | AGPLv3, 일부 EE 라이선스 |
| ⛔ 대안 권고 | 유지중단/라이선스 불확실 → 교체 검토 | (아카이브·비표준) |

> **핵심 패턴**: 우리 설계는 대부분 오픈소스를 **모듈 임베드(라이브러리)** 또는 **별도 서비스(네트워크 API)** 로 사용합니다. AGPL 컴포넌트도 **S3/HTTP 같은 네트워크 프로토콜로 분리 호출**하고 **소스를 수정하지 않으면** 일반적으로 의무가 발생하지 않습니다(5장 참조). 다만 케이스별 법무 확인이 필요합니다.

---

## 3. 단계별 채택 모듈 일람

### 3.1 데이터 상류 (L1~L3)
| 구성요소 | 오픈소스 | 채용 모듈 | 라이선스 | 판정 |
|---|---|---|---|---|
| 빅데이터 적재 | Apache Spark | `pyspark` | Apache 2.0 | ✅ |
| 스트리밍(선택) | Apache Kafka | consumer | Apache 2.0 | ✅ |
| 라벨링 | Argilla | `argilla` SDK | Apache 2.0 | ✅ |
| 라벨링(범용) | Label Studio (Community) | 서버/SDK | Apache 2.0 | ✅ (Enterprise는 상용) |
| 품질검증 | Great Expectations | `great_expectations` | Apache 2.0 | ✅ |
| 근사중복 | datasketch | `MinHash`,`MinHashLSH` | MIT | ✅ |
| PII 마스킹 | Presidio | `presidio_analyzer` | MIT | ✅ |
| 언어감지 | fastText | `fasttext` | MIT | ✅ |
| 증강(선택) | nlpaug | `nlpaug` | MIT | ✅ |
| 데이터셋 | HF datasets | `datasets` | Apache 2.0 | ✅ |
| 데이터 버전 | DVC | `dvc.api` | Apache 2.0 | ✅ |

### 3.2 학습 / 튜닝 (L4)
| 구성요소 | 오픈소스 | 채용 모듈 | 라이선스 | 판정 |
|---|---|---|---|---|
| 베이스 프레임워크 | PyTorch | `torch` | BSD-3 | ✅ |
| 트랜스포머 | Transformers | `AutoModelForCausalLM` 등 | Apache 2.0 | ✅ |
| 어댑터 | PEFT | `LoraConfig`,`get_peft_model` | Apache 2.0 | ✅ |
| 트레이너 | TRL | `SFTTrainer`,`DPOTrainer`,`GRPOTrainer` | Apache 2.0 | ✅ |
| 학습 가속 | Unsloth | `FastLanguageModel` | Apache 2.0 | ✅ |
| 양자화 | bitsandbytes | `bitsandbytes` | MIT | ✅ |
| (대규모 분산) | DeepSpeed | `deepspeed` | Apache 2.0 | ✅ |
| (러너, 선택) | Axolotl | YAML 러너 | Apache 2.0 | ✅ |
| HPO | Optuna | `create_study` 외 | MIT | ✅ |

### 3.3 평가 (L5)
| 구성요소 | 오픈소스 | 채용 모듈 | 라이선스 | 판정 |
|---|---|---|---|---|
| RAG 평가 | Ragas | `ragas.metrics.*` | Apache 2.0 | ✅ |
| LLM 유닛테스트 | DeepEval | `deepeval.metrics.*` | Apache 2.0 | ✅ |
| 프롬프트/레드팀 | promptfoo | CLI/config | MIT | ✅ |
| 표준 벤치마크 | lm-evaluation-harness | CLI/lib | MIT | ✅ |
| 표준 벤치마크 | lighteval | CLI/lib | MIT | ✅ |

### 3.4 서빙 / 게이트웨이 / RAG (L7)
| 구성요소 | 오픈소스 | 채용 모듈 | 라이선스 | 판정 |
|---|---|---|---|---|
| 서빙 엔진 | vLLM | `AsyncLLMEngine`,`LoRARequest` | Apache 2.0 | ✅ |
| 서빙(선택) | SGLang | 서버 | Apache 2.0 | ✅ |
| 게이트웨이 | LiteLLM | `litellm.Router` | MIT | ✅ |
| 오케스트레이션 | LangChain | `langchain-core`만 | MIT | ✅ |
| 검색/인덱싱 | LlamaIndex | `llama-index-core` | MIT | ✅ |
| 벡터 DB | Qdrant | `qdrant-client` / 서버 | Apache 2.0 | ✅ |
| 벡터 DB(선택) | Milvus / Chroma | client/서버 | Apache 2.0 | ✅ |
| 임베딩 서버 | **TEI** | 서버 | **확인 필요** | ⚠️ → Infinity(MIT)/vLLM 대체 |
| 임베딩 서버(대안) | Infinity | 서버 | MIT | ✅ |
| 임베딩 모델 | bge-m3 (FlagEmbedding) | 가중치 | MIT | ✅ (한국어 양호) |
| 프롬프트 최적화(선택) | DSPy | 컴파일러 | MIT | ✅ |

### 3.5 레지스트리 / 파이프라인 / 배포 (L6)
| 구성요소 | 오픈소스 | 채용 모듈 | 라이선스 | 판정 |
|---|---|---|---|---|
| 실험추적/레지스트리 | MLflow | `mlflow` client | Apache 2.0 | ✅ |
| 파이프라인 | Argo Workflows | WorkflowTemplate | Apache 2.0 | ✅ (CNCF) |
| 이벤트 트리거 | Argo Events | EventSource/Sensor | Apache 2.0 | ✅ |
| GitOps 배포 | Argo CD | Application | Apache 2.0 | ✅ (CNCF) |
| 모델 변환 | ONNX / ONNX Runtime | export | MIT/Apache | ✅ |
| 컨테이너 레지스트리 | Harbor | 서버 | Apache 2.0 | ✅ (CNCF) |

### 3.6 관측 / 인프라 (횡단)
| 구성요소 | 오픈소스 | 채용 모듈 | 라이선스 | 판정 |
|---|---|---|---|---|
| LLM 트레이싱 | **Langfuse** | 서버 + SDK | **MIT(코어)** + 소수 EE | ✅ (자체호스팅 상업 사용 가능) |
| 계측 표준 | OpenTelemetry | SDK | Apache 2.0 | ✅ |
| 계측(LLM) | OpenLLMetry | SDK | Apache 2.0 | ✅ |
| 인프라 메트릭 | Prometheus | 서버 | Apache 2.0 | ✅ (CNCF) |
| 대시보드 | **Grafana** | 서버 | **AGPLv3(코어)** | ⚠️ 내부 미수정 사용 가능 / 대안 Perses |
| 오브젝트 스토리지 | **MinIO** | (boto3 접근) | **AGPLv3 + 아카이브** | ⛔ **대안 교체 권고** |
| 컨테이너 | Kubernetes | — | Apache 2.0 | ✅ (CNCF) |
| GPU | NVIDIA GPU Operator | — | Apache 2.0 | ✅ |

---

## 4. 라이선스 리스크 상세 & 대응

### 4.1 🔴 MinIO — AGPLv3 + 커뮤니티판 아카이브
MinIO는 서버·게이트웨이·콘솔이 **AGPLv3**이며(클라이언트 SDK는 Apache 2.0), 더 중요한 점은 **커뮤니티(FOSS) 에디션이 2025년 말 유지보수 모드를 거쳐 2026년 초 아카이브**되었고 사전 컴파일 바이너리 제공이 중단되어 소스 빌드만 가능하다는 것입니다. 즉 라이선스 리스크에 더해 **장기 유지보수 리스크**가 큽니다.
- **대응(권고)**: 온프레미스 S3 백본을 **SeaweedFS(Apache 2.0)**, **Apache Ozone(Apache 2.0)**, 또는 **Ceph + RADOS Gateway(LGPL)** 로 교체. 모두 S3 호환이므로 우리 설계의 S3 seam(boto3)에 무손실 적용. Phase 2에서는 어차피 **NCP Object Storage(S3 호환)** 로 전환되므로 영향 최소.

### 4.2 🟠 Grafana — 코어 AGPLv3 (open-core)
Grafana 코어(및 Loki·Tempo)는 2021년 Apache 2.0 → **AGPLv3** 로 전환되었고, 고급 보안/거버넌스 기능은 Enterprise(상용)입니다. AGPLv3 의무는 **"Grafana 소스를 수정하여 네트워크로 제공할 때"** 발생합니다.
- **대응**: 우리는 Grafana를 **수정 없이 별도 모니터링 서비스로 사용**하므로 일반적으로 의무가 발생하지 않습니다. 완전 회피가 필요하면 ① **Grafana Enterprise 무료 바이너리**(proprietary, 무상, 추가 기능은 유료) 또는 ② **Perses(Apache 2.0, CNCF)** 대시보드 사용.

### 4.3 🟡 TEI(Text Embeddings Inference) — 라이선스 확인 필요
HF TEI는 과거 비표준 라이선스 이력이 있어 현재 조건을 확정 확인해야 합니다.
- **대응**: 임베딩 서빙을 **vLLM(Apache 2.0) 임베딩 엔드포인트** 또는 **Infinity(MIT)** 로 대체하면 라이선스 명확. 임베딩 모델은 **bge-m3(MIT)** 권장.

### 4.4 open-core 일반 (Langfuse 등)
**Langfuse**는 2025년 6월 대부분의 제품 기능을 **MIT로 오픈소스화**했고, 자체호스팅(Docker) 코어는 **상업적 사용 가능**합니다. 소수의 엔터프라이즈 전용 기능만 `/ee` 디렉터리의 상용 라이선스 대상입니다. → **코어 채택 안전**. (다만 향후 EE 경계는 릴리즈별로 확인)

---

## 5. AGPL 의무 발생 조건 (오해 방지)

AGPLv3의 핵심은 **Section 13(네트워크 상호작용)** 입니다. 실무 판단:

| 사용 형태 | 소스공개 의무 |
|---|---|
| 미수정 + 내부 사용 | 일반적으로 **미발생** |
| 미수정 + 네트워크 API로 호출(우리 설계: S3/HTTP) | 일반적으로 **미발생**(우리 코드는 파생물 아님) |
| **소스 수정** + 외부/사용자에게 네트워크 제공 | **발생**(수정분 공개 의무) |
| 바이너리 재배포 | GPL/AGPL 조건 적용 |

> 우리 설계는 AGPL 컴포넌트(MinIO·Grafana)를 **수정하지 않고 별도 서비스로 분리 호출**하므로 위험이 낮습니다. 단, ① 실제 배포 형태, ② 지역 법규, ③ 파생물 해석에 따라 달라질 수 있어 **법무 검토를 권장**합니다. (본 문서는 법률 자문이 아닙니다.)

---

## 6. 최종 채택 권고 스택 (클린)

라이선스·유지보수 안전성을 반영한 권고 구성:

```
데이터 상류 : Spark · Argilla · Label Studio(CE) · Great Expectations · datasketch · Presidio · DVC · HF datasets   [전부 ✅]
학습/튜닝   : PyTorch · Transformers · PEFT · TRL · Unsloth · bitsandbytes · (Axolotl) · Optuna                    [전부 ✅]
평가        : Ragas · DeepEval · promptfoo · lm-eval/lighteval                                                     [전부 ✅]
서빙/RAG    : vLLM · LiteLLM · langchain-core · llama-index-core · Qdrant · Infinity(임베딩) · bge-m3              [전부 ✅]
배포/파이프 : MLflow · Argo Workflows/Events/CD · ONNX · Harbor                                                    [전부 ✅]
관측        : Langfuse(코어 MIT) · OpenTelemetry · Prometheus · (Grafana 또는 Perses)                              [✅ / Grafana ⚠️]
스토리지    : SeaweedFS(또는 Ceph/Ozone)  ← MinIO 대체                                                            [✅]
인프라      : Kubernetes · NVIDIA GPU Operator                                                                    [전부 ✅]
```

**교체 요약**: MinIO → **SeaweedFS**(S3 호환·Apache 2.0), 임베딩 TEI → **Infinity/vLLM**, Grafana는 유지(내부 미수정) 또는 **Perses**.

> 라이선스는 버전·시점에 따라 변동될 수 있으므로, 도입 확정 시점에 각 프로젝트 `LICENSE` 파일로 재확인하고 법무 검토를 거친다.
