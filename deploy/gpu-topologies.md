# GPU 토폴로지 구성 (쿠버네티스 미사용 · Docker 기반)

GPU 잡 스케줄러(`llmops_core/orchestration/scheduler.py`)가 `LLMOPS_ORCH__NODES`(JSON) 설정으로
노드·GPU 풀을 구성한다. 학습/평가/병합/HPO 잡은 비어있는 디바이스에 핀닝(`--gpus device=N`)되어
**가용 GPU 수만큼 병렬 실행**되고, 서빙은 GPU를 영구 점유한다.

## 1) 단일 서버 H200 × 8

```bash
# 한 노드에 8 GPU — 최대 8개 1-GPU 잡 또는 1개 8-GPU 잡 동시
LLMOPS_ORCH__NODES='[{"name":"local","gpus":8}]'
```

compose(app 서비스 environment)에 추가:
```yaml
environment:
  LLMOPS_ORCH__NODES: '[{"name":"local","gpus":8}]'
```
- 콘솔/게이트웨이가 도는 호스트의 로컬 Docker 소켓 사용(docker_host 생략).
- 학습 컨테이너는 콘솔 프로세스가 `docker run --gpus device=N`으로 기동(호스트에서 콘솔 실행 또는 docker.sock 마운트).

## 2) 2노드 H200 × 4 (4 + 4)

각 노드에서 Docker 데몬을 TCP로 노출하고, 스케줄러가 노드별로 디스패치한다.

```bash
LLMOPS_ORCH__NODES='[
  {"name":"node-a","docker_host":"tcp://10.0.0.1:2375","gpus":4},
  {"name":"node-b","docker_host":"tcp://10.0.0.2:2375","gpus":4}
]'
```
- 잡은 `docker -H tcp://노드 run --gpus device=N`으로 해당 노드에서 실행.
- **전제(분산 FS)**: 학습 산출물(`.runs`)·HF 캐시·`repo`가 두 노드에서 동일 경로로 보이도록 **공유 스토리지(NFS 등)** 또는 노드별 동기화 필요. 모델 아티팩트는 S3(MinIO/NCP)로 공유 권장.
- 각 노드 Docker 데몬 TCP 노출은 보안 구간(사설망/ mTLS)에서만.

## 동작 요약
| 토폴로지 | nodes 설정 | 병렬 잡 |
|---|---|---|
| H200×8 단일 | `[{name:local, gpus:8}]` | 최대 8 |
| H200×4 ×2 | 2개 노드(gpus:4) + docker_host | 노드별 4, 합 8 |
| 개발(GB10) | 기본 `[{name:local, gpus:1}]` | 1 |

## 확인
- 콘솔 **대시보드 → GPU 스케줄러** 카드에서 노드별 점유/여유 실시간 확인.
- `GET /api/infra` 의 `scheduler` 필드(노드·busy·free).
