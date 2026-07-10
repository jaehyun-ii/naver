# llmops_core.etl — MinerU hybrid 문서 ETL

`noksan_ax`의 MinerU 추출 파이프라인을 이식한 것. PDF를 우리 청커
(`scripts/kr_rule_chunker.py`, `scripts/abs_guide_chunker.py`)의 입력인
`content_list.json`으로 뽑아낸다.

**동일 결과 보장(파이프라인 동일)**: MinerU 자체는 stock(비수정)이라 같은 버전만
설치하면 된다. 병합 후처리(`postprocess.py`)는 noksan_ax `etl_postprocess.py`를
**바이트 단위로 복사**하고 FastAPI/DB 커플링만 벗겼다. mineru CLI 명령·플래그도
noksan과 동일하다.

## 구조

| 파일 | 역할 |
|---|---|
| `mineru.py` | MinerU 2.5 CLI 호출 (`-b hybrid-http-client`, 동일 플래그) |
| `postprocess.py` | pdftext 병합 후처리(결정적) — noksan `etl_postprocess.py` 이식 |
| `pipeline.py` | 오케스트레이션: mineru → raw 스냅샷 → 후처리 → 산출물 수집 |
| `config.py` | 설정(env 이름·기본값을 noksan `config.py`와 일치) |
| `compare.py` | 두 산출물 디렉터리 동일성 비교(검증용, noksan 스크립트 이식) |

## 2단계 구성

```
PDF ──①MinerU VLM 추론(별도 GPU 서버)──▶ raw content_list.json
    ──②pdftext 병합 후처리(이 코드, 결정적)──▶ 최종 content_list.json + .md + images/
```

- **① 추론**은 GPU 서버가 필요(비결정 요소는 여기뿐). 아래처럼 naver에 서버를 띄운다.
- **② 후처리**는 순수 파이썬(결정적). 서버 없이도 재현·검증 가능.

## MinerU VLM 서버 (naver에 새로 띄우기)

`hybrid-http-client`의 `-u` 엔드포인트 = `MinerU2.5-Pro-2605-1.2B`를 서빙하는
OpenAI 호환 VLM 서버. GPU 박스에서:

```bash
# 서버측(GPU): mineru sglang 서버
uv pip install "mineru[sglang]"
export MINERU_MODEL_SOURCE=local          # 또는 modelscope / huggingface
mineru-sglang-server --port 8002
# → 엔드포인트: http://<gpu-host>:8002
```

naver의 vLLM 인프라로 대체 서빙해도 됨(OpenAI 호환이면 동일). 서버 주소를
`PARSER_ENDPOINT`로 넘긴다.

## 실행

```bash
uv pip install -e '.[etl]'                # mineru[pipeline] + pdftext + pymupdf
export PARSER_ENDPOINT=http://<gpu-host>:8002

# 단일 PDF
python -m llmops_core.etl KR/1편_2025.pdf -o data

# 디렉터리 일괄
python -m llmops_core.etl ./pdfs -o data

# 산출물: data/<name>/<name>_content_list.json  ← 청커 입력
python scripts/kr_rule_chunker.py data/<name>/<name>_content_list.json -o data_chunks/x.jsonl
```

기본 설정(noksan `config.py`와 동일): effort=high, parse_method=auto,
image_analysis/formula/table=true. `--no-formula` 등으로 끌 수 있고 env로도 제어
(`PARSER_MINERU_EFFORT`, `PARSER_PARSE_METHOD`, `PARSER_IMAGE_ANALYSIS`, …).

## 동일성 검증

`compare.py`로 기준 산출물과 후보 산출물을 비교(파일 SHA256 + 블록/표/수식 지표):

```bash
python -m llmops_core.etl.compare \
    --left  data/AccidentalLoadAnalysisandDesignforOffshoreStructures-v1 \
    --right data/<new-run>/<name> \
    --left-label baseline --right-label ported --show-markdown-diff
```

**후처리 이식 검증 완료**: 이식한 `postprocess.py`를 실제 KR PDF(462p)+content_list에
돌려 pdftext 462p 추출·8350 슬롯 중 6967 매칭, 구조 불변식(topLevelListLength·
recursiveShape·pageSet) 전부 통과 확인. 병합 로직이 raw MinerU 텍스트를 1873슬롯
개선(설계대로). 전체 e2e(① 추론 포함) 동일성은 VLM 서버를 띄운 뒤 위 compare로 확인.
