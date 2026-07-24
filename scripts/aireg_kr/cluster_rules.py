"""7개 발행처 규정 조항의 의미 클러스터 실측 — 학습셋 규모 산정·train/test 분리 기반.

"18k 조항은 실제로 몇 개의 requirement family인가"를 가설이 아니라 측정으로 답한다.
- 대상: data_chunks/*/ 의 실질 요건 parent 청크(GUIDE 미러·정의류 제외)
- 임베딩: BAAI/bge-m3 (다국어 — KR 한국어 vs 타 선급 영어 간 cross-lingual 매칭 포함)
- 클러스터: 코사인 유사도 임계값별 연결 요소(union-find). GPU 블록 행렬곱.
- 출력: data_aireg/cluster_report.json + 조항→클러스터 매핑(cluster_map.jsonl,
  threshold 0.85 기준) — 이후 층화 샘플링과 requirement-family 단위 분리에 사용.

컨테이너 실행(GPU):
    docker run --rm --gpus all --entrypoint python \
      -v /home/jaehyun/Dev/naver:/work -v ~/.cache/huggingface:/hf -w /work \
      llmops/hf-serving -m scripts.aireg_kr.cluster_rules
"""
from __future__ import annotations

import glob
import json
import re
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "data_aireg"
NON_REQ = re.compile(r"정의|용어|일반사항$|적용$|제출도면")
THRESHOLDS = [0.95, 0.90, 0.85, 0.80]
MAP_THRESHOLD = 0.85


def load_parents() -> list[dict]:
    rows = []
    for soc_dir in sorted(glob.glob(str(ROOT / "data_chunks" / "*") + "/")):
        soc = Path(soc_dir).name
        if soc.startswith("_"):
            continue
        for f in glob.glob(soc_dir + "*_chunks.jsonl"):
            for l in open(f, encoding="utf-8"):
                if '"parent"' not in l:
                    continue
                r = json.loads(l)
                if (r.get("chunk_level") == "parent"
                        and not r["chunk_id"].startswith("GUIDE")
                        and r.get("content_tokens", 0) >= 150
                        and not NON_REQ.search(r.get("article_title") or "")):
                    rows.append({"soc": soc, "chunk_id": r["chunk_id"],
                                 "title": r.get("article_title") or "",
                                 "text": (r.get("article_title") or "") + "\n" + r["content"][:2000]})
    return rows


@torch.no_grad()
def embed_all(texts: list[str], device: str) -> torch.Tensor:
    tok = AutoTokenizer.from_pretrained("BAAI/bge-m3")
    model = AutoModel.from_pretrained("BAAI/bge-m3", torch_dtype=torch.float16).to(device).eval()
    vecs = []
    B = 32
    for i in range(0, len(texts), B):
        enc = tok(texts[i:i + B], padding=True, truncation=True, max_length=512,
                  return_tensors="pt").to(device)
        out = model(**enc).last_hidden_state[:, 0]  # CLS (bge-m3 dense)
        vecs.append(torch.nn.functional.normalize(out, dim=-1).cpu())
        if (i // B) % 50 == 0:
            print(f"  embed {i}/{len(texts)}", flush=True)
    return torch.cat(vecs)


class UF:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def cluster(vecs: torch.Tensor, thr: float, device: str) -> list[int]:
    n = vecs.shape[0]
    uf = UF(n)
    gv = vecs.to(device)
    B = 2048
    for i in range(0, n, B):
        sims = gv[i:i + B] @ gv.T  # (b, n)
        idx = (sims >= thr).nonzero()
        for a, b in idx.tolist():
            if i + a < b:
                uf.union(i + a, b)
    return [uf.find(i) for i in range(n)]


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = load_parents()
    print(f"대상 조항: {len(rows)}", flush=True)
    vecs = embed_all([r["text"] for r in rows], device)

    report = {"n_articles": len(rows),
              "per_society": {},
              "thresholds": {}}
    for r in rows:
        report["per_society"][r["soc"]] = report["per_society"].get(r["soc"], 0) + 1

    for thr in THRESHOLDS:
        roots = cluster(vecs, thr, device)
        clusters = {}
        for i, root in enumerate(roots):
            clusters.setdefault(root, []).append(i)
        sizes = sorted((len(v) for v in clusters.values()), reverse=True)
        cross = sum(1 for v in clusters.values() if len({rows[i]["soc"] for i in v}) > 1)
        report["thresholds"][str(thr)] = {
            "n_clusters": len(clusters),
            "singletons": sum(1 for s in sizes if s == 1),
            "top_sizes": sizes[:10],
            "cross_society_clusters": cross,
        }
        print(f"thr={thr}: 클러스터 {len(clusters)} (싱글턴 {report['thresholds'][str(thr)]['singletons']}, "
              f"선급 교차 {cross})", flush=True)
        if thr == MAP_THRESHOLD:
            with (OUT / "cluster_map.jsonl").open("w", encoding="utf-8") as f:
                for i, root in enumerate(roots):
                    f.write(json.dumps({"chunk_id": rows[i]["chunk_id"], "soc": rows[i]["soc"],
                                        "title": rows[i]["title"], "cluster": int(root)},
                                       ensure_ascii=False) + "\n")

    (OUT / "cluster_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print("완료 → data_aireg/cluster_report.json, cluster_map.jsonl", flush=True)


if __name__ == "__main__":
    main()
