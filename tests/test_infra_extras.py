"""인프라 확장 단위 테스트 — NCP 엔드포인트 전환, 마이크로배칭 (GPU/네트워크 불필요)."""

from __future__ import annotations

from llmops_core.common.config import S3Settings
from llmops_core.serving.batching import MicroBatcher, make_request


def test_s3_endpoint_minio_default():
    s = S3Settings()
    assert s.provider == "minio"
    assert s.effective_endpoint() == "http://localhost:9000"


def test_s3_endpoint_switches_to_ncp():
    s = S3Settings(provider="ncp")
    # provider=ncp이고 endpoint 미지정 → NCP 기본 엔드포인트
    assert s.effective_endpoint().endswith("ncloudstorage.com")


def test_s3_endpoint_explicit_override_wins():
    s = S3Settings(provider="ncp", endpoint_url="https://custom.example.com")
    assert s.effective_endpoint() == "https://custom.example.com"


def test_microbatcher_batches_and_returns():
    seen_batch_sizes = []

    def fake_generate(batch):
        seen_batch_sizes.append(len(batch))
        return [{"text": f"ans:{r.messages}", "n": i} for i, r in enumerate(batch)]

    mb = MicroBatcher(fake_generate, window_ms=200, max_batch=8)
    try:
        import threading

        results = {}
        ready = threading.Barrier(4)

        def call(idx):
            req = make_request([{"role": "user", "content": str(idx)}], 32, 0.0, 0.95)
            ready.wait()  # 4개 스레드를 동시에 제출시켜 배칭 보장
            results[idx] = mb.submit(req)

        threads = [threading.Thread(target=call, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 4
        assert all("ans:" in r["text"] for r in results.values())
        # 동시 도착분이 한 배치로 묶였는지(최소 한 번은 2건 이상)
        assert max(seen_batch_sizes) >= 2
    finally:
        mb.stop()
