"""grpo_smoke 지원 코드 — 환경 판정·샘플 선별·정렬 검증 로직 (GPU 불필요 단위)."""
import json

from scripts.aireg_kr.grpo_smoke import (collect_environment, env_ready,
                                         make_rewards, select_samples)


def test_collect_environment_keys():
    env = collect_environment()
    for k in ("python", "torch", "transformers", "trl", "peft",
              "datasets", "accelerate", "cuda_available"):
        assert k in env


def test_env_ready_reports_missing():
    assert env_ready({"torch": "2", "transformers": "4", "trl": "1", "peft": "0",
                      "datasets": "5", "accelerate": "1", "cuda_available": True}) == []
    missing = env_ready({"torch": None, "cuda_available": False})
    assert "torch" in missing and "cuda" in missing


def test_select_samples_covers_required_mix():
    samples, coverage = select_samples()
    # §13 필수 6유형 전부 실데이터/합성/부재 중 하나로 명시 보고
    assert set(coverage) == {"APPLICABLE", "NOT_APPLICABLE", "INSUFFICIENT_INFORMATION",
                             "EXCEPTION_APPLIES", "EXCEPTION_PARTIALLY_MET",
                             "NUMERIC_BOUNDARY"}
    # 예외·경계값 유형은 합성 대체 금지 — real 또는 MISSING만
    for k in ("EXCEPTION_APPLIES", "EXCEPTION_PARTIALLY_MET", "NUMERIC_BOUNDARY"):
        assert coverage[k] in ("real", "MISSING")
    for s in samples:  # smoke 필수 컬럼
        for key in ("prompt", "gold_label", "sample_id", "canonical_rule_group_id",
                    "task_type", "answer_format"):
            assert key in s, (s.get("sample_id"), key)


def test_metadata_alignment_reward_validates(tmp_path):
    rewards, f = make_rewards(tmp_path / "debug.jsonl")
    align = rewards[0]
    out = align(["p1", "p1"], ["c1", "c2"], gold_label=["a", "a"],
                sample_id=["s", "s"], canonical_rule_group_id=["g", "g"])
    assert out == [0.0, 0.0]
    f.close()
    recs = [json.loads(l) for l in (tmp_path / "debug.jsonl").read_text().splitlines()]
    assert len(recs) == 2 and recs[0]["sample_id"] == "s"
    # 필수 메타 누락 시 명확한 실패
    rewards2, f2 = make_rewards(tmp_path / "d2.jsonl")
    try:
        rewards2[0](["p"], ["c"])  # gold_label 없음
        raise SystemExit("도달하면 안 됨")
    except AssertionError as e:
        assert "gold_label" in str(e)
    finally:
        f2.close()


def test_label_accuracy_reward_parses_decision(tmp_path):
    rewards, f = make_rewards(tmp_path / "d.jsonl")
    label_fn = rewards[1]
    good = "판단:\n적합\n\n근거:\n인용"
    uncertain = "판단:\n판단 불가\n\n추가 확인 정보:\n- 선종"
    out = label_fn(["p"] * 3, [good, uncertain, "자유 서술"],
                   gold_label=["compliant", "uncertain", "compliant"])
    assert out == [1.0, 1.0, 0.0]
    f.close()
