"""build_master — canonical 그룹·결정적 split·leakage·fallback 임계값 단위 검증."""
import pytest

from scripts.aireg_kr.build_master import (assign_split, check_fallback,
                                           stable_bucket, validate)
from scripts.aireg_kr.grouping import (build_canonical_rule_group_id,
                                       group_id_for_sample, normalize_key_part)

RATIOS = (0.8, 0.1, 0.1)


def _sample(sid: str, gid: str, *, source_text: str = "", question: str = ""):
    return {"sample_id": sid, "canonical_rule_group_id": gid,
            "task": {"task_type": "compliance_judgment", "question": question},
            "source": {"source_text": source_text}}


# ── canonical_rule_group_id ─────────────────────────────────────────────
def test_judgment_and_suite_converge_to_same_id():
    # 판정 트랙: rule_uid 원문 좌표 / 스위트: evidence chunk_id — 같은 조항이면 동일 ID
    judgment = {"sample_id": "RULE_A::case1::compliant",
                "source": {"publisher": "KR", "source_file": "5편_2025_chunks",
                           "edition": None, "section_path": "규칙 > 제5편 > 201.",
                           "chunk_id": "RULE_P5_C7_S2_A201"},
                "rule": {"rule_id": "RULE_P5_C7_S2_A201"}}
    suite = {"sample_id": "RULE_A::spec::q1",
             "source": {"publisher": "KR", "source_file": "5편_2025_chunks",
                        "edition": None, "section_path": "규칙 > 제5편 > 201.",
                        "chunk_id": "RULE_P5_C7_S2_A201"},
             "rule": {"rule_id": "RULE_P5_C7_S2_A201"}}
    assert group_id_for_sample(judgment)[0] == group_id_for_sample(suite)[0]


def test_non_kr_uid_prefix_stripped_converges_with_chunk_id():
    # 비-KR 판정 트랙 rule_uid는 publisher__file__chunk 접두 — 접두 제거 후 수렴
    a = build_canonical_rule_group_id(publisher="ABS", document_id="cyber_2024",
                                      edition=None, rule_uid="ABS__cyber_2024__ABS_S2_3")
    b = build_canonical_rule_group_id(publisher="ABS", document_id="cyber_2024",
                                      edition=None, rule_uid="ABS_S2_3")
    assert a[0] == b[0]


def test_publisher_and_edition_differentiate():
    kw = dict(document_id="rules", rule_uid="pt5-ch7-601")
    kr = build_canonical_rule_group_id(publisher="KR", edition="2025", **kw)
    nk = build_canonical_rule_group_id(publisher="NK", edition="2025", **kw)
    kr26 = build_canonical_rule_group_id(publisher="KR", edition="2026", **kw)
    assert len({kr[0], nk[0], kr26[0]}) == 3


def test_normalization_absorbs_representation_diffs():
    assert normalize_key_part("규칙 > 제5편 / 201.") == normalize_key_part("규칙>제5편/201.")
    assert normalize_key_part("  Pt.5   Ch.7 ") == normalize_key_part("pt.5 ch.7")
    a = build_canonical_rule_group_id(publisher=" KR ", document_id="5편_2025",
                                      edition=None, section_path="규칙 > 제5편 > 201.")
    b = build_canonical_rule_group_id(publisher="kr", document_id="5편_2025",
                                      edition=None, section_path="규칙>제5편>201.")
    assert a[0] == b[0]


def test_priority_and_fallback_order():
    full = dict(publisher="KR", document_id="d", edition="2025")
    assert build_canonical_rule_group_id(**full, rule_uid="R1", article_id="A1")[1] == "rule_uid"
    assert build_canonical_rule_group_id(**full, article_id="A1",
                                         section_path="s")[1] == "article_id"
    assert build_canonical_rule_group_id(**full, section_path="s")[1] == "section_path"
    assert build_canonical_rule_group_id(**full, parent_rule_id="p")[1] == "parent_rule_id"
    assert build_canonical_rule_group_id(**full, chunk_id="c")[1] == "chunk_id"
    with pytest.raises(ValueError):
        build_canonical_rule_group_id(**full)


# ── split ────────────────────────────────────────────────────────────────
def test_same_group_same_split_deterministic():
    keys = [f"crg_{i:04d}" for i in range(200)]
    s42 = [assign_split(42, k, RATIOS) for k in keys]
    assert s42 == [assign_split(42, k, RATIOS) for k in keys]      # 같은 seed 동일
    assert s42 != [assign_split(7, k, RATIOS) for k in keys]       # 다른 seed 일부 변경
    assert stable_bucket(42, "X") == stable_bucket(42, "X")
    counts = {sp: s42.count(sp) for sp in ("train", "validation", "test")}
    assert counts["train"] > counts["validation"] and counts["train"] > counts["test"]


# ── leakage / fallback ──────────────────────────────────────────────────
def test_validate_passes_and_warns():
    samples = [_sample("a", "G1"), _sample("b", "G1"),
               _sample("c", "G2", source_text="같은 원문"),
               _sample("d", "G3", source_text="같은 원문")]
    warns = validate(samples, {"a": "train", "b": "train", "c": "train", "d": "test"})
    assert any("source_text" in w for w in warns)   # 본문 정확중복 split 교차 경고


def test_validate_fails_on_group_across_splits():
    samples = [_sample("a", "G1"), _sample("b", "G1")]
    with pytest.raises(SystemExit, match="canonical 그룹"):
        validate(samples, {"a": "train", "b": "test"})


def test_check_fallback_thresholds():
    assert check_fallback(0.01, 0.05, 0.20, strict=False) == []
    assert len(check_fallback(0.08, 0.05, 0.20, strict=False)) == 1   # 경고
    assert len(check_fallback(0.30, 0.05, 0.20, strict=False)) == 1   # non-strict는 경고만
    with pytest.raises(SystemExit, match="fallback"):
        check_fallback(0.30, 0.05, 0.20, strict=True)                 # strict 실패
