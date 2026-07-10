from __future__ import annotations

import copy
from dataclasses import dataclass
from difflib import SequenceMatcher
import json
from pathlib import Path
import re
from typing import Any, Callable

TEXT_KEYS = ("text", "content")
LIST_KEYS = ("list_items", "table_caption", "table_footnote", "image_caption", "image_footnote")
INLINE_MATH_RE = re.compile(r"\\\(.+?\\\)")
PLAIN_VARIABLE_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"t\s*a\s*s\s*-\s*b\s*u\s*i\s*l\s*t", r"\(t _ {a s - b u i l t}\)"),
    (r"t\s*v\s*o\s*l\s*a\s*d\s*d", r"\(t _ {v o l a d d}\)"),
    (r"(?<![A-Za-z])t\s*r\s*e\s*n(?![A-Za-z])", r"\(t _ {r e n}\)"),
    (r"(?<![A-Za-z])t\s*c(?![A-Za-z])", r"\(t _ {c}\)"),
    (r"(?<![A-Za-z])t\s*m(?![A-Za-z])", r"\(t _ {m}\)"),
)


@dataclass
class TextSlot:
    page_idx: int
    block_index: int
    key: str
    list_index: int | None
    original: str
    bbox: list[float] | None
    replacement: str | None = None
    score: float = 0.0
    text_score: float = 0.0
    spatial_score: float = 0.0
    inline_math_repaired: bool = False


@dataclass
class PdfTextSegment:
    text: str
    bbox: list[float] | None = None


def postprocess_mineru_output_dir(
    *,
    input_path: Path,
    output_dir: Path,
    doc_name: str,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    min_score: float = 0.28,
) -> dict[str, Any]:
    files = [path for path in output_dir.rglob("*") if path.is_file()]
    content_list_path = _find_output(files, ("content_list.json",), ("content_list_v2.json",))
    markdown_path = _find_output(files, (".md",), ())
    if content_list_path is None:
        return _write_report(output_dir, {"enabled": False, "reason": "content_list.json not found"})

    content_list = _read_json_list(content_list_path)
    total_pages = len(page_set(content_list))
    _notify(progress_callback, total_pages, 0, 0, "MinerU 산출물 후처리를 시작합니다.")

    pdftext_pages: list[dict[str, Any]] = []
    pdftext_error: str | None = None
    if input_path.suffix.lower() == ".pdf":
        try:
            pdftext_pages = generate_pdftext_pages(input_path)
        except Exception as exc:
            pdftext_error = str(exc)

    if pdftext_pages:
        merged, merge_report = merge_content_list_with_pdftext(content_list, pdftext_pages, min_score=min_score, progress_callback=lambda processed, current: _notify(progress_callback, total_pages, processed, current, "MinerU 산출물 후처리 중입니다."))
    else:
        merged = copy.deepcopy(content_list)
        merge_report = {
            "summary": {
                "blocks": len(content_list),
                "pagesInContentList": total_pages,
                "pagesInPdftext": 0,
                "textSlots": 0,
                "matchedSlots": 0,
                "changedSlots": 0,
                "unmatchedSlots": 0,
                "inlineMathRepairedSlots": 0,
            },
            "tests": _shape_tests(content_list, merged),
            "sampleInlineMathRepairs": [],
            "pdftextError": pdftext_error,
        }

    image_report = ensure_mineru_images(input_path, output_dir, merged)
    markdown = markdown_from_content_list(merged, doc_name)
    content_list_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if markdown_path is None:
        markdown_path = output_dir / f"{Path(doc_name).stem or doc_name}.md"
    markdown_path.write_text(markdown, encoding="utf-8")
    _notify(progress_callback, total_pages, total_pages, total_pages, "MinerU 산출물 후처리 완료")

    report = {
        "enabled": True,
        "stage": "etl_postprocess",
        "contentListPath": str(content_list_path),
        "markdownPath": str(markdown_path),
        "pdftextSource": "generated-with-bbox" if pdftext_pages else "unavailable",
        "merge": merge_report,
        "imageCopy": image_report,
    }
    return _write_report(output_dir, report)


def generate_pdftext_pages(input_path: Path) -> list[dict[str, Any]]:
    from pdftext.extraction import dictionary_output

    output = dictionary_output(str(input_path), sort=True, keep_chars=True, workers=1)
    return normalize_pdftext_dictionary_output(output)


def normalize_pdftext_dictionary_output(output: Any) -> list[dict[str, Any]]:
    if isinstance(output, dict) and isinstance(output.get("pages"), list):
        pages = output["pages"]
    elif isinstance(output, list):
        pages = output
    else:
        return []
    normalized = []
    for index, page in enumerate(pages):
        if isinstance(page, dict) and isinstance(page.get("blocks"), list):
            next_page = copy.deepcopy(page)
            next_page["page"] = index + 1
            next_page["text"] = page_text(next_page)
            normalized.append(next_page)
        else:
            normalized.append({"page": index + 1, "text": page_text(page)})
    return normalized


def page_text(page: Any) -> str:
    if isinstance(page, str):
        return page
    if isinstance(page, dict):
        if isinstance(page.get("text"), str):
            return page["text"]
        values = []
        for block in page.get("blocks") or []:
            if isinstance(block, dict):
                for line in block.get("lines") or []:
                    text = line_text_from_pdftext(line) if isinstance(line, dict) else ""
                    if text:
                        values.append(text)
        return "\n".join(values)
    return ""


def merge_content_list_with_pdftext(
    content_list: list[dict[str, Any]],
    pdftext_pages: list[dict[str, Any]],
    *,
    min_score: float,
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    merged = copy.deepcopy(content_list)
    slots = collect_text_slots(merged)
    page_segments = {int(page.get("page") or 0) - 1: build_pdftext_segments(page) for page in pdftext_pages if isinstance(page, dict)}
    page_slots: dict[int, list[TextSlot]] = {}
    for slot in slots:
        page_slots.setdefault(slot.page_idx, []).append(slot)

    matched = changed = unmatched = low_score = inline_math_repairs = 0
    scores: list[float] = []
    bbox_segments = sum(1 for segments in page_segments.values() for segment in segments if segment.bbox)
    bbox_matched = 0
    processed_pages = 0
    for page_idx in sorted(page_slots):
        cursor = 0
        segments = page_segments.get(page_idx, [])
        for slot in page_slots[page_idx]:
            replacement, score, start, end, text_score, spatial_score = best_replacement(slot.original, slot.bbox, segments, cursor, min_score)
            slot.score = score
            slot.text_score = text_score
            slot.spatial_score = spatial_score
            if replacement is not None:
                replacement, repaired = restore_inline_math(slot.original, replacement)
                slot.inline_math_repaired = repaired
                inline_math_repairs += 1 if repaired else 0
                slot.replacement = replacement
                set_slot_value(merged, slot, replacement)
                cursor = end or cursor
                matched += 1
                if start is not None and end is not None and any(segment.bbox for segment in segments[start:end]):
                    bbox_matched += 1
                scores.append(score)
                changed += 1 if normalize_compare(replacement) != normalize_compare(slot.original) else 0
            else:
                unmatched += 1
                if score > 0:
                    low_score += 1
                    scores.append(score)
        processed_pages += 1
        if progress_callback:
            progress_callback(processed_pages, page_idx + 1)

    return merged, {
        "summary": {
            "blocks": len(content_list),
            "pagesInContentList": len(page_set(content_list)),
            "pagesInPdftext": len(page_segments),
            "textSlots": len(slots),
            "matchedSlots": matched,
            "changedSlots": changed,
            "unmatchedSlots": unmatched,
            "lowScoreRejectedSlots": low_score,
            "minScore": min_score,
            "averageAcceptedScore": round(sum(scores) / len(scores), 4) if scores else None,
            "pdftextBboxSegments": bbox_segments,
            "bboxMatchedSlots": bbox_matched,
            "inlineMathRepairedSlots": inline_math_repairs,
        },
        "tests": _shape_tests(content_list, merged),
        "sampleInlineMathRepairs": sample_inline_math_repairs(slots),
    }


def collect_text_slots(content_list: list[dict[str, Any]]) -> list[TextSlot]:
    slots: list[TextSlot] = []
    for block_index, block in enumerate(content_list):
        if not isinstance(block, dict):
            continue
        page_idx = int(block.get("page_idx") or block.get("page") or 0)
        for key in TEXT_KEYS:
            value = block.get(key)
            if isinstance(value, str) and value.strip():
                slots.append(TextSlot(page_idx, block_index, key, None, value, normalize_bbox(block.get("bbox"))))
                break
        for key in LIST_KEYS:
            value = block.get(key)
            if isinstance(value, list):
                for list_index, item in enumerate(value):
                    item_text = item.get("text") if isinstance(item, dict) else item
                    if isinstance(item_text, str) and item_text.strip():
                        slots.append(TextSlot(page_idx, block_index, key, list_index, item_text, normalize_bbox(block.get("bbox"))))
            elif isinstance(value, str) and value.strip():
                slots.append(TextSlot(page_idx, block_index, key, None, value, normalize_bbox(block.get("bbox"))))
    return slots


def set_slot_value(content_list: list[dict[str, Any]], slot: TextSlot, value: str) -> None:
    block = content_list[slot.block_index]
    if slot.list_index is None:
        block[slot.key] = value
        return
    items = block[slot.key]
    item = items[slot.list_index]
    if isinstance(item, dict):
        next_item = dict(item)
        next_item["text" if "text" in next_item else "content"] = value
        items[slot.list_index] = next_item
    else:
        items[slot.list_index] = value


def build_pdftext_segments(page: dict[str, Any]) -> list[PdfTextSegment]:
    if isinstance(page.get("blocks"), list):
        segments = []
        width = float(page.get("width") or 0)
        height = float(page.get("height") or 0)
        for block in page.get("blocks") or []:
            if not isinstance(block, dict):
                continue
            for line in block.get("lines") or []:
                if not isinstance(line, dict):
                    continue
                text = normalize_visible(line_text_from_pdftext(line))
                bbox = normalize_pdftext_bbox(line.get("bbox"), width, height)
                if text:
                    segments.append(PdfTextSegment(text=text, bbox=bbox))
        if segments:
            return segments
    return [PdfTextSegment(text=text) for text in split_pdftext_segments(str(page.get("text") or ""))]


def line_text_from_pdftext(line: dict[str, Any]) -> str:
    if isinstance(line.get("text"), str):
        return line["text"]
    return "".join(span.get("text", "") for span in line.get("spans") or [] if isinstance(span, dict))


def split_pdftext_segments(text: str) -> list[str]:
    return [line for line in (normalize_visible(raw) for raw in text.splitlines()) if line]


def best_replacement(original: str, slot_bbox: list[float] | None, segments: list[PdfTextSegment], cursor: int, min_score: float) -> tuple[str | None, float, int | None, int | None, float, float]:
    if not segments:
        return None, 0.0, None, None, 0.0, 0.0
    target = normalize_compare(original)
    if not target:
        return None, 0.0, None, None, 0.0, 0.0
    best: tuple[str | None, float, int | None, int | None, float, float] = (None, 0.0, None, None, 0.0, 0.0)
    has_bbox = any(segment.bbox for segment in segments)
    start_min = 0 if has_bbox and slot_bbox else max(0, cursor - 2)
    start_max = len(segments) if has_bbox and slot_bbox else min(len(segments), cursor + 30)
    marker = leading_marker(original)
    for start in range(start_min, start_max):
        for end in range(start + 1, min(len(segments), start + 9) + 1):
            candidate_segments = segments[start:end]
            candidate = " ".join(segment.text for segment in candidate_segments).strip()
            if marker and leading_marker(candidate) != marker:
                continue
            text_score = similarity(target, normalize_compare(candidate))
            if text_score < 0.18:
                continue
            spatial_score = 0.0
            score = text_score
            candidate_bbox = union_bbox([segment.bbox for segment in candidate_segments if segment.bbox])
            if slot_bbox and candidate_bbox:
                spatial_score = bbox_spatial_score(slot_bbox, candidate_bbox)
                if spatial_score < 0.15:
                    continue
                score = text_score * 0.72 + spatial_score * 0.28
            if score > best[1]:
                best = (candidate, score, start, end, text_score, spatial_score)
            if len(normalize_compare(candidate)) > max(80, len(target) * 1.8):
                break
    return best if best[0] is not None and best[1] >= min_score else (None, best[1], best[2], best[3], best[4], best[5])


def restore_inline_math(original: str, replacement: str) -> tuple[str, bool]:
    formulas = unique_formulas([formula for formula in (canonical_inline_formula(token) for token in INLINE_MATH_RE.findall(original)) if formula] + extract_plain_variable_formulas(original))
    if not formulas or INLINE_MATH_RE.search(replacement):
        return replacement, False
    repaired = repair_missing_variable_list(replacement, formulas)
    if repaired != replacement:
        return repaired, True
    if starts_with_inline_formula(original):
        body = definition_body_from_replacement(replacement)
        repaired = f"{formulas[0]} : {body}" if body else f"{formulas[0]} {replacement}".strip()
        return repaired, normalize_compare(repaired) != normalize_compare(replacement)
    changed = False
    tm_formula = first_formula_named(formulas, "m")
    if tm_formula and "계측두께" in original and "계측두께" in repaired:
        next_value = re.sub(r"(계측두께)\s*(은|는)", lambda match: f"{match.group(1)} {tm_formula} {match.group(2)}", repaired, count=1)
        next_value = re.sub(r"(계측두께)\s+(다음)", lambda match: f"{match.group(1)} {tm_formula} {match.group(2)}", next_value, count=1)
        if next_value != repaired:
            repaired = next_value
            changed = True
    threshold_formula = first_threshold_formula(formulas)
    if threshold_formula and "남아있는 두께" in original and "남아있는 두께" in repaired:
        next_value = re.sub(r"(남아있는 두께가)\s*mm\s*(이상)", lambda match: f"{match.group(1)} {threshold_formula} {match.group(2)}", repaired, count=1)
        if next_value != repaired:
            repaired = next_value
            changed = True
    return repaired, changed


def unique_formulas(formulas: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for formula in formulas:
        key = normalize_compare(formula)
        if key not in seen:
            seen.add(key)
            result.append(formula)
    return result


def extract_plain_variable_formulas(value: str) -> list[str]:
    normalized = value.replace("−", "-").replace("–", "-")
    matches: list[tuple[int, str]] = []
    for pattern, formula in PLAIN_VARIABLE_PATTERNS:
        for match in re.finditer(pattern, normalized, flags=re.IGNORECASE):
            matches.append((match.start(), formula))
    return unique_formulas([formula for _, formula in sorted(matches, key=lambda item: item[0])])


def repair_missing_variable_list(value: str, formulas: list[str]) -> str:
    if len(formulas) < 2:
        return value
    match = re.match(r"^\s*(?:[,，]\s*){2,}(?:및\s*)?은\s*(.+)$", value)
    return f"{', '.join(formulas[:-1])} 및 {formulas[-1]} 은 {match.group(1).strip()}" if match else value


def starts_with_inline_formula(value: str) -> bool:
    return bool(re.match(r"^\s*\\\(.+?\\\)", value, flags=re.DOTALL))


def definition_body_from_replacement(value: str) -> str:
    body = normalize_visible(value)
    body = re.sub(r"^여기서,\s*", "", body).lstrip(":：;； ")
    for delimiter in (" : ", ":", "："):
        if delimiter in body:
            body = body.rsplit(delimiter, 1)[-1]
            break
    return body.lstrip(":：;； ").strip()


def canonical_inline_formula(token: str) -> str | None:
    inside = token[2:-2].strip()
    compact = re.sub(r"\s+", "", inside).lower()
    if not compact:
        return None
    if "t_{as-built}" in compact or "t_{as-built" in compact:
        return r"\(t _ {a s - b u i l t}\)"
    if "t_{voladd}" in compact or "t_{voladd" in compact:
        return r"\(t _ {v o l a d d}\)"
    if "t_{ren}-1" in compact or ("t_{ren}" in compact and "-1" in compact):
        return r"\(t _ {r e n} - 1 \mathrm{mm}\)"
    if "t_{ren}" in compact:
        return r"\(t _ {r e n}\)"
    if "t_{c}" in compact:
        return r"\(t _ {c}\)"
    if "t_{m}" in compact:
        return r"\(t _ {m}\)"
    return token


def first_formula_named(formulas: list[str], name: str) -> str | None:
    needle = f"_ {{{name}}}"
    return next((formula for formula in formulas if needle in formula), None)


def first_threshold_formula(formulas: list[str]) -> str | None:
    return next((formula for formula in formulas if "t_{ren}-1" in re.sub(r"\s+", "", formula)), None)


def ensure_mineru_images(input_path: Path, output_dir: Path, content_list: list[dict[str, Any]]) -> dict[str, Any]:
    image_blocks = {
        block["img_path"].strip(): block
        for block in content_list
        if isinstance(block, dict) and isinstance(block.get("img_path"), str) and block.get("img_path").strip()
    }
    existing = 0
    cropped = 0
    missing: list[str] = []
    for relative_path, block in sorted(image_blocks.items()):
        destination = output_dir / relative_path
        if destination.exists():
            existing += 1
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        if input_path.suffix.lower() == ".pdf" and crop_image_from_pdf(input_path, block, destination):
            cropped += 1
        else:
            missing.append(relative_path)
    return {"referencedImages": len(image_blocks), "existingImages": existing, "croppedImages": cropped, "missingImages": missing}


def crop_image_from_pdf(input_path: Path, block: dict[str, Any], destination: Path) -> bool:
    bbox = normalize_bbox(block.get("bbox"))
    if not bbox or block.get("page_idx") is None:
        return False
    try:
        import fitz

        pdf = fitz.open(str(input_path))
        page = pdf.load_page(int(block["page_idx"]))
        rect = page.rect
        clip = fitz.Rect(bbox[0] / 1000 * rect.width, bbox[1] / 1000 * rect.height, bbox[2] / 1000 * rect.width, bbox[3] / 1000 * rect.height)
        pixmap = page.get_pixmap(matrix=fitz.Matrix(3, 3), clip=clip, alpha=False)
        pixmap.save(destination)
        pdf.close()
        return destination.exists()
    except Exception:
        return False


def markdown_from_content_list(content_list: list[dict[str, Any]], file_name: str) -> str:
    pages: dict[int, list[dict[str, Any]]] = {}
    for block in content_list:
        if isinstance(block, dict):
            pages.setdefault(int(block.get("page_idx") or block.get("page") or 0), []).append(block)
    parts = [f"# {file_name}"]
    for page_idx in sorted(pages):
        body = blocks_to_markdown(pages[page_idx])
        parts.append(f"## Page {page_idx + 1}")
        if body:
            parts.append(body)
    return "\n\n".join(parts).strip() + "\n"


def blocks_to_markdown(blocks: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for block in blocks:
        block_type = str(block.get("type") or "")
        if block_type == "table":
            value = table_block_markdown(block)
        elif block_type in {"image", "chart"}:
            value = image_block_markdown(block)
        else:
            value = block_text(block)
            if block_type == "title" and value:
                value = f"# {value}"
        if value:
            parts.append(value)
    return "\n\n".join(parts).strip()


def table_block_markdown(block: dict[str, Any]) -> str:
    parts: list[str] = []
    parts.extend(list_or_string_values(block.get("table_caption")))
    body = block.get("table_body") or block.get("content") or block.get("text")
    if isinstance(body, str) and body.strip():
        parts.append(body.strip())
    parts.extend(list_or_string_values(block.get("table_footnote")))
    return "\n\n".join(parts)


def image_block_markdown(block: dict[str, Any]) -> str:
    parts: list[str] = []
    captions = list_or_string_values(block.get("image_caption"))
    img_path = block.get("img_path")
    if isinstance(img_path, str) and img_path.strip():
        parts.append(f"![{normalize_visible(captions[0]) if captions else 'image'}]({img_path.strip()})")
    image = block.get("image")
    if isinstance(image, dict) and image.get("storageKey"):
        parts.append(f"![image]({image['storageKey']})")
    parts.extend(captions)
    return "\n\n".join(parts)


def block_text(block: dict[str, Any]) -> str:
    list_items = block.get("list_items")
    if isinstance(list_items, list):
        values = []
        for item in list_items:
            value = item.get("text") or item.get("content") if isinstance(item, dict) else item
            if isinstance(value, str) and value.strip():
                values.append(value.strip())
        if values:
            return "\n".join(values)
    value = block.get("text") or block.get("content")
    return value.strip() if isinstance(value, str) else ""


def list_or_string_values(value: Any) -> list[str]:
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    if isinstance(value, list):
        return [item.get("text", "").strip() if isinstance(item, dict) else str(item).strip() for item in value if (item.get("text", "").strip() if isinstance(item, dict) else str(item).strip())]
    return []


def _find_output(files: list[Path], suffixes: tuple[str, ...], exclude_suffixes: tuple[str, ...]) -> Path | None:
    matches = []
    for path in files:
        normalized = path.name.lower()
        if exclude_suffixes and any(normalized.endswith(suffix) for suffix in exclude_suffixes):
            continue
        if any(normalized.endswith(suffix) for suffix in suffixes):
            matches.append(path)
    return sorted(matches, key=lambda item: (len(item.parts), len(item.name), item.name))[0] if matches else None


def _read_json_list(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        items = payload.get("content_list") or payload.get("content") or payload.get("items")
        if isinstance(items, list):
            return [item for item in items if isinstance(item, dict)]
    return []


def _write_report(output_dir: Path, report: dict[str, Any]) -> dict[str, Any]:
    path = output_dir / "etl_postprocess_report.json"
    report["reportPath"] = str(path)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def _notify(callback: Callable[[dict[str, Any]], None] | None, total_pages: int, processed_pages: int, current_page: int, message: str) -> None:
    if callback is None:
        return
    callback({"stage": "etl_postprocess", "totalPages": total_pages, "processedPages": processed_pages, "currentPage": current_page, "message": message})


def _shape_tests(original: Any, merged: Any) -> dict[str, bool]:
    tests = {
        "topLevelListLengthIdentical": isinstance(original, list) and isinstance(merged, list) and len(original) == len(merged),
        "recursiveShapeIdentical": shape_signature(original) == shape_signature(merged),
        "pageSetIdentical": sorted(page_set(original if isinstance(original, list) else [])) == sorted(page_set(merged if isinstance(merged, list) else [])),
    }
    tests["allPassed"] = all(tests.values())
    return tests


def sample_inline_math_repairs(slots: list[TextSlot], limit: int = 12) -> list[dict[str, Any]]:
    samples = []
    for slot in slots:
        if slot.inline_math_repaired and slot.replacement is not None:
            samples.append({"page": slot.page_idx + 1, "blockIndex": slot.block_index, "key": slot.key, "listIndex": slot.list_index, "from": slot.original[:220], "to": slot.replacement[:220]})
        if len(samples) >= limit:
            break
    return samples


def leading_marker(value: str) -> str | None:
    normalized = normalize_visible(re.sub(r"<[^>]+>", "", value))
    match = re.match(r"^([0-9]+(?:-[0-9]+)?|제\s*[0-9]+\s*[장절]|[(][0-9]+[)]|[-•])", normalized)
    return re.sub(r"\s+", "", match.group(1)) if match else None


def similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    if left in right or right in left:
        return min(len(left), len(right)) / max(len(left), len(right))
    return SequenceMatcher(None, left, right).ratio()


def normalize_bbox(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        bbox = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    return [item * 1000 for item in bbox] if max(bbox) <= 1 else bbox


def normalize_pdftext_bbox(value: Any, width: float, height: float) -> list[float] | None:
    bbox = normalize_bbox(value)
    if not bbox or width <= 0 or height <= 0:
        return None
    if max(bbox) <= 1000 and width <= 1000 and height <= 1000:
        return [bbox[0] / width * 1000, bbox[1] / height * 1000, bbox[2] / width * 1000, bbox[3] / height * 1000]
    return bbox


def union_bbox(boxes: list[list[float]]) -> list[float] | None:
    return [min(box[0] for box in boxes), min(box[1] for box in boxes), max(box[2] for box in boxes), max(box[3] for box in boxes)] if boxes else None


def bbox_spatial_score(slot_bbox: list[float], candidate_bbox: list[float]) -> float:
    slot_center = ((slot_bbox[0] + slot_bbox[2]) / 2, (slot_bbox[1] + slot_bbox[3]) / 2)
    candidate_center = ((candidate_bbox[0] + candidate_bbox[2]) / 2, (candidate_bbox[1] + candidate_bbox[3]) / 2)
    if slot_bbox[0] - 20 <= candidate_center[0] <= slot_bbox[2] + 20 and slot_bbox[1] - 20 <= candidate_center[1] <= slot_bbox[3] + 20:
        return 1.0
    slot_w = max(1.0, slot_bbox[2] - slot_bbox[0])
    slot_h = max(1.0, slot_bbox[3] - slot_bbox[1])
    dx = abs(slot_center[0] - candidate_center[0]) / max(120.0, slot_w)
    dy = abs(slot_center[1] - candidate_center[1]) / max(60.0, slot_h)
    return max(0.0, 1.0 - ((dx + dy) / 2))


def shape_signature(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: shape_signature(child) for key, child in sorted(value.items())}
    if isinstance(value, list):
        return [shape_signature(item) for item in value]
    return type(value).__name__


def page_set(content_list: list[dict[str, Any]]) -> set[int]:
    return {int(block.get("page_idx") or block.get("page") or 0) for block in content_list if isinstance(block, dict)}


def normalize_visible(value: str) -> str:
    return re.sub(r"\s+", " ", value.replace("\u00a0", " ")).strip()


def normalize_compare(value: str) -> str:
    value = re.sub(r"<[^>]+>", "", value)
    return re.sub(r"\s+", "", value.replace("\u00a0", " ")).strip()
