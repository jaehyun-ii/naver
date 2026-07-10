from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


TARGETS = {
    "markdown": (".md",),
    "content_list": ("content_list.json",),
    "content_list_v2": ("content_list_v2.json",),
    "middle": ("middle.json",),
    "model": ("model.json",),
    "layout_pdf": ("layout.pdf",),
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare two MinerU output directories.")
    parser.add_argument("--left", required=True, type=Path, help="Baseline output directory, e.g. HF Space output.")
    parser.add_argument("--right", required=True, type=Path, help="Candidate output directory, e.g. local service output.")
    parser.add_argument("--left-label", default="left")
    parser.add_argument("--right-label", default="right")
    parser.add_argument("--output-json", type=Path, help="Write the comparison report as JSON.")
    parser.add_argument("--output-markdown", type=Path, help="Write a human-readable Markdown report.")
    parser.add_argument("--markdown-diff-output", type=Path, help="Write Markdown unified diff to a file.")
    parser.add_argument("--show-markdown-diff", action="store_true")
    parser.add_argument("--max-diff-lines", type=int, default=120)
    args = parser.parse_args()

    left = inspect_output(args.left)
    right = inspect_output(args.right)
    report = build_report(left, right, args.left_label, args.right_label)
    report_json = json.dumps(report, ensure_ascii=False, indent=2)
    print(report_json)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(report_json + "\n", encoding="utf-8")
    if args.output_markdown:
        args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
        args.output_markdown.write_text(build_markdown_report(report), encoding="utf-8")

    if args.show_markdown_diff or args.markdown_diff_output:
        left_md = read_text(left["files"].get("markdown"))
        right_md = read_text(right["files"].get("markdown"))
        diff_lines = list(
            difflib.unified_diff(
                left_md.splitlines(),
                right_md.splitlines(),
                fromfile=args.left_label,
                tofile=args.right_label,
                lineterm="",
            )
        )
        if args.markdown_diff_output:
            args.markdown_diff_output.parent.mkdir(parents=True, exist_ok=True)
            args.markdown_diff_output.write_text("\n".join(diff_lines) + ("\n" if diff_lines else ""), encoding="utf-8")
        if args.show_markdown_diff:
            print("\n--- markdown unified diff ---")
            for index, line in enumerate(diff_lines):
                if index >= args.max_diff_lines:
                    print(f"... truncated after {args.max_diff_lines} diff lines")
                    break
                print(line)
    return 0


def inspect_output(root: Path) -> dict[str, Any]:
    files = {name: find_file(root, suffixes) for name, suffixes in TARGETS.items()}
    markdown = read_text(files.get("markdown"))
    content_list = read_json(files.get("content_list"), default=[])
    content_list_v2 = read_json(files.get("content_list_v2"), default=[])
    middle = read_json(files.get("middle"), default={})
    model = read_json(files.get("model"), default=[])

    return {
        "root": str(root),
        "files": files,
        "file_report": file_report(files),
        "markdown": markdown_metrics(markdown),
        "content_list": block_metrics(content_list),
        "content_list_v2": block_metrics(content_list_v2),
        "middle": json_metrics(middle),
        "model": block_metrics(model),
    }


def build_report(left: dict[str, Any], right: dict[str, Any], left_label: str, right_label: str) -> dict[str, Any]:
    return {
        "labels": {"left": left_label, "right": right_label},
        "roots": {"left": left["root"], "right": right["root"]},
        "files": compare_dicts(left["file_report"], right["file_report"]),
        "markdown": compare_dicts(left["markdown"], right["markdown"]),
        "content_list": compare_dicts(left["content_list"], right["content_list"]),
        "content_list_v2": compare_dicts(left["content_list_v2"], right["content_list_v2"]),
        "middle": compare_dicts(left["middle"], right["middle"]),
        "model": compare_dicts(left["model"], right["model"]),
        "interpretation": interpretation(left, right),
    }


def build_markdown_report(report: dict[str, Any]) -> str:
    left_label = report["labels"]["left"]
    right_label = report["labels"]["right"]
    lines = [
        f"# MinerU Output Comparison: {left_label} vs {right_label}",
        "",
        "## Roots",
        "",
        f"- {left_label}: `{report['roots']['left']}`",
        f"- {right_label}: `{report['roots']['right']}`",
        "",
        "## Interpretation",
        "",
    ]
    for note in report["interpretation"]:
        lines.append(f"- {note}")

    lines.extend(["", "## Key Metrics", ""])
    metric_sections = [
        ("markdown", ["chars", "html_table_count", "latex_inline_count", "latex_display_count", "image_markdown_count", "spaced_formula_token_count"]),
        ("content_list", ["block_count", "page_count", "bbox_count", "text_chars", "html_chars", "latex_chars"]),
        ("content_list_v2", ["block_count", "page_count", "bbox_count", "text_chars", "html_chars", "latex_chars"]),
        ("model", ["block_count", "page_count", "bbox_count", "text_chars", "html_chars", "latex_chars"]),
    ]
    for section, keys in metric_sections:
        lines.extend([f"### {section}", "", "| Metric | Left | Right | Same |", "|---|---:|---:|:---:|"])
        section_report = report.get(section) or {}
        for key in keys:
            item = section_report.get(key)
            if not item:
                continue
            same = "Y" if item["same"] else "N"
            lines.append(f"| {key} | {format_markdown_value(item['left'])} | {format_markdown_value(item['right'])} | {same} |")
        lines.append("")

    lines.extend(["## Files", "", "| Artifact | Left Exists | Right Exists | Same SHA256 |", "|---|:---:|:---:|:---:|"])
    for artifact, item in sorted((report.get("files") or {}).items()):
        left = item.get("left") or {}
        right = item.get("right") or {}
        same_hash = bool(left.get("sha256")) and left.get("sha256") == right.get("sha256")
        lines.append(f"| {artifact} | {bool_mark(left.get('exists'))} | {bool_mark(right.get('exists'))} | {bool_mark(same_hash)} |")
    lines.append("")
    return "\n".join(lines)


def format_markdown_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        return str(value)
    return "`" + str(value).replace("`", "\\`") + "`"


def bool_mark(value: Any) -> str:
    return "Y" if bool(value) else "N"


def interpretation(left: dict[str, Any], right: dict[str, Any]) -> list[str]:
    notes: list[str] = []
    for key in TARGETS:
        if left["file_report"][key]["exists"] and not right["file_report"][key]["exists"]:
            notes.append(f"candidate is missing {key}")
        if not left["file_report"][key]["exists"] and right["file_report"][key]["exists"]:
            notes.append(f"candidate has {key}, baseline does not")
    left_blocks = left["content_list"]["block_count"]
    right_blocks = right["content_list"]["block_count"]
    if left_blocks and right_blocks and abs(left_blocks - right_blocks) / max(left_blocks, right_blocks) > 0.1:
        notes.append("content_list block count differs by more than 10 percent")
    left_tables = left["markdown"]["html_table_count"]
    right_tables = right["markdown"]["html_table_count"]
    if left_tables != right_tables:
        notes.append("HTML table count differs")
    left_latex = left["markdown"]["latex_inline_count"] + left["markdown"]["latex_display_count"]
    right_latex = right["markdown"]["latex_inline_count"] + right["markdown"]["latex_display_count"]
    if left_latex != right_latex:
        notes.append("LaTeX delimiter count differs")
    if right["markdown"]["spaced_formula_token_count"] > left["markdown"]["spaced_formula_token_count"]:
        notes.append("candidate has more spaced formula tokens, which can indicate formula OCR degradation")
    if not notes:
        notes.append("no high-level metric differences detected")
    return notes


def compare_dicts(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    keys = sorted(set(left) | set(right))
    return {key: {"left": left.get(key), "right": right.get(key), "same": left.get(key) == right.get(key)} for key in keys}


def file_report(files: dict[str, Path | None]) -> dict[str, dict[str, Any]]:
    report: dict[str, dict[str, Any]] = {}
    for key, path in files.items():
        report[key] = {
            "path": str(path) if path else None,
            "exists": path is not None and path.exists(),
            "size": path.stat().st_size if path and path.exists() else 0,
            "sha256": sha256(path) if path and path.exists() else None,
        }
    return report


def markdown_metrics(text: str) -> dict[str, Any]:
    lines = text.splitlines()
    return {
        "chars": len(text),
        "non_empty_lines": sum(1 for line in lines if line.strip()),
        "heading_count": sum(1 for line in lines if line.lstrip().startswith("#")),
        "html_table_count": len(re.findall(r"<table\b", text, flags=re.IGNORECASE)),
        "latex_inline_count": len(re.findall(r"\\\(", text)),
        "latex_display_count": len(re.findall(r"\\\[", text)),
        "dollar_formula_count": len(re.findall(r"\$[^$\n]+\$", text)),
        "image_markdown_count": len(re.findall(r"!\[[^\]]*\]\(", text)),
        "spaced_formula_token_count": len(re.findall(r"\b[a-zA-Z](?:\s+[_{}a-zA-Z0-9-]){4,}", text)),
        "sample_spaced_formula_tokens": re.findall(r"\b[a-zA-Z](?:\s+[_{}a-zA-Z0-9-]){4,}", text)[:10],
    }


def block_metrics(payload: Any) -> dict[str, Any]:
    blocks = payload if isinstance(payload, list) else []
    type_counts = Counter(str(block.get("type") or "unknown") for block in blocks if isinstance(block, dict))
    pages = set()
    bbox_count = 0
    text_chars = 0
    html_chars = 0
    latex_chars = 0
    for block in blocks:
        if not isinstance(block, dict):
            continue
        page = first_not_none(block.get("page_idx"), block.get("page"), block.get("pageNumber"), block.get("page_no"))
        if page is not None:
            pages.add(str(page))
        if block.get("bbox"):
            bbox_count += 1
        text_chars += len(str(block.get("text") or block.get("content") or ""))
        html_chars += len(str(block.get("html") or ""))
        latex_chars += len(str(block.get("latex") or ""))
    return {
        "block_count": len(blocks),
        "page_count": len(pages),
        "bbox_count": bbox_count,
        "text_chars": text_chars,
        "html_chars": html_chars,
        "latex_chars": latex_chars,
        "type_counts": dict(sorted(type_counts.items())),
    }


def json_metrics(payload: Any) -> dict[str, Any]:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True) if payload else ""
    return {
        "json_chars": len(encoded),
        "top_level_type": type(payload).__name__,
        "top_level_len": len(payload) if isinstance(payload, (list, dict)) else 0,
    }


def find_file(root: Path, suffixes: tuple[str, ...]) -> Path | None:
    if not root.exists():
        return None
    matches = []
    for path in root.rglob("*"):
        if path.is_file() and any(path.name.endswith(suffix) for suffix in suffixes):
            matches.append(path)
    if not matches:
        return None
    return sorted(matches, key=lambda item: (len(item.parts), str(item)))[0]


def read_text(path: Path | None) -> str:
    if path is None or not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def read_json(path: Path | None, default: Any) -> Any:
    if path is None or not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def first_not_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


if __name__ == "__main__":
    raise SystemExit(main())
