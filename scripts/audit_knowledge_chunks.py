#!/usr/bin/env python3
"""Audit knowledge chunk structure before rechunking or rebuilding embeddings."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FILES = [
    ROOT / "data" / "knowledge" / "book_2026_chunks.json",
    ROOT / "data" / "knowledge" / "bei_di_ling_qian_chunks.json",
]


def percentile(values: list[int], fraction: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[max(0, min(len(ordered) - 1, int(len(ordered) * fraction) - 1))]


def audit(path: Path) -> dict:
    chunks = json.loads(path.read_text(encoding="utf-8"))
    lengths = [len(str(chunk.get("text", "")).strip()) for chunk in chunks]
    duplicate_counts = Counter(str(chunk.get("text", "")).strip() for chunk in chunks)
    required_fields = ["id", "text", "source", "topic"]
    missing = {
        field: sum(not chunk.get(field) for chunk in chunks)
        for field in required_fields
    }
    flags = []
    for chunk in chunks:
        text = str(chunk.get("text", "")).strip()
        reasons = []
        if len(text) < 80:
            reasons.append("too_short")
        if len(text) > 4000:
            reasons.append("too_long")
        if not chunk.get("source"):
            reasons.append("missing_source")
        if reasons:
            flags.append({"id": chunk.get("id"), "reasons": reasons})

    return {
        "file": str(path.relative_to(ROOT)),
        "count": len(chunks),
        "length": {
            "min": min(lengths, default=0),
            "median": statistics.median(lengths) if lengths else 0,
            "p95": percentile(lengths, 0.95),
            "max": max(lengths, default=0),
        },
        "missing_required_fields": missing,
        "missing_page_metadata": sum(chunk.get("page_start") is None for chunk in chunks),
        "empty_text": sum(not str(chunk.get("text", "")).strip() for chunk in chunks),
        "duplicate_texts": sum(count - 1 for count in duplicate_counts.values() if count > 1),
        "under_80_chars": sum(length < 80 for length in lengths),
        "over_4000_chars": sum(length > 4000 for length in lengths),
        "topic_counts": dict(sorted(Counter(chunk.get("topic") for chunk in chunks).items(), key=lambda item: str(item[0]))),
        "flags": flags,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="*", type=Path, default=DEFAULT_FILES)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    report = {"files": [audit(path if path.is_absolute() else ROOT / path) for path in args.files]}
    output = json.dumps(report, ensure_ascii=False, indent=2)
    print(output)
    if args.json_output:
        args.json_output.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
