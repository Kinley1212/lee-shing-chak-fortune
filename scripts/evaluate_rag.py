#!/usr/bin/env python3
"""Run the labelled retrieval evaluation set against the current hybrid RAG."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from dotenv import load_dotenv
from google import genai

from rag import RAGEngine

DEFAULT_CASES = ROOT / "tests" / "fixtures" / "rag_eval_cases.json"


def build_embedder():
    load_dotenv(ROOT / ".env")
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("GEMINI_API_KEY is required for semantic evaluation")
    model = os.getenv("RAG_EMBEDDING_MODEL", "gemini-embedding-001")
    dimensions = int(os.getenv("RAG_EMBEDDING_DIMENSIONS", "768"))
    client = genai.Client(api_key=api_key)

    def embed(query: str) -> list[float]:
        response = client.models.embed_content(
            model=model,
            contents=query,
            config={"task_type": "RETRIEVAL_QUERY", "output_dimensionality": dimensions},
        )
        return list(response.embeddings[0].values or [])

    return embed, model, dimensions


def evaluate_case(engine: RAGEngine, case: dict) -> tuple[bool, str]:
    query = engine.enrich_query(case["query"])
    kind = case["kind"]

    if kind == "clarify":
        passed = engine.needs_zodiac_clarification(query)
        return passed, "clarification detected" if passed else "clarification was not detected"

    if engine.needs_zodiac_clarification(query):
        return False, "unexpected clarification"

    results = engine.search(query, top_k=1 if case.get("expected_zodiac") else 3)
    if kind == "reject":
        return (not results), "rejected" if not results else f"unexpected result {results[0].get('id')}"
    if not results:
        return False, "no retrieval result"

    first = results[0]
    if kind == "identity":
        passed = first.get("source") == "身份資料"
        return passed, first.get("source", "missing source")
    if kind == "hexagram":
        expected = case["expected_hexagram"]
        passed = first.get("hexagram") == expected
        return passed, f"expected {expected}, got {first.get('hexagram')}"

    expected_topics = set(case.get("expected_topics", []))
    if expected_topics and first.get("topic") not in expected_topics:
        return False, f"expected top-1 topic {sorted(expected_topics)}, got {first.get('topic')}"

    expected_zodiac = case.get("expected_zodiac")
    if expected_zodiac and first.get("zodiac") != expected_zodiac:
        return False, f"expected zodiac {expected_zodiac}, got {first.get('zodiac')}"
    if case.get("forbid_zodiac") and any(result.get("zodiac") for result in results):
        return False, "returned zodiac-specific content without a zodiac"

    return True, f"top={first.get('id')} topic={first.get('topic')} score={first.get('_semantic_score')}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--offline", action="store_true", help="Disable semantic query embeddings")
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    embedder = None
    engine_kwargs = {}
    if not args.offline:
        embedder, model, dimensions = build_embedder()
        engine_kwargs = {
            "expected_embedding_model": model,
            "expected_embedding_dimensions": dimensions,
        }
    engine = RAGEngine(query_embedder=embedder, **engine_kwargs)
    cases = json.loads(args.cases.read_text(encoding="utf-8"))

    outcomes = []
    by_kind = Counter()
    passed_by_kind = Counter()
    for case in cases:
        passed, detail = evaluate_case(engine, case)
        by_kind[case["kind"]] += 1
        passed_by_kind[case["kind"]] += int(passed)
        outcomes.append({"id": case["id"], "kind": case["kind"], "passed": passed, "detail": detail})
        print(f"{'PASS' if passed else 'FAIL'} {case['id']}: {detail}")

    passed_count = sum(item["passed"] for item in outcomes)
    summary = {
        "passed": passed_count,
        "total": len(outcomes),
        "accuracy": round(passed_count / len(outcomes), 4) if outcomes else 0,
        "by_kind": {
            kind: {"passed": passed_by_kind[kind], "total": total}
            for kind, total in sorted(by_kind.items())
        },
        "failures": [item for item in outcomes if not item["passed"]],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.json_output:
        args.json_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    raise SystemExit(0 if passed_count == len(outcomes) else 1)


if __name__ == "__main__":
    main()
