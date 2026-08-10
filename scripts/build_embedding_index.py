#!/usr/bin/env python3
"""Build the checked-in semantic index used by the hybrid RAG retriever."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from dotenv import load_dotenv
from google import genai

from rag import CHUNKS_PATH, EMBEDDINGS_PATH, LINGQIAN_PATH, chunk_key, corpus_hash


def _load_json(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def _document_text(chunk: dict) -> str:
    metadata = " ".join(
        str(value)
        for value in (
            chunk.get("zodiac"),
            chunk.get("topic"),
            chunk.get("chapter"),
            chunk.get("hexagram"),
        )
        if value
    )
    return f"{metadata}\n{chunk.get('text', '')}".strip()


def main() -> None:
    load_dotenv(ROOT / ".env")
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("GEMINI_API_KEY is required in .env or the environment")

    model = os.getenv("RAG_EMBEDDING_MODEL", "gemini-embedding-001")
    dimensions = int(os.getenv("RAG_EMBEDDING_DIMENSIONS", "768"))
    batch_size = int(os.getenv("RAG_EMBEDDING_BATCH_SIZE", "50"))
    chunks = _load_json(CHUNKS_PATH)
    lingqian_chunks = _load_json(LINGQIAN_PATH)
    documents = [
        (chunk_key(collection, chunk, position), _document_text(chunk))
        for collection, items in (("book", chunks), ("lingqian", lingqian_chunks))
        for position, chunk in enumerate(items)
    ]

    client = genai.Client(api_key=api_key)
    embeddings: dict[str, list[float]] = {}
    for start in range(0, len(documents), batch_size):
        batch = documents[start:start + batch_size]
        response = client.models.embed_content(
            model=model,
            contents=[text for _, text in batch],
            config={
                "task_type": "RETRIEVAL_DOCUMENT",
                "output_dimensionality": dimensions,
            },
        )
        vectors = response.embeddings or []
        if len(vectors) != len(batch):
            raise RuntimeError(f"Embedding API returned {len(vectors)} vectors for {len(batch)} documents")
        for (key, _), embedding in zip(batch, vectors):
            embeddings[key] = list(embedding.values or [])
        print(f"Embedded {min(start + len(batch), len(documents))}/{len(documents)} documents")

    index = {
        "version": 1,
        "model": model,
        "dimensions": dimensions,
        "corpus_hash": corpus_hash(chunks, lingqian_chunks),
        "embeddings": embeddings,
    }
    output_path = Path(EMBEDDINGS_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=output_path.parent, delete=False) as temp:
        json.dump(index, temp, ensure_ascii=False, separators=(",", ":"))
        temp_path = Path(temp.name)
    temp_path.replace(output_path)
    print(f"Wrote {len(embeddings)} vectors to {output_path}")


if __name__ == "__main__":
    main()
