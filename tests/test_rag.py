import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from rag import RAGEngine, chunk_key, corpus_hash


class HybridRAGTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        base = Path(self.temp_dir.name)
        self.chunks = [
            {
                "id": "money",
                "text": "今年收入穩定，投資應保持審慎。",
                "zodiac": "龍",
                "topic": "財運",
                "source": "測試全書",
                "page_start": 10,
                "page_end": 11,
            },
            {
                "id": "career",
                "text": "工作環境將有變化，適合尋找新的發展方向。",
                "zodiac": "龍",
                "topic": "事業",
                "source": "測試全書",
                "page_start": 12,
                "page_end": 12,
            },
            {
                "id": "generic-career",
                "text": "轉換職涯方向前應評估長遠發展。",
                "zodiac": None,
                "topic": "事業",
                "source": "測試全書",
                "page_start": 20,
                "page_end": 20,
            },
        ]
        self.lingqian = [
            {"id": "hex-1", "hexagram": "乾", "text": "乾卦內容", "source": "測試靈簽"}
        ]
        self.paths = {
            "chunks_path": str(base / "chunks.json"),
            "lingqian_path": str(base / "lingqian.json"),
            "lingqian_old_path": str(base / "missing-old.json"),
            "profile_path": str(base / "profile.md"),
            "embeddings_path": str(base / "embeddings.json"),
        }
        Path(self.paths["chunks_path"]).write_text(
            json.dumps(self.chunks, ensure_ascii=False), encoding="utf-8"
        )
        Path(self.paths["lingqian_path"]).write_text(
            json.dumps(self.lingqian, ensure_ascii=False), encoding="utf-8"
        )
        Path(self.paths["profile_path"]).write_text("博士資料", encoding="utf-8")
        index = {
            "version": 1,
            "model": "fake-embedding",
            "dimensions": 2,
            "corpus_hash": corpus_hash(self.chunks, self.lingqian),
            "embeddings": {
                chunk_key("book", self.chunks[0]): [1.0, 0.0],
                chunk_key("book", self.chunks[1]): [0.0, 1.0],
                chunk_key("book", self.chunks[2]): [0.0, 0.9],
                chunk_key("lingqian", self.lingqian[0]): [0.5, 0.5],
            },
        }
        Path(self.paths["embeddings_path"]).write_text(json.dumps(index), encoding="utf-8")

    def tearDown(self):
        self.temp_dir.cleanup()

    def make_engine(self, embedder=lambda _query: [0.0, 1.0], **overrides):
        kwargs = dict(self.paths)
        kwargs.update(overrides)
        return RAGEngine(query_embedder=embedder, **kwargs)

    def test_semantic_paraphrase_finds_relevant_chunk(self):
        engine = self.make_engine()
        results = engine.search("前途會否另有出路", top_k=1)
        self.assertEqual(results[0]["id"], "generic-career")
        self.assertEqual(results[0]["_retrieval_method"], "semantic")

    def test_metadata_filter_remains_first_priority(self):
        calls = []
        engine = self.make_engine(embedder=lambda query: calls.append(query) or [1.0, 0.0])
        results = engine.search("屬龍事業", top_k=1)
        self.assertEqual(results[0]["topic"], "事業")
        self.assertEqual(calls, [])

    def test_exact_metadata_does_not_add_lower_priority_topics(self):
        engine = self.make_engine()
        results = engine.search("屬龍財運", top_k=3)
        self.assertEqual([result["id"] for result in results], ["money"])

    def test_topic_alias_does_not_return_unspecified_zodiac(self):
        engine = self.make_engine()
        results = engine.search("我想轉工", top_k=3)
        self.assertTrue(results)
        self.assertTrue(all(not chunk.get("zodiac") for chunk in results))

    def test_unrelated_query_returns_no_context(self):
        engine = self.make_engine(embedder=lambda _query: [-1.0, -1.0], min_semantic_score=0.4)
        self.assertEqual(engine.search("完全無關內容"), [])
        self.assertIn("沒有檢索到", engine.get_context("完全無關內容"))

    def test_embedding_failure_falls_back_to_keyword(self):
        def failing_embedder(_query):
            raise RuntimeError("temporary API error")

        engine = self.make_engine(embedder=failing_embedder)
        results = engine.search("轉換職涯方向", top_k=1)
        self.assertEqual(results[0]["id"], "generic-career")
        self.assertEqual(results[0]["_retrieval_method"], "keyword")

    def test_context_includes_source_and_page_range(self):
        engine = self.make_engine(embedder=lambda _query: [1.0, 0.0])
        context = engine.get_context("屬龍財運", top_k=1)
        self.assertIn("來源：測試全書", context)
        self.assertIn("頁碼：10-11", context)

    def test_stale_index_disables_semantic_search(self):
        path = Path(self.paths["embeddings_path"])
        index = json.loads(path.read_text(encoding="utf-8"))
        index["corpus_hash"] = "stale"
        path.write_text(json.dumps(index), encoding="utf-8")
        engine = self.make_engine()
        self.assertFalse(engine.semantic_enabled)
        self.assertEqual(engine.embedding_count, 0)

    def test_personal_query_without_zodiac_requests_clarification(self):
        engine = self.make_engine()
        self.assertTrue(engine.needs_zodiac_clarification("我今年財運如何？"))
        self.assertFalse(engine.needs_zodiac_clarification("屬龍今年財運如何？"))

    def test_history_zodiac_is_reused(self):
        engine = self.make_engine()
        enriched = engine.enrich_query(
            "今年事業如何？",
            [{"role": "user", "content": "我是屬龍的"}],
        )
        self.assertIn("屬龍", enriched)
        self.assertFalse(engine.needs_zodiac_clarification(enriched))

    def test_structured_citations_do_not_include_chunk_text(self):
        engine = self.make_engine(embedder=lambda _query: [1.0, 0.0])
        results = engine.search("屬龍財運", top_k=1)
        citations = engine.citations(results)
        self.assertEqual(citations[0]["source"], "測試全書")
        self.assertEqual(citations[0]["page_start"], 10)
        self.assertNotIn("text", citations[0])


if __name__ == "__main__":
    unittest.main()
