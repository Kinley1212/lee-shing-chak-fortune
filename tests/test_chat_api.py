import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

import main


class ChatAPITests(unittest.TestCase):
    def setUp(self):
        self.client = main.app.test_client()
        with self.client.session_transaction() as session:
            session["authenticated"] = True

    def test_missing_zodiac_returns_clarification_without_generation(self):
        with patch.object(main, "call_gemini") as generate:
            response = self.client.post("/chat", json={"message": "我今年財運如何？", "history": []})
        payload = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertTrue(payload["needs_clarification"])
        self.assertFalse(payload["grounded"])
        self.assertEqual(payload["citations"], [])
        generate.assert_not_called()

    def test_no_evidence_stops_generation(self):
        with (
            patch.object(main.rag, "enrich_query", return_value="無關問題"),
            patch.object(main.rag, "needs_zodiac_clarification", return_value=False),
            patch.object(main.rag, "search", return_value=[]),
            patch.object(main, "call_gemini") as generate,
        ):
            response = self.client.post("/chat", json={"message": "無關問題", "history": []})
        payload = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertFalse(payload["grounded"])
        self.assertEqual(payload["citations"], [])
        generate.assert_not_called()

    def test_grounded_answer_returns_structured_citation(self):
        result = {
            "id": "dragon-wealth",
            "text": "財運原文",
            "source": "測試全書",
            "page_start": 10,
            "page_end": 11,
            "zodiac": "龍",
            "topic": "財運",
            "_retrieval_method": "metadata",
            "_retrieval_score": 1.0,
        }
        with (
            patch.object(main.rag, "enrich_query", return_value="屬龍財運"),
            patch.object(main.rag, "needs_zodiac_clarification", return_value=False),
            patch.object(main.rag, "search", return_value=[result]),
            patch.object(main, "call_gemini", return_value="有依據的回答"),
        ):
            response = self.client.post("/chat", json={"message": "屬龍財運", "history": []})
        payload = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertTrue(payload["grounded"])
        self.assertEqual(payload["reply"], "有依據的回答")
        self.assertEqual(payload["citations"][0]["source"], "測試全書")
        self.assertEqual(payload["citations"][0]["page_start"], 10)

    def test_report_sources_cover_all_six_sections(self):
        sections, context, citations, missing = main.retrieve_report_sources("龍")
        self.assertEqual(missing, [])
        self.assertEqual(set(sections), {"overall", "wealth", "career", "love", "health", "remedy"})
        self.assertEqual(len(context), 6)
        self.assertEqual(len(citations), 6)
        self.assertTrue(all(citation["zodiac"] == "龍" for citation in citations.values()))

    def test_report_question_citations_collapse_duplicate_source_labels(self):
        citations = [
            {"id": "feng-shui-1", "source": "李丞責2026全書", "topic": "風水", "zodiac": None,
             "page_start": None, "page_end": None},
            {"id": "feng-shui-2", "source": "李丞責2026全書", "topic": "風水", "zodiac": None,
             "page_start": None, "page_end": None},
        ]
        result = main.dedupe_display_citations(citations)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["id"], "feng-shui-1")


if __name__ == "__main__":
    unittest.main()
