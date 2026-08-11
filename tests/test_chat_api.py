import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
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
            patch.object(main, "call_gemini", return_value="**有依據的回答**"),
        ):
            response = self.client.post("/chat", json={"message": "屬龍財運", "history": []})
        payload = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertTrue(payload["grounded"])
        self.assertEqual(payload["reply"], "有依據的回答")
        self.assertEqual(payload["citations"][0]["source"], "測試全書")
        self.assertEqual(payload["citations"][0]["page_start"], 10)

    def test_chat_reply_is_plain_text(self):
        self.assertEqual(
            main.clean_chat_reply("## 財運\n\n**今年穩定**\n- 量入為出\n`不要冒險`"),
            "財運\n\n今年穩定\n• 量入為出\n不要冒險",
        )

    def test_chat_continues_once_when_model_hits_output_limit(self):
        first = SimpleNamespace(
            text="回答上半段，",
            candidates=[SimpleNamespace(finish_reason=SimpleNamespace(name="MAX_TOKENS"))],
        )
        second = SimpleNamespace(
            text="這是完成的下半段。",
            candidates=[SimpleNamespace(finish_reason=SimpleNamespace(name="STOP"))],
        )
        with patch.object(main.gemini_client.models, "generate_content", side_effect=[first, second]) as generate:
            reply = main.call_gemini("系統", "參考", "屬馬事業如何？", [
                {"role": "user", "content": "屬馬事業如何？"},
            ])
        self.assertEqual(reply, "回答上半段，這是完成的下半段。")
        self.assertEqual(generate.call_count, 2)
        first_contents = generate.call_args_list[0].kwargs["contents"]
        self.assertEqual(len(first_contents), 1)

    def test_report_and_chat_pages_link_to_each_other_without_quick_questions(self):
        report_html = self.client.get("/").get_data(as_text=True)
        chat_html = self.client.get("/chat").get_data(as_text=True)
        self.assertIn('href="/chat"', report_html)
        self.assertIn('src="/static/portrait.png"', chat_html)
        self.assertIn('href="/"', chat_html)
        self.assertNotIn('id="menu"', chat_html)
        self.assertNotIn('/menu/', chat_html)
        self.assertNotIn("已核對 6/6", report_html)
        self.assertIn("請先輸入生肖或完整出生日期", chat_html)

    def test_chat_can_infer_zodiac_from_birth_date_and_recent_history(self):
        current = main.enrich_chat_birth_date("1990年5月20日出生，今年財運如何？", [])
        follow_up = main.enrich_chat_birth_date("那事業呢？", [
            {"role": "user", "content": "我的出生日期是1990-05-20"},
        ])
        self.assertIn("生肖屬馬", current)
        self.assertIn("生肖屬馬", follow_up)

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
