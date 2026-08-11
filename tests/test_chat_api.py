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
        self.assertNotIn("資料依據", report_html)
        self.assertNotIn("資料依據", chat_html)
        self.assertNotIn("report-source", report_html)
        self.assertNotIn("citations-title", chat_html)
        self.assertIn("請先輸入生肖或完整出生日期", chat_html)
        self.assertNotIn('id="memory-card"', chat_html)
        self.assertNotIn('id="memory-clear"', chat_html)

    def test_prompts_do_not_request_visible_source_attribution(self):
        self.assertNotIn("每次回答末端加", main.SYSTEM_PROMPT)
        self.assertIn("不要在回答中顯示", main.SYSTEM_PROMPT)

    def test_greeting_and_self_introduction_do_not_use_rag(self):
        social_messages = [
            "你好",
            "你可以自我介紹一下嗎？",
            "你會做啥？",
            "介紹一下你自己",
            "你識做咩㗎？",
            "你有什么功能？",
            "你是AI嗎？",
            "你能幹什麼？",
        ]
        with (
            patch.object(main.rag, "search") as search,
            patch.object(main, "call_gemini") as generate,
        ):
            responses = [
                self.client.post("/chat", json={"message": message, "history": []})
                for message in social_messages
            ]
        for message, response in zip(social_messages, responses):
            with self.subTest(message=message):
                self.assertEqual(response.status_code, 200)
                self.assertTrue(response.get_json()["social"])
                self.assertIn("AI玄學問答助手", response.get_json()["reply"])
        search.assert_not_called()
        generate.assert_not_called()

    def test_casual_checkin_does_not_use_rag(self):
        with (
            patch.object(main.rag, "search") as search,
            patch.object(main, "call_gemini") as generate,
        ):
            response = self.client.post("/chat", json={"message": "你好嗎？", "history": []})
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.get_json()["social"])
        self.assertIn("謝謝你關心", response.get_json()["reply"])
        search.assert_not_called()
        generate.assert_not_called()

    def test_chat_can_infer_zodiac_from_birth_date_and_recent_history(self):
        current = main.enrich_chat_birth_date("1990年5月20日出生，今年財運如何？", [])
        follow_up = main.enrich_chat_birth_date("那事業呢？", [
            {"role": "user", "content": "我的出生日期是1990-05-20"},
        ])
        self.assertIn("生肖屬馬", current)
        self.assertIn("生肖屬馬", follow_up)

    def test_structured_profile_persists_across_requests(self):
        result = {
            "id": "horse-wealth", "text": "財運原文", "source": "測試全書",
            "zodiac": "馬", "topic": "財運", "_retrieval_method": "metadata",
            "_retrieval_score": 1.0,
        }
        with (
            patch.object(main.rag, "search", return_value=[result]) as search,
            patch.object(main, "call_gemini", return_value="完整回答"),
        ):
            first = self.client.post("/chat", json={
                "message": "我是男性，1990年5月20日出生，今年財運如何？", "history": [],
            })
            second = self.client.post("/chat", json={"message": "那事業呢？", "history": []})

        self.assertEqual(first.status_code, 200)
        self.assertEqual(first.get_json()["profile"]["birth_date"], "1990-05-20")
        self.assertEqual(first.get_json()["profile"]["zodiac"], "馬")
        self.assertEqual(first.get_json()["profile"]["gender"], "男")
        self.assertTrue(second.get_json()["memory_used"])
        self.assertIn("屬馬", search.call_args_list[1].args[0])

    def test_new_identity_replaces_old_profile_silently(self):
        result = {
            "id": "source", "text": "原文", "source": "測試全書",
            "zodiac": "牛", "topic": "事業", "_retrieval_method": "metadata",
            "_retrieval_score": 1.0,
        }
        with (
            patch.object(main.rag, "search", return_value=[result]),
            patch.object(main, "call_gemini", return_value="完整回答"),
        ):
            self.client.post("/chat", json={
                "message": "我是男性，1990年5月20日出生，今年財運如何？", "history": [],
            })
            changed = self.client.post("/chat", json={
                "message": "改為幫另一位問：我是女性，屬牛，今年事業如何？", "history": [],
            }).get_json()["profile"]
        self.assertIsNone(changed["birth_date"])
        self.assertEqual(changed["zodiac"], "牛")
        self.assertEqual(changed["gender"], "女")
        self.assertEqual(changed["topics"], ["事業"])

    def test_conditional_reranker_skips_exact_match_and_runs_for_close_semantic_results(self):
        exact = [
            {"id": "exact", "zodiac": "馬", "topic": "財運", "_retrieval_method": "metadata", "_retrieval_score": 1.0},
            {"id": "other", "zodiac": "馬", "topic": "事業", "_retrieval_method": "hybrid", "_retrieval_score": 0.98},
        ]
        ambiguous = [
            {"id": "a", "zodiac": None, "topic": "風水", "_retrieval_method": "hybrid", "_retrieval_score": 0.72},
            {"id": "b", "zodiac": None, "topic": "五行", "_retrieval_method": "semantic", "_retrieval_score": 0.69},
        ]
        with patch.object(main, "GEMINI_API_KEY", "test-key"):
            self.assertFalse(main.should_rerank("屬馬財運", exact))
            self.assertTrue(main.should_rerank("屋企氣場應該點改善？", ambiguous))

    def test_reranker_reorders_candidates_from_model_json(self):
        candidates = [
            {"id": "a", "text": "候選甲", "topic": "風水", "_retrieval_method": "hybrid", "_retrieval_score": 0.72},
            {"id": "b", "text": "候選乙", "topic": "五行", "_retrieval_method": "semantic", "_retrieval_score": 0.69},
        ]
        response = SimpleNamespace(text='["c2","c1"]')
        with (
            patch.object(main, "should_rerank", return_value=True),
            patch.object(main.gemini_client.models, "generate_content", return_value=response),
        ):
            ordered, reranked = main.conditional_rerank("屋企氣場", candidates)
        self.assertTrue(reranked)
        self.assertEqual([item["id"] for item in ordered], ["b", "a"])

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
