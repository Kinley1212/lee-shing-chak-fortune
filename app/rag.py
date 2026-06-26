#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG 搜尋引擎：四優先級匹配策略
優先級1：生肖+主題雙匹配 → 優先級2：生肖匹配 → 優先級3：主題匹配 → 優先級4：全文關鍵字
"""

import json
import os
import re

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CHUNKS_PATH   = os.path.join(BASE, "data", "knowledge", "book_2026_chunks.json")
LINGQIAN_PATH = os.path.join(BASE, "data", "knowledge", "bei_di_ling_qian.json")
PROFILE_PATH  = os.path.join(BASE, "data", "knowledge", "dr_lee_profile.md")

ZODIACS = ["鼠", "牛", "虎", "兔", "龍", "蛇", "馬", "羊", "猴", "雞", "狗", "豬"]
TOPICS  = ["財運", "感情", "事業", "健康", "風水", "預言", "佈局"]

ZODIAC_ALIASES: dict[str, list[str]] = {
    "鼠": ["鼠", "子鼠", "屬鼠"],
    "牛": ["牛", "丑牛", "屬牛"],
    "虎": ["虎", "寅虎", "屬虎", "老虎"],
    "兔": ["兔", "卯兔", "屬兔", "兔子"],
    "龍": ["龍", "辰龍", "屬龍"],
    "蛇": ["蛇", "巳蛇", "屬蛇"],
    "馬": ["馬", "午馬", "屬馬"],
    "羊": ["羊", "未羊", "屬羊"],
    "猴": ["猴", "申猴", "屬猴", "猴子"],
    "雞": ["雞", "酉雞", "屬雞"],
    "狗": ["狗", "戌狗", "屬狗"],
    "豬": ["豬", "亥豬", "屬豬"],
}

IDENTITY_KEYWORDS = ["李丞責", "師傅", "博士", "你是誰", "你係", "介紹自己", "你嘅資料"]


class RAGEngine:
    def __init__(self):
        self.chunks: list[dict] = []
        self.profile_text: str = ""
        self.lingqian_text: str = ""
        self._load_all()

    def _load_all(self):
        if os.path.exists(CHUNKS_PATH):
            with open(CHUNKS_PATH, encoding="utf-8") as f:
                self.chunks = json.load(f)
            print(f"[RAG] 知識庫載入：{len(self.chunks)} 塊")
        else:
            print(f"[RAG] 警告：找不到 {CHUNKS_PATH}")

        if os.path.exists(PROFILE_PATH):
            with open(PROFILE_PATH, encoding="utf-8") as f:
                self.profile_text = f.read()
            print("[RAG] 身份資料載入完成")

        if os.path.exists(LINGQIAN_PATH):
            with open(LINGQIAN_PATH, encoding="utf-8") as f:
                self.lingqian_text = json.load(f)["text"]
            print("[RAG] 北帝靈簽載入完成")

    # ── 解析查詢 ─────────────────────────────────────

    def _extract_zodiacs(self, query: str) -> list[str]:
        found = []
        for zodiac, aliases in ZODIAC_ALIASES.items():
            if any(a in query for a in aliases):
                found.append(zodiac)
        return found

    def _extract_topics(self, query: str) -> list[str]:
        return [t for t in TOPICS if t in query]

    def _keyword_score(self, chunk: dict, query: str) -> int:
        """全文關鍵字得分：2字以上詞組在文本中出現次數"""
        text = chunk.get("text", "")
        score = 0
        for word in re.findall(r"[一-鿿]{2,}", query):
            score += text.count(word)
        return score

    # ── 四優先級搜尋 ──────────────────────────────────

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        if not self.chunks:
            return []

        # 特殊路由：靈簽
        if "靈簽" in query or "北帝" in query:
            return [{"text": self.lingqian_text[:1500], "source": "北帝靈簽", "zodiac": None}]

        # 特殊路由：身份查詢
        if any(kw in query for kw in IDENTITY_KEYWORDS):
            return [{"text": self.profile_text, "source": "身份資料", "zodiac": None}]

        zodiacs = self._extract_zodiacs(query)
        topics  = self._extract_topics(query)

        def rank(pool: list[dict]) -> list[dict]:
            scored = [(self._keyword_score(c, query), c) for c in pool]
            scored.sort(key=lambda x: x[0], reverse=True)
            return [c for _, c in scored]

        seen: set = set()
        results: list[dict] = []

        def add(pool: list[dict]):
            for c in pool:
                cid = c.get("id", id(c))
                if cid not in seen:
                    seen.add(cid)
                    results.append(c)
                if len(results) >= top_k:
                    return

        # 優先級1：生肖 + 主題雙匹配
        if zodiacs and topics:
            add(rank([c for c in self.chunks
                      if c.get("zodiac") in zodiacs and c.get("topic") in topics]))

        # 優先級2：只匹配生肖
        if zodiacs and len(results) < top_k:
            add(rank([c for c in self.chunks if c.get("zodiac") in zodiacs]))

        # 優先級3：只匹配主題
        if topics and len(results) < top_k:
            add(rank([c for c in self.chunks if c.get("topic") in topics]))

        # 優先級4：全文關鍵字搜尋
        if len(results) < top_k:
            add(rank(self.chunks))

        if not results:
            return [{"text": self.profile_text, "source": "身份資料", "zodiac": None}]

        return results[:top_k]

    def get_context(self, query: str, top_k: int = 3) -> str:
        results = self.search(query, top_k=top_k)
        parts = []
        for i, chunk in enumerate(results, 1):
            source = chunk.get("source") or chunk.get("chapter", "2026全書")
            zodiac = chunk.get("zodiac")
            topic  = chunk.get("topic", "")
            if zodiac:
                label = f"屬{zodiac}" + (f"·{topic}" if topic and topic != "其他" else "")
            else:
                label = source
            parts.append(f"【參考資料{i}｜{label}】\n{chunk['text']}")

        return (
            f"以下是參考資料：\n\n"
            + "\n\n".join(parts)
            + "\n\n請根據以上資料回答用戶問題。"
        )

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)
