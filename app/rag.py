#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG 搜尋引擎：四優先級匹配策略
優先級1：生肖+主題雙匹配 → 優先級2：生肖匹配 → 優先級3：主題匹配 → 優先級4：全文關鍵字
"""

import hashlib
import json
import math
import os
import re
from collections.abc import Callable

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CHUNKS_PATH        = os.path.join(BASE, "data", "knowledge", "book_2026_chunks.json")
LINGQIAN_PATH      = os.path.join(BASE, "data", "knowledge", "bei_di_ling_qian_chunks.json")
LINGQIAN_OLD_PATH  = os.path.join(BASE, "data", "knowledge", "bei_di_ling_qian.json")
PROFILE_PATH       = os.path.join(BASE, "data", "knowledge", "dr_lee_profile.md")
EMBEDDINGS_PATH    = os.path.join(BASE, "data", "knowledge", "embedding_index.json")

ZODIACS = ["鼠", "牛", "虎", "兔", "龍", "蛇", "馬", "羊", "猴", "雞", "狗", "豬"]

TOPIC_ALIASES: dict[str, list[str]] = {
    "整體運勢": ["整體運勢", "總運", "全年運勢", "今年運程", "流年運勢", "個人流年"],
    "財運": ["財運", "金錢", "收入", "投資", "求財", "賺錢", "搵錢", "搵多啲錢", "理財"],
    "感情": ["感情", "愛情", "桃花", "姻緣", "婚姻", "伴侶", "戀愛", "拍拖", "對象", "兩個人", "相處", "爭執"],
    "事業": ["事業", "工作", "職場", "職涯", "跑道", "前途", "出路", "轉工", "跳槽", "升職", "創業"],
    "健康": ["健康", "身體", "不舒服", "疾病", "睡眠", "精神狀態"],
    "風水": ["風水", "屋企", "住宅", "家居", "氣場", "方位", "擺設"],
    "預言": ["預言", "趨勢", "大環境"],
    "佈局": ["佈局", "布局", "擺陣", "開運"],
    "化解建議": ["化解建議", "化解", "改善方法", "補救方法"],
    "每月運勢": ["每月運勢", "月份運勢", "月份", "幾月", "幾個月", "哪個月", "哪幾個月", "注意月份"],
    "紫微斗數": ["紫微斗數", "紫微", "命宮", "星曜"],
    "九宮飛星": ["九宮飛星", "飛星", "九星"],
    "五行": ["五行", "金木水火土", "缺金", "缺木", "缺水", "缺火", "缺土"],
    "天干地支": ["天干地支", "天干", "地支", "甲乙丙丁", "子丑寅卯"],
    "太歲": ["太歲", "犯太歲", "攝太歲", "拜太歲"],
}

PERSONAL_FORTUNE_TOPICS = {"整體運勢", "財運", "感情", "事業", "健康", "化解建議", "每月運勢"}
PERSONAL_QUERY_MARKERS = ["我", "本人", "自己", "今年", "2026", "流年"]

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

IDENTITY_KEYWORDS  = ["李丞責", "師傅", "博士", "你是誰", "你係", "介紹自己", "你嘅資料"]
LINGQIAN_KEYWORDS  = ["靈簽", "卜籤", "求籤", "北帝", "上籤", "中籤", "下籤", "占卜"]

HEXAGRAM_ALIASES: dict[str, list[str]] = {
    "乾": ["乾卦", "乾"],   "坤": ["坤卦", "坤"],   "屯": ["屯卦"],    "蒙": ["蒙卦"],
    "需": ["需卦"],          "訟": ["訟卦"],          "師": ["師卦"],    "比": ["比卦"],
    "小畜": ["小畜卦"],      "履": ["履卦"],          "泰": ["泰卦"],    "否": ["否卦"],
    "同人": ["同人卦"],       "大有": ["大有卦"],       "謙": ["謙卦"],    "豫": ["豫卦"],
    "隨": ["隨卦"],          "蠱": ["蠱卦"],          "臨": ["臨卦"],    "觀": ["觀卦"],
    "噬嗑": ["噬嗑卦"],      "賁": ["賁卦"],          "剝": ["剝卦"],    "復": ["復卦"],
    "無妄": ["無妄卦"],       "大畜": ["大畜卦"],       "頤": ["頤卦"],    "大過": ["大過卦"],
    "坎": ["坎卦"],          "離": ["離卦"],          "咸": ["咸卦"],    "恒": ["恒卦"],
    "遯": ["遯卦"],          "大壯": ["大壯卦"],       "晉": ["晉卦"],    "明夷": ["明夷卦"],
    "家人": ["家人卦"],       "睽": ["睽卦"],          "蹇": ["蹇卦"],    "解": ["解卦"],
    "損": ["損卦"],          "益": ["益卦"],          "夬": ["夬卦"],    "姤": ["姤卦"],
    "萃": ["萃卦"],          "升": ["升卦"],          "困": ["困卦"],    "井": ["井卦"],
    "革": ["革卦"],          "鼎": ["鼎卦"],          "震": ["震卦"],    "艮": ["艮卦"],
    "漸": ["漸卦"],          "豐": ["豐卦"],          "旅": ["旅卦"],
    "巽": ["巽卦"],          "兌": ["兌卦"],          "渙": ["渙卦"],    "節": ["節卦"],
    "中孚": ["中孚卦"],       "小過": ["小過卦"],       "既濟": ["既濟卦"], "未濟": ["未濟卦"],
}


def chunk_key(collection: str, chunk: dict, position: int = 0) -> str:
    """Return a stable key shared by the offline indexer and runtime retriever."""
    return f"{collection}:{chunk.get('id') or position}"


def corpus_hash(chunks: list[dict], lingqian_chunks: list[dict]) -> str:
    """Fingerprint retrieval-relevant content so stale vector indexes are rejected."""
    digest = hashlib.sha256()
    for collection, items in (("book", chunks), ("lingqian", lingqian_chunks)):
        for position, chunk in enumerate(items):
            payload = {
                "key": chunk_key(collection, chunk, position),
                "text": chunk.get("text", ""),
                "zodiac": chunk.get("zodiac"),
                "topic": chunk.get("topic"),
                "chapter": chunk.get("chapter"),
                "hexagram": chunk.get("hexagram"),
            }
            digest.update(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8"))
    return digest.hexdigest()


class RAGEngine:
    def __init__(
        self,
        query_embedder: Callable[[str], list[float]] | None = None,
        *,
        chunks_path: str = CHUNKS_PATH,
        lingqian_path: str = LINGQIAN_PATH,
        lingqian_old_path: str = LINGQIAN_OLD_PATH,
        profile_path: str = PROFILE_PATH,
        embeddings_path: str = EMBEDDINGS_PATH,
        semantic_weight: float = 0.65,
        min_semantic_score: float = 0.62,
        unscoped_min_semantic_score: float = 0.68,
        expected_embedding_model: str | None = None,
        expected_embedding_dimensions: int | None = None,
    ):
        self.chunks: list[dict] = []
        self.lingqian_chunks: list[dict] = []
        self.profile_text: str = ""
        self.query_embedder = query_embedder
        self.semantic_weight = min(max(semantic_weight, 0.0), 1.0)
        self.min_semantic_score = min_semantic_score
        self.unscoped_min_semantic_score = max(unscoped_min_semantic_score, min_semantic_score)
        self.expected_embedding_model = expected_embedding_model
        self.expected_embedding_dimensions = expected_embedding_dimensions
        self.embedding_model = ""
        self.embeddings: dict[str, list[float]] = {}
        self.semantic_enabled = False
        self._query_cache: dict[str, list[float]] = {}
        self._paths = {
            "chunks": chunks_path,
            "lingqian": lingqian_path,
            "lingqian_old": lingqian_old_path,
            "profile": profile_path,
            "embeddings": embeddings_path,
        }
        self._load_all()

    def _load_all(self):
        chunks_path = self._paths["chunks"]
        profile_path = self._paths["profile"]
        lingqian_path = self._paths["lingqian"]
        lingqian_old_path = self._paths["lingqian_old"]

        if os.path.exists(chunks_path):
            with open(chunks_path, encoding="utf-8") as f:
                self.chunks = json.load(f)
            print(f"[RAG] 知識庫載入：{len(self.chunks)} 塊")
        else:
            print(f"[RAG] 警告：找不到 {chunks_path}")

        if os.path.exists(profile_path):
            with open(profile_path, encoding="utf-8") as f:
                self.profile_text = f.read()
            print("[RAG] 身份資料載入完成")

        if os.path.exists(lingqian_path):
            with open(lingqian_path, encoding="utf-8") as f:
                self.lingqian_chunks = json.load(f)
            print(f"[RAG] 北帝靈簽載入完成（{len(self.lingqian_chunks)} 卦）")
        elif os.path.exists(lingqian_old_path):
            # 向後兼容：舊版全文格式
            with open(lingqian_old_path, encoding="utf-8") as f:
                old = json.load(f)
            self.lingqian_chunks = [{"hexagram": "全文", "text": old.get("text", "")[:2000],
                                     "source": "北帝靈簽詳解"}]
            print("[RAG] 北帝靈簽（舊版全文）載入完成")

        self._load_embeddings()

    def _load_embeddings(self):
        path = self._paths["embeddings"]
        if not self.query_embedder or not os.path.exists(path):
            print("[RAG] 語義檢索未啟用，使用規則與關鍵字模式")
            return
        try:
            with open(path, encoding="utf-8") as f:
                index = json.load(f)
            expected_hash = corpus_hash(self.chunks, self.lingqian_chunks)
            if index.get("corpus_hash") != expected_hash:
                print("[RAG] 向量索引已過期，請重新執行 build_embedding_index.py")
                return
            if self.expected_embedding_model and index.get("model") != self.expected_embedding_model:
                print("[RAG] 向量索引模型不符，請重新執行 build_embedding_index.py")
                return
            if self.expected_embedding_dimensions and index.get("dimensions") != self.expected_embedding_dimensions:
                print("[RAG] 向量索引維度不符，請重新執行 build_embedding_index.py")
                return
            embeddings = index.get("embeddings", {})
            if not isinstance(embeddings, dict) or not embeddings:
                print("[RAG] 向量索引為空，使用關鍵字模式")
                return
            self.embeddings = embeddings
            self.embedding_model = index.get("model", "")
            self.semantic_enabled = True
            print(f"[RAG] 語義索引載入：{len(embeddings)} 個向量（{self.embedding_model}）")
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            print(f"[RAG] 向量索引載入失敗，使用關鍵字模式：{exc}")

    # ── 解析查詢 ─────────────────────────────────────

    def _extract_zodiacs(self, query: str) -> list[str]:
        found = []
        for zodiac, aliases in ZODIAC_ALIASES.items():
            if any(a in query for a in aliases):
                found.append(zodiac)
        return found

    def _extract_topics(self, query: str) -> list[str]:
        return [topic for topic, aliases in TOPIC_ALIASES.items() if any(alias in query for alias in aliases)]

    def enrich_query(self, query: str, history: list[dict] | None = None) -> str:
        """Reuse a zodiac mentioned earlier and append canonical metadata terms."""
        enriched = query.strip()
        zodiacs = self._extract_zodiacs(enriched)
        if not zodiacs and history:
            for turn in reversed(history[-10:]):
                if turn.get("role") != "user":
                    continue
                previous = str(turn.get("content", ""))
                found = self._extract_zodiacs(previous)
                if found:
                    zodiacs = found
                    enriched = f"{enriched} 屬{found[0]}"
                    break

        topics = self._extract_topics(enriched)
        canonical_terms = [*topics]
        canonical_terms.extend(f"屬{zodiac}" for zodiac in zodiacs)
        if canonical_terms:
            enriched = f"{enriched} {' '.join(dict.fromkeys(canonical_terms))}"
        return enriched

    def needs_zodiac_clarification(self, query: str) -> bool:
        """Return True when a personal annual-fortune answer needs missing zodiac data."""
        if self._extract_zodiacs(query):
            return False
        topics = set(self._extract_topics(query))
        return bool(topics & PERSONAL_FORTUNE_TOPICS) and any(marker in query for marker in PERSONAL_QUERY_MARKERS)

    def query_metadata(self, query: str) -> dict:
        """Return non-sensitive intent metadata for retrieval observability."""
        return {
            "zodiacs": self._extract_zodiacs(query),
            "topics": self._extract_topics(query),
            "query_length": len(query),
        }

    def _keyword_score(self, chunk: dict, query: str) -> float:
        """關鍵字得分 × 塊權重（第六章週運塊 weight=0.5，其餘 weight=1.0）"""
        text = chunk.get("text", "")
        terms: set[str] = set(re.findall(r"[A-Za-z0-9]{2,}", query.lower()))
        for sequence in re.findall(r"[一-鿿]+", query):
            if len(sequence) >= 2:
                terms.add(sequence)
                terms.update(sequence[index:index + 2] for index in range(len(sequence) - 1))
        score = sum(text.lower().count(term) for term in terms)
        weight = chunk.get("weight", 1.0)
        return score * weight

    @staticmethod
    def _cosine_similarity(left: list[float], right: list[float]) -> float:
        if not left or len(left) != len(right):
            return -1.0
        dot = sum(a * b for a, b in zip(left, right))
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        if not left_norm or not right_norm:
            return -1.0
        return dot / (left_norm * right_norm)

    def _embed_query(self, query: str) -> list[float] | None:
        if not self.semantic_enabled or not self.query_embedder:
            return None
        if query in self._query_cache:
            return self._query_cache[query]
        try:
            vector = self.query_embedder(query)
            if not vector:
                return None
            if len(self._query_cache) >= 128:
                self._query_cache.pop(next(iter(self._query_cache)))
            self._query_cache[query] = vector
            return vector
        except Exception as exc:
            # Retrieval must remain available when the embedding API is unavailable.
            print(f"[RAG] 查詢向量生成失敗，回退關鍵字模式：{exc}")
            return None

    def _rank(
        self,
        pool: list[dict],
        query: str,
        *,
        collection: str,
        allow_zero_match: bool = False,
        use_semantic: bool = True,
        min_semantic_score: float | None = None,
    ) -> list[dict]:
        if not pool:
            return []

        query_vector = self._embed_query(query) if use_semantic else None
        semantic_floor = self.min_semantic_score if min_semantic_score is None else min_semantic_score
        lexical_scores = [self._keyword_score(chunk, query) for chunk in pool]
        semantic_scores: list[float] = []

        for position, chunk in enumerate(pool):
            vector = self.embeddings.get(chunk_key(collection, chunk, position))
            semantic_scores.append(
                self._cosine_similarity(query_vector, vector) if query_vector and vector else -1.0
            )

        lexical_order = sorted(range(len(pool)), key=lambda index: lexical_scores[index], reverse=True)
        semantic_order = sorted(range(len(pool)), key=lambda index: semantic_scores[index], reverse=True)
        lexical_ranks = {index: rank for rank, index in enumerate(lexical_order, 1) if lexical_scores[index] > 0}
        semantic_ranks = {
            index: rank
            for rank, index in enumerate(semantic_order, 1)
            if semantic_scores[index] >= semantic_floor
        }
        rrf_k = 60
        best_possible_rrf = 1 / (rrf_k + 1)
        scored: list[tuple[float, dict]] = []

        for position, chunk in enumerate(pool):
            lexical_raw = lexical_scores[position]
            semantic = semantic_scores[position]
            has_semantic_match = semantic >= semantic_floor

            if not allow_zero_match:
                # When semantic retrieval is available, lexical overlap alone must not
                # admit an out-of-domain passage. Keyword-only fallback remains available
                # when the embedding API is disabled or temporarily fails.
                if query_vector and not has_semantic_match:
                    continue
                if not query_vector and lexical_raw <= 0:
                    continue

            lexical_rrf = 1 / (rrf_k + lexical_ranks[position]) if position in lexical_ranks else 0.0
            semantic_rrf = 1 / (rrf_k + semantic_ranks[position]) if position in semantic_ranks else 0.0

            if semantic >= -0.5:
                weighted_rrf = (1 - self.semantic_weight) * lexical_rrf + self.semantic_weight * semantic_rrf
                score = weighted_rrf / best_possible_rrf
                method = "hybrid" if lexical_raw > 0 else "semantic"
            else:
                score = lexical_rrf / best_possible_rrf
                method = "keyword" if lexical_raw > 0 else "metadata"

            result = dict(chunk)
            result["_retrieval_score"] = round(score, 4)
            result["_semantic_score"] = round(semantic, 4) if semantic >= -0.5 else None
            result["_retrieval_method"] = method
            scored.append((score, result))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [chunk for _, chunk in scored]

    def _search_lingqian(self, query: str) -> list[dict]:
        """搜尋靈簽：優先匹配具名卦，否則按關鍵字評分"""
        # 直接指定卦名
        for hexagram, aliases in HEXAGRAM_ALIASES.items():
            if any(a in query for a in aliases):
                for c in self.lingqian_chunks:
                    if c.get("hexagram") == hexagram:
                        result = dict(c)
                        result["_retrieval_score"] = 1.0
                        result["_retrieval_method"] = "exact"
                        return [result]

        # 沒有具名卦時用混合檢索；無相關結果不再錯誤返回第一卦。
        return self._rank(self.lingqian_chunks, query, collection="lingqian")[:1]

    # ── 四優先級搜尋 ──────────────────────────────────

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        if not self.chunks:
            return []

        # 特殊路由：靈簽（含靈簽關鍵字 或 直接提及卦名）
        is_lingqian = any(kw in query for kw in LINGQIAN_KEYWORDS)
        if not is_lingqian:
            is_lingqian = any(
                any(a in query for a in aliases)
                for aliases in HEXAGRAM_ALIASES.values()
            )
        if is_lingqian:
            return self._search_lingqian(query)

        # 特殊路由：身份查詢
        if any(kw in query for kw in IDENTITY_KEYWORDS):
            return [{"text": self.profile_text, "source": "身份資料", "zodiac": None}]

        zodiacs = self._extract_zodiacs(query)
        topics  = self._extract_topics(query)

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
            exact_results = self._rank(
                [c for c in self.chunks if c.get("zodiac") in zodiacs and c.get("topic") in topics],
                query, collection="book", allow_zero_match=True, use_semantic=False,
            )
            if exact_results:
                return exact_results[:top_k]

        # 優先級2：只匹配生肖
        if zodiacs and len(results) < top_k:
            add(self._rank(
                [c for c in self.chunks if c.get("zodiac") in zodiacs],
                query, collection="book", allow_zero_match=True,
            ))

        # 優先級3：只匹配主題
        if topics and len(results) < top_k:
            topic_pool = [
                c for c in self.chunks
                if c.get("topic") in topics and (zodiacs or not c.get("zodiac"))
            ]
            add(self._rank(
                topic_pool,
                query, collection="book", allow_zero_match=True,
            ))

        # 優先級4：全文關鍵字搜尋
        if len(results) < top_k:
            fallback_pool = self.chunks if zodiacs else [c for c in self.chunks if not c.get("zodiac")]
            add(self._rank(
                fallback_pool,
                query,
                collection="book",
                min_semantic_score=self.unscoped_min_semantic_score if not zodiacs and not topics else None,
            ))

        return results[:top_k]

    def format_context(self, results: list[dict]) -> str:
        if not results:
            return "知識庫中沒有檢索到足夠相關的參考資料。請勿虛構書本內容，並向用戶說明資料不足。"
        parts = []
        for i, chunk in enumerate(results, 1):
            source = chunk.get("source") or chunk.get("chapter", "2026全書")
            zodiac = chunk.get("zodiac")
            topic  = chunk.get("topic", "")
            if zodiac:
                label = f"屬{zodiac}" + (f"·{topic}" if topic and topic != "其他" else "")
            else:
                label = source
            source = chunk.get("source") or chunk.get("chapter", "2026全書")
            page_start = chunk.get("page_start")
            page_end = chunk.get("page_end")
            if page_start is not None:
                pages = str(page_start) if page_end in (None, page_start) else f"{page_start}-{page_end}"
                citation = f"｜來源：{source}，頁碼：{pages}"
            else:
                citation = f"｜來源：{source}"
            parts.append(f"【參考資料{i}｜{label}{citation}】\n{chunk['text']}")

        return (
            f"以下是參考資料：\n\n"
            + "\n\n".join(parts)
            + "\n\n請根據以上資料回答用戶問題。"
        )

    def get_context(self, query: str, top_k: int = 3) -> str:
        return self.format_context(self.search(query, top_k=top_k))

    @staticmethod
    def citations(results: list[dict]) -> list[dict]:
        """Return safe, structured citation metadata for API and UI consumers."""
        citations = []
        seen = set()
        for chunk in results:
            chunk_id = chunk.get("id")
            source = chunk.get("source") or chunk.get("chapter", "2026全書")
            key = (chunk_id, source, chunk.get("page_start"), chunk.get("page_end"))
            if key in seen:
                continue
            seen.add(key)
            citations.append({
                "id": chunk_id,
                "source": source,
                "chapter": chunk.get("chapter"),
                "page_start": chunk.get("page_start"),
                "page_end": chunk.get("page_end"),
                "zodiac": chunk.get("zodiac"),
                "topic": chunk.get("topic"),
                "hexagram": chunk.get("hexagram"),
                "retrieval_method": chunk.get("_retrieval_method"),
                "score": chunk.get("_retrieval_score"),
            })
        return citations

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)

    @property
    def lingqian_count(self) -> int:
        return len(self.lingqian_chunks)

    @property
    def embedding_count(self) -> int:
        return len(self.embeddings)
