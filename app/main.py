#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
李丞責2026運勢AI — Flask 後端（google.genai 新SDK版）
路由：GET / → 主網站  GET /chat → 聊天介面  POST /analyze → 運勢分析  POST /chat → 聊天API
"""

import os
import re
import sys
import time
import json
import secrets
import socket
import smtplib
import ssl
from threading import Lock
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

from google import genai
from google.genai import types
from flask import Flask, request, jsonify, render_template, session, redirect, url_for
from flask_cors import CORS
from rag import RAGEngine
from bazi import calculate_bazi

# ── Gemini 設定 ─────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL   = "gemini-2.5-flash"
EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "gemini-embedding-001")
EMBEDDING_DIMENSIONS = int(os.getenv("RAG_EMBEDDING_DIMENSIONS", "768"))
RAG_LOG_RETRIEVAL = os.getenv("RAG_LOG_RETRIEVAL", "1") == "1"
RAG_RERANK_ENABLED = os.getenv("RAG_RERANK_ENABLED", "1") == "1"
RAG_RERANK_SCORE_MARGIN = float(os.getenv("RAG_RERANK_SCORE_MARGIN", "0.08"))
gemini_client  = genai.Client(api_key=GEMINI_API_KEY)

# ── Gmail SMTP ────────────────────────────────────────────
GMAIL_USER = os.getenv("GMAIL_USER", "byondhk@gmail.com")
GMAIL_PASS = os.getenv("GMAIL_PASS", "pvrhyewvwprfqmnx")

# ── 載入 system prompt ──────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _load_system_prompt() -> str:
    path = os.path.join(BASE, "system_prompt.md")
    if not os.path.exists(path):
        return "你是李丞責博士本人，香港著名玄學風水專家，以繁體中文書面語回答問題。"
    with open(path, encoding="utf-8") as f:
        raw = f.read()
    marker = "## 正式System Prompt（複製此段用於代碼）"
    if marker in raw:
        raw = raw[raw.index(marker) + len(marker):].strip()
    menu_marker = "## 選單設計"
    if menu_marker in raw:
        raw = raw[:raw.index(menu_marker)].strip()
    return raw

SYSTEM_PROMPT = _load_system_prompt()

# ── Flask App ──────────────────────────────────────────
app = Flask(__name__)
app.json.ensure_ascii = False
app.secret_key = os.getenv("SECRET_KEY", "lczai-2026-site-gate-secret")
app.permanent_session_lifetime = timedelta(days=30)
CORS(app)

# Chat profile data stays server-side. The signed session cookie only stores a
# random lookup ID, not the user's birth date or other profile fields.
CHAT_PROFILE_TTL_SECONDS = int(os.getenv("CHAT_PROFILE_TTL_SECONDS", "86400"))
CHAT_PROFILES: dict[str, dict] = {}
CHAT_PROFILES_LOCK = Lock()

def embed_retrieval_query(query: str) -> list[float]:
    """Embed one search query; document vectors are precomputed offline."""
    response = gemini_client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=query,
        config={
            "task_type": "RETRIEVAL_QUERY",
            "output_dimensionality": EMBEDDING_DIMENSIONS,
        },
    )
    if not response.embeddings or not response.embeddings[0].values:
        raise RuntimeError("Embedding API returned an empty query vector")
    return list(response.embeddings[0].values)


rag = RAGEngine(
    query_embedder=embed_retrieval_query if GEMINI_API_KEY else None,
    semantic_weight=float(os.getenv("RAG_SEMANTIC_WEIGHT", "0.65")),
    min_semantic_score=float(os.getenv("RAG_MIN_SEMANTIC_SCORE", "0.62")),
    unscoped_min_semantic_score=float(os.getenv("RAG_UNSCOPED_MIN_SEMANTIC_SCORE", "0.68")),
    expected_embedding_model=EMBEDDING_MODEL,
    expected_embedding_dimensions=EMBEDDING_DIMENSIONS,
)


def log_retrieval(query: str, results: list[dict], status: str, elapsed: float) -> None:
    """Log retrieval diagnostics without storing the user's message or personal data."""
    if not RAG_LOG_RETRIEVAL:
        return
    payload = {
        **rag.query_metadata(query),
        "status": status,
        "elapsed": elapsed,
        "result_count": len(results),
        "results": [
            {
                "id": result.get("id"),
                "topic": result.get("topic"),
                "zodiac": result.get("zodiac"),
                "method": result.get("_retrieval_method"),
                "score": result.get("_retrieval_score"),
                "semantic_score": result.get("_semantic_score"),
            }
            for result in results
        ],
    }
    print("[RAG_METRIC] " + json.dumps(payload, ensure_ascii=False, separators=(",", ":")))


REPORT_TOPIC_MAP = [
    ("整體運勢", "overall"),
    ("財運", "wealth"),
    ("事業", "career"),
    ("感情", "love"),
    ("健康", "health"),
    ("化解建議", "remedy"),
]


def retrieve_report_sources(shengxiao: str) -> tuple[dict, list[str], dict, list[str]]:
    """Retrieve and validate all six deterministic source sections for a report."""
    sections = {}
    context_parts = []
    citations = {}
    missing = []

    for topic_zh, topic_key in REPORT_TOPIC_MAP:
        query = f"屬{shengxiao} {topic_zh}"
        retrieval_t0 = time.time()
        results = rag.search(query, top_k=1)
        retrieval_elapsed = round(time.time() - retrieval_t0, 3)
        exact = next(
            (
                chunk for chunk in results
                if chunk.get("zodiac") == shengxiao and chunk.get("topic") == topic_zh
            ),
            None,
        )
        log_retrieval(query, [exact] if exact else [], "report_source" if exact else "missing_report_source", retrieval_elapsed)
        if not exact:
            missing.append(topic_zh)
            sections[topic_key] = ""
            continue

        sections[topic_key] = exact["text"]
        context_parts.append(f"【{topic_zh}｜屬{shengxiao}】\n{exact['text']}")
        citation = rag.citations([exact])
        citations[topic_key] = citation[0] if citation else None

    return sections, context_parts, citations, missing


def dedupe_display_citations(citations: list[dict]) -> list[dict]:
    """Collapse chunk-level citations that render as the same human-readable source."""
    unique = []
    seen = set()
    for citation in citations:
        key = (
            citation.get("source"),
            citation.get("page_start"),
            citation.get("page_end"),
            citation.get("zodiac"),
            citation.get("topic"),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(citation)
    return unique


def _empty_chat_profile() -> dict:
    return {"birth_date": None, "zodiac": None, "gender": None, "topics": [], "updated_at": time.time()}


def _chat_memory_id(create: bool = False) -> str | None:
    memory_id = session.get("chat_memory_id")
    if not memory_id and create:
        memory_id = secrets.token_urlsafe(18)
        session["chat_memory_id"] = memory_id
    return memory_id


def get_chat_profile(create: bool = False) -> dict:
    """Return a copy of the current session's server-side structured profile."""
    memory_id = _chat_memory_id(create=create)
    if not memory_id:
        return _empty_chat_profile()
    now = time.time()
    with CHAT_PROFILES_LOCK:
        profile = CHAT_PROFILES.get(memory_id)
        if profile and now - profile.get("updated_at", 0) > CHAT_PROFILE_TTL_SECONDS:
            CHAT_PROFILES.pop(memory_id, None)
            profile = None
        if profile is None:
            profile = _empty_chat_profile()
            if create:
                CHAT_PROFILES[memory_id] = profile
        return dict(profile)


def save_chat_profile(profile: dict) -> dict:
    memory_id = _chat_memory_id(create=True)
    stored = {
        "birth_date": profile.get("birth_date"),
        "zodiac": profile.get("zodiac"),
        "gender": profile.get("gender"),
        "topics": list(dict.fromkeys(profile.get("topics") or []))[-5:],
        "updated_at": time.time(),
    }
    with CHAT_PROFILES_LOCK:
        expired_ids = [
            key for key, value in CHAT_PROFILES.items()
            if stored["updated_at"] - value.get("updated_at", 0) > CHAT_PROFILE_TTL_SECONDS
        ]
        for expired_id in expired_ids:
            CHAT_PROFILES.pop(expired_id, None)
        CHAT_PROFILES[memory_id] = stored
    return dict(stored)


def clear_chat_profile() -> None:
    memory_id = session.pop("chat_memory_id", None)
    if memory_id:
        with CHAT_PROFILES_LOCK:
            CHAT_PROFILES.pop(memory_id, None)


def public_chat_profile(profile: dict) -> dict:
    return {
        "birth_date": profile.get("birth_date"),
        "zodiac": profile.get("zodiac"),
        "gender": profile.get("gender"),
        "topics": profile.get("topics") or [],
    }


def _extract_birth_date(text: str) -> date | None:
    patterns = (
        r"(?<!\d)(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})(?!\d)",
        r"(?<!\d)(\d{4})年\s*(\d{1,2})月\s*(\d{1,2})日?",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        try:
            return date(*map(int, match.groups()))
        except ValueError:
            continue
    return None


def _extract_gender(text: str) -> str | None:
    if re.search(r"(?:我是|本人是?|性別[:：]?\s*)(?:男性|男士|男)(?:\b|人|生)?", text):
        return "男"
    if re.search(r"(?:我是|本人是?|性別[:：]?\s*)(?:女性|女士|女)(?:\b|人|生)?", text):
        return "女"
    return None


def update_chat_profile(user_msg: str, history: list[dict]) -> tuple[dict, bool]:
    """Extract stable profile fields and report whether stored memory enriched this turn."""
    profile = get_chat_profile(create=True)
    original = public_chat_profile(profile)
    metadata = rag.query_metadata(user_msg)
    birth = _extract_birth_date(user_msg)
    explicit_zodiac = metadata["zodiacs"][0] if metadata["zodiacs"] else None

    if birth:
        inferred_zodiac = calculate_bazi(birth.year, birth.month, birth.day)["shengxiao"]
        if (
            (profile.get("birth_date") and profile["birth_date"] != birth.isoformat())
            or (profile.get("zodiac") and profile["zodiac"] != inferred_zodiac)
        ):
            profile = _empty_chat_profile()
        profile["birth_date"] = birth.isoformat()
        profile["zodiac"] = inferred_zodiac
    elif explicit_zodiac:
        if profile.get("zodiac") and profile["zodiac"] != explicit_zodiac:
            profile = _empty_chat_profile()
        profile["zodiac"] = explicit_zodiac
    elif not profile.get("zodiac"):
        # Smoothly migrate context created before structured memory was enabled.
        for turn in reversed(history[-10:]):
            if turn.get("role") != "user":
                continue
            previous = str(turn.get("content", ""))
            previous_birth = _extract_birth_date(previous)
            previous_zodiacs = rag.query_metadata(previous)["zodiacs"]
            if previous_birth:
                profile["birth_date"] = previous_birth.isoformat()
                profile["zodiac"] = calculate_bazi(
                    previous_birth.year, previous_birth.month, previous_birth.day
                )["shengxiao"]
                break
            if previous_zodiacs:
                profile["zodiac"] = previous_zodiacs[0]
                break

    gender = _extract_gender(user_msg)
    if gender:
        profile["gender"] = gender
    if metadata["topics"]:
        profile["topics"] = [*(profile.get("topics") or []), *metadata["topics"]]

    memory_used = bool(
        original.get("zodiac")
        and not birth
        and not explicit_zodiac
    )
    return save_chat_profile(profile), memory_used


def apply_chat_profile(user_msg: str, profile: dict) -> str:
    enriched = user_msg
    if profile.get("zodiac") and not rag.query_metadata(enriched)["zodiacs"]:
        enriched = f"{enriched} 生肖屬{profile['zodiac']}"
    return enriched


def format_chat_profile_context(profile: dict) -> str:
    fields = []
    if profile.get("birth_date"):
        fields.append(f"出生日期：{profile['birth_date']}")
    if profile.get("zodiac"):
        fields.append(f"生肖：屬{profile['zodiac']}")
    if profile.get("gender"):
        fields.append(f"性別：{profile['gender']}")
    return "【本次會話資料】\n" + "\n".join(fields) if fields else ""

# ── 網站密碼保護 ─────────────────────────────────────────
SITE_PASSWORD = os.getenv("SITE_PASSWORD", "88888888")

# 不需要登入即可訪問的路由（登入頁本身、靜態檔案、健康檢查）
PUBLIC_ENDPOINTS = {"login", "static", "health"}
# 前端 AJAX 呼叫的 API：未登入時回傳 401 JSON，而非導向登入頁
JSON_ENDPOINTS = {"chat", "analyze", "test_email"}


@app.before_request
def require_login():
    if request.endpoint in PUBLIC_ENDPOINTS:
        return
    if session.get("authenticated"):
        return
    if request.endpoint in JSON_ENDPOINTS:
        return jsonify({"error": "請先登入"}), 401
    return redirect(url_for("login", next=request.path))


@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        if request.form.get("password", "") == SITE_PASSWORD:
            session.permanent = True
            session["authenticated"] = True
            return redirect(request.args.get("next") or url_for("index"))
        error = "密碼錯誤，請重新輸入"
    return render_template("login.html", error=error)


@app.route("/logout")
def logout():
    clear_chat_profile()
    session.pop("authenticated", None)
    return redirect(url_for("login"))


# ── 工具函數 ───────────────────────────────────────────

def call_gemini_raw(prompt: str, max_tokens: int = 2048,
                    disable_thinking: bool = False) -> str:
    """單輪呼叫 Gemini（無歷史）"""
    cfg_kwargs: dict = dict(temperature=0.7, max_output_tokens=max_tokens)
    if disable_thinking:
        cfg_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
    response = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[types.Content(role="user", parts=[types.Part(text=prompt)])],
        config=types.GenerateContentConfig(**cfg_kwargs),
    )
    return response.text


def call_gemini(system_prompt: str, context: str,
                user_msg: str, history: list[dict]) -> str:
    """多輪對話呼叫 Gemini"""
    format_rules = """回答格式補充：
- 使用純文字，不要使用 Markdown 語法，不要輸出星號、井號、反引號或 Markdown 表格。
- 回答要完整、有結論，通常控制在 300 至 700 個中文字內；不要在句子中途停止。
- 需要分點時使用中文序號「一、二、三」或全形圓點「•」。"""
    full_user_msg = f"{system_prompt}\n\n{format_rules}\n\n{context}\n\n用戶問題：{user_msg}"
    contents: list[types.Content] = []
    for turn in history[-5:]:
        role    = turn.get("role", "user")
        content = turn.get("content", "").strip()
        if role not in ("user", "model") or not content:
            continue
        # 舊版前端會把當前問題也放進 history；避免模型重複看到同一句。
        if role == "user" and content == user_msg:
            continue
        contents.append(types.Content(role=role, parts=[types.Part(text=content)]))
    contents.append(types.Content(role="user", parts=[types.Part(text=full_user_msg)]))
    config = types.GenerateContentConfig(
        temperature=0.55,
        max_output_tokens=4096,
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    )
    response = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=contents,
        config=config,
    )
    reply = response.text or ""

    candidates = getattr(response, "candidates", None) or []
    finish_reason = getattr(candidates[0], "finish_reason", None) if candidates else None
    finish_name = getattr(finish_reason, "name", str(finish_reason or ""))
    if "MAX_TOKENS" in finish_name and reply:
        continuation_contents = [
            *contents,
            types.Content(role="model", parts=[types.Part(text=reply)]),
            types.Content(
                role="user",
                parts=[types.Part(text="請只從剛才中斷的位置接續，完成尚未說完的內容；不要重複前文。")],
            ),
        ]
        continuation = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=continuation_contents,
            config=types.GenerateContentConfig(
                temperature=0.45,
                max_output_tokens=2048,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        reply = reply.rstrip() + (continuation.text or "").lstrip()

    return reply


def clean_chat_reply(text: str) -> str:
    """Render model output as clean plain text even if it ignores format instructions."""
    cleaned = (text or "").replace("\r\n", "\n")
    cleaned = re.sub(r"```(?:[\w+-]+)?\s*", "", cleaned)
    cleaned = cleaned.replace("```", "").replace("**", "").replace("__", "").replace("`", "")
    cleaned = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", cleaned)
    cleaned = re.sub(r"(?m)^\s*[-*+]\s+", "• ", cleaned)
    cleaned = re.sub(r"(?m)^\s*>\s?", "", cleaned)
    cleaned = cleaned.replace("~~", "").replace("*", "")
    cleaned = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1（\2）", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def chat_social_reply(user_msg: str) -> str | None:
    """Handle lightweight conversation without pretending it needs book evidence."""
    normalized = re.sub(r"[\s，。！？!?、,.：:～~]+", "", user_msg.lower())
    greetings = {"你好", "您好", "哈囉", "嗨", "hi", "hello", "早晨", "早安", "午安", "晚安", "在嗎", "喂"}
    thanks = {"謝謝", "多謝", "唔該", "thankyou", "thanks", "明白了", "知道了"}
    goodbyes = {"再見", "拜拜", "bye", "下次再聊", "遲啲再傾"}
    identity_phrases = ("你是誰", "你係邊個", "介紹一下自己", "自我介紹", "你可以做什麼", "你可以做咩", "你識做咩")

    if normalized in greetings or (len(normalized) <= 18 and any(phrase in normalized for phrase in identity_phrases)):
        return (
            "你好，我是李丞責博士的AI玄學問答助手。我會根據李丞責博士的著作，"
            "回答2026年生肖運勢、財運、事業、感情、健康、風水、五行、太歲及北帝靈籤等問題。"
            "如果你想了解個人運勢，可以告訴我生肖或完整出生日期，再說明想問的事情。"
        )
    if normalized in thanks:
        return "不用客氣，很高興能幫到你。如果還想了解其他運勢或風水問題，直接告訴我便可以。"
    if normalized in goodbyes:
        return "再見，祝你事事順利。有需要時再回來找我。"
    return None


def enrich_chat_birth_date(user_msg: str, history: list[dict]) -> str:
    """Append a zodiac inferred from a full birth date in the current chat context."""
    if rag.query_metadata(user_msg)["zodiacs"]:
        return user_msg

    user_texts = [user_msg]
    user_texts.extend(
        str(turn.get("content", ""))
        for turn in reversed(history[-10:])
        if turn.get("role") == "user"
    )
    for text in user_texts:
        birth = _extract_birth_date(text)
        if not birth:
            continue
        zodiac = calculate_bazi(birth.year, birth.month, birth.day)["shengxiao"]
        return f"{user_msg} 生肖屬{zodiac}"
    return user_msg


def should_rerank(query: str, results: list[dict]) -> bool:
    """Only spend a model call when retrieval produces genuinely ambiguous candidates."""
    if not RAG_RERANK_ENABLED or not GEMINI_API_KEY or len(results) < 2:
        return False

    metadata = rag.query_metadata(query)
    top = results[0]
    if (
        len(metadata["zodiacs"]) == 1
        and len(metadata["topics"]) == 1
        and top.get("zodiac") == metadata["zodiacs"][0]
        and top.get("topic") == metadata["topics"][0]
    ):
        return False

    top_score = float(top.get("_retrieval_score") or 0)
    second_score = float(results[1].get("_retrieval_score") or 0)
    close_scores = top_score - second_score <= RAG_RERANK_SCORE_MARGIN
    semantic_candidates = any(
        result.get("_retrieval_method") in {"semantic", "hybrid"}
        for result in results[:4]
    )
    multi_intent = len(metadata["topics"]) > 1
    return multi_intent or (semantic_candidates and close_scores)


def conditional_rerank(query: str, results: list[dict], top_k: int = 3) -> tuple[list[dict], bool]:
    """Rerank ambiguous candidates with Gemini; safely fall back on any failure."""
    if not should_rerank(query, results):
        return results[:top_k], False

    candidates = [
        {
            "key": f"c{index}",
            "zodiac": result.get("zodiac"),
            "topic": result.get("topic"),
            "text": result.get("text", "")[:1200],
        }
        for index, result in enumerate(results[:8], 1)
    ]
    prompt = f"""你是檢索排序器。根據用戶問題，將候選資料按「能否直接支持回答」由高至低排序。
只輸出 JSON 字串陣列，例如 ["c2","c1","c3"]，不要解釋，不要加入 Markdown。
用戶問題：{query}
候選資料：{json.dumps(candidates, ensure_ascii=False)}"""
    try:
        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[types.Content(role="user", parts=[types.Part(text=prompt)])],
            config=types.GenerateContentConfig(
                temperature=0,
                max_output_tokens=256,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        match = re.search(r"\[[\s\S]*?\]", response.text or "")
        ordered_keys = json.loads(match.group(0)) if match else []
        if not isinstance(ordered_keys, list):
            raise ValueError("reranker returned a non-list payload")
        by_key = {candidate["key"]: result for candidate, result in zip(candidates, results[:8])}
        ordered = [by_key[key] for key in ordered_keys if key in by_key]
        ordered_ids = {id(result) for result in ordered}
        ordered.extend(result for result in results[:8] if id(result) not in ordered_ids)
        return ordered[:top_k], True
    except Exception as exc:
        print(f"[RAG] Reranker 失敗，保留原排序：{exc}")
        return results[:top_k], False


def _parse_fortune_sections(text: str) -> dict[str, str]:
    """
    從 Gemini 回覆中提取各段。
    使用 SECTION_ 前綴標記 + lookahead，避免內文星名【】截斷問題。
    """
    sections = {"overall": "", "wealth": "", "career": "", "love": "", "health": "", "advice": ""}

    # 前瞻：下一個 SECTION（不含自身）或字串結尾
    next_sec = r"(?=SECTION_(?:財運|事業|感情|健康|化解建議)|$)"

    patterns = {
        "overall": rf"SECTION_整體運勢\s*(.*?)\s*(?=SECTION_(?:財運|事業|感情|健康|化解建議)|$)",
        "wealth":  rf"SECTION_財運\s*(.*?)\s*(?=SECTION_(?:事業|感情|健康|化解建議)|$)",
        "career":  rf"SECTION_事業\s*(.*?)\s*(?=SECTION_(?:感情|健康|化解建議)|$)",
        "love":    rf"SECTION_感情\s*(.*?)\s*(?=SECTION_(?:健康|化解建議)|$)",
        "health":  rf"SECTION_健康\s*(.*?)\s*(?=SECTION_化解建議|$)",
        "advice":  rf"SECTION_化解建議\s*(.*?)(?:\s*（本內容.*?）)?\s*$",
    }

    found_any = False
    for key, pat in patterns.items():
        m = re.search(pat, text, re.DOTALL)
        if m:
            sections[key] = m.group(1).strip()
            found_any = True

    if not found_any:
        sections["overall"] = text.strip()

    return sections


# ── 路由 ──────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index_new.html")


@app.route("/chat", methods=["GET"])
def chat_page():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "chunks_loaded": rag.chunk_count,
        "semantic_search": rag.semantic_enabled,
        "embeddings_loaded": rag.embedding_count,
        "gemini_model": GEMINI_MODEL,
        "api_key_set": bool(GEMINI_API_KEY),
    })


@app.route("/chat", methods=["POST"])
def chat():
    data     = request.get_json(silent=True) or {}
    user_msg = data.get("message", "").strip()
    history  = data.get("history", [])

    if not user_msg:
        return jsonify({"error": "message 不能為空"}), 400

    social_reply = chat_social_reply(user_msg)
    if social_reply:
        profile, _ = update_chat_profile(user_msg, history)
        log_retrieval(user_msg, [], "social", 0)
        return jsonify({
            "reply": social_reply,
            "elapsed": 0,
            "citations": [],
            "grounded": False,
            "social": True,
            "retrieval_elapsed": 0,
            "profile": public_chat_profile(profile),
            "memory_used": False,
            "reranked": False,
        })

    retrieval_t0 = time.time()
    profile, memory_used = update_chat_profile(user_msg, history)
    profile_query = apply_chat_profile(user_msg, profile)
    retrieval_query = rag.enrich_query(profile_query, history)
    if rag.needs_zodiac_clarification(retrieval_query):
        retrieval_elapsed = round(time.time() - retrieval_t0, 3)
        log_retrieval(retrieval_query, [], "needs_clarification", retrieval_elapsed)
        return jsonify({
            "reply": "要提供個人化的2026年運勢分析，我需要先知道你的生肖或完整出生日期。請告訴我其中一項，我再根據書本資料回答。",
            "elapsed": 0,
            "citations": [],
            "needs_clarification": True,
            "grounded": False,
            "retrieval_elapsed": retrieval_elapsed,
            "profile": public_chat_profile(profile),
            "memory_used": memory_used,
            "reranked": False,
        })

    candidates = rag.search(retrieval_query, top_k=8)
    results, reranked = conditional_rerank(retrieval_query, candidates, top_k=3)
    retrieval_elapsed = round(time.time() - retrieval_t0, 3)
    citations = dedupe_display_citations(rag.citations(results))
    if not results:
        log_retrieval(retrieval_query, [], "no_evidence", retrieval_elapsed)
        return jsonify({
            "reply": "目前知識庫中沒有找到足夠相關的資料，因此我不會把回答說成來自李丞責博士的著作。你可以改問2026年生肖運勢、財運、事業、感情、健康、風水、五行或北帝靈籤。",
            "elapsed": 0,
            "citations": [],
            "grounded": False,
            "retrieval_elapsed": retrieval_elapsed,
            "profile": public_chat_profile(profile),
            "memory_used": memory_used,
            "reranked": reranked,
        })

    log_retrieval(retrieval_query, results, "grounded_reranked" if reranked else "grounded", retrieval_elapsed)
    profile_context = format_chat_profile_context(profile)
    context = "\n\n".join(part for part in [profile_context, rag.format_context(results)] if part)

    t0 = time.time()
    try:
        reply = clean_chat_reply(call_gemini(SYSTEM_PROMPT, context, user_msg, history))
    except Exception as e:
        return jsonify({"error": f"Gemini API 錯誤：{e}"}), 500
    elapsed = round(time.time() - t0, 2)

    return jsonify({
        "reply": reply,
        "elapsed": elapsed,
        "citations": citations,
        "grounded": True,
        "retrieval_elapsed": retrieval_elapsed,
        "profile": public_chat_profile(profile),
        "memory_used": memory_used,
        "reranked": reranked,
    })


def send_report_email(to_addr: str, full_name: str, shengxiao: str,
                      bazi_str: str, lunar: str, wuxing_summary: str,
                      sections: dict, question: str, question_answer: str) -> bool:
    """使用 Gmail SMTP 發送運勢報告，返回是否成功。"""
    if not to_addr:
        return False

    def p(text: str) -> str:
        return "".join(f"<p>{line}</p>" for line in text.splitlines() if line.strip())

    qa_block = ""
    if question and question_answer:
        qa_block = f"""
        <div style="margin-top:24px;padding:20px 24px;background:#fff8f0;border-left:4px solid #990f23;border-radius:8px;">
          <h3 style="color:#990f23;margin:0 0 8px;">❓ 您的問題解答</h3>
          <p style="color:#555;font-style:italic;margin:0 0 12px;">「{question}」</p>
          {p(question_answer)}
        </div>"""

    html = f"""<!DOCTYPE html>
<html lang="zh-Hant">
<head><meta charset="UTF-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/></head>
<body style="margin:0;padding:0;background:#f5f0eb;font-family:'PingFang TC','Microsoft JhengHei',sans-serif;color:#333;">
<table width="100%" cellpadding="0" cellspacing="0" style="background:#f5f0eb;padding:32px 0;">
<tr><td align="center">
<table width="600" cellpadding="0" cellspacing="0" style="background:#fff;border-radius:12px;overflow:hidden;box-shadow:0 4px 24px rgba(0,0,0,.1);">

  <!-- Header -->
  <tr><td style="background:linear-gradient(135deg,#7a0a1b,#990f23);padding:36px 32px;text-align:center;">
    <p style="margin:0 0 4px;color:rgba(211,168,98,.8);font-size:13px;letter-spacing:2px;">李丞責博士</p>
    <h1 style="margin:0;color:#fff;font-size:22px;letter-spacing:1px;">2026馬年個人運程分析報告</h1>
    <p style="margin:8px 0 0;color:rgba(255,255,255,.7);font-size:13px;">結合紫微斗數・奇門遁甲・玄學智慧</p>
  </td></tr>

  <!-- 命盤資料 -->
  <tr><td style="padding:28px 32px 0;">
    <h2 style="color:#990f23;font-size:15px;border-bottom:2px solid #f0e0c8;padding-bottom:8px;margin:0 0 16px;">▸ 命盤基本資料</h2>
    <table width="100%" cellpadding="6" cellspacing="0">
      <tr><td style="color:#888;width:90px;">姓名</td><td style="font-weight:700;">{full_name}</td>
          <td style="color:#888;width:90px;">生肖</td><td style="font-weight:700;">屬{shengxiao}</td></tr>
      <tr><td style="color:#888;">農曆</td><td colspan="3">{lunar}</td></tr>
      <tr><td style="color:#888;">八字</td><td colspan="3">{bazi_str}</td></tr>
      <tr><td style="color:#888;">五行</td><td colspan="3">{wuxing_summary}</td></tr>
    </table>
  </td></tr>

  <!-- 六大運勢 -->
  <tr><td style="padding:24px 32px 0;">
    {''.join(f"""
    <div style="margin-bottom:20px;">
      <h3 style="color:#990f23;font-size:14px;margin:0 0 8px;padding:6px 12px;background:#fff5f5;border-radius:6px;">{icon} {title}</h3>
      <div style="font-size:14px;line-height:1.8;color:#444;">{p(content)}</div>
    </div>""" for icon, title, content in [
        ("🌟","整體運勢", sections.get("overall","")),
        ("💰","財運分析", sections.get("wealth","")),
        ("💼","事業分析", sections.get("career","")),
        ("❤️","感情分析", sections.get("love","")),
        ("🌿","健康提示", sections.get("health","")),
        ("✦","化解建議", sections.get("remedy","")),
    ] if content)}
  </td></tr>

  <!-- 問題解答 -->
  <tr><td style="padding:0 32px;">{qa_block}</td></tr>

  <!-- Footer -->
  <tr><td style="padding:28px 32px;text-align:center;border-top:1px solid #f0e0c8;margin-top:24px;">
    <p style="margin:0;font-size:12px;color:#aaa;">玄學內容僅供參考，一切以個人判斷為準。</p>
    <p style="margin:4px 0 0;font-size:12px;color:#aaa;">© 2026 李丞責中華風水文化基金會</p>
  </td></tr>

</table>
</td></tr>
</table>
</body></html>"""

    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"【李丞責博士】{full_name} 的2026馬年個人運程分析報告"
    msg["From"]    = f"李丞責博士 <{GMAIL_USER}>"
    msg["To"]      = to_addr
    msg.attach(MIMEText(html, "html", "utf-8"))

    try:
        # 強制 IPv4 避免 Render 的 IPv6 路由問題
        ip = socket.gethostbyname("smtp.gmail.com")
        ctx = ssl.create_default_context()
        with smtplib.SMTP(ip, 587, timeout=20) as server:
            server.ehlo("smtp.gmail.com")
            server.starttls(context=ctx)
            server.ehlo("smtp.gmail.com")
            server.login(GMAIL_USER, GMAIL_PASS)
            server.sendmail(GMAIL_USER, to_addr, msg.as_string())
        print(f"[EMAIL OK] 已發送至 {to_addr}", flush=True)
        return True
    except Exception as e:
        print(f"[EMAIL ERROR] {type(e).__name__}: {e}", flush=True)
        return False


@app.route("/test-email")
def test_email():
    """Gmail SMTP 發信測試"""
    to = request.args.get("to", "")
    if not to:
        return jsonify({"error": "需要 ?to=收件人郵箱"}), 400
    import traceback
    try:
        ip = socket.gethostbyname("smtp.gmail.com")
        msg = MIMEMultipart("alternative")
        msg["Subject"] = "李丞責博士 · 郵件測試"
        msg["From"]    = f"李丞責博士 <{GMAIL_USER}>"
        msg["To"]      = to
        msg.attach(MIMEText("<h1>測試成功</h1><p>郵件發送功能正常。</p>", "html", "utf-8"))
        ctx = ssl.create_default_context()
        with smtplib.SMTP(ip, 587, timeout=20) as server:
            server.ehlo("smtp.gmail.com")
            server.starttls(context=ctx)
            server.ehlo("smtp.gmail.com")
            server.login(GMAIL_USER, GMAIL_PASS)
            server.sendmail(GMAIL_USER, to, msg.as_string())
        return jsonify({"status": "success", "to": to})
    except Exception as e:
        return jsonify({"status": "error", "error": f"{type(e).__name__}: {e}",
                        "trace": traceback.format_exc()})


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json(silent=True) or {}

    surname    = data.get("surname", "").strip()
    name       = data.get("name", "").strip()
    birth_date = data.get("birth_date", "")      # "YYYY-MM-DD"
    birth_time = data.get("birth_time")           # "HH:MM" or None
    gender     = data.get("gender", "不透露").strip()
    question   = data.get("question", "").strip()
    email      = data.get("email", "")

    if not all([surname, name, birth_date]):
        return jsonify({"error": "姓名、出生日期為必填"}), 400

    # ── 1. 計算八字 ──
    try:
        y, mo, d = [int(x) for x in birth_date.split("-")]
    except Exception:
        return jsonify({"error": "出生日期格式錯誤，應為 YYYY-MM-DD"}), 400

    hour = minute = None
    if birth_time:
        try:
            hh, mm = birth_time.split(":")
            hour, minute = int(hh), int(mm)
        except Exception:
            hour = minute = None

    try:
        bazi = calculate_bazi(y, mo, d, hour, minute or 0)
    except Exception as e:
        return jsonify({"error": f"八字計算錯誤：{e}"}), 400

    full_name = surname + name

    # ── 2. RAG 搜尋：六主題各取一塊原書內容（直接回傳，不經 Gemini 改寫）──
    shengxiao = bazi['shengxiao']
    rag_sections, context_parts, rag_citations, missing_topics = retrieve_report_sources(shengxiao)
    if missing_topics:
        return jsonify({
            "error": "知識庫缺少報告所需的原書資料：" + "、".join(missing_topics) + "。為避免生成無依據內容，報告已停止。"
        }), 503

    question_citations = []
    question_supported = False
    if question:
        question_query = rag.enrich_query(f"屬{shengxiao} {question}")
        if rag.query_metadata(question_query)["topics"]:
            question_results = rag.search(question_query, top_k=3)
            question_supported = bool(question_results)
            existing_ids = {citation.get("id") for citation in rag_citations.values() if citation}
            extra_results = [chunk for chunk in question_results if chunk.get("id") not in existing_ids]
            if extra_results:
                context_parts.append("【用戶問題補充參考】\n" + "\n\n".join(chunk["text"] for chunk in extra_results))
                question_citations = dedupe_display_citations(rag.citations(extra_results))

    context = "\n\n".join(context_parts)

    # ── 3. 計算五行強弱摘要 ──
    wuxing = bazi["wuxing"]
    wuxing_parts = []
    for elem, cnt in wuxing.items():
        if cnt == 0:
            wuxing_parts.append(f"缺{elem}")
        elif cnt >= 3:
            wuxing_parts.append(f"{elem}旺")
    wuxing_summary = "、".join(wuxing_parts) if wuxing_parts else "五行較為均衡"
    wuxing_detail = " ".join(f"{e}{wuxing.get(e,0)}" for e in ["金","木","水","火","土"])

    hour_pillar_str = f" {bazi['hour_pillar']}" if bazi.get("hour_pillar") else ""
    bazi_str = f"{bazi['year_pillar']} {bazi['month_pillar']} {bazi['day_pillar']}{hour_pillar_str}"

    if question and question_supported:
        question_section = f"""【問題解答】
針對用戶問題「{question}」，結合以上所有分析，給出深入詳盡的回答。300-400字，分2-3個要點展開。
嚴格禁止在此段提及任何星曜名稱，包括但不限於：祿勳、擎天、病符、亡神、的煞、大耗、天解、解神、豹尾、天狗、吊客、月煞、浮沉、血刃、天廚、唐符、歲破等，即使加任何括號或標點均不可。只說「收入有望增加」「有機會晉升」「需注意健康」等實際影響。
末尾加：「（本內容以李丞責著作及八字五行原理為依據，玄學僅供參考。如需深入個人命盤分析，歡迎預約李丞責博士親身批算。）」"""
    elif question:
        question_section = f"""【問題解答】
用戶問題「{question}」不屬於目前知識庫可驗證的運勢、財運、事業、感情、健康、風水、五行、紫微斗數、太歲或靈籤主題。只需清楚說明原書資料不足，不能把一般模型知識說成李丞責博士著作內容；不要自行延伸回答。"""
    else:
        question_section = ""

    # ── 4. 組合 Prompt（七段：六運勢 + 問題解答）──
    prompt = f"""你是李丞責博士本人，現在為用戶提供2026丙午年個人運勢分析。

用戶資料：
- 姓名：{full_name}（{gender}）
- 出生：{bazi['lunar_date']}，生肖屬{shengxiao}
- 八字四柱：{bazi_str}
- 五行狀況：{wuxing_summary}（{wuxing_detail}）
{f"- 用戶問題：{question}" if question else ""}
【書本參考資料】（以下內容來自李丞責2026全書，是分析的最高依據）：
{context}

請根據以上資料，以李丞責博士第一人稱，用繁體中文書面語，為{full_name}提供2026年個人運勢分析。

分析原則：
1. 以書本生肖運勢為主軸和最高依據
2. 在此基礎上，結合用戶的八字四柱和五行狀況，提供個人化的補充分析
3. 如果八字五行與生肖運勢方向一致，可加強說明
4. 如果八字五行與生肖運勢有出入，以生肖運勢為準，用融合的語言表達，例如「雖然你的八字根基如此，但今年的流年氣場⋯⋯」，絕對不可以直接說兩者矛盾或衝突
5. 引用書本中的具體星曜名稱（如唐符、天廚、歲破等），增加可信度

必須嚴格按以下格式輸出{"七" if question else "六"}個部分，不可增刪標題：

【整體運勢】
根據生肖流年運勢，結合八字日主強弱，說明整體氣場走向。150-200字。

重要寫作規則：
- 【整體運勢】可以提及星曜名稱（如祿勳、擎天、病符等）
- 以下六個部分絕對不可提及任何星曜名稱，無論用何種標點符號（【】「」()等）均不可。只描述實際影響，例如說「收入有望增加」而非「有祿勳入命」，說「需注意高危活動」而非「有亡神」，說「有機會獲得重要職責」而非「有擎天」：
  【財運分析】【事業分析】【感情分析】【健康提示】【化解建議】【問題解答】

【財運分析】
以書本財運指引為主，結合五行喜忌，說明進財方向和注意事項。150-200字。

【事業分析】
以書本事業運勢為主，結合四柱特質，說明發展方向和把握時機。150-200字。

【感情分析】
以書本感情運勢為主，結合八字中的感情宮位特質，提供建議。150-200字。

【健康提示】
以書本健康警示為主，結合五行缺失，說明需要注意的身體部位。150-200字。

【化解建議】
以書本的化解方法為主，結合五行補救，提供具體開運建議。150-200字。

{question_section}"""

    # ── 5. 呼叫 Gemini ──
    t0 = time.time()
    try:
        raw_reply = call_gemini_raw(prompt, max_tokens=5000, disable_thinking=True)
    except Exception as e:
        return jsonify({"error": f"Gemini API 錯誤：{e}"}), 500
    elapsed = round(time.time() - t0, 2)

    # ── 6. 解析輸出（【標題】格式，清除 markdown）──
    raw_reply = re.sub(r"\*+", "", raw_reply)

    def _extract(text: str, key: str) -> str:
        # 只在行首的【才視為新節點，避免正文內的【星曜】被截斷
        m = re.search(rf"【{key}】\s*\n(.*?)(?=\n【|\Z)", text, re.S)
        return m.group(1).strip() if m else ""

    gemini_sections = {
        "overall": _extract(raw_reply, "整體運勢"),
        "wealth":  _extract(raw_reply, "財運分析"),
        "career":  _extract(raw_reply, "事業分析"),
        "love":    _extract(raw_reply, "感情分析"),
        "health":  _extract(raw_reply, "健康提示"),
        "remedy":  _extract(raw_reply, "化解建議"),
    }
    # 模型偶爾會在沒有提問時自行追加問題解答；只在用戶真的提問時展示。
    question_answer = _extract(raw_reply, "問題解答") if question else ""

    # 郵件由前端 EmailJS 發送，後端不處理

    return jsonify({
        "name":            full_name,
        "shengxiao":       bazi["shengxiao"],
        "bazi":            bazi["bazi_string"],
        "year_pillar":     bazi["year_pillar"],
        "month_pillar":    bazi["month_pillar"],
        "day_pillar":      bazi["day_pillar"],
        "hour_pillar":     bazi.get("hour_pillar"),
        "lunar":           bazi["lunar_date"],
        "wuxing":          bazi["wuxing"],
        "wuxing_summary":  wuxing_summary,
        "rag_sections":    rag_sections,
        "rag_citations":   rag_citations,
        "question_citations": question_citations,
        "question_grounded": question_supported,
        "gemini_sections": gemini_sections,
        "question":        question,
        "question_answer": question_answer,
        "elapsed":         elapsed,
    })


if __name__ == "__main__":
    print(f"System Prompt 載入（{len(SYSTEM_PROMPT)} 字）")
    print(f"知識庫：{rag.chunk_count} 塊")
    print(f"Gemini 模型：{GEMINI_MODEL}")
    print(f"API Key：{'已設定' if GEMINI_API_KEY else '未設定！'}")
    app.run(debug=True, host="0.0.0.0", port=5000)
