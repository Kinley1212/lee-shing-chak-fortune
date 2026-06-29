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
from datetime import date

sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

from google import genai
from google.genai import types
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from rag import RAGEngine
from bazi import calculate_bazi

# ── Gemini 設定 ─────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL   = "gemini-2.5-flash"
gemini_client  = genai.Client(api_key=GEMINI_API_KEY)

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

# ── 選單定義 ─────────────────────────────────────────────
MENUS: dict[str, list[str]] = {
    "main": ["紫微斗數流年", "奇門遁甲", "塔羅牌占卜", "卜卦（六爻）", "今日運勢", "關於李丞責博士"],
    "ziwei": ["今年財運", "今年感情", "今年事業", "注意月份", "化解方法", "返回主選單"],
    "qimen": ["問事業", "問財運", "問感情", "問搬遷", "如何增強運勢", "返回主選單"],
}

# ── Flask App ──────────────────────────────────────────
app = Flask(__name__)
app.json.ensure_ascii = False
CORS(app)

rag = RAGEngine()


# ── 工具函數 ───────────────────────────────────────────

def detect_menu(msg: str) -> str:
    if any(k in msg for k in ["紫微", "斗數", "流年", "命盤"]):
        return "ziwei"
    if any(k in msg for k in ["奇門", "遁甲"]):
        return "qimen"
    if "塔羅" in msg:
        return "tarot"
    if any(k in msg for k in ["卜卦", "六爻"]):
        return "gua"
    if any(k in msg for k in ["返回", "主選單"]):
        return "main"
    if any(k in msg for k in ["財運", "感情", "事業", "健康", "月份", "化解", "運勢", "生肖"]):
        return "ziwei"
    return "main"


def call_gemini_raw(prompt: str, max_tokens: int = 2048) -> str:
    """單輪呼叫 Gemini（無歷史）"""
    response = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[types.Content(role="user", parts=[types.Part(text=prompt)])],
        config=types.GenerateContentConfig(
            temperature=0.7,
            max_output_tokens=max_tokens,
        ),
    )
    return response.text


def call_gemini(system_prompt: str, context: str,
                user_msg: str, history: list[dict]) -> str:
    """多輪對話呼叫 Gemini"""
    full_user_msg = f"{system_prompt}\n\n{context}\n\n用戶問題：{user_msg}"
    contents: list[types.Content] = []
    for turn in history[-5:]:
        role    = turn.get("role", "user")
        content = turn.get("content", "").strip()
        if role not in ("user", "model") or not content:
            continue
        contents.append(types.Content(role=role, parts=[types.Part(text=content)]))
    contents.append(types.Content(role="user", parts=[types.Part(text=full_user_msg)]))
    response = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=contents,
        config=types.GenerateContentConfig(temperature=0.7, max_output_tokens=2048),
    )
    return response.text


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
        "gemini_model": GEMINI_MODEL,
        "api_key_set": bool(GEMINI_API_KEY),
    })


@app.route("/menu/<menu_name>")
def menu(menu_name: str):
    buttons = MENUS.get(menu_name, MENUS["main"])
    items = [
        {"label": btn, "action": "nav_main" if btn == "返回主選單" else "send"}
        for btn in buttons
    ]
    return jsonify({"menu": menu_name, "buttons": items})


@app.route("/chat", methods=["POST"])
def chat():
    data     = request.get_json(silent=True) or {}
    user_msg = data.get("message", "").strip()
    history  = data.get("history", [])

    if not user_msg:
        return jsonify({"error": "message 不能為空"}), 400

    context = rag.get_context(user_msg)

    t0 = time.time()
    try:
        reply = call_gemini(SYSTEM_PROMPT, context, user_msg, history)
    except Exception as e:
        return jsonify({"error": f"Gemini API 錯誤：{e}"}), 500
    elapsed = round(time.time() - t0, 2)

    return jsonify({"reply": reply, "menu": detect_menu(user_msg), "elapsed": elapsed})


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

    if not all([surname, name, birth_date, question]):
        return jsonify({"error": "姓名、出生日期、問題為必填"}), 400

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

    # ── 2. RAG 搜尋（生肖 + 問題關鍵字）──
    rag_query = f"屬{bazi['shengxiao']} {question}"
    context   = rag.get_context(rag_query, top_k=2)

    # ── 3. 組合分析 Prompt ──
    hour_pillar_str = f"、時柱{bazi['hour_pillar']}" if bazi.get("hour_pillar") else ""

    prompt = f"""你是李丞責博士，現在為用戶提供2026丙午年個人運勢分析。

用戶資料：
- 姓名：{full_name}
- 性別：{gender}
- 生肖：屬{bazi['shengxiao']}
- 八字：{bazi['bazi_string']}（年柱{bazi['year_pillar']}、月柱{bazi['month_pillar']}、日柱{bazi['day_pillar']}{hour_pillar_str}）
- 農曆：{bazi['lunar_date']}
- 用戶問題：{question}

參考資料（來自李丞責2026全書）：
{context}

請根據以上資料，以李丞責博士第一人稱，用繁體中文書面語，
為{full_name}提供詳細的2026年運勢分析。

必須嚴格按以下格式輸出六個段落，每段80-100字（精簡有力），缺一不可：

SECTION_整體運勢
（整體氣場，本年吉凶星曜）

SECTION_財運
（正偏財走勢，投資建議）

SECTION_事業
（事業機遇，需把握時機）

SECTION_感情
（姻緣或夫妻關係）

SECTION_健康
（注意事項，養生建議）

SECTION_化解建議
（開運化煞方法，方位擺設）

嚴格要求：
1. 每段控制在80-100字，不可超過，必須精簡
2. 必須輸出全部六個SECTION，不可省略
3. 有實質分析，不可說「建議預約諮詢」
4. 末尾加：「（本內容以李丞責著作及玄學原理為依據，玄學僅供參考。）」"""

    # ── 4. 呼叫 Gemini ──
    t0 = time.time()
    try:
        raw_reply = call_gemini_raw(prompt, max_tokens=6000)
    except Exception as e:
        return jsonify({"error": f"Gemini API 錯誤：{e}"}), 500
    elapsed = round(time.time() - t0, 2)

    # ── 5. 解析分段 ──
    sections = _parse_fortune_sections(raw_reply)

    return jsonify({
        "name":         full_name,
        "shengxiao":    bazi["shengxiao"],
        "bazi":         bazi["bazi_string"],
        "year_pillar":  bazi["year_pillar"],
        "month_pillar": bazi["month_pillar"],
        "day_pillar":   bazi["day_pillar"],
        "hour_pillar":  bazi.get("hour_pillar"),
        "lunar":        bazi["lunar_date"],
        "wuxing":       bazi["wuxing"],
        "fortune": {
            "overall": sections["overall"],
            "wealth":  sections["wealth"],
            "career":  sections["career"],
            "love":    sections["love"],
            "health":  sections["health"],
        },
        "advice":      sections["advice"],
        "email_sent":  False,
        "elapsed":     elapsed,
    })


if __name__ == "__main__":
    print(f"System Prompt 載入（{len(SYSTEM_PROMPT)} 字）")
    print(f"知識庫：{rag.chunk_count} 塊")
    print(f"Gemini 模型：{GEMINI_MODEL}")
    print(f"API Key：{'已設定' if GEMINI_API_KEY else '未設定！'}")
    app.run(debug=True, host="0.0.0.0", port=5000)
