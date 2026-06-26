#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
李丞責2026運勢AI — Flask 後端（google.genai 新SDK版）
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

from google import genai
from google.genai import types
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from rag import RAGEngine

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
    "main": [
        "紫微斗數流年",
        "奇門遁甲",
        "塔羅牌占卜",
        "卜卦（六爻）",
        "今日運勢",
        "關於李丞責博士",
    ],
    "ziwei": [
        "今年財運",
        "今年感情",
        "今年事業",
        "注意月份",
        "化解方法",
        "返回主選單",
    ],
    "qimen": [
        "問事業",
        "問財運",
        "問感情",
        "問搬遷",
        "如何增強運勢",
        "返回主選單",
    ],
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


def call_gemini(system_prompt: str, context: str,
                user_msg: str, history: list[dict]) -> str:
    """呼叫 Gemini API，帶入 system prompt、RAG context 及對話歷史"""

    # 把 system prompt + RAG context 合入 user 的首條訊息
    full_user_msg = (
        f"{system_prompt}\n\n"
        f"{context}\n\n"
        f"用戶問題：{user_msg}"
    )

    # 建立對話內容列表（新SDK格式）
    contents: list[types.Content] = []

    # 加入歷史（最近5輪，跳過空白）
    for turn in history[-5:]:
        role    = turn.get("role", "user")
        content = turn.get("content", "").strip()
        if role not in ("user", "model") or not content:
            continue
        contents.append(
            types.Content(role=role, parts=[types.Part(text=content)])
        )

    # 加入本次用戶訊息
    contents.append(
        types.Content(role="user", parts=[types.Part(text=full_user_msg)])
    )

    response = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            temperature=0.7,
            max_output_tokens=2048,
        ),
    )
    return response.text


# ── 路由 ──────────────────────────────────────────────

@app.route("/")
def index():
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

    return jsonify({
        "reply":   reply,
        "menu":    detect_menu(user_msg),
        "elapsed": elapsed,
    })


if __name__ == "__main__":
    print(f"System Prompt 載入（{len(SYSTEM_PROMPT)} 字）")
    print(f"知識庫：{rag.chunk_count} 塊")
    print(f"Gemini 模型：{GEMINI_MODEL}")
    print(f"API Key：{'已設定' if GEMINI_API_KEY else '未設定！'}")
    app.run(debug=True, host="0.0.0.0", port=5000)
