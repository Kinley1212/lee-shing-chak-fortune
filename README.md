# 李丞責 2026 丙午年運勢 AI

一個從零到一、實際上線運行中的 AI 運勢問答網站。使用者輸入生辰資訊後，AI 會結合玄學知識庫給出個人化五行運勢分析，並可選擇留下 email 收取完整報告。

**🔗 Live Demo：** https://lczai-chatbot.onrender.com
（網站有進站密碼保護，非公開流量，如需試用密碼請與我聯繫）

![運勢查詢主頁](./docs/screenshot-form.png)
![運勢報告範例](./docs/screenshot-report-1.png)
![運勢報告細節](./docs/screenshot-report-2.png)

## 這個專案在做什麼

不是一個 demo 或練習作品，而是**真實客戶（命理師）委託開發並持續維運中**的產品，涵蓋從需求溝通、功能迭代到上線維護的完整流程：

- 使用者輸入生辰八字，AI 根據書籍內容 + RAG 知識庫給出個人化五行運勢解讀
- 可選填 email，自動寄送 HTML 格式完整運勢報告到信箱
- 進站密碼保護，避免未授權流量消耗免費資源
- 隨著客戶反饋持續打磨 UI（淺色主題、紅金配色、時間輸入方式優化等）

## 功能特色

- 🔮 **RAG 知識庫問答**：以命理書籍內容為知識庫，結合關鍵字檢索 + Gemini API 生成個人化回答
- 📧 **自動寄信**：使用者留 email 後，前端直接透過 EmailJS 寄送 HTML 運勢報告，無需後端 SMTP
- 🔒 **進站密碼保護**：全站登入驗證，未授權無法瀏覽頁面或呼叫 API
- 🎨 **持續迭代的 UI**：從初版逐步優化為淺色主題、紅金主色調、鍵盤輸入時間選擇器
- ☁️ **雲端部署**：透過 Render 自動化部署，GitHub push 後自動上線

## 技術棧

- **後端**：Python / Flask
- **AI**：Google Gemini 1.5 Flash
- **知識庫檢索**：JSON + 關鍵字匹配 RAG
- **PDF 資料提取**：PyMuPDF (fitz)
- **寄信**：EmailJS（瀏覽器端直接發信）
- **部署**：Render（GitHub 自動部署）

## 專案結構

```
lee-shing-chak-fortune/
├── data/
│   ├── raw/              原始PDF及提取文字
│   └── knowledge/        處理好的JSON知識庫
├── scripts/
│   ├── extract_pdf.py    PDF文字提取
│   ├── build_knowledge.py 知識庫建立
│   └── generate_qa.py    訓練數據生成
├── app/
│   ├── main.py           Flask主程式
│   ├── rag.py            RAG搜尋引擎
│   └── templates/
│       ├── index.html    聊天主介面
│       └── login.html    進站密碼頁
├── training/
│   └── qa_data.jsonl     Fine-tuning訓練數據
├── .env.example
├── requirements.txt
└── README.md
```

## 本機執行

```bash
pip3 install -r requirements.txt
cp .env.example .env
# 編輯 .env，填入 Gemini API Key、SITE_PASSWORD 等設定

python3 scripts/extract_pdf.py       # 提取PDF文字
python3 scripts/build_knowledge.py   # 建立知識庫

cd app
python3 main.py
```

瀏覽器打開 http://localhost:5000

## 作者

[@Kinley1212](https://github.com/Kinley1212) — 獨立開發、部署與維運此專案

---
玄學內容僅供參考，一切以個人判斷為準。
