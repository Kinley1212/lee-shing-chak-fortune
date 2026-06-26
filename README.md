# 李丞責2026運勢AI（lczai）

基於RAG技術的AI運勢聊天機器人，根據李丞責2026年運程書籍回答問題。

## 快速開始

### 1. 安裝依賴

```bash
pip3 install -r requirements.txt
```

### 2. 設置API Key

```bash
cp .env.example .env
# 編輯 .env，填入你的 Gemini API Key
```

### 3. 放置PDF文件

將 `2026全書.pdf` 放到 `data/raw/` 目錄下。

### 4. 提取PDF文字

```bash
python3 scripts/extract_pdf.py
```

### 5. 建立知識庫

```bash
python3 scripts/build_knowledge.py
```

### 6. 啟動聊天機器人

```bash
cd app
python3 main.py
```

瀏覽器打開 http://localhost:5000

## 項目結構

```
chatbot/
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
│       └── index.html    前端介面
├── training/
│   └── qa_data.jsonl     Fine-tuning訓練數據
├── .env.example
├── requirements.txt
└── README.md
```

## 技術棧

- **後端**：Python / Flask
- **AI**：Google Gemini 1.5 Flash
- **知識庫**：JSON + 關鍵字匹配RAG
- **PDF提取**：PyMuPDF (fitz)
