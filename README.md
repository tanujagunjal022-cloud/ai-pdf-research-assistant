# 🤖 AI PDF Research Assistant

A portfolio-ready RAG-style PDF question-answering application.

## Features
- Upload one or more PDFs
- Extract and chunk document text
- Retrieve relevant passages with TF-IDF
- Ask questions about documents
- Display source passages and relevance
- Optional Azure OpenAI integration
- Local fallback mode without an API key

## Tech Stack
Python • Streamlit • PyPDF • scikit-learn • Azure OpenAI (optional)

## Run
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

For Azure OpenAI mode, copy `.env.example` to `.env` and add your own values.

## Architecture
PDF → Extraction → Chunking → TF-IDF Retrieval → Context → AI Answer
