import os
from typing import List
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

st.set_page_config(page_title="AI PDF Research Assistant", page_icon="🤖", layout="wide")
st.title("🤖 AI PDF Research Assistant")
st.caption("Ask questions about your uploaded documents with transparent retrieval.")

def extract_chunks(files):
    chunks, sources = [], []
    for file in files:
        reader = PdfReader(file)
        for page_no, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            words = text.split()
            for start in range(0, len(words), 180):
                chunk = " ".join(words[start:start + 180])
                if chunk:
                    chunks.append(chunk)
                    sources.append(f"{file.name} — page {page_no}")
    return chunks, sources

def retrieve(question, chunks, sources, k=4):
    if not chunks:
        return []
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(chunks + [question])
    scores = cosine_similarity(matrix[-1], matrix[:-1]).ravel()
    indices = scores.argsort()[::-1][:k]
    return [(chunks[i], sources[i], float(scores[i])) for i in indices if scores[i] > 0]

def local_answer(results):
    if not results:
        return "I could not find a relevant passage in the uploaded documents."
    return "Relevant information found in the documents:\n\n" + "\n\n".join(
        f"• {text}" for text, _, _ in results[:3]
    )

def azure_answer(question, results):
    if not results:
        return None
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    key = os.getenv("AZURE_OPENAI_API_KEY")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    if not all([endpoint, key, deployment]):
        return None
    try:
        from openai import AzureOpenAI
        client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=key,
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")
        )
        context = "\n\n".join(f"[{s}]\n{t}" for t, s, _ in results)
        response = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": "Answer only from the supplied context. If insufficient, say so."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
            ],
            temperature=0.2
        )
        return response.choices[0].message.content
    except Exception as exc:
        st.warning(f"Azure AI unavailable; using local retrieval mode. ({exc})")
        return None

files = st.file_uploader("Upload PDF documents", type=["pdf"], accept_multiple_files=True)

if files:
    chunks, sources = extract_chunks(files)
    st.success(f"Indexed {len(chunks)} chunks from {len(files)} document(s).")
    question = st.text_input("Ask a question about your documents")

    if question:
        results = retrieve(question, chunks, sources)
        answer = azure_answer(question, results) or local_answer(results)
        st.subheader("Answer")
        st.write(answer)

        st.subheader("Retrieved sources")
        for text, source, score in results:
            with st.expander(f"{source} • relevance {score:.2f}"):
                st.write(text)
else:
    st.info("Upload a PDF to begin.")
