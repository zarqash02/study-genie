---
title: Study Genie
emoji: 📘
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: AI study assistant with RAG over your own PDFs
tags:
  - rag
  - study
  - notebooklm
  - pinecone
  - huggingface
  - fastapi
  - streamlit
---

# 📘 Study Genie

**Your Intelligent Learning Companion** — a NotebookLM-style study assistant that grounds every answer in your own uploaded materials.

Upload your PDFs and Study Genie will:
- Answer questions with citation-backed responses (RAG)
- Generate structured summaries
- Create quizzes and flashcards for active recall
- Keep you focused by refusing to answer outside your material

---

## 🏗 Architecture

```
┌────────────────┐    HTTP    ┌──────────────────┐
│ Streamlit UI   │ ─────────► │ FastAPI Backend  │
│ (port 7860)    │            │ (port 8000)      │
└────────────────┘            └────────┬─────────┘
                                       │
                     ┌─────────────────┼────────────────────┐
                     ▼                 ▼                    ▼
            ┌────────────────┐ ┌──────────────┐   ┌────────────────────┐
            │ sentence-      │ │ Pinecone     │   │ HuggingFace        │
            │ transformers   │ │ (vector DB)  │   │ Inference API      │
            │ (embeddings)   │ │              │   │ (generation)       │
            └────────────────┘ └──────────────┘   └────────────────────┘
```

**Flow:** PDF → PyMuPDF extraction → sliding-window chunks → MiniLM embeddings → Pinecone upsert → query retrieves top-k → Mistral-7B grounded answer with sources.

---

## 📁 Folder Structure

```
study-genie/
├── backend/
│   ├── __init__.py
│   ├── main.py                     # FastAPI app + endpoints
│   ├── config.py                   # env config
│   └── services/
│       ├── __init__.py
│       ├── pdf_service.py          # extract + chunk
│       ├── embedding_service.py    # sentence-transformers
│       ├── pinecone_service.py     # vector DB
│       └── llm_service.py          # HF Inference + prompts
├── frontend/
│   └── app.py                      # Streamlit UI
├── Dockerfile                      # HF Spaces (Docker SDK)
├── start.sh                        # runs FastAPI + Streamlit
├── requirements.txt
├── .env.example
├── .dockerignore
└── README.md
```

---

## 🔧 Local Setup

### 1. Clone and install

```bash
git clone https://github.com/zarqash02/study-genie study-genie
cd study-genie
python -m venv .venv
.venv\Scripts\activate          # MacOS: source .venv/bin/activate
pip install pymupdf --only-binary :all:
pip install numpy --only-binary :all:
pip install pydantic-core pillow tokenizers --prefer-binary
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 2. Get API keys

- **Pinecone** — create a free account at [pinecone.io](https://www.pinecone.io), grab your API key. The index is created automatically on first run.
- **HuggingFace** — create a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) with **read** scope.

### 3. Configure environment

```bash
cp .env.example .env
# edit .env and paste your keys
```

### 4. Run backend and frontend (two terminals)

Terminal 1 — FastAPI:
```bash
uvicorn backend.main:app --reload --port 8000
```

Terminal 2 — Streamlit:
```bash
streamlit run frontend/app.py
```

Open [http://localhost:8501](http://localhost:8501).

### Single-container run (like HF Spaces)

```bash
chmod +x start.sh
./start.sh
```
App on [http://localhost:7860](http://localhost:7860).

---

## 🐳 Docker

```bash
docker build -t study-genie .
docker run -p 7860:7860 --env-file .env study-genie
```

---

## ☁️ Deploy to Hugging Face Spaces

1. Create a new Space → **SDK: Docker**.
2. Push this repo to the Space.
3. In **Settings → Variables and secrets**, add:
   - `PINECONE_API_KEY`
   - `PINECONE_INDEX_NAME` (optional, default `study-genie`)
   - `PINECONE_CLOUD` (optional, default `aws`)
   - `PINECONE_REGION` (optional, default `us-east-1`)
   - `HF_TOKEN`
   - `HF_GENERATION_MODEL` (optional, default Mistral-7B-Instruct-v0.3)
4. The Space builds the Dockerfile and serves on port 7860.

The YAML front matter at the top of this README is the Space config — HF reads it automatically.

---

## 🔌 API Reference

Base URL: `http://localhost:8000`

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/health` | — | `{status}` |
| POST | `/upload` | multipart `file` (PDF) | `{doc_id, doc_name, chunks}` |
| POST | `/ask` | `{query, doc_ids?, top_k?}` | `{answer, sources[]}` |
| POST | `/summarize` | `{doc_ids?}` | `{summary}` |
| POST | `/quiz` | `{doc_ids?, n?}` | `{quiz}` |
| POST | `/flashcards` | `{doc_ids?, n?}` | `{flashcards}` |
| DELETE | `/document/{doc_id}` | — | `{deleted}` |

Full interactive docs at `/docs` (Swagger UI).

---

## 🧠 RAG Prompt

```
You are a helpful study assistant. Answer ONLY using the provided context.
If the answer is not in the context, say 'I don't know based on the provided material.'

Context:
{retrieved_chunks}

Question:
{user_query}

Answer:
```

---

## ⚙️ Configuration

| Variable | Default | Purpose |
|---|---|---|
| `CHUNK_SIZE` | 500 | words per chunk |
| `CHUNK_OVERLAP` | 50 | word overlap between chunks |
| `TOP_K` | 4 | retrieved chunks per query |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | 384-dim, fast, runs on CPU |
| `EMBEDDING_DIM` | 384 | must match embedding model |
| `HF_GENERATION_MODEL` | `Mistral-7B-Instruct-v0.3` | any HF Inference-API text-generation model |

To swap the embedding model (e.g. to `BAAI/bge-small-en-v1.5`), update **both** `EMBEDDING_MODEL` and `EMBEDDING_DIM`, and re-create your Pinecone index.

---

## 🚫 Constraints (by design)

- No audio features
- Answers are strictly grounded — no hallucinated content
- Refuses to answer outside uploaded material
- Minimal, distraction-free UI

---

## 💾 Memory Notes for HF Spaces Free Tier

- Embedding model runs locally (~90 MB) — fits in free-tier RAM.
- Generation is offloaded to HF Inference API — **no model download**.
- Pinecone stores all vectors — no disk usage.
- Use `--no-cache-dir` in Dockerfile keeps image slim.

---

## 📜 License

MIT