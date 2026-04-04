# AI Document Processing Microservice

Production-style **FastAPI** backend that ingests **PDF** and **TXT** files, chunks text (~500 tokens with ~50 overlap), creates **OpenAI** embeddings (`text-embedding-3-small`), stores them in a local **FAISS** index with JSON metadata, and exposes **semantic search** plus optional **RAG** (`/ask`).

## Requirements

- Python **3.11+** (3.12 recommended; tested on 3.14)
- An [OpenAI API key](https://platform.openai.com/api-keys)
- **Node.js 20+** and npm (for the optional DocuMind web UI in `frontend/`)

## Setup

```bash
cd AI_docs_processing_microservices
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env and set OPENAI_API_KEY
```

Optional environment variables (see `app/config.py`):

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI key | _(required for upload/query/ask)_ |
| `VECTOR_STORE_DIR` | FAISS + metadata directory | `./data/vector_store` |
| `CHUNK_SIZE_TOKENS` | Chunk size | `500` |
| `CHUNK_OVERLAP_TOKENS` | Overlap | `50` |
| `QUERY_TOP_K` | Chunks returned for `/query` and `/ask` context | `5` |
| `EMBEDDING_MODEL` | Embeddings model | `text-embedding-3-small` |
| `CHAT_MODEL` | Chat model for RAG | `gpt-4o-mini` |

## Run

```bash
source .venv/bin/activate
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

- API docs: [http://localhost:8000/docs](http://localhost:8000/docs)
- Health: `GET /health`

CORS is enabled for the Vite dev origin (`http://127.0.0.1:5173` and `http://localhost:5173`).

### DocuMind web UI (optional)

In a second terminal:

```bash
cd frontend
cp .env.example .env   # optional; defaults to http://127.0.0.1:8000
npm install
npm run dev
```

Open [http://localhost:5173](http://localhost:5173). Upload a PDF or TXT file; the app calls `/upload`, then `/ask` for an automatic summary, and uses `/ask` for chat and quick actions. The **×** control calls `DELETE /documents/{document_id}` to remove that document’s vectors from the index.

## API overview

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/upload` | Upload PDF/TXT, chunk, embed, append to FAISS |
| `POST` | `/query` | Semantic search (top‑K chunks + scores) |
| `POST` | `/ask` | RAG answer using retrieved chunks |
| `DELETE` | `/documents/{document_id}` | Remove all chunks for one upload from FAISS |
| `GET` | `/health` | Liveness |

Each upload gets a unique `document_id`. The FAISS index and `metadata.json` are **persisted** under `VECTOR_STORE_DIR` after each upload and on shutdown.

## Usage examples (`curl`)

Replace `sample.pdf` / `notes.txt` with your files.

**1. Upload a document**

```bash
curl -s -X POST "http://localhost:8000/upload" \
  -F "file=@sample.pdf"
```

Example response:

```json
{
  "document_id": "b3c1f2a0-....",
  "filename": "sample.pdf",
  "chunks_created": 12,
  "message": "Document processed successfully."
}
```

**2. Semantic search**

```bash
curl -s -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the payment terms?"}'
```

**3. RAG question**

```bash
curl -s -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"query": "Summarize the main obligations in one paragraph."}'
```

**4. Remove a document**

```bash
curl -s -X DELETE "http://localhost:8000/documents/<document_id>"
```

**5. Health**

```bash
curl -s "http://localhost:8000/health"
```

## Project layout

```
main.py                 # FastAPI app + lifespan (CORS, wiring)
frontend/               # Vite + React DocuMind UI
app/
  api/routes.py         # /upload, /query, /ask, /documents, /health
  models/schemas.py     # Pydantic request/response models
  config.py             # Settings from environment
  services/
    document_processor.py   # PDF/TXT extract + token chunking
    embedding_service.py    # OpenAI embeddings + query cache
    vector_store.py         # FAISS IndexFlatIP + JSON persistence
    rag_service.py          # Retrieval + chat completion
```

## Behavior notes

- **Similarity**: Embeddings are L2‑normalized; FAISS `IndexFlatIP` scores correspond to **cosine similarity** (higher is better).
- **Errors**: Invalid file types, empty PDFs, and API failures return **4xx/502** with a clear `detail` message. Missing `OPENAI_API_KEY` yields **503** on `/upload`, `/query`, and `/ask`.
- **Logging**: INFO logs cover extraction, chunk counts, index size, and RAG completion.

## Non-goals (by design)

No authentication or deployment manifests—the API and local persistence remain the core; the UI is a thin client for local use.
# AI-Document-Processing-Model
