# RecruitAI

> Staffing assistant for associations — analyzes CVs and groups candidates by domain using semantic RAG.

## What it does

RecruitAI reads CVs (PDF/DOCX) from the `cvs/` folder, extracts candidate information using GPT-4o-mini, embeds them into a vector space with `all-MiniLM-L6-v2`, stores them in Qdrant, and retrieves the best-matching candidates per domain using cosine similarity.

## Tech stack

| Layer | Technology |
|---|---|
| Backend | FastAPI + Uvicorn |
| Frontend | Vanilla JS / HTML / CSS |
| LLM extraction | GPT-4o-mini via OpenRouter |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector DB | Qdrant Cloud |
| CV parsing | pdfplumber + python-docx |

## Pipeline

```
CVs (PDF/DOCX)
   └─► pdfplumber / python-docx          (text extraction)
       └─► GPT-4o-mini (OpenRouter)      (structured info: name, skills, summary)
           └─► all-MiniLM-L6-v2          (384-dim embeddings)
               └─► Qdrant Cloud          (vector store + cosine similarity search)
                   └─► FastAPI           (REST API)
                       └─► Web UI        (results by domain)
```

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

Create a `.env` file at the project root:

```env
OPENROUTER_API_KEY=your_openrouter_key
QDRANT_HOST=https://your-cluster.qdrant.io:6333
QDRANT_API_KEY=your_qdrant_api_key
ENABLE_SEMANTIC_RAG=1
```

- **OpenRouter key**: [openrouter.ai](https://openrouter.ai) — used for GPT-4o-mini CV extraction
- **Qdrant**: [cloud.qdrant.io](https://cloud.qdrant.io) — create a free cluster

### 3. Add CVs

Drop PDF or DOCX files into the `cvs/` folder.

### 4. Run

```bash
python -m uvicorn server:app --host 0.0.0.0 --port 7000
```

Open **http://localhost:7000** in your browser.

> Avoid port 6000 (blocked by Chromium) and 8080/8082 (commonly used by proxies).

## Usage

1. **Select domains** — choose which association domains to staff (Education, Numérique, Social, etc.)
2. **Set group size** — how many candidates per domain
3. **Analyse** — the system embeds all CVs, queries Qdrant per domain, and returns ranked groups with AI-generated explanations

## Fallback behavior

| Condition | Behavior |
|---|---|
| OpenRouter rate-limited / unavailable | Regex fallback extracts name, email, phone from CV text |
| Qdrant unavailable | In-memory cosine similarity |
| Embedding model unavailable | Keyword frequency matching |

## Project structure

```
├── server.py          # FastAPI entry point
├── core/
│   ├── analyzer.py    # LLM-based CV info extraction (OpenRouter / regex fallback)
│   ├── parser.py      # PDF and DOCX text extraction
│   ├── rag.py         # Semantic grouping — Qdrant / in-memory / keyword fallback
│   └── vectorize.py   # Embedding model + Qdrant client (singleton pattern)
├── web/
│   ├── index.html     # Single-page app
│   ├── app.js         # Frontend logic
│   └── styles.css     # Dark glassmorphism UI
├── cvs/               # Drop your CV files here (PDF or DOCX)
├── requirements.txt
├── Makefile
├── .env.example
└── .gitignore
```
