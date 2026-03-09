# DocsChat Implementation Plan and Process

> Detailed plan and implementation record for a RAG Chat service using LangChain + ChromaDB + Docker Compose

---

## 1. Project Overview

### Goals
- Upload various documents (TXT, PDF, Web URL) and store them as vectors in ChromaDB
- When users ask questions, search relevant document chunks (RAG) and have the LLM provide answers
- LLM is user-selectable (OpenAI, Anthropic, Google, Ollama)
- Deploy services with Docker Compose, provide demo UI with Streamlit

### Requirements Summary
| Requirement | Details |
|---------|------|
| Vector DB | ChromaDB (Docker HTTP server mode) |
| Document Support | TXT, PDF, Web URL |
| LLM | OpenAI GPT / Anthropic Claude / Google Gemini / Ollama (local) |
| Deployment | Docker Compose |
| Demo UI | Streamlit |

---

## 2. Architecture Design

```
┌─────────────────────────────────────────────────────────────┐
│                    Docker Compose Network                    │
│                                                             │
│  ┌─────────────────┐         ┌──────────────────────────┐  │
│  │  docschat-app   │         │   docschat-chromadb      │  │
│  │  (Streamlit)    │◄───────►│   (ChromaDB HTTP Server) │  │
│  │  Port: 8501     │         │   Port: 8000             │  │
│  └────────┬────────┘         └──────────────────────────┘  │
│           │                                                  │
│           │ (Optional)                                       │
│           ▼                                                  │
│  ┌─────────────────┐                                        │
│  │  docschat-ollama│                                        │
│  │  (Local LLM)    │                                        │
│  │  Port: 11434    │                                        │
│  └─────────────────┘                                        │
└─────────────────────────────────────────────────────────────┘
             │
             │ (External API calls)
             ▼
  OpenAI / Anthropic / Google API
```

### Data Flow

```
Document Ingestion Pipeline:
  [File/URL] → [DocumentLoader] → [TextSplitter] → [Embeddings] → [ChromaDB]

Query Pipeline:
  [User Question] → [Embeddings] → [ChromaDB Search] → [Relevant Chunks]
                 → [RAG Prompt] → [LLM] → [Streaming Answer]
```

---

## 3. Technology Stack

| Layer | Technology | Version |
|--------|------|------|
| **UI** | Streamlit | ≥1.31.0 |
| **RAG Framework** | LangChain | ≥0.3.0 |
| **Vector DB** | ChromaDB | ≥0.5.0 |
| **LangChain-Chroma** | langchain-chroma | ≥0.1.0 |
| **Embeddings** | HuggingFace sentence-transformers / OpenAI | - |
| **LLM** | OpenAI / Anthropic / Google / Ollama | - |
| **PDF Parsing** | pypdf | ≥4.0.0 |
| **Web Scraping** | beautifulsoup4, requests, lxml | - |
| **Container** | Docker Compose | ≥2.0 |

---

## 4. Project File Structure

```
DocsChat/
├── app.py                     # Streamlit main application
├── core/
│   ├── __init__.py
│   ├── document_loader.py     # TXT/PDF/Web document loader
│   ├── embeddings.py          # Embedding factory (HuggingFace/OpenAI)
│   ├── llm_factory.py         # LLM factory (4 providers)
│   ├── vector_store.py        # ChromaDB connection and management
│   └── rag_engine.py          # RAG pipeline (LCEL chain)
├── config/
│   ├── __init__.py
│   └── settings.py            # Environment variable-based settings
├── docs/
│   ├── plan.md                # This file: plan and implementation process
│   ├── vector_db.md           # Vector DB comparative analysis
│   ├── vector_db_docker.md    # Building Vector DB with Docker
│   ├── demo.md                # Streamlit/Gradio demo guide
│   └── service.md             # Full service setup guide
├── docker-compose.yml         # Service orchestration
├── Dockerfile                 # App container image
├── requirements.txt           # Python dependencies
├── .env.example               # Environment variable template
├── .gitignore
└── README.md
```

---

## 5. Core Module Design

### 5.1 `core/document_loader.py`

Module that integrates loaders for different document types and splits them into chunks.

- **TXT**: `TextLoader` (UTF-8 encoding)
- **PDF**: `PyPDFLoader` (pypdf-based, pure Python, no system dependencies)
- **Web URL**: `WebBaseLoader` (HTML parsing with requests + BeautifulSoup4)
- **Common**: `RecursiveCharacterTextSplitter` for chunk splitting

### 5.2 `core/embeddings.py`

Embedding factory pattern supporting two providers:

- **HuggingFace** (default): `sentence-transformers/all-MiniLM-L6-v2` - free, local execution
- **OpenAI**: `text-embedding-3-small` - paid, high performance

> HuggingFace models are cached in Docker volume (`huggingface_cache`) to avoid re-downloading on restart

### 5.3 `core/llm_factory.py`

Integrates 4 LLM providers using Factory pattern:

| Provider | Default Model | Streaming |
|--------|---------|---------|
| OpenAI | gpt-4o-mini | ✅ |
| Anthropic | claude-3-5-sonnet-20241022 | ✅ |
| Google | gemini-1.5-flash | ✅ |
| Ollama | llama3.2 | ✅ |

### 5.4 `core/vector_store.py`

Connects to ChromaDB in HTTP client mode:
- Docker environment: connects using service name via `CHROMA_HOST` environment variable
- Local environment: requires ChromaDB server running at `localhost:8000`

Includes collection initialization feature (delete all data and recreate).

### 5.5 `core/rag_engine.py`

RAG chain based on LangChain LCEL (LangChain Expression Language):

```
retriever → format_docs
                          ↘
                            prompt → LLM → StrOutputParser
                          ↗
RunnablePassthrough (question)
```

- `stream_query()`: Streaming response + source document return
- `query()`: Regular response (non-streaming)

### 5.6 `app.py` (Streamlit UI)

**Sidebar Layout:**
- LLM settings (provider, model, API Key)
- Embedding settings (provider, model)
- Search settings (Top-K, collection name)
- Document upload (file upload + web URL)
- Index/reset buttons

**Main Area:**
- Chat tab: streaming responses, source document display
- Info tab: usage instructions, supported format guide

**Streamlit Caching Strategy:**
- `@st.cache_resource`: Embedding models, ChromaDB connection (reused throughout app lifetime)
- LLM: Created per query based on sidebar settings (reflects API Key changes)

---

## 6. Docker Compose Configuration

### Service Configuration

| Service | Image | Port | Profile | Description |
|--------|--------|------|---------|------|
| `chromadb` | chromadb/chroma:latest | 8000 | (default) | Vector DB |
| `app` | (build) | 8501 | (default) | Streamlit UI |
| `ollama` | ollama/ollama:latest | 11434 | `ollama` | Local LLM |

### Volumes

| Volume | Purpose |
|------|------|
| `chroma_data` | ChromaDB persistent data |
| `huggingface_cache` | HuggingFace model cache |
| `ollama_models` | Ollama model storage |

---

## 7. Implementation Process Record

### Step 1: Document Structure Analysis (docs/ folder review)
Analyzed 4 MD files in existing docs folder to determine architecture design direction:
- `vector_db.md`: Vector DB comparison → Selected Chroma (optimal for development/prototyping)
- `service.md`: Service architecture, LLM Factory pattern, RAG chain configuration
- `demo.md`: Streamlit UI design, basic RAG engine structure
- `vector_db_docker.md`: Chroma Docker configuration

### Step 2: Project Structure Design
Designed modular structure based on recommended structure from docs:
- `core/` module for business logic separation
- `config/` module for environment variable management separation
- Separation of Streamlit app and business logic

### Step 3: Core Module Implementation
1. `config/settings.py` - Environment variable-based settings (os.getenv)
2. `core/document_loader.py` - Multi-format document loader
3. `core/embeddings.py` - Embedding factory
4. `core/llm_factory.py` - LLM factory
5. `core/vector_store.py` - Chroma HTTP client connection
6. `core/rag_engine.py` - LCEL-based RAG pipeline

### Step 4: Streamlit UI Implementation
- Caching embeddings/vectorstore with `@st.cache_resource`
- LLM/embedding/search settings in sidebar
- Streaming responses displayed with `st.write_stream()`
- Source documents displayed in expanders

### Step 5: Docker Environment Configuration
- `Dockerfile`: python:3.11-slim + PyTorch CPU (size optimization)
- `docker-compose.yml`: Chroma + App + Ollama (optional)
- HuggingFace model cache volume for performance optimization on restart

### Step 6: Documentation
- `docs/plan.md` (this file): Full plan and implementation process
- `README.md`: Complete documentation for installation, running, and usage
- `.gitignore`: Exclude sensitive information (.env), cache, temporary files

---

## 7-1. Issues Found and Resolved During Build & Deployment

### Issue 1: ChromaDB Health Check API Path Change (v1 → v2)
- **Symptom**: docker-compose.yml health check `GET /api/v1/heartbeat` failed → chromadb `unhealthy`
- **Cause**: ChromaDB 1.0.0+ changed API path from `/api/v1/` to `/api/v2/`
- **Solution**: Changed health check to bash `/dev/tcp` method (handles containers without curl/python3)
  ```yaml
  test: ["CMD", "bash", "-c", "echo > /dev/tcp/localhost/8000"]
  ```

### Issue 2: Streamlit Port Conflict (8501)
- **Symptom**: `Bind for 0.0.0.0:8501 failed: port is already allocated`
- **Cause**: Another Streamlit service on the same host occupying 8501
- **Solution**: Changed host port to 8502 (container internal port stays 8501)
  ```yaml
  ports:
    - "8502:8501"   # host:container
  ```

### Issue 3: TRANSFORMERS_CACHE Environment Variable Deprecated Warning
- **Symptom**: `FutureWarning: Using TRANSFORMERS_CACHE is deprecated, use HF_HOME instead`
- **Solution**: Removed `TRANSFORMERS_CACHE` from docker-compose.yml, use only `HF_HOME`

---

## 8. Key Design Decisions

### Choosing ChromaDB HTTP Mode
In Docker Compose environment, the app and ChromaDB are separated as distinct services, so HTTP client mode is used. Even for local development, you can use the same interface by running only ChromaDB with `docker compose up chromadb -d`.

### Choosing HuggingFace Embeddings as Default
- Usable immediately without an API Key → lowers entry barrier
- `all-MiniLM-L6-v2`: lightweight (~90MB) + fast inference + multilingual support
- Docker volume model cache → no re-download on restart

### Choosing PyTorch CPU Version
Installing CPU-only version with `torch --index-url https://download.pytorch.org/whl/cpu`:
- Saves ~700MB compared to CUDA version
- CPU provides sufficient performance for embedding inference

### Choosing LangChain LCEL Chain
Using LCEL chain instead of `RetrievalQA`:
- More natural streaming support
- Can return source documents separately by separating retrieval and generation steps
- Latest LangChain pattern (RetrievalQA is being deprecated)

---

## 9. Known Limitations

1. **Embedding Consistency**: The embedding model used during indexing must match the model used during retrieval. Collection initialization required when changing embedding settings in UI.

2. **Multi-user**: ChromaDB collection is shared. Multiple users using the same demo reference the same document pool. Production needs per-user collection separation.

3. **Web URL Limitations**: SPAs requiring JavaScript rendering cannot be collected with `WebBaseLoader`. Only static HTML pages are supported.

4. **Image Size**: Docker image is ~1.5GB due to PyTorch + sentence-transformers. Can be reduced to ~300MB by switching to OpenAI embeddings if needed.

---

## 10. Future Plans

- [ ] Per-document metadata filtering (date, source, etc.)
- [ ] Query reformulation based on conversation history (ConversationalRetrievalChain)
- [ ] Add DOCX, MD file support
- [ ] Hybrid search (vector + BM25 keyword)
- [ ] Per-user collection separation (multi-tenancy)
- [ ] Add Gradio UI version

---

## 11. References

- [LangChain LCEL Official Docs](https://python.langchain.com/docs/concepts/lcel)
- [LangChain RAG Tutorial](https://python.langchain.com/docs/tutorials/rag)
- [ChromaDB Official Docs](https://docs.trychroma.com)
- [langchain-chroma GitHub](https://github.com/langchain-ai/langchain-chroma)
- [Streamlit Chat Elements](https://docs.streamlit.io/develop/api-reference/chat)
- [sentence-transformers Model Hub](https://huggingface.co/sentence-transformers)
