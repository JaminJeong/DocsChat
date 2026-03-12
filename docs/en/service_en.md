# How to Build a RAG Chat Service

> A production RAG service guide using LangChain + Vector DB + Docker Compose

---

## 1. Overall Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     DocsChat Service                         │
│                                                             │
│  ┌───────────┐    ┌──────────────┐    ┌─────────────────┐  │
│  │   Demo    │    │  RAG Service │    │   Vector DB     │  │
│  │ (Streamlit│───▶│ (LangChain)  │───▶│ (Chroma/Qdrant/ │  │
│  │ /Gradio)  │    │              │    │  Weaviate)      │  │
│  └───────────┘    └──────┬───────┘    └─────────────────┘  │
│                          │                                  │
│                    ┌─────▼──────┐                           │
│                    │    LLM     │                           │
│                    │ (OpenAI/   │                           │
│                    │  Ollama/   │                           │
│                    │  Claude)   │                           │
│                    └────────────┘                           │
└─────────────────────────────────────────────────────────────┘
```

### Key Components

| Component | Role | Technology Options |
|---------|------|------------|
| **Document Loader** | Parse documents in various formats | LangChain Document Loaders |
| **Text Splitter** | Split into chunks | RecursiveCharacterTextSplitter |
| **Embeddings** | Text → Vector conversion | OpenAI, HuggingFace, Ollama |
| **Vector Store** | Store and search embeddings | Chroma, Qdrant, Weaviate |
| **Retriever** | Search similar documents | Similarity, MMR, Hybrid |
| **LLM** | Generate answers | GPT-4o, Claude, Gemini, Llama |
| **Frontend** | User interface | Streamlit, Gradio |

---

## 2. Document Ingestion Pipeline

### 2.1 Supported Document Formats

```python
from langchain_community.document_loaders import (
    PyPDFLoader,          # PDF
    TextLoader,           # TXT
    UnstructuredWordDocumentLoader,  # DOCX
    UnstructuredMarkdownLoader,      # MD
    UnstructuredHTMLLoader,          # HTML
    CSVLoader,            # CSV
    JSONLoader,           # JSON
)
```

### 2.2 Text Splitting Strategies

```python
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,  # Generally recommended
    CharacterTextSplitter,
    TokenTextSplitter,               # Token-based
    MarkdownHeaderTextSplitter,      # Preserve Markdown structure
)

# Recommended settings
splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,        # Chunk size (tokens)
    chunk_overlap=50,      # Overlap (preserve context)
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
)
```

### 2.3 Embedding Model Selection

```python
# OpenAI (paid, high performance)
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# HuggingFace (free, local)
from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

# Ollama (free, fully local)
from langchain_ollama import OllamaEmbeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text")
```

---

## 3. LLM Selection and Integration

### 3.1 Supported LLM List

| LLM | Provider | Cost | Features |
|-----|--------|------|------|
| GPT-4o | OpenAI | Paid | High performance, multimodal |
| GPT-4o-mini | OpenAI | Paid (affordable) | Fast, low cost |
| Claude 3.5 Sonnet | Anthropic | Paid | Long context, accuracy |
| Gemini 1.5 Pro | Google | Paid | Very long context |
| Llama 3.2 | Meta (Ollama) | Free | Fully local |
| Mistral | Mistral AI | Paid/Free | Lightweight |
| Qwen2 | Alibaba (Ollama) | Free | Excellent for multilingual |

### 3.2 LLM Factory Pattern

```python
from abc import ABC, abstractmethod
from langchain_core.language_models import BaseChatModel

class LLMFactory:
    @staticmethod
    def create(provider: str, model: str = None, api_key: str = None) -> BaseChatModel:
        match provider:
            case "openai":
                from langchain_openai import ChatOpenAI
                return ChatOpenAI(model=model or "gpt-4o", api_key=api_key)
            case "anthropic":
                from langchain_anthropic import ChatAnthropic
                return ChatAnthropic(model=model or "claude-3-5-sonnet-20241022", api_key=api_key)
            case "google":
                from langchain_google_genai import ChatGoogleGenerativeAI
                return ChatGoogleGenerativeAI(model=model or "gemini-1.5-pro", api_key=api_key)
            case "ollama":
                from langchain_ollama import ChatOllama
                return ChatOllama(model=model or "llama3.2", base_url="http://ollama:11434")
            case _:
                raise ValueError(f"Unsupported LLM provider: {provider}")
```

---

## 4. Vector Store Integration

### 4.1 Vector Store Factory Pattern

```python
from langchain_core.vectorstores import VectorStore
from langchain_core.embeddings import Embeddings

class VectorStoreFactory:
    @staticmethod
    def create(db_type: str, embeddings: Embeddings, **kwargs) -> VectorStore:
        match db_type:
            case "chroma":
                from langchain_chroma import Chroma
                return Chroma(
                    collection_name=kwargs.get("collection", "documents"),
                    embedding_function=embeddings,
                    persist_directory=kwargs.get("persist_dir", "./chroma_db")
                )
            case "qdrant":
                from langchain_qdrant import QdrantVectorStore
                from qdrant_client import QdrantClient
                client = QdrantClient(url=kwargs.get("url", "http://localhost:6333"))
                return QdrantVectorStore(
                    client=client,
                    collection_name=kwargs.get("collection", "documents"),
                    embedding=embeddings
                )
            case "weaviate":
                import weaviate
                from langchain_weaviate.vectorstores import WeaviateVectorStore
                client = weaviate.connect_to_local(
                    host=kwargs.get("host", "localhost"),
                    port=kwargs.get("port", 8080)
                )
                return WeaviateVectorStore(
                    client=client,
                    index_name=kwargs.get("index", "Document"),
                    text_key="content",
                    embedding=embeddings
                )
            case _:
                raise ValueError(f"Unsupported Vector DB: {db_type}")
```

---

## 5. RAG Chain Construction

### 5.1 Basic RAG Chain (LangChain LCEL)

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

def create_rag_chain(llm, vectorstore, search_type="similarity", k=5):
    retriever = vectorstore.as_retriever(
        search_type=search_type,  # "similarity", "mmr", "similarity_score_threshold"
        search_kwargs={"k": k}
    )

    prompt = ChatPromptTemplate.from_template("""
You are an AI assistant that provides accurate and helpful answers based on the given documents.
Use the context information below to answer the question.
If the answer is not in the context, honestly say you don't know.

Context:
{context}

Question: {question}

Answer:""")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever
```

### 5.2 RAG with Conversation History

```python
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

def create_conversational_rag(llm, vectorstore):
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
        memory=memory,
        return_source_documents=True,
        verbose=False
    )
    return chain
```

---

## 6. Full Docker Compose Service Configuration

### 6.1 docker-compose.yml

```yaml
version: '3.8'

services:
  # ─────────────── Frontend ───────────────
  app:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: docschat-app
    ports:
      - "8501:8501"
    environment:
      - LLM_PROVIDER=${LLM_PROVIDER:-openai}
      - LLM_MODEL=${LLM_MODEL:-gpt-4o}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - GOOGLE_API_KEY=${GOOGLE_API_KEY}
      - VECTOR_DB_TYPE=${VECTOR_DB_TYPE:-chroma}
      - CHROMA_HOST=chromadb
      - CHROMA_PORT=8000
      - QDRANT_HOST=qdrant
      - QDRANT_PORT=6333
      - WEAVIATE_HOST=weaviate
      - WEAVIATE_PORT=8080
      - OLLAMA_HOST=ollama
    volumes:
      - ./data:/app/data
      - ./uploads:/app/uploads
    depends_on:
      - chromadb
      - qdrant
    restart: unless-stopped
    networks:
      - docschat-network

  # ─────────────── Vector DB ───────────────
  chromadb:
    image: chromadb/chroma:latest
    container_name: docschat-chromadb
    volumes:
      - chroma_data:/chroma/chroma
    ports:
      - "8000:8000"
    environment:
      - ALLOW_RESET=TRUE
    restart: unless-stopped
    networks:
      - docschat-network

  qdrant:
    image: qdrant/qdrant:latest
    container_name: docschat-qdrant
    volumes:
      - qdrant_storage:/qdrant/storage
    ports:
      - "6333:6333"
      - "6334:6334"
    restart: unless-stopped
    networks:
      - docschat-network
    profiles: ["qdrant"]

  weaviate:
    image: cr.weaviate.io/semitechnologies/weaviate:1.24.0
    container_name: docschat-weaviate
    volumes:
      - weaviate_data:/var/lib/weaviate
    ports:
      - "8080:8080"
    environment:
      AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED: 'true'
      PERSISTENCE_DATA_PATH: '/var/lib/weaviate'
      DEFAULT_VECTORIZER_MODULE: 'none'
    restart: unless-stopped
    networks:
      - docschat-network
    profiles: ["weaviate"]

  # ─────────────── Local LLM (Optional) ───────────────
  ollama:
    image: ollama/ollama:latest
    container_name: docschat-ollama
    volumes:
      - ollama_models:/root/.ollama
    ports:
      - "11434:11434"
    restart: unless-stopped
    networks:
      - docschat-network
    profiles: ["ollama"]
    # Uncomment below to enable GPU:
    # deploy:
    #   resources:
    #     reservations:
    #       devices:
    #         - driver: nvidia
    #           count: 1
    #           capabilities: [gpu]

networks:
  docschat-network:
    driver: bridge

volumes:
  chroma_data:
  qdrant_storage:
  weaviate_data:
  ollama_models:
```

### 6.2 .env File

```env
# LLM settings
LLM_PROVIDER=openai          # openai | anthropic | google | ollama
LLM_MODEL=gpt-4o             # Model name to use
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AIza...

# Vector DB settings
VECTOR_DB_TYPE=chroma        # chroma | qdrant | weaviate

# Embedding settings
EMBEDDING_PROVIDER=openai    # openai | huggingface | ollama
EMBEDDING_MODEL=text-embedding-3-small
```

### 6.3 Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
```

### 6.4 requirements.txt

```txt
# Frontend
streamlit>=1.31.0

# LangChain core
langchain>=0.3.0
langchain-core>=0.3.0
langchain-community>=0.3.0

# LLM providers
langchain-openai>=0.2.0
langchain-anthropic>=0.2.0
langchain-google-genai>=2.0.0
langchain-ollama>=0.2.0

# Embeddings
langchain-huggingface>=0.1.0
sentence-transformers>=3.0.0

# Vector DB clients
langchain-chroma>=0.1.0
qdrant-client>=1.9.0
langchain-qdrant>=0.1.0
weaviate-client>=4.6.0
langchain-weaviate>=0.0.3

# Document processing
pypdf>=4.0.0
python-docx>=1.0.0
unstructured>=0.14.0

# Utilities
python-dotenv>=1.0.0
```

---

## 7. Running the Service

```bash
# Basic run (Chroma + OpenAI)
cp .env.example .env
# Set API keys in .env file
docker compose up -d

# Run with Qdrant
docker compose --profile qdrant up -d

# Run with Ollama (local LLM)
docker compose --profile ollama up -d
docker exec -it docschat-ollama ollama pull llama3.2

# Run all services
docker compose --profile qdrant --profile ollama up -d

# View logs
docker compose logs -f app
```

---

## 8. Recommended Project Structure

```
DocsChat/
├── app.py                    # Streamlit main
├── gradio_app.py             # Gradio main (optional)
├── core/
│   ├── rag_engine.py         # RAG pipeline
│   ├── llm_factory.py        # LLM factory
│   ├── vector_store.py       # Vector Store factory
│   ├── document_loader.py    # Document loader
│   └── embeddings.py         # Embedding settings
├── config/
│   └── settings.py           # Environment variable-based settings
├── data/                     # Persistent data
├── uploads/                  # Temporary upload files
├── docs/                     # Project documentation
│   ├── vector_db.md
│   ├── vector_db_docker.md
│   ├── demo.md
│   └── service.md
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── .env.example
```

---

## 9. References

- [LangChain Official Docs](https://python.langchain.com/docs)
- [LangChain RAG Tutorial](https://python.langchain.com/docs/tutorials/rag)
- [LangChain LCEL](https://python.langchain.com/docs/concepts/lcel)
- [Ollama Official Site](https://ollama.ai)
- [Docker Compose Docs](https://docs.docker.com/compose)
