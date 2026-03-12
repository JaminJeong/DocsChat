# RAG Chat 서비스 구축 방법

> LangChain + Vector DB + Docker Compose를 활용한 프로덕션 RAG 서비스 가이드

---

## 1. 전체 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                     DocsChat 서비스                          │
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

### 주요 컴포넌트

| 컴포넌트 | 역할 | 기술 선택지 |
|---------|------|------------|
| **문서 로더** | 다양한 포맷 문서 파싱 | LangChain Document Loaders |
| **텍스트 분할** | 청크 단위 분割 | RecursiveCharacterTextSplitter |
| **임베딩** | 텍스트 → 벡터 변환 | OpenAI, HuggingFace, Ollama |
| **벡터 저장소** | 임베딩 저장 및 검색 | Chroma, Qdrant, Weaviate |
| **리트리버** | 유사 문서 검색 | Similarity, MMR, Hybrid |
| **LLM** | 답변 생성 | GPT-4o, Claude, Gemini, Llama |
| **프론트엔드** | 사용자 인터페이스 | Streamlit, Gradio |

---

## 2. 문서 인제스트 파이프라인

### 2.1 지원 문서 포맷

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

### 2.2 텍스트 분할 전략

```python
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,  # 일반적으로 권장
    CharacterTextSplitter,
    TokenTextSplitter,               # 토큰 기반
    MarkdownHeaderTextSplitter,      # Markdown 구조 유지
)

# 권장 설정
splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,        # 청크 크기 (토큰)
    chunk_overlap=50,      # 오버랩 (문맥 유지)
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
)
```

### 2.3 임베딩 모델 선택

```python
# OpenAI (유료, 고성능)
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# HuggingFace (무료, 로컬)
from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

# Ollama (무료, 완전 로컬)
from langchain_ollama import OllamaEmbeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text")
```

---

## 3. LLM 선택 및 연동

### 3.1 지원 LLM 목록

| LLM | 제공자 | 비용 | 특징 |
|-----|--------|------|------|
| GPT-4o | OpenAI | 유료 | 고성능, 멀티모달 |
| GPT-4o-mini | OpenAI | 유료(저렴) | 빠름, 저비용 |
| Claude 3.5 Sonnet | Anthropic | 유료 | 긴 컨텍스트, 정확성 |
| Gemini 1.5 Pro | Google | 유료 | 매우 긴 컨텍스트 |
| Llama 3.2 | Meta (Ollama) | 무료 | 완전 로컬 |
| Mistral | Mistral AI | 유료/무료 | 경량화 |
| Qwen2 | Alibaba (Ollama) | 무료 | 한국어 우수 |

### 3.2 LLM Factory 패턴

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
                raise ValueError(f"지원하지 않는 LLM 제공자: {provider}")
```

---

## 4. Vector Store 연동

### 4.1 Vector Store Factory 패턴

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
                raise ValueError(f"지원하지 않는 Vector DB: {db_type}")
```

---

## 5. RAG 체인 구성

### 5.1 기본 RAG 체인 (LangChain LCEL)

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
당신은 주어진 문서를 기반으로 정확하고 유용한 답변을 제공하는 AI 어시스턴트입니다.
아래의 컨텍스트 정보를 활용하여 질문에 답변하세요.
컨텍스트에 없는 내용은 솔직하게 모른다고 말하세요.

컨텍스트:
{context}

질문: {question}

답변:""")
    
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

### 5.2 대화 히스토리 지원 RAG

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

## 6. Docker Compose 전체 서비스 구성

### 6.1 docker-compose.yml

```yaml
version: '3.8'

services:
  # ─────────────── 프론트엔드 ───────────────
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

  # ─────────────── 로컬 LLM (선택적) ───────────────
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
    # GPU 사용 시 아래 주석 해제:
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

### 6.2 .env 파일

```env
# LLM 설정
LLM_PROVIDER=openai          # openai | anthropic | google | ollama
LLM_MODEL=gpt-4o             # 사용할 모델명
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AIza...

# Vector DB 설정
VECTOR_DB_TYPE=chroma        # chroma | qdrant | weaviate

# 임베딩 설정
EMBEDDING_PROVIDER=openai    # openai | huggingface | ollama
EMBEDDING_MODEL=text-embedding-3-small
```

### 6.3 Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# 시스템 의존성 설치
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
# 프론트엔드
streamlit>=1.31.0

# LangChain 코어
langchain>=0.3.0
langchain-core>=0.3.0
langchain-community>=0.3.0

# LLM 제공자
langchain-openai>=0.2.0
langchain-anthropic>=0.2.0
langchain-google-genai>=2.0.0
langchain-ollama>=0.2.0

# 임베딩
langchain-huggingface>=0.1.0
sentence-transformers>=3.0.0

# Vector DB 클라이언트
langchain-chroma>=0.1.0
qdrant-client>=1.9.0
langchain-qdrant>=0.1.0
weaviate-client>=4.6.0
langchain-weaviate>=0.0.3

# 문서 처리
pypdf>=4.0.0
python-docx>=1.0.0
unstructured>=0.14.0

# 유틸리티
python-dotenv>=1.0.0
```

---

## 7. 서비스 실행 방법

```bash
# 기본 실행 (Chroma + OpenAI)
cp .env.example .env
# .env 파일에 API 키 설정 후
docker compose up -d

# Qdrant 포함 실행
docker compose --profile qdrant up -d

# Ollama(로컬 LLM) 포함 실행
docker compose --profile ollama up -d
docker exec -it docschat-ollama ollama pull llama3.2

# 모든 서비스 실행
docker compose --profile qdrant --profile ollama up -d

# 로그 확인
docker compose logs -f app
```

---

## 8. 프로젝트 구조 (권장)

```
DocsChat/
├── app.py                    # Streamlit 메인
├── gradio_app.py             # Gradio 메인 (선택)
├── core/
│   ├── rag_engine.py         # RAG 파이프라인
│   ├── llm_factory.py        # LLM 팩토리
│   ├── vector_store.py       # Vector Store 팩토리
│   ├── document_loader.py    # 문서 로더
│   └── embeddings.py         # 임베딩 설정
├── config/
│   └── settings.py           # 환경변수 기반 설정
├── data/                     # 영구 데이터
├── uploads/                  # 업로드 임시 파일
├── docs/                     # 프로젝트 문서
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

## 9. 참고 자료

- [LangChain 공식 문서](https://python.langchain.com/docs)
- [LangChain RAG Tutorial](https://python.langchain.com/docs/tutorials/rag)
- [LangChain LCEL](https://python.langchain.com/docs/concepts/lcel)
- [Ollama 공식 사이트](https://ollama.ai)
- [Docker Compose 문서](https://docs.docker.com/compose)
