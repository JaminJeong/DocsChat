# 📚 DocsChat

> 문서 기반 RAG(Retrieval-Augmented Generation) 채팅 서비스
> LangChain + ChromaDB + Docker Compose + Streamlit

---

## 개요

DocsChat은 PDF, TXT, 웹 페이지 등 다양한 문서를 업로드하고, 해당 문서의 내용을 기반으로 AI와 대화할 수 있는 RAG 채팅 서비스입니다.

```
┌─────────────────────────────────────────────────────────────┐
│  사용자 질문                                                  │
│      │                                                       │
│      ▼                                                       │
│  [ChromaDB 검색] ──► [관련 문서 청크]                         │
│      │                      │                               │
│      └──────────────────────►                               │
│                             ▼                               │
│                        [RAG 프롬프트]                         │
│                             │                               │
│                             ▼                               │
│                    [LLM (GPT/Claude/Gemini/Ollama)]          │
│                             │                               │
│                             ▼                               │
│                        [스트리밍 답변]                        │
└─────────────────────────────────────────────────────────────┘
```

---

## 기능

- **다양한 문서 지원**: PDF, TXT, 웹 URL
- **LLM 선택**: OpenAI GPT / Anthropic Claude / Google Gemini / Ollama(로컬)
- **임베딩 선택**: HuggingFace(무료/로컬) / OpenAI(유료)
- **벡터 DB**: ChromaDB (Docker HTTP 서버 모드, 데이터 영구 보존)
- **스트리밍 응답**: 실시간 답변 생성
- **소스 표시**: 답변 근거가 된 문서 청크 표시
- **Docker Compose**: 원클릭 배포

---

## 빠른 시작

### 1. 저장소 복제

```bash
git clone https://github.com/DocsChat.git
cd DocsChat
```

### 2. 환경변수 설정

```bash
cp .env.example .env
```

`.env` 파일을 편집하여 API Key를 설정합니다:

```env
# 사용할 LLM 제공자의 API Key만 설정하면 됩니다
OPENAI_API_KEY=sk-...        # OpenAI 사용 시
ANTHROPIC_API_KEY=sk-ant-... # Anthropic 사용 시
GOOGLE_API_KEY=AIza...       # Google 사용 시
```

### 3. 서비스 실행

```bash
# 기본 실행 (ChromaDB + Streamlit 앱)
docker compose up -d

# 로그 확인
docker compose logs -f app
```

### 4. 브라우저 접속

```
http://localhost:8501
```

---

## 사용 방법

### 문서 인덱싱

1. 사이드바에서 **LLM 설정** (제공자, 모델, API Key)
2. **파일 업로드** (PDF, TXT) 또는 **웹 URL** 입력
3. **📥 인덱싱** 버튼 클릭
4. 인덱싱 완료 메시지 확인

### 채팅

1. 채팅 탭의 입력창에 질문 입력
2. AI가 관련 문서를 검색하여 스트리밍으로 답변
3. 답변 하단의 **📎 참고 문서** 에서 근거 확인

---

## 지원 LLM

| 제공자 | 모델 | API Key | 특징 |
|--------|------|---------|------|
| **OpenAI** | gpt-4o-mini, gpt-4o | 필요 | 빠름, 저비용 |
| **Anthropic** | claude-3-5-sonnet-20241022 | 필요 | 긴 컨텍스트 |
| **Google** | gemini-1.5-flash, gemini-1.5-pro | 필요 | 무료 티어 존재 |
| **Ollama** | llama3.2, mistral 등 | 불필요 | 완전 로컬 실행 |

---

## Ollama (로컬 LLM) 사용

```bash
# Ollama 포함 실행
docker compose --profile ollama up -d

# 모델 다운로드 (예: llama3.2)
docker exec -it docschat-ollama ollama pull llama3.2

# 사용 가능한 모델 목록 확인
docker exec -it docschat-ollama ollama list
```

---

## 지원 임베딩

| 제공자 | 기본 모델 | 비용 | 특징 |
|--------|---------|------|------|
| **HuggingFace** | all-MiniLM-L6-v2 | 무료 | 로컬 실행, 최초 다운로드 필요 |
| **OpenAI** | text-embedding-3-small | 유료 | 고성능, API 호출 |

> HuggingFace 임베딩 모델은 첫 실행 시 자동 다운로드되며, Docker 볼륨에 캐시됩니다.

---

## 아키텍처

```
DocsChat/
├── app.py                     # Streamlit 메인 앱
├── core/
│   ├── document_loader.py     # TXT/PDF/Web 문서 로더
│   ├── embeddings.py          # 임베딩 팩토리
│   ├── llm_factory.py         # LLM 팩토리
│   ├── vector_store.py        # ChromaDB 연결/관리
│   └── rag_engine.py          # RAG 파이프라인 (LCEL)
├── config/
│   └── settings.py            # 환경변수 기반 설정
├── docs/
│   ├── plan.md                # 구현 계획 및 과정
│   ├── vector_db.md           # Vector DB 비교
│   ├── demo.md                # 데모 가이드
│   └── service.md             # 서비스 구축 가이드
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── .env.example
```

---

## Docker Compose 서비스

| 서비스 | 이미지 | 포트 | 설명 |
|--------|--------|------|------|
| `chromadb` | chromadb/chroma:latest | 8000 | Vector DB |
| `app` | (로컬 빌드) | 8501 | Streamlit UI |
| `ollama` | ollama/ollama:latest | 11434 | 로컬 LLM (선택) |

### 볼륨

| 볼륨 | 용도 |
|------|------|
| `chroma_data` | ChromaDB 문서 데이터 (영구 보존) |
| `huggingface_cache` | HuggingFace 임베딩 모델 캐시 |
| `ollama_models` | Ollama LLM 모델 저장소 |

---

## 환경변수

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `LLM_PROVIDER` | `openai` | LLM 제공자 |
| `LLM_MODEL` | (제공자별 기본) | LLM 모델명 |
| `OPENAI_API_KEY` | - | OpenAI API Key |
| `ANTHROPIC_API_KEY` | - | Anthropic API Key |
| `GOOGLE_API_KEY` | - | Google API Key |
| `EMBEDDING_PROVIDER` | `huggingface` | 임베딩 제공자 |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | 임베딩 모델 |
| `CHROMA_HOST` | `chromadb` | ChromaDB 호스트 |
| `CHROMA_PORT` | `8000` | ChromaDB 포트 |
| `CHROMA_COLLECTION` | `docschat` | 컬렉션 이름 |
| `OLLAMA_HOST` | `ollama` | Ollama 호스트 |
| `OLLAMA_PORT` | `11434` | Ollama 포트 |

---

## 로컬 개발 (Docker 없이)

```bash
# ChromaDB는 Docker로 실행
docker run -d -p 8000:8000 chromadb/chroma:latest

# Python 가상환경 생성
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# PyTorch CPU 설치 (HuggingFace 임베딩용)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 의존성 설치
pip install -r requirements.txt

# 환경변수 설정 (CHROMA_HOST를 localhost로)
export CHROMA_HOST=localhost

# 실행
streamlit run app.py
```

---

## 유용한 명령어

```bash
# 서비스 상태 확인
docker compose ps

# 앱 로그 확인
docker compose logs -f app

# ChromaDB 로그 확인
docker compose logs -f chromadb

# 서비스 중지
docker compose down

# 데이터 포함 전체 삭제 (주의: 인덱싱된 문서 삭제됨)
docker compose down -v

# 이미지 재빌드 (코드 변경 후)
docker compose up -d --build app

# ChromaDB API 직접 접근
curl http://localhost:8000/api/v1/heartbeat
```

---

## 관련 문서

- [구현 계획 및 과정](docs/plan.md)
- [Vector DB 비교 분석](docs/vector_db.md)
- [Vector DB Docker 구성](docs/vector_db_docker.md)
- [데모 UI 가이드](docs/demo.md)
- [서비스 구축 가이드](docs/service.md)

---

## 기술 스택

- [LangChain](https://python.langchain.com) - RAG 프레임워크
- [ChromaDB](https://docs.trychroma.com) - 벡터 데이터베이스
- [Streamlit](https://streamlit.io) - 웹 UI
- [sentence-transformers](https://sbert.net) - HuggingFace 임베딩
- [Docker Compose](https://docs.docker.com/compose) - 컨테이너 오케스트레이션
