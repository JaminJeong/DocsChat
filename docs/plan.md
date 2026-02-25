# DocsChat 구현 계획 및 과정

> LangChain + ChromaDB + Docker Compose를 활용한 RAG Chat 서비스 상세 계획 및 구현 기록

---

## 1. 프로젝트 개요

### 목표
- 다양한 문서(TXT, PDF, 웹 URL)를 업로드하여 ChromaDB에 벡터로 저장
- 사용자가 질문하면 관련 문서 청크를 검색(RAG)하여 LLM이 답변
- LLM은 사용자가 선택 가능 (OpenAI, Anthropic, Google, Ollama)
- Docker Compose로 서비스 배포, Streamlit으로 데모 UI 제공

### 요구사항 정리
| 요구사항 | 상세 |
|---------|------|
| Vector DB | ChromaDB (Docker HTTP 서버 모드) |
| 문서 지원 | TXT, PDF, 웹 URL |
| LLM | OpenAI GPT / Anthropic Claude / Google Gemini / Ollama(로컬) |
| 배포 | Docker Compose |
| 데모 UI | Streamlit |

---

## 2. 아키텍처 설계

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
│           │ (선택적)                                         │
│           ▼                                                  │
│  ┌─────────────────┐                                        │
│  │  docschat-ollama│                                        │
│  │  (로컬 LLM)     │                                        │
│  │  Port: 11434    │                                        │
│  └─────────────────┘                                        │
└─────────────────────────────────────────────────────────────┘
             │
             │ (외부 API 호출)
             ▼
  OpenAI / Anthropic / Google API
```

### 데이터 흐름

```
문서 입력 파이프라인:
  [파일/URL] → [DocumentLoader] → [TextSplitter] → [Embeddings] → [ChromaDB]

질의 파이프라인:
  [사용자 질문] → [Embeddings] → [ChromaDB 검색] → [관련 청크]
               → [RAG 프롬프트] → [LLM] → [스트리밍 답변]
```

---

## 3. 기술 스택

| 레이어 | 기술 | 버전 |
|--------|------|------|
| **UI** | Streamlit | ≥1.31.0 |
| **RAG 프레임워크** | LangChain | ≥0.3.0 |
| **Vector DB** | ChromaDB | ≥0.5.0 |
| **LangChain-Chroma** | langchain-chroma | ≥0.1.0 |
| **임베딩** | HuggingFace sentence-transformers / OpenAI | - |
| **LLM** | OpenAI / Anthropic / Google / Ollama | - |
| **PDF 파싱** | pypdf | ≥4.0.0 |
| **웹 스크래핑** | beautifulsoup4, requests, lxml | - |
| **컨테이너** | Docker Compose | ≥2.0 |

---

## 4. 프로젝트 파일 구조

```
DocsChat/
├── app.py                     # Streamlit 메인 애플리케이션
├── core/
│   ├── __init__.py
│   ├── document_loader.py     # TXT/PDF/Web 문서 로더
│   ├── embeddings.py          # 임베딩 팩토리 (HuggingFace/OpenAI)
│   ├── llm_factory.py         # LLM 팩토리 (4개 제공자)
│   ├── vector_store.py        # ChromaDB 연결 및 관리
│   └── rag_engine.py          # RAG 파이프라인 (LCEL 체인)
├── config/
│   ├── __init__.py
│   └── settings.py            # 환경변수 기반 설정
├── docs/
│   ├── plan.md                # 이 파일: 계획 및 구현 과정
│   ├── vector_db.md           # Vector DB 비교 분석
│   ├── vector_db_docker.md    # Docker로 Vector DB 구축
│   ├── demo.md                # Streamlit/Gradio 데모 가이드
│   └── service.md             # 전체 서비스 구축 가이드
├── docker-compose.yml         # 서비스 오케스트레이션
├── Dockerfile                 # 앱 컨테이너 이미지
├── requirements.txt           # Python 의존성
├── .env.example               # 환경변수 템플릿
├── .gitignore
└── README.md
```

---

## 5. 핵심 모듈 설계

### 5.1 `core/document_loader.py`

문서 유형별 로더를 통합하고 청크로 분할하는 모듈.

- **TXT**: `TextLoader` (UTF-8 인코딩)
- **PDF**: `PyPDFLoader` (pypdf 기반, 순수 Python, 시스템 의존성 없음)
- **웹 URL**: `WebBaseLoader` (requests + BeautifulSoup4로 HTML 파싱)
- **공통**: `RecursiveCharacterTextSplitter`로 청크 분할

### 5.2 `core/embeddings.py`

임베딩 팩토리 패턴으로 두 가지 제공자 지원:

- **HuggingFace** (기본값): `sentence-transformers/all-MiniLM-L6-v2` - 무료, 로컬 실행
- **OpenAI**: `text-embedding-3-small` - 유료, 고성능

> HuggingFace 모델은 Docker 볼륨(`huggingface_cache`)에 캐시되어 재시작 시 재다운로드 불필요

### 5.3 `core/llm_factory.py`

4개 LLM 제공자를 Factory 패턴으로 통합:

| 제공자 | 기본 모델 | 스트리밍 |
|--------|---------|---------|
| OpenAI | gpt-4o-mini | ✅ |
| Anthropic | claude-3-5-sonnet-20241022 | ✅ |
| Google | gemini-1.5-flash | ✅ |
| Ollama | llama3.2 | ✅ |

### 5.4 `core/vector_store.py`

ChromaDB를 HTTP 클라이언트 모드로 연결:
- Docker 환경: `CHROMA_HOST` 환경변수 → 서비스 이름으로 연결
- 로컬 환경: `localhost:8000`에 ChromaDB 서버 필요

컬렉션 초기화 기능 포함 (데이터 전체 삭제 후 재생성).

### 5.5 `core/rag_engine.py`

LangChain LCEL(LangChain Expression Language) 기반 RAG 체인:

```
retriever → format_docs
                          ↘
                            prompt → LLM → StrOutputParser
                          ↗
RunnablePassthrough (question)
```

- `stream_query()`: 스트리밍 응답 + 소스 문서 반환
- `query()`: 일반 응답 (비스트리밍)

### 5.6 `app.py` (Streamlit UI)

**사이드바 구성:**
- LLM 설정 (제공자, 모델, API Key)
- 임베딩 설정 (제공자, 모델)
- 검색 설정 (Top-K, 컬렉션 이름)
- 문서 업로드 (파일 업로드 + 웹 URL)
- 인덱싱/초기화 버튼

**메인 영역:**
- 채팅 탭: 스트리밍 응답, 소스 문서 표시
- 정보 탭: 사용 방법, 지원 형식 안내

**Streamlit 캐싱 전략:**
- `@st.cache_resource`: 임베딩 모델, ChromaDB 연결 (앱 수명 동안 재사용)
- LLM: 각 쿼리마다 사이드바 설정에 따라 생성 (API Key 변경 반영)

---

## 6. Docker Compose 구성

### 서비스 구성

| 서비스 | 이미지 | 포트 | Profile | 설명 |
|--------|--------|------|---------|------|
| `chromadb` | chromadb/chroma:latest | 8000 | (기본) | Vector DB |
| `app` | (빌드) | 8501 | (기본) | Streamlit UI |
| `ollama` | ollama/ollama:latest | 11434 | `ollama` | 로컬 LLM |

### 볼륨

| 볼륨 | 용도 |
|------|------|
| `chroma_data` | ChromaDB 영구 데이터 |
| `huggingface_cache` | HuggingFace 모델 캐시 |
| `ollama_models` | Ollama 모델 저장소 |

---

## 7. 구현 과정 기록

### Step 1: 문서 구조 분석 (docs/ 폴더 검토)
기존 docs 폴더의 4개 MD 파일을 분석하여 아키텍처 설계 방향 결정:
- `vector_db.md`: Vector DB 비교 → Chroma 선택 (개발/프로토타입 최적)
- `service.md`: 서비스 아키텍처, LLM Factory 패턴, RAG 체인 구성
- `demo.md`: Streamlit UI 설계, RAG 엔진 기본 구조
- `vector_db_docker.md`: Chroma Docker 설정

### Step 2: 프로젝트 구조 설계
docs의 권장 구조를 기반으로 모듈화된 구조 설계:
- `core/` 모듈로 비즈니스 로직 분리
- `config/` 모듈로 환경변수 관리 분리
- Streamlit 앱과 비즈니스 로직 분리

### Step 3: 핵심 모듈 구현
1. `config/settings.py` - 환경변수 기반 설정 (os.getenv)
2. `core/document_loader.py` - 멀티포맷 문서 로더
3. `core/embeddings.py` - 임베딩 팩토리
4. `core/llm_factory.py` - LLM 팩토리
5. `core/vector_store.py` - Chroma HTTP 클라이언트 연결
6. `core/rag_engine.py` - LCEL 기반 RAG 파이프라인

### Step 4: Streamlit UI 구현
- `@st.cache_resource`로 임베딩/벡터스토어 캐싱
- 사이드바에서 LLM/임베딩/검색 설정
- `st.write_stream()`으로 스트리밍 응답 표시
- 소스 문서를 expander로 표시

### Step 5: Docker 환경 구성
- `Dockerfile`: python:3.11-slim + PyTorch CPU (용량 최적화)
- `docker-compose.yml`: Chroma + App + Ollama(선택)
- HuggingFace 모델 캐시 볼륨으로 재시작 시 성능 최적화

### Step 6: 문서화
- `docs/plan.md` (이 파일): 전체 계획 및 구현 과정
- `README.md`: 설치, 실행, 사용법 전체 문서화
- `.gitignore`: 민감 정보(.env), 캐시, 임시파일 제외

---

## 8. 주요 설계 결정

### ChromaDB HTTP 모드 선택
Docker Compose 환경에서 앱과 ChromaDB가 별도 서비스로 분리되므로 HTTP 클라이언트 모드 사용. 로컬 개발 시에도 `docker compose up chromadb -d`로 ChromaDB만 실행하면 동일한 인터페이스 사용 가능.

### HuggingFace 임베딩을 기본값으로 선택
- API Key 없이 바로 사용 가능 → 진입 장벽 낮춤
- `all-MiniLM-L6-v2`: 경량(~90MB) + 빠른 추론 + 다국어 지원
- Docker 볼륨으로 모델 캐시 → 재시작 시 재다운로드 불필요

### PyTorch CPU 버전 선택
`torch --index-url https://download.pytorch.org/whl/cpu`로 CPU 전용 버전 설치:
- CUDA 버전 대비 ~700MB 절약
- 임베딩 추론에는 CPU로 충분한 성능

### LangChain LCEL 체인 선택
`RetrievalQA` 대신 LCEL 체인 사용:
- 스트리밍 지원이 더 자연스러움
- 검색과 생성 단계를 분리하여 소스 문서를 별도로 반환 가능
- 최신 LangChain 패턴 (RetrievalQA는 deprecated 방향)

---

## 9. 알려진 제약사항

1. **임베딩 일관성**: 인덱싱 시 사용한 임베딩 모델과 조회 시 임베딩 모델이 동일해야 함. UI에서 임베딩 설정 변경 시 컬렉션 초기화 필요.

2. **멀티유저**: ChromaDB 컬렉션은 공유됨. 여러 사용자가 같은 데모를 쓰면 같은 문서 풀을 참조. 프로덕션에서는 사용자별 컬렉션 분리 필요.

3. **웹 URL 제한**: JavaScript 렌더링이 필요한 SPA는 `WebBaseLoader`로 수집 불가. 정적 HTML 페이지만 지원.

4. **이미지 크기**: PyTorch + sentence-transformers로 인해 Docker 이미지가 ~1.5GB. 필요 시 OpenAI 임베딩으로 교체하면 ~300MB로 감소.

---

## 10. 확장 계획 (향후)

- [ ] 문서별 메타데이터 필터링 (날짜, 출처 등)
- [ ] 대화 히스토리 기반 쿼리 재구성 (ConversationalRetrievalChain)
- [ ] DOCX, MD 파일 지원 추가
- [ ] Hybrid 검색 (벡터 + BM25 키워드)
- [ ] 사용자별 컬렉션 분리 (멀티테넌시)
- [ ] Gradio UI 버전 추가

---

## 11. 참고 자료

- [LangChain LCEL 공식 문서](https://python.langchain.com/docs/concepts/lcel)
- [LangChain RAG Tutorial](https://python.langchain.com/docs/tutorials/rag)
- [ChromaDB 공식 문서](https://docs.trychroma.com)
- [langchain-chroma GitHub](https://github.com/langchain-ai/langchain-chroma)
- [Streamlit Chat Elements](https://docs.streamlit.io/develop/api-reference/chat)
- [sentence-transformers 모델 허브](https://huggingface.co/sentence-transformers)
