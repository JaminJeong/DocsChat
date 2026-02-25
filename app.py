"""
DocsChat - 문서 기반 RAG 채팅 서비스
Streamlit UI + LangChain + ChromaDB

지원:
  - 문서: TXT, PDF, 웹 URL
  - LLM: OpenAI, Anthropic, Google, Ollama
  - 임베딩: HuggingFace (무료), OpenAI (유료)
"""
import os

import streamlit as st

from config.settings import settings
from core.document_loader import load_uploaded_file, load_web, split_documents
from core.embeddings import get_embeddings, DEFAULT_MODELS as EMB_DEFAULT_MODELS
from core.llm_factory import create_llm, MODEL_OPTIONS, PROVIDER_LABELS
from core.rag_engine import RAGEngine
from core.vector_store import (
    check_connection,
    get_document_count,
    get_vectorstore,
    reset_collection,
)

# ── 페이지 설정 ────────────────────────────────────────────────
st.set_page_config(
    page_title="DocsChat",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "DocsChat - RAG 기반 문서 채팅 서비스\nhttps://github.com/DocsChat",
    },
)


# ── 캐시된 리소스 ─────────────────────────────────────────────
@st.cache_resource(show_spinner="임베딩 모델 로딩 중...")
def _get_cached_embeddings(provider: str, model: str, api_key: str, ollama_url: str):
    """임베딩 모델을 캐싱합니다 (HuggingFace 모델은 첫 실행 시 다운로드)."""
    return get_embeddings(
        provider=provider,
        model=model or None,
        api_key=api_key or None,
        ollama_base_url=ollama_url,
    )


@st.cache_resource(show_spinner="ChromaDB 연결 중...")
def _get_cached_vectorstore(
    emb_provider: str,
    emb_model: str,
    emb_api_key: str,
    collection_name: str,
    ollama_url: str,
):
    """벡터스토어 인스턴스를 캐싱합니다 (설정 변경 시 자동 재생성)."""
    embeddings = _get_cached_embeddings(emb_provider, emb_model, emb_api_key, ollama_url)
    return get_vectorstore(
        embeddings=embeddings,
        collection_name=collection_name,
    )


# ── 사이드바 ──────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ 설정")

    # ── ChromaDB 연결 상태 ─────────────────────────────────────
    if check_connection():
        st.success("ChromaDB 연결됨", icon="✅")
    else:
        st.error(
            f"ChromaDB 연결 실패 ({settings.chroma_host}:{settings.chroma_port})\n\n"
            "ChromaDB 서버를 시작하세요:\n`docker compose up chromadb -d`",
            icon="❌",
        )

    st.divider()

    # ── LLM 설정 ──────────────────────────────────────────────
    with st.expander("🤖 LLM 설정", expanded=True):
        llm_provider = st.selectbox(
            "제공자",
            options=list(PROVIDER_LABELS.keys()),
            format_func=lambda x: PROVIDER_LABELS[x],
            index=list(PROVIDER_LABELS.keys()).index(settings.llm_provider)
            if settings.llm_provider in PROVIDER_LABELS
            else 0,
            key="llm_provider",
        )

        default_model = settings.llm_model or MODEL_OPTIONS[llm_provider][0]
        if default_model not in MODEL_OPTIONS[llm_provider]:
            default_model = MODEL_OPTIONS[llm_provider][0]

        llm_model = st.selectbox(
            "모델",
            options=MODEL_OPTIONS[llm_provider],
            index=MODEL_OPTIONS[llm_provider].index(default_model),
            key="llm_model",
        )

        # API Key (Ollama는 불필요)
        if llm_provider != "ollama":
            env_key_map = {
                "openai": "OPENAI_API_KEY",
                "anthropic": "ANTHROPIC_API_KEY",
                "google": "GOOGLE_API_KEY",
            }
            llm_api_key = st.text_input(
                "API Key",
                value=os.getenv(env_key_map.get(llm_provider, ""), ""),
                type="password",
                placeholder="API Key를 입력하세요",
                key="llm_api_key",
            )
        else:
            llm_api_key = ""
            st.info(
                f"Ollama URL: `http://{settings.ollama_host}:{settings.ollama_port}`\n\n"
                "Ollama 사용 시: `docker compose --profile ollama up -d`",
                icon="ℹ️",
            )

        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.1,
            help="값이 낮을수록 결정적, 높을수록 창의적인 답변",
            key="temperature",
        )

    # ── 임베딩 설정 ────────────────────────────────────────────
    with st.expander("🔢 임베딩 설정", expanded=False):
        emb_provider = st.selectbox(
            "제공자",
            options=["huggingface", "openai"],
            format_func=lambda x: {
                "huggingface": "HuggingFace (무료/로컬)",
                "openai": "OpenAI (유료)",
            }[x],
            index=0 if settings.embedding_provider == "huggingface" else 1,
            key="emb_provider",
        )

        emb_model_options = {
            "huggingface": [
                "sentence-transformers/all-MiniLM-L6-v2",
                "BAAI/bge-m3",
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            ],
            "openai": [
                "text-embedding-3-small",
                "text-embedding-3-large",
                "text-embedding-ada-002",
            ],
        }

        default_emb_model = settings.embedding_model or EMB_DEFAULT_MODELS.get(emb_provider, "")
        emb_opts = emb_model_options[emb_provider]
        emb_idx = emb_opts.index(default_emb_model) if default_emb_model in emb_opts else 0

        emb_model = st.selectbox(
            "모델",
            options=emb_opts,
            index=emb_idx,
            key="emb_model",
        )

        emb_api_key = ""
        if emb_provider == "openai":
            emb_api_key = st.text_input(
                "OpenAI API Key (임베딩용)",
                value=os.getenv("OPENAI_API_KEY", ""),
                type="password",
                key="emb_api_key",
            )

        st.caption(
            "⚠️ 임베딩 설정을 변경하면 기존에 인덱싱된 문서와 호환되지 않을 수 있습니다. "
            "변경 후 컬렉션을 초기화하세요."
        )

    # ── 검색 설정 ──────────────────────────────────────────────
    with st.expander("🔍 검색 설정", expanded=False):
        top_k = st.slider(
            "Top-K (검색 결과 수)",
            min_value=1,
            max_value=15,
            value=5,
            key="top_k",
            help="RAG 검색 시 가져올 문서 청크 수",
        )

        collection_name = st.text_input(
            "컬렉션 이름",
            value=settings.chroma_collection,
            key="collection_name",
            help="ChromaDB 컬렉션 이름. 다른 문서 세트는 다른 컬렉션 이름을 사용하세요.",
        )

        chunk_size = st.number_input(
            "청크 크기 (문자 수)",
            min_value=256,
            max_value=4096,
            value=1000,
            step=256,
            key="chunk_size",
        )

        chunk_overlap = st.number_input(
            "청크 오버랩 (문자 수)",
            min_value=0,
            max_value=512,
            value=200,
            step=50,
            key="chunk_overlap",
        )

    st.divider()

    # ── 문서 관리 ──────────────────────────────────────────────
    st.subheader("📄 문서 관리")

    uploaded_files = st.file_uploader(
        "파일 업로드",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        help="PDF, TXT 파일을 업로드하세요",
    )

    web_url = st.text_input(
        "웹 URL",
        placeholder="https://example.com/page",
        help="텍스트를 추출할 웹 페이지 URL",
    )

    col1, col2 = st.columns(2)

    with col1:
        index_btn = st.button(
            "📥 인덱싱",
            type="primary",
            use_container_width=True,
            help="선택한 문서를 ChromaDB에 저장",
        )

    with col2:
        reset_btn = st.button(
            "🗑️ 초기화",
            type="secondary",
            use_container_width=True,
            help="컬렉션의 모든 문서를 삭제",
        )

    # ── 인덱싱 처리 ────────────────────────────────────────────
    if index_btn:
        if not uploaded_files and not web_url:
            st.warning("파일 또는 웹 URL을 입력해주세요.")
        else:
            with st.spinner("문서를 인덱싱 중..."):
                try:
                    ollama_url = f"http://{settings.ollama_host}:{settings.ollama_port}"
                    vs = _get_cached_vectorstore(
                        emb_provider, emb_model, emb_api_key,
                        collection_name, ollama_url,
                    )

                    all_docs = []

                    # 파일 로드
                    for f in (uploaded_files or []):
                        with st.spinner(f"`{f.name}` 로딩 중..."):
                            docs = load_uploaded_file(f.read(), f.name)
                            all_docs.extend(docs)
                            st.caption(f"✓ {f.name}: {len(docs)}개 페이지/섹션")

                    # 웹 URL 로드
                    if web_url:
                        with st.spinner(f"웹 페이지 로딩 중..."):
                            docs = load_web(web_url)
                            all_docs.extend(docs)
                            st.caption(f"✓ {web_url}: {len(docs)}개 섹션")

                    if all_docs:
                        # 청크 분할
                        chunks = split_documents(
                            all_docs,
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap,
                        )
                        # ChromaDB에 저장
                        vs.add_documents(chunks)

                        total = st.session_state.get("indexed_total", 0) + len(chunks)
                        st.session_state["indexed_total"] = total
                        st.session_state["vectorstore_ready"] = True
                        st.success(f"✅ {len(chunks)}개 청크 인덱싱 완료!")

                except Exception as e:
                    st.error(f"인덱싱 오류: {e}")

    # ── 컬렉션 초기화 처리 ─────────────────────────────────────
    if reset_btn:
        try:
            reset_collection(collection_name)
            st.session_state["indexed_total"] = 0
            st.session_state["vectorstore_ready"] = False
            st.session_state["messages"] = []
            # 캐시 무효화
            _get_cached_vectorstore.clear()
            st.success(f"컬렉션 '{collection_name}'이 초기화되었습니다.")
            st.rerun()
        except Exception as e:
            st.error(f"초기화 오류: {e}")

    # ── 문서 상태 표시 ─────────────────────────────────────────
    try:
        ollama_url = f"http://{settings.ollama_host}:{settings.ollama_port}"
        vs = _get_cached_vectorstore(
            emb_provider, emb_model, emb_api_key, collection_name, ollama_url,
        )
        doc_count = get_document_count(vs)
        if doc_count > 0:
            st.info(f"📊 저장된 청크: **{doc_count}개**", icon="📚")
        else:
            st.caption("아직 인덱싱된 문서가 없습니다.")
    except Exception:
        pass


# ── 메인 영역 ─────────────────────────────────────────────────
st.title("📚 DocsChat")
st.caption("문서를 업로드하거나 웹 URL을 입력하고, AI에게 질문하세요")

tab_chat, tab_info = st.tabs(["💬 채팅", "ℹ️ 사용 방법"])


# ── 정보 탭 ──────────────────────────────────────────────────
with tab_info:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
### 🚀 빠른 시작

1. **사이드바** 에서 LLM 제공자와 API Key를 설정하세요
2. **파일을 업로드**하거나 **웹 URL**을 입력하세요
3. **📥 인덱싱** 버튼을 클릭하세요
4. **채팅 탭**에서 질문하세요!

---

### 📋 지원 형식

| 형식 | 설명 |
|------|------|
| 📄 PDF | 텍스트 기반 PDF 파일 |
| 📝 TXT | 일반 텍스트 파일 |
| 🌐 웹 URL | 정적 HTML 웹 페이지 |
        """)

    with col2:
        st.markdown("""
### 🤖 지원 LLM

| 제공자 | 모델 | API Key 필요 |
|--------|------|-------------|
| OpenAI | GPT-4o, GPT-4o-mini | ✅ |
| Anthropic | Claude 3.5 Sonnet | ✅ |
| Google | Gemini 1.5 Flash/Pro | ✅ |
| Ollama | Llama3, Mistral 등 | ❌ (로컬) |

---

### 🔧 임베딩 제공자

| 제공자 | 비용 | 특징 |
|--------|------|------|
| HuggingFace | 무료 | 로컬 실행, API Key 불필요 |
| OpenAI | 유료 | 고성능, API 호출 필요 |
        """)

    st.divider()
    st.markdown("""
### ⚠️ 주의사항

- **임베딩 일관성**: 인덱싱 시 사용한 임베딩 모델과 질문 시 모델이 동일해야 합니다.
  임베딩 설정 변경 후에는 반드시 컬렉션을 초기화(🗑️)하고 재인덱싱하세요.
- **웹 URL**: JavaScript로 렌더링되는 SPA 페이지는 지원하지 않습니다 (정적 HTML만 지원).
- **Ollama**: `docker compose --profile ollama up -d` 로 Ollama 서비스를 먼저 시작하세요.
    """)


# ── 채팅 탭 ──────────────────────────────────────────────────
with tab_chat:
    # 채팅 히스토리 초기화
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # 채팅 히스토리 표시
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📎 참고 문서", expanded=False):
                    for src in msg["sources"]:
                        st.caption(src)

    # 대화 초기화 버튼
    if st.session_state["messages"]:
        if st.button("🗑️ 대화 초기화", key="clear_chat"):
            st.session_state["messages"] = []
            st.rerun()

    # 채팅 입력
    if prompt := st.chat_input("문서에 대해 질문하세요..."):
        # 사용자 메시지 표시
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 어시스턴트 응답 생성
        with st.chat_message("assistant"):
            try:
                ollama_url = f"http://{settings.ollama_host}:{settings.ollama_port}"

                # 벡터스토어 연결
                vs = _get_cached_vectorstore(
                    emb_provider, emb_model, emb_api_key,
                    collection_name, ollama_url,
                )

                # 인덱싱된 문서 확인
                doc_count = get_document_count(vs)
                if doc_count == 0:
                    msg = (
                        "아직 인덱싱된 문서가 없습니다. "
                        "사이드바에서 파일을 업로드하거나 웹 URL을 입력한 후 "
                        "**📥 인덱싱** 버튼을 클릭해주세요."
                    )
                    st.warning(msg)
                    st.session_state["messages"].pop()  # 사용자 메시지 제거
                else:
                    # LLM 생성
                    llm = create_llm(
                        provider=llm_provider,
                        model=llm_model,
                        api_key=llm_api_key or None,
                        ollama_base_url=ollama_url,
                        temperature=temperature,
                        streaming=True,
                    )

                    # RAG 엔진 생성 및 스트리밍 쿼리
                    rag = RAGEngine(llm=llm, vectorstore=vs)
                    stream, source_docs = rag.stream_query(prompt, top_k=top_k)

                    # 스트리밍으로 답변 표시
                    response = st.write_stream(stream)

                    # 소스 문서 표시
                    source_summaries = []
                    if source_docs:
                        with st.expander("📎 참고 문서", expanded=False):
                            for i, doc in enumerate(source_docs, 1):
                                source = doc.metadata.get("source", "알 수 없음")
                                preview = doc.page_content[:200].replace("\n", " ")
                                summary = f"**[{i}] {source}**: {preview}..."
                                st.caption(summary)
                                source_summaries.append(f"[{i}] {source}: {preview}...")

                    # 히스토리에 저장
                    st.session_state["messages"].append({
                        "role": "assistant",
                        "content": response,
                        "sources": source_summaries,
                    })

            except Exception as e:
                error_msg = f"오류가 발생했습니다: {str(e)}"
                st.error(error_msg)
                # 에러 메시지도 히스토리에 저장
                st.session_state["messages"].append({
                    "role": "assistant",
                    "content": f"❌ {error_msg}",
                })
