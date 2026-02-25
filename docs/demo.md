# RAG Chat 데모 구축 방법

> Streamlit과 Gradio를 활용한 RAG Chat 서비스 데모 가이드

---

## 1. Streamlit vs Gradio 비교

| 항목 | Streamlit | Gradio |
|------|-----------|--------|
| **장점** | 풍부한 UI 컴포넌트, 상태 관리 용이 | 빠른 프로토타입, 공유 URL 제공 |
| **채팅 UI** | `st.chat_message`, `st.chat_input` | `gr.ChatInterface` |
| **파일 업로드** | `st.file_uploader` | `gr.File` |
| **스트리밍** | ✅ (`st.write_stream`) | ✅ |
| **커스터마이징** | 높음 | 중간 |
| **공유** | URL 공유 가능 | 1-click 공유 링크 제공 |

---

## 2. Streamlit 기반 RAG Chat 데모

### 2.1 설치

```bash
pip install streamlit langchain langchain-openai langchain-community chromadb pypdf
```

### 2.2 프로젝트 구조

```
docschat/
├── app.py              # Streamlit 메인 앱
├── rag_engine.py       # RAG 로직
├── Dockerfile
└── requirements.txt
```

### 2.3 Streamlit 앱 (app.py)

```python
import streamlit as st
from rag_engine import RAGEngine

st.set_page_config(page_title="DocsChat", page_icon="📚", layout="wide")
st.title("📚 DocsChat - RAG Chat")

# 사이드바 설정
with st.sidebar:
    st.header("⚙️ 설정")
    llm_provider = st.selectbox(
        "LLM 제공자",
        ["OpenAI (GPT-4o)", "Anthropic (Claude)", "Ollama (로컬)", "Google (Gemini)"]
    )
    api_key = st.text_input("API Key", type="password")
    vector_db = st.selectbox("Vector DB", ["Chroma", "Qdrant", "Weaviate"])
    top_k = st.slider("검색 결과 수 (Top-K)", 1, 10, 5)

    st.header("📄 문서 업로드")
    uploaded_files = st.file_uploader(
        "PDF, TXT, MD 파일 업로드",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True
    )
    if uploaded_files and st.button("문서 인덱싱", type="primary"):
        with st.spinner("처리 중..."):
            st.session_state.rag = RAGEngine(llm_provider, vector_db, api_key)
            count = st.session_state.rag.index_documents(uploaded_files)
            st.success(f"✅ {count}개 청크 인덱싱 완료!")

# 채팅 히스토리
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 사용자 입력 처리
if prompt := st.chat_input("문서에 대해 질문하세요..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        if "rag" not in st.session_state:
            st.warning("먼저 문서를 업로드하고 인덱싱해주세요!")
        else:
            response, sources = st.session_state.rag.query(prompt, top_k)
            st.markdown(response)
            if sources:
                with st.expander("📎 참고 문서"):
                    for s in sources:
                        st.info(s)
            st.session_state.messages.append({"role": "assistant", "content": response})
```

### 2.4 RAG 엔진 (rag_engine.py)

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
import tempfile, os

class RAGEngine:
    def __init__(self, llm_provider: str, vector_db: str, api_key: str = None):
        self.llm = self._init_llm(llm_provider, api_key)
        self.embeddings = OpenAIEmbeddings(api_key=api_key)
        self.vectorstore = None

    def _init_llm(self, provider, api_key):
        if "OpenAI" in provider:
            return ChatOpenAI(model="gpt-4o", api_key=api_key)
        elif "Anthropic" in provider:
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(model="claude-3-5-sonnet-20241022", api_key=api_key)
        elif "Ollama" in provider:
            from langchain_ollama import ChatOllama
            return ChatOllama(model="llama3.2")
        elif "Google" in provider:
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(model="gemini-1.5-pro", api_key=api_key)

    def index_documents(self, files) -> int:
        documents = []
        for file in files:
            suffix = f".{file.name.split('.')[-1]}"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(file.read())
                tmp_path = tmp.name
            loader = PyPDFLoader(tmp_path) if file.name.endswith(".pdf") else TextLoader(tmp_path)
            documents.extend(loader.load())
            os.unlink(tmp_path)

        splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
        chunks = splitter.split_documents(documents)
        self.vectorstore = Chroma.from_documents(chunks, self.embeddings)
        return len(chunks)

    def query(self, question: str, top_k: int = 5):
        chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": top_k}),
            return_source_documents=True
        )
        result = chain.invoke({"query": question})
        sources = [doc.page_content[:200] for doc in result["source_documents"]]
        return result["result"], sources
```

### 2.5 Dockerfile

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

---

## 3. Gradio 기반 RAG Chat 데모

### 3.1 설치

```bash
pip install gradio langchain langchain-openai chromadb
```

### 3.2 Gradio 앱 (gradio_app.py)

```python
import gradio as gr
from rag_engine import RAGEngine

rag_engine = None

def init_rag(files, llm_provider, api_key, vector_db):
    global rag_engine
    rag_engine = RAGEngine(llm_provider=llm_provider, vector_db=vector_db, api_key=api_key)
    count = rag_engine.index_documents(files)
    return f"✅ {len(files)}개 파일, {count}개 청크 인덱싱 완료!"

def chat(message, history, top_k):
    global rag_engine
    if rag_engine is None:
        return history + [[message, "❌ 먼저 문서를 업로드하고 인덱싱해주세요."]]
    response, sources = rag_engine.query(message, top_k=top_k)
    src_text = "\n\n📎 **참고:**\n" + "\n".join([f"- {s[:150]}..." for s in sources])
    return history + [[message, response + src_text]]

with gr.Blocks(title="DocsChat", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 📚 DocsChat - RAG Chat 서비스")
    with gr.Row():
        with gr.Column(scale=1):
            llm_dd = gr.Dropdown(
                ["OpenAI (GPT-4o)", "Anthropic (Claude)", "Ollama (로컬)", "Google (Gemini)"],
                label="LLM 제공자", value="OpenAI (GPT-4o)"
            )
            api_key_tb = gr.Textbox(label="API Key", type="password")
            vdb_dd = gr.Dropdown(["Chroma", "Qdrant", "Weaviate"], label="Vector DB", value="Chroma")
            top_k_sl = gr.Slider(1, 10, value=5, label="Top-K")
            file_up = gr.File(file_count="multiple", file_types=[".pdf", ".txt", ".md"])
            idx_btn = gr.Button("📥 문서 인덱싱", variant="primary")
            idx_status = gr.Textbox(label="상태", interactive=False)
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(height=500, label="대화")
            msg_tb = gr.Textbox(label="메시지", placeholder="질문을 입력하세요...")
            with gr.Row():
                send_btn = gr.Button("전송", variant="primary")
                clear_btn = gr.Button("초기화")

    idx_btn.click(init_rag, [file_up, llm_dd, api_key_tb, vdb_dd], [idx_status])
    send_btn.click(chat, [msg_tb, chatbot, top_k_sl], [chatbot]).then(lambda: "", outputs=[msg_tb])
    msg_tb.submit(chat, [msg_tb, chatbot, top_k_sl], [chatbot]).then(lambda: "", outputs=[msg_tb])
    clear_btn.click(lambda: [], outputs=[chatbot])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
```

---

## 4. Docker Compose 통합

```yaml
version: '3.8'
services:
  streamlit:
    build: .
    ports:
      - "8501:8501"
    environment:
      - CHROMA_HOST=chromadb
    depends_on:
      - chromadb

  chromadb:
    image: chromadb/chroma:latest
    volumes:
      - chroma_data:/chroma/chroma
    ports:
      - "8000:8000"

volumes:
  chroma_data:
```

---

## 5. 핵심 기능 체크리스트

- [ ] 파일 업로드: PDF, TXT, DOCX, MD 지원
- [ ] LLM 선택: OpenAI, Anthropic, Ollama, Google
- [ ] Vector DB 선택: Chroma, Qdrant, Weaviate
- [ ] 멀티턴 채팅 지원
- [ ] 소스 문서 표시 (답변 근거)
- [ ] 스트리밍 응답
- [ ] 대화 초기화

---

## 6. 참고 자료

- [Streamlit 공식 문서](https://docs.streamlit.io)
- [Streamlit Chat Elements](https://docs.streamlit.io/develop/api-reference/chat)
- [Gradio 공식 문서](https://www.gradio.app/docs)
- [Gradio ChatInterface](https://www.gradio.app/docs/gradio/chatinterface)
- [LangChain Streamlit Integration](https://python.langchain.com/docs/integrations/callbacks/streamlit)
