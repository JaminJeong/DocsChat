"""
RAG 파이프라인 모듈
LangChain LCEL(LangChain Expression Language)을 사용한 RAG 체인을 구현합니다.

흐름:
  [질문] → [Retriever] → [Context 포맷팅]
                                         → [Prompt] → [LLM] → [답변]
  [질문] → [RunnablePassthrough]        ↗
"""
from typing import Generator, List, Tuple

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import VectorStore


RAG_SYSTEM_PROMPT = """\
당신은 주어진 문서를 기반으로 정확하고 유용한 답변을 제공하는 AI 어시스턴트입니다.

규칙:
- 아래 '참고 문서' 섹션의 내용만을 근거로 답변하세요.
- 참고 문서에 없는 내용은 "제공된 문서에서 해당 정보를 찾을 수 없습니다."라고 솔직하게 말하세요.
- 답변은 명확하고 구조적으로 작성하세요.
- 한국어로 질문하면 한국어로, 영어로 질문하면 영어로 답변하세요.

참고 문서:
{context}

질문: {question}

답변:"""


class RAGEngine:
    """RAG 파이프라인을 관리하는 엔진 클래스."""

    def __init__(self, llm: BaseChatModel, vectorstore: VectorStore):
        """
        Args:
            llm: LangChain 호환 LLM 인스턴스
            vectorstore: LangChain 호환 VectorStore 인스턴스
        """
        self.llm = llm
        self.vectorstore = vectorstore
        self._prompt = ChatPromptTemplate.from_template(RAG_SYSTEM_PROMPT)

    def _get_retriever(self, top_k: int = 5, search_type: str = "similarity"):
        """벡터 유사도 검색 Retriever를 반환합니다."""
        return self.vectorstore.as_retriever(
            search_type=search_type,
            search_kwargs={"k": top_k},
        )

    @staticmethod
    def _format_docs(docs: List[Document]) -> str:
        """검색된 Document 리스트를 프롬프트에 넣을 문자열로 포맷합니다."""
        if not docs:
            return "관련 문서를 찾을 수 없습니다."
        parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "알 수 없음")
            parts.append(f"[{i}] 출처: {source}\n{doc.page_content}")
        return "\n\n---\n\n".join(parts)

    def retrieve(self, question: str, top_k: int = 5) -> List[Document]:
        """질문과 관련된 문서 청크를 검색합니다."""
        retriever = self._get_retriever(top_k)
        return retriever.invoke(question)

    def query(self, question: str, top_k: int = 5) -> Tuple[str, List[Document]]:
        """
        RAG 체인을 실행하여 답변을 생성합니다 (비스트리밍).

        Returns:
            (answer: str, source_docs: List[Document])
        """
        docs = self.retrieve(question, top_k)
        context = self._format_docs(docs)

        chain = self._prompt | self.llm | StrOutputParser()
        answer = chain.invoke({"context": context, "question": question})

        return answer, docs

    def stream_query(
        self,
        question: str,
        top_k: int = 5,
    ) -> Tuple[Generator, List[Document]]:
        """
        RAG 체인을 실행하여 스트리밍 답변을 생성합니다.

        Returns:
            (stream: Generator[str], source_docs: List[Document])
            stream은 st.write_stream()에 직접 전달 가능
        """
        # 검색은 스트리밍 전에 완료 (소스 문서를 즉시 반환하기 위해)
        docs = self.retrieve(question, top_k)
        context = self._format_docs(docs)

        chain = self._prompt | self.llm | StrOutputParser()
        stream = chain.stream({"context": context, "question": question})

        return stream, docs
