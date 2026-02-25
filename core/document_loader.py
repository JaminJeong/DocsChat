"""
문서 로더 모듈
TXT, PDF, 웹 URL을 LangChain Document로 로드하고 청크로 분할합니다.
"""
import os
import tempfile
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


def load_txt(file_path: str) -> List[Document]:
    """TXT 파일을 로드합니다."""
    from langchain_community.document_loaders import TextLoader
    loader = TextLoader(file_path, encoding="utf-8")
    return loader.load()


def load_pdf(file_path: str) -> List[Document]:
    """PDF 파일을 로드합니다 (pypdf 기반)."""
    from langchain_community.document_loaders import PyPDFLoader
    loader = PyPDFLoader(file_path)
    return loader.load()


def load_web(url: str) -> List[Document]:
    """웹 URL에서 텍스트를 스크래핑하여 로드합니다."""
    from langchain_community.document_loaders import WebBaseLoader
    import ssl
    # SSL 인증서 검증 문제 우회 (개발환경용)
    try:
        loader = WebBaseLoader(
            web_paths=[url],
            bs_kwargs={"features": "lxml"},
        )
        loader.requests_kwargs = {"verify": True, "timeout": 30}
        docs = loader.load()
    except Exception:
        # lxml이 없을 경우 html.parser 사용
        loader = WebBaseLoader(
            web_paths=[url],
            bs_kwargs={"features": "html.parser"},
        )
        loader.requests_kwargs = {"verify": True, "timeout": 30}
        docs = loader.load()

    # URL을 소스 메타데이터로 설정
    for doc in docs:
        doc.metadata["source"] = url
    return docs


def load_uploaded_file(file_bytes: bytes, filename: str) -> List[Document]:
    """
    업로드된 파일 바이트를 Document로 로드합니다.
    지원 형식: .txt, .pdf
    """
    suffix = Path(filename).suffix.lower()
    if suffix not in (".txt", ".pdf"):
        raise ValueError(f"지원하지 않는 파일 형식: {suffix} (지원: .txt, .pdf)")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        if suffix == ".pdf":
            docs = load_pdf(tmp_path)
        else:  # .txt
            docs = load_txt(tmp_path)

        # 원본 파일명을 소스 메타데이터로 설정
        for doc in docs:
            doc.metadata["source"] = filename
        return docs
    finally:
        os.unlink(tmp_path)


def split_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Document]:
    """
    문서를 지정된 크기의 청크로 분할합니다.

    Args:
        documents: 분할할 Document 리스트
        chunk_size: 청크 최대 길이 (문자 수)
        chunk_overlap: 인접 청크 간 오버랩 길이 (문맥 유지용)

    Returns:
        분할된 Document 리스트
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)
