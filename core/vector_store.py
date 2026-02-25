"""
ChromaDB 벡터스토어 모듈
ChromaDB HTTP 클라이언트를 통해 벡터스토어에 연결하고 관리합니다.
"""
import os
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma


def get_chroma_client(host: str | None = None, port: int | None = None):
    """ChromaDB HTTP 클라이언트를 생성합니다."""
    import chromadb
    h = host or os.getenv("CHROMA_HOST", "localhost")
    p = port or int(os.getenv("CHROMA_PORT", "8000"))
    return chromadb.HttpClient(host=h, port=p)


def get_vectorstore(
    embeddings: Embeddings,
    collection_name: str | None = None,
    chroma_host: str | None = None,
    chroma_port: int | None = None,
) -> Chroma:
    """
    ChromaDB 벡터스토어 인스턴스를 반환합니다.

    Args:
        embeddings: 임베딩 모델 인스턴스
        collection_name: ChromaDB 컬렉션 이름
        chroma_host: ChromaDB 서버 호스트 (None이면 환경변수 사용)
        chroma_port: ChromaDB 서버 포트 (None이면 환경변수 사용)

    Returns:
        LangChain Chroma 벡터스토어 인스턴스
    """
    name = collection_name or os.getenv("CHROMA_COLLECTION", "docschat")
    client = get_chroma_client(chroma_host, chroma_port)

    return Chroma(
        client=client,
        collection_name=name,
        embedding_function=embeddings,
    )


def get_document_count(vectorstore: Chroma) -> int:
    """벡터스토어에 저장된 문서 청크 수를 반환합니다."""
    try:
        return vectorstore._collection.count()
    except Exception:
        return 0


def reset_collection(
    collection_name: str,
    chroma_host: str | None = None,
    chroma_port: int | None = None,
) -> None:
    """
    지정된 컬렉션을 삭제하고 재생성합니다 (전체 초기화).

    Args:
        collection_name: 초기화할 컬렉션 이름
        chroma_host: ChromaDB 서버 호스트
        chroma_port: ChromaDB 서버 포트
    """
    client = get_chroma_client(chroma_host, chroma_port)
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass  # 컬렉션이 없으면 무시
    client.create_collection(collection_name)


def check_connection(
    chroma_host: str | None = None,
    chroma_port: int | None = None,
) -> bool:
    """ChromaDB 서버 연결 상태를 확인합니다."""
    try:
        client = get_chroma_client(chroma_host, chroma_port)
        client.heartbeat()
        return True
    except Exception:
        return False
