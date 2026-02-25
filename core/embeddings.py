"""
임베딩 팩토리 모듈
HuggingFace(기본, 무료) 또는 OpenAI 임베딩 모델을 생성합니다.
"""
from langchain_core.embeddings import Embeddings


# 각 제공자별 기본 모델
DEFAULT_MODELS = {
    "huggingface": "sentence-transformers/all-MiniLM-L6-v2",
    "openai": "text-embedding-3-small",
    "ollama": "nomic-embed-text",
}


def get_embeddings(
    provider: str = "huggingface",
    model: str | None = None,
    api_key: str | None = None,
    ollama_base_url: str = "http://localhost:11434",
) -> Embeddings:
    """
    임베딩 모델 인스턴스를 생성합니다.

    Args:
        provider: 임베딩 제공자 ('huggingface' | 'openai' | 'ollama')
        model: 모델명 (None이면 제공자별 기본값 사용)
        api_key: API Key (openai 제공자 전용)
        ollama_base_url: Ollama 서버 URL (ollama 제공자 전용)

    Returns:
        LangChain Embeddings 인스턴스
    """
    model = model or DEFAULT_MODELS.get(provider, DEFAULT_MODELS["huggingface"])

    if provider == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name=model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    elif provider == "openai":
        import os
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            model=model,
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
        )

    elif provider == "ollama":
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(
            model=model,
            base_url=ollama_base_url,
        )

    else:
        raise ValueError(
            f"지원하지 않는 임베딩 제공자: '{provider}'. "
            f"지원 목록: huggingface, openai, ollama"
        )
