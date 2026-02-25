"""
환경변수 기반 설정 모듈
.env 파일 또는 실제 환경변수에서 설정을 읽어옵니다.
"""
import os
from pathlib import Path

# .env 파일 로드 (python-dotenv가 설치된 경우)
try:
    from dotenv import load_dotenv
    _env_file = Path(__file__).parent.parent / ".env"
    if _env_file.exists():
        load_dotenv(_env_file)
except ImportError:
    pass


class Settings:
    # ── LLM 설정 ──────────────────────────────────
    @property
    def llm_provider(self) -> str:
        return os.getenv("LLM_PROVIDER", "openai")

    @property
    def llm_model(self) -> str | None:
        return os.getenv("LLM_MODEL")

    @property
    def openai_api_key(self) -> str | None:
        return os.getenv("OPENAI_API_KEY")

    @property
    def anthropic_api_key(self) -> str | None:
        return os.getenv("ANTHROPIC_API_KEY")

    @property
    def google_api_key(self) -> str | None:
        return os.getenv("GOOGLE_API_KEY")

    # ── 임베딩 설정 ────────────────────────────────
    @property
    def embedding_provider(self) -> str:
        return os.getenv("EMBEDDING_PROVIDER", "huggingface")

    @property
    def embedding_model(self) -> str | None:
        return os.getenv("EMBEDDING_MODEL")

    # ── ChromaDB 설정 ──────────────────────────────
    @property
    def chroma_host(self) -> str:
        return os.getenv("CHROMA_HOST", "localhost")

    @property
    def chroma_port(self) -> int:
        return int(os.getenv("CHROMA_PORT", "8000"))

    @property
    def chroma_collection(self) -> str:
        return os.getenv("CHROMA_COLLECTION", "docschat")

    # ── Ollama 설정 ────────────────────────────────
    @property
    def ollama_host(self) -> str:
        return os.getenv("OLLAMA_HOST", "localhost")

    @property
    def ollama_port(self) -> int:
        return int(os.getenv("OLLAMA_PORT", "11434"))

    @property
    def ollama_base_url(self) -> str:
        return f"http://{self.ollama_host}:{self.ollama_port}"


settings = Settings()
