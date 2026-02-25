"""
LLM 팩토리 모듈
OpenAI, Anthropic, Google, Ollama LLM 인스턴스를 생성합니다.
"""
import os
from langchain_core.language_models import BaseChatModel


# 각 제공자별 기본 모델
DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-3-5-sonnet-20241022",
    "google": "gemini-1.5-flash",
    "ollama": "llama3.2",
}

# 사용자에게 표시할 모델 목록
MODEL_OPTIONS = {
    "openai": ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
    "anthropic": ["claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022", "claude-3-haiku-20240307"],
    "google": ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash"],
    "ollama": ["llama3.2", "llama3.1", "mistral", "qwen2.5", "deepseek-r1"],
}

# 사용자 표시명
PROVIDER_LABELS = {
    "openai": "OpenAI (GPT)",
    "anthropic": "Anthropic (Claude)",
    "google": "Google (Gemini)",
    "ollama": "Ollama (로컬)",
}


def create_llm(
    provider: str,
    model: str | None = None,
    api_key: str | None = None,
    ollama_base_url: str = "http://localhost:11434",
    temperature: float = 0.1,
    streaming: bool = True,
) -> BaseChatModel:
    """
    LLM 인스턴스를 생성합니다.

    Args:
        provider: LLM 제공자 ('openai' | 'anthropic' | 'google' | 'ollama')
        model: 모델명 (None이면 제공자별 기본값 사용)
        api_key: API Key (ollama 제외)
        ollama_base_url: Ollama 서버 URL (ollama 전용)
        temperature: 응답 다양성 (0.0 = 결정적, 1.0 = 창의적)
        streaming: 스트리밍 활성화 여부

    Returns:
        LangChain BaseChatModel 인스턴스
    """
    model = model or DEFAULT_MODELS.get(provider, "")

    match provider:
        case "openai":
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=model,
                api_key=api_key or os.getenv("OPENAI_API_KEY"),
                temperature=temperature,
                streaming=streaming,
            )

        case "anthropic":
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(
                model=model,
                api_key=api_key or os.getenv("ANTHROPIC_API_KEY"),
                temperature=temperature,
                streaming=streaming,
            )

        case "google":
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(
                model=model,
                google_api_key=api_key or os.getenv("GOOGLE_API_KEY"),
                temperature=temperature,
            )

        case "ollama":
            from langchain_ollama import ChatOllama
            return ChatOllama(
                model=model,
                base_url=ollama_base_url,
                temperature=temperature,
            )

        case _:
            raise ValueError(
                f"지원하지 않는 LLM 제공자: '{provider}'. "
                f"지원 목록: openai, anthropic, google, ollama"
            )
