from enum import Enum
from typing import Optional
from dataclasses import dataclass

class ModelProvider(Enum):
    OPENAI = "openai"
    GOOGLE = "google"
    DEEPSEEK = "deepseek"

@dataclass
class ModelConfig:
    provider: ModelProvider
    model_name: str
    api_key_env: str
    max_tokens: int = 8192
    temperature: float = 0.7

DEFAULT_MODELS = {
    ModelProvider.OPENAI: ModelConfig(
        provider=ModelProvider.OPENAI,
        model_name="gpt-4o-mini",
        api_key_env="OPENAI_API_KEY",
    ),
    ModelProvider.GOOGLE: ModelConfig(
        provider=ModelProvider.GOOGLE,
        model_name="gemini-2.0-flash-thinking-exp-01-21",
        api_key_env="GOOGLE_API_KEY",
    ),
    ModelProvider.DEEPSEEK: ModelConfig(
        provider=ModelProvider.DEEPSEEK,
        model_name="deepseek/deepseek-reasoner",
        api_key_env="DEEPSEEK_API_KEY",
    ),
}

def get_model_config(provider: Optional[str] = None) -> ModelConfig:
    """Get model configuration based on provider name."""
    if not provider:
        return DEFAULT_MODELS[ModelProvider.OPENAI]
    
    try:
        provider_enum = ModelProvider(provider.lower())
        return DEFAULT_MODELS[provider_enum]
    except (ValueError, KeyError):
        raise ValueError(f"Unsupported provider: {provider}. Available providers: {[p.value for p in ModelProvider]}")