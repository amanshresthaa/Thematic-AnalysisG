from dataclasses import dataclass
from typing import Any, Callable, Type, Optional

@dataclass
class OptimizerConfig:
    max_bootstrapped_demos: int = 4
    max_labeled_demos: int = 4
    num_candidate_programs: int = 10
    num_threads: int = 1
    temperature: float = 0.7
    max_tokens: int = 8192
    retrieval_k: int = 20

@dataclass
class ModelConfig:
    """Configuration for the language model to be used."""
    provider: str = "openai"  # Default to OpenAI
    model_name: Optional[str] = None  # If None, will use provider's default model
    api_key_env: Optional[str] = None  # Environment variable name for API key
    
    def __post_init__(self):
        # Set default values based on provider
        if self.provider == "openai":
            self.model_name = self.model_name or "gpt-4"
            self.api_key_env = self.api_key_env or "OPENAI_API_KEY"
        elif self.provider == "google":
            self.model_name = self.model_name or "gemini-2.0-flash-thinking-exp-01-21"
            self.api_key_env = self.api_key_env or "GOOGLE_API_KEY"
        elif self.provider == "deepseek":
            self.model_name = self.model_name or "deepseek/deepseek-reasoner"
            self.api_key_env = self.api_key_env or "DEEPSEEK_API_KEY"
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

@dataclass
class ModuleConfig:
    index_name: str
    codebase_chunks_file: str
    queries_file_standard: str
    evaluation_set_file: str
    output_filename_primary: str
    training_data: str
    optimized_program_path: str
    module_class: Type[Any]
    conversion_func: Optional[Callable[[str, str, str], None]] = None
    model_config: Optional[ModelConfig] = None  # Added field for model configuration

    def __post_init__(self):
        # If no model config is provided, use default OpenAI configuration
        if self.model_config is None:
            self.model_config = ModelConfig()