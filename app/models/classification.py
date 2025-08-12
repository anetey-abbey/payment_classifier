import uuid
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, PositiveFloat, PositiveInt, field_validator

from app.core.config import load_config


class ActionType(str, Enum):
    CLASSIFY = "classify"
    SEARCH = "search"


class ModelType(str, Enum):
    LOCAL = "local"
    CLOUD = "cloud"


class ClassificationRequest(BaseModel):
    model_config = {"protected_namespaces": ()}

    payment_text: str = Field(
        ..., min_length=1, description="Payment description to classify"
    )
    categories: List[str] = Field(
        ..., min_length=1, description="Available categories to classify into"
    )
    model_type: ModelType = Field(
        ..., description="Type of model to use for classification"
    )
    model_name: str = Field(
        ...,
        min_length=1,
        description="Specific model name (e.g., 'qwen2.5:1.5b', 'gemini-pro')",
    )
    use_search: bool = Field(
        default=False, description="Enable internet search for local models"
    )

    @field_validator("categories")
    @classmethod
    def validate_categories(cls, v):
        if not v:
            raise ValueError("Categories list cannot be empty")

        cleaned = []
        seen = set()
        for cat in v:
            clean_cat = cat.strip()
            if clean_cat and clean_cat.lower() not in seen:
                seen.add(clean_cat.lower())
                cleaned.append(clean_cat)

        if not cleaned:
            raise ValueError("At least one valid category must be provided")
        if len(cleaned) > 20:
            raise ValueError("Maximum 20 categories allowed")

        return cleaned

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, v, info):
        model_type = info.data.get("model_type")

        if not v or not v.strip():
            raise ValueError("Model name cannot be empty")

        config = load_config()
        valid_models_config = config.get("valid_models", {})
        valid_models = {
            ModelType.LOCAL: valid_models_config.get("local", []),
            ModelType.CLOUD: valid_models_config.get("cloud", []),
        }

        if model_type and model_type in valid_models:
            if v not in valid_models[model_type]:
                raise ValueError(
                    f"Invalid model '{v}' for type '{model_type}'. "
                    f"Valid models: {', '.join(valid_models[model_type])}"
                )

        return v.strip()

    @field_validator("use_search")
    @classmethod
    def validate_search_options(cls, v, info):
        model_type = info.data.get("model_type")

        if v and model_type == ModelType.CLOUD:
            raise ValueError("use_search is not supported for cloud models")

        return v


class PaymentDecision(BaseModel):
    action: ActionType
    category: Optional[str] = None
    search_query: Optional[str] = None
    reasoning: str


class PaymentClassification(BaseModel):
    category: str
    reasoning: str
    search_used: bool = False


class ClassificationResult(BaseModel):
    category: str
    reasoning: str
    confidence: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    correlation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    model_used: Optional[str] = None
    processing_time_ms: Optional[float] = None


class LLMProviderType(Enum):
    OLLAMA = "ollama"
    GEMINI = "gemini"
    OPENAI = "openai"


class BaseLLMConfig(BaseModel):

    timeout: PositiveFloat = 30.0
    max_retries: PositiveInt = 3
    max_concurrent_requests: PositiveInt = 10
    max_categories: PositiveInt = 50
    max_payment_text_length: PositiveInt = 10000
    enable_request_logging: bool = True
    enable_response_logging: bool = False


class OllamaConfig(BaseLLMConfig):
    base_url: str = "http://localhost:11434"
    model: str = "qwen2.5:1.5b"
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    google_api_key: Optional[str] = None
    google_search_engine_id: Optional[str] = None


class GeminiConfig(BaseLLMConfig):
    api_key: Optional[str] = None
    model: Literal["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.5-flash"] = (
        "gemini-2.5-flash"
    )
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_output_tokens: PositiveInt = 1024


class OpenAIConfig(BaseLLMConfig):
    api_key: Optional[str] = None
    model: str = "gpt-4o-mini"
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: PositiveInt = 1024
