from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator

from app.utils.config import load_config


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
    use_vector_db: bool = Field(
        default=False, description="Enable vector database search for local models"
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

    @field_validator("use_search", "use_vector_db")
    @classmethod
    def validate_search_options(cls, v, info):
        model_type = info.data.get("model_type")
        field_name = info.field_name

        if v and model_type == ModelType.CLOUD:
            raise ValueError(f"{field_name} is not supported for cloud models")

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
