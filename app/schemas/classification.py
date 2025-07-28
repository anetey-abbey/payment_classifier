from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class ActionType(str, Enum):
    CLASSIFY = "classify"
    SEARCH = "search"


class ClassificationRequest(BaseModel):
    payment_text: str = Field(
        ..., min_length=1, description="Payment description to classify"
    )
    categories: List[str] = Field(
        ..., min_length=1, description="Available categories to classify into"
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


class PaymentDecision(BaseModel):
    action: ActionType
    category: Optional[str] = None
    search_query: Optional[str] = None
    reasoning: str


class PaymentClassification(BaseModel):
    category: str
    reasoning: str
    search_used: bool = False
