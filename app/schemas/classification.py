from enum import Enum
from typing import Optional

from pydantic import BaseModel


class ActionType(str, Enum):
    CLASSIFY = "classify"
    SEARCH = "search"


class PaymentData(BaseModel):
    payment_text: str


class PaymentDecision(BaseModel):
    action: ActionType
    category: Optional[str] = None
    search_query: Optional[str] = None
    reasoning: str


class PaymentClassification(BaseModel):
    category: str
    reasoning: str
    search_used: bool = False
