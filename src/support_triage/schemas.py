from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class Category(str, Enum):
    billing = "billing"
    technical = "technical"
    account = "account"
    legal = "legal"
    security = "security"
    refund_dispute = "refund_dispute"
    other = "other"


class Ticket(BaseModel):
    ticket_id: str
    customer_id: str
    subject: str
    body: str


class TriageResult(BaseModel):
    category: Category
    priority: int = Field(ge=1, le=4, description="1 = urgent, 4 = low")
    sentiment: Literal["calm", "frustrated", "angry"]
    summary: str = Field(description="One sentence, no personal data")


class DraftReply(BaseModel):
    reply: str
    sources: list[str] = Field(description="KB document names used")
    self_score: float = Field(ge=0, le=1, description="Does the reply fully answer using sources?")


class GuardrailResult(BaseModel):
    passed: bool
    reasons: list[str] = Field(default_factory=list)
    pii_types: list[str] = Field(default_factory=list)
    masked_text: str | None = None


class Decision(BaseModel):
    action: Literal["auto_send", "human_review"]
    confidence: float
    reasons: list[str]