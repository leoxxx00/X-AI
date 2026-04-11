from typing import TypedDict, Any
from pydantic import BaseModel
from typing import Optional

class QAItem(BaseModel):
    question: str
    answer: str
    context: Optional[str] = None
    source: Optional[str] = None

class FinalResult(TypedDict):
    summary: str
    capacity: dict[str, Any]
    metrics: dict[str, Any]
    accepted_pairs: list[dict[str, Any]]
    artifacts: dict[str, Any]