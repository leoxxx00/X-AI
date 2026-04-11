from pydantic import BaseModel, HttpUrl, Field
from typing import Optional, Dict, Any, List


class EvaluatorRequest(BaseModel):
    url: HttpUrl
    strictness: str = Field(default="Standard", pattern="^(Strict|Standard|Lenient)$")


class EvaluatorResponse(BaseModel):
    url: str
    title: str
    status: str
    training_grade_pairs: int
    raw_extractable_pairs: int
    predicted_pairs: int
    predicted_min: int
    predicted_max: int
    confidence: float
    method: str
    metrics: Dict[str, Any]
    notes: List[str]
    elapsed_seconds: float
    llm_used: bool = False
    llm_error: Optional[str] = None
    llm_details: Optional[Dict[str, Any]] = None
    quality_view: Optional[Dict[str, Any]] = None