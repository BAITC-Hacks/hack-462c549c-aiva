from typing import Literal

from pydantic import BaseModel, Field


EvidenceType = Literal["FACT", "INFERENCE", "RISK_FLAG"]


class Fragment(BaseModel):
    fragment_id: str
    document_name: str
    page_number: int
    clause_id: str | None = None
    raw_text: str
    cleaned_text: str


class FunctionRecord(BaseModel):
    function_id: str
    unit_name: str | None = None
    text: str
    fragment_id: str
    clause_id: str | None = None


class ComparisonRow(BaseModel):
    before: FunctionRecord | None = None
    after: FunctionRecord | None = None
    status: str
    evidence_type: EvidenceType
    explanation: str
    confidence: float | None = Field(default=None, ge=0, le=1)


class ParsedDocument(BaseModel):
    document_name: str
    fragments: list[Fragment]
    clauses: dict[str, Fragment]
    units: list[dict]
    functions: list[FunctionRecord]
