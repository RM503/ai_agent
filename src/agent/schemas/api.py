# Store schemas for data validation for FastAPI
from uuid import UUID

from pydantic import BaseModel

class ChatRequest(BaseModel):
    """Data format for request body from UI"""
    session_id: str | UUID
    message: str
    file_id: str | None=None

class ChatResponse(BaseModel):
    """Data format for response body from UI"""
    session_id: str | UUID
    route: str # tells the frontend which agent branch handled the request
    response: str
    transcription_text: str | None=None
    summary_text: str | None=None
    analysis_text: str | None=None
