# Store schemas for data validation for FastAPI
from typing import Optional
from uuid import UUID

from pydantic import BaseModel

class ChatRequest(BaseModel):
    """Data format for request body from UI"""
    session_id: str | UUID
    message: str
    file_id: Optional[str]=None

class ChatResponse(BaseModel):
    """Data format for response body from UI"""
    session_id: str | UUID
    route: str # tells the frontend which agent branch handled the request
    transcription_text: Optional[str]=None
    summary_text: Optional[str]=None
    analysis_text: Optional[str]=None
