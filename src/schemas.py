from pydantic import BaseModel, Field
from typing import Optional, List

class ChatMessage(BaseModel):
    """Tek bir mesajin yapisini belirten sema."""
    role: str = Field(..., description="Mesaji gonderen rol: 'user', 'assistant' veya 'system'")
    content: str = Field(..., description="Mesaj icerigi")

class AgentResponse(BaseModel):
    """Ajanin dondurecegi yapilandirilmis yanit semasi."""
    agent_name: str
    reply: str
    status_code: int = 200
    sources: Optional[List[str]] = Field(default_factory=list, description="RAG kaynaklari")