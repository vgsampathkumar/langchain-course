from typing import List
from pydantic import BaseModel, Field

class Source(BaseModel):
    """Schema for a source of information."""
    url: str = Field(description="The URL of the source.")

class AgentResponse(BaseModel):
    """Schema for a source of information."""
    answer: str = Field(description="The agent answer to the query.")
    sources: List[Source] = Field(
        default_factory=list, 
        description="The list of sources used to answer the query."
        )