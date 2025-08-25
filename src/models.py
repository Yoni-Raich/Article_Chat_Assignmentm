from pydantic import BaseModel, Field
from typing import List, Optional, Annotated, Sequence
from typing_extensions import TypedDict
from datetime import datetime
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class ArticleMetadata(BaseModel):
    """Structured metadata for LLM extraction"""
    summary: str = Field(description="2-3 sentence summary of the article")
    keywords: List[str] = Field(description="5-7 key topics from the article")
    entities: List[str] = Field(description="5-10 important named entities (people, organizations, locations, products)", default_factory=list)
    sentiment: float = Field(description="Sentiment score between -1 (negative) and 1 (positive)", ge=-1, le=1)
    category: str = Field(description="Article category")


class Article(BaseModel):
    """Article data model with metadata"""
    id: str
    url: str
    title: str
    content: str
    metadata: ArticleMetadata
    entities: List[str] = Field(default_factory=list)
    processed_at: datetime = Field(default_factory=datetime.now)

class QueryRequest(BaseModel):
    """User query request"""
    query: str = Field(description="User's question or request")


class QueryResponse(BaseModel):
    """Response to user query"""
    answer: str = Field(description="AI agent's response")
    sources: List[str] = Field(default_factory=list, description="URLs of articles used in response")
    confidence: float = Field(default=1.0, description="Confidence score", ge=0, le=1)
    tools_used: List[str] = Field(default_factory=list, description="Tools used by agent")


class IngestRequest(BaseModel):
    """Article ingestion request"""
    url: str = Field(description="URL of article to ingest")


class IngestResponse(BaseModel):
    """Article ingestion response"""
    success: bool = Field(description="Whether ingestion was successful")
    message: str = Field(description="Status message")
    article_id: Optional[str] = Field(default=None, description="ID of ingested article")
    title: Optional[str] = Field(default=None, description="Title of ingested article")


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(description="Error type")
    message: str = Field(description="Error message")
    detail: Optional[str] = Field(default=None, description="Additional error details")


# State definition with message history for LangGraph
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    current_query: str
    final_answer: str
    sources: List[str]
