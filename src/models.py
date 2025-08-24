from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class ArticleMetadata(BaseModel):
    """Structured metadata for LLM extraction"""
    summary: str = Field(description="2-3 sentence summary of the article")
    keywords: List[str] = Field(description="5-7 key topics from the article")
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
    query: str

class QueryResponse(BaseModel):
    """Response to user query"""
    answer: str
    sources: List[str] = []
    confidence: float = 1.0