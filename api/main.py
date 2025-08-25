"""
FastAPI main application for Article Chat System.

This module provides REST API endpoints for:
- Chat functionality with the AI agent
- Article ingestion from URLs
- Health checks and status
"""

import os
import sys
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import Optional
import uvicorn

# Get the project root directory and add to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.agent import ArticleAnalysisAgent
from src.models import QueryRequest, QueryResponse, IngestRequest, IngestResponse, ErrorResponse
from src.ingestion import ArticleProcessor
from src.vector_store import VectorStore

# Global instances
agent: Optional[ArticleAnalysisAgent] = None
article_processor: Optional[ArticleProcessor] = None
vector_store: Optional[VectorStore] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup and shutdown events for the FastAPI application.
    """
    # Startup
    global agent, article_processor, vector_store
    
    print("🚀 Starting Article Chat API...")
    
    try:
        # Initialize components
        print("🔧 Initializing vector store...")
        vector_store = VectorStore()
        
        print("🔧 Initializing article processor...")
        article_processor = ArticleProcessor()
        
        print("🔧 Initializing AI agent...")
        agent = ArticleAnalysisAgent()
        
        print("✅ All components initialized successfully!")
        
    except Exception as e:
        print(f"❌ Failed to initialize application: {e}")
        raise
    
    yield
    
    # Shutdown
    print("🛑 Shutting down Article Chat API...")


# Create FastAPI app with lifespan management
app = FastAPI(
    title="Article Chat API",
    description="AI-powered article analysis and chat system",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """
    Root endpoint - API health check.
    """
    return {
        "message": "Article Chat API is running!",
        "version": "1.0.0",
        "status": "healthy",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """
    Health check endpoint for monitoring.
    """
    return {
        "status": "healthy",
        "agent_ready": agent is not None,
        "vector_store_ready": vector_store is not None,
        "processor_ready": article_processor is not None
    }


@app.post("/chat", response_model=QueryResponse)
async def chat(request: QueryRequest):
    """
    Chat endpoint - send a query to the AI agent and get a response.
    
    This endpoint processes user questions about articles using the AI agent
    with access to the vector database and various analysis tools.
    """
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI agent not initialized"
        )
    
    try:
        # Get response from agent
        answer = agent.query(request.query)
        
        # Get tools used in the query
        tools_used = agent.get_used_tools()
        
        return QueryResponse(
            answer=answer,
            tools_used=tools_used,
            confidence=1.0,
            sources=[]  # Could be enhanced to extract sources from tools
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing query: {str(e)}"
        )


@app.post("/ingest", response_model=IngestResponse)
async def ingest_article(request: IngestRequest):
    """
    Article ingestion endpoint - add a new article by URL.
    
    This endpoint fetches an article from the provided URL, processes it
    with AI to extract metadata, and adds it to the vector database.
    """
    if not article_processor or not vector_store:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Article processor or vector store not initialized"
        )
    
    try:
        # Check if article already exists
        article_id = request.url.replace("https://", "").replace("/", "_")
        if vector_store.article_exists(article_id):
            return IngestResponse(
                success=False,
                message="Article already exists in database",
                article_id=article_id
            )
        
        # Process the article
        article = article_processor.process_url(request.url)
        
        if not article:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to fetch or process article from URL"
            )
        
        # Add to vector store
        success = vector_store.add_article(article)
        
        if success:
            return IngestResponse(
                success=True,
                message="Article successfully processed and added to database",
                article_id=article.id,
                title=article.title
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to add article to vector database"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error ingesting article: {str(e)}"
        )


if __name__ == "__main__":
    # For development - run with: python api/main.py
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )