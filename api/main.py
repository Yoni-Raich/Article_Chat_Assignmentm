"""
FastAPI application for Article Chat System.

This module provides REST API endpoints for:
- Chat functionality with the AI agent
- Article ingestion from URLs
- Health checks and status
"""

# Standard library imports
import os
import sys
from contextlib import asynccontextmanager
from typing import Optional
from datetime import datetime

# Third-party imports
import uvicorn
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Get the project root directory and add to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Local imports
from src.agent import ArticleAnalysisAgent
from src.models import QueryRequest, QueryResponse, IngestRequest, IngestResponse, ArticleSource
from src.ingestion import ArticleProcessor
from src.vector_store import VectorStore
from src.cache import QueryCache
from scripts.initialize_articles import ArticleInitializer, ARTICLE_URLS

# Global instances
agent: Optional[ArticleAnalysisAgent] = None
article_processor: Optional[ArticleProcessor] = None
vector_store: Optional[VectorStore] = None
query_cache: Optional[QueryCache] = None


async def initialize_articles_if_needed():
    """
    Initialize articles in the database if it's empty or has very few articles.
    This ensures the system has content to work with on startup.
    """
    global vector_store
    
    if not vector_store:
        print("❌ Vector store not initialized")
        return
    
    # Check current article count
    current_count = len(vector_store.get_all_articles())
    expected_count = len(ARTICLE_URLS)
    
    print(f"📊 Current articles: {current_count}, Expected: {expected_count}")
    
    # If we have less than 80% of expected articles, initialize
    if current_count < (expected_count * 0.8):
        print(f"📥 Initializing article database ({current_count}/{expected_count} articles found)...")
        
        try:
            # Initialize articles using the existing initializer
            initializer = ArticleInitializer(max_workers=2, retry_attempts=1)
            summary = initializer.initialize_all_articles(ARTICLE_URLS)
            
            print(f"✅ Article initialization complete!")
            print(f"   📊 Successful: {summary['successful']}")
            print(f"   ❌ Failed: {summary['failed']}")
            print(f"   ⏭️  Skipped: {summary['skipped']}")
            
        except Exception as e:
            print(f"❌ Failed to initialize articles: {e}")
            # Don't fail the startup, just log the error
    else:
        print(f"✅ Article database already populated ({current_count} articles)")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup and shutdown events for the FastAPI application.
    """
    # Startup
    global agent, article_processor, vector_store, query_cache

    print("🚀 Starting Article Chat API...")

    try:
        # Initialize query cache first
        print("🔧 Initializing query cache...")
        query_cache = QueryCache(max_size=200, ttl_seconds=3600)  # 1 hour TTL

        # Initialize components
        print("🔧 Initializing vector store...")
        vector_store = VectorStore()

        print("🔧 Initializing article processor...")
        article_processor = ArticleProcessor()

        print("🔧 Initializing AI agent...")
        agent = ArticleAnalysisAgent()

        # Auto-initialize articles if needed
        print("🔧 Checking article database...")
        await initialize_articles_if_needed()

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

# Mount static files for web UI
web_dir = os.path.join(project_root, "web")
if os.path.exists(web_dir):
    app.mount("/static", StaticFiles(directory=web_dir), name="static")


@app.get("/")
async def root():
    """
    Root endpoint - serves the web UI.
    """
    web_file = os.path.join(project_root, "web", "index.html")
    if os.path.exists(web_file):
        return FileResponse(web_file)
    return {
        "message": "Article Chat API is running!",
        "version": "1.0.0",
        "status": "healthy",
        "docs": "/docs"
    }


@app.get("/stats")
async def get_stats():
    """
    Get system statistics.
    """
    try:
        # Import here to avoid circular imports during startup
        from src.vector_store import VectorStore

        vector_store = VectorStore()
        articles = vector_store.get_all_articles()

        return {
            "article_count": len(articles),
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
    except Exception as e:
        print(f"Error getting stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve system statistics"
        )


@app.get("/health")
async def health_check():
    """
    Health check endpoint for monitoring.
    """
    return {
        "status": "healthy",
        "agent_ready": agent is not None,
        "vector_store_ready": vector_store is not None,
        "processor_ready": article_processor is not None,
        "cache_ready": query_cache is not None
    }


@app.post("/chat", response_model=QueryResponse)
async def chat(request: QueryRequest):
    """
    Chat endpoint - send a query to the AI agent and get a response.

    This endpoint processes user questions about articles using the AI agent
    with access to the vector database and various analysis tools.

    Features automatic caching of responses for repeated queries.
    """
    global agent, vector_store, query_cache

    if not agent:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI agent not initialized"
        )

    try:
        # Check cache first for repeated queries
        if query_cache:
            cached_response = query_cache.get(request.query, request.max_articles)
            if cached_response:
                print(f"🚀 Cache HIT for query: '{request.query[:50]}...'")
                return QueryResponse(**cached_response)
            else:
                print(f"🔍 Cache MISS for query: '{request.query[:50]}...'")

        # Get response from agent (cache miss or no cache)
        answer = agent.query(request.query)

        # Get tools used in the query
        tools_used = agent.get_used_tools()

        # Get sources from vector store with similarity search
        sources = []
        if vector_store:
            # Search for relevant articles
            search_results = vector_store.search(request.query, k=request.max_articles)

            for result in search_results:
                article_source = ArticleSource(
                    title=result.get('title', 'Unknown Title'),
                    url=result.get('url', 'Unknown URL'),
                    relevance_score=result.get('similarity_score', 0.0)
                )
                sources.append(article_source)

        # Prepare response data
        response_data = {
            "response": answer,
            "sources": [source.dict() for source in sources],
            "confidence": 1.0,
            "tools_used": tools_used
        }

        # Cache the response for future identical queries
        if query_cache:
            query_cache.set(request.query, response_data, request.max_articles)
            print(f"💾 Response cached for query: '{request.query[:50]}...'")

        return QueryResponse(**response_data)

    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing query: {str(e)}"
        )


@app.get("/cache/stats")
async def get_cache_stats():
    """
    Get cache statistics and performance metrics.
    """
    global query_cache

    if not query_cache:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Cache not initialized"
        )

    try:
        return query_cache.stats()
    except Exception as e:
        print(f"Error getting cache stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve cache statistics"
        )


@app.post("/cache/clear")
async def clear_cache():
    """
    Clear all cached query responses.
    """
    global query_cache

    if not query_cache:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Cache not initialized"
        )

    try:
        query_cache.clear()
        return {"message": "Cache cleared successfully"}
    except Exception as e:
        print(f"Error clearing cache: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to clear cache"
        )


@app.post("/cache/cleanup")
async def cleanup_cache():
    """
    Remove expired entries from cache.
    """
    global query_cache

    if not query_cache:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Cache not initialized"
        )

    try:
        removed_count = query_cache.cleanup_expired()
        return {
            "message": f"Cache cleanup completed",
            "expired_entries_removed": removed_count
        }
    except Exception as e:
        print(f"Error cleaning up cache: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to cleanup cache"
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
