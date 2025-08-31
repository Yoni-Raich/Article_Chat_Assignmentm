"""
Tools module for the article analysis agent.

This module provides a focused set of tools for the agent to interact
with the chunk-based vector store.
"""

from typing import List, Dict, Optional
from langchain.tools import tool
from .vector_store import VectorStore
from .logger import logger

# Global VectorStore instance
VECTOR_STORE: Optional[VectorStore] = None

def init_tools(vector_store_instance: VectorStore = None):
    """Initialize tools with the VectorStore instance."""
    global VECTOR_STORE
    VECTOR_STORE = vector_store_instance or VectorStore()
    logger.info("Tools initialized with VectorStore.")

def get_list_of_tools():
    """Get a list of all available tools for the agent."""
    return [
        search_article_chunks,
        get_article_details_by_id,
        list_all_articles,
        get_chunks_for_article,
    ]

@tool
def search_article_chunks(query: str, max_results: int = 5) -> List[Dict]:
    """
    Searches for the most relevant text chunks from articles based on a query.
    This is the primary tool for finding specific information within articles.
    Returns a list of chunks, each with its content and parent article info.
    """
    if not VECTOR_STORE:
        return {"error": "Vector store not initialized"}
    
    logger.info(f"Searching for chunks with query: '{query}'")
    return VECTOR_STORE.search_chunks(query, k=max_results)

@tool
def get_article_details_by_id(article_id: str) -> Dict:
    """
    Retrieves the full details of a single article by its unique ID.
    Use this to get the summary, metadata, and full content of an article
    after finding a relevant chunk.
    """
    if not VECTOR_STORE:
        return {"error": "Vector store not initialized"}
    
    logger.info(f"Getting article details for ID: {article_id}")
    article = VECTOR_STORE.get_by_id(article_id)
    if not article:
        return {"error": f"Article with ID '{article_id}' not found."}
    return article

@tool
def list_all_articles() -> List[Dict]:
    """
    Lists all articles currently in the database.
    Returns a list of articles with their title, ID, and summary.
    Useful for getting a general overview of the available content.
    """
    if not VECTOR_STORE:
        return {"error": "Vector store not initialized"}
    
    logger.info("Listing all articles.")
    articles = VECTOR_STORE.get_all_articles()
    # Return a lighter version of the article data
    return [
        {
            "id": a.get("id"),
            "title": a.get("title"),
            "summary": a.get("summary"),
            "category": a.get("category"),
        }
        for a in articles
    ]

@tool
def get_chunks_for_article(article_id: str) -> List[Dict]:
    """
    Retrieves all text chunks for a specific article, in order.
    Useful when the agent needs to read or analyze the entire content of a
    single article.
    """
    if not VECTOR_STORE:
        return {"error": "Vector store not initialized"}
        
    logger.info(f"Getting all chunks for article ID: {article_id}")
    return VECTOR_STORE.get_chunks_by_article_id(article_id)
