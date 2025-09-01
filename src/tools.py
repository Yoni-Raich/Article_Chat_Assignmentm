"""
Tools module for the article analysis agent.

This module provides a comprehensive set of tools for the agent to interact
with the chunk-based vector store, optimized for three query types:
1. Single article queries
2. Multiple article queries  
3. Database-wide queries
"""

from typing import List, Dict, Optional, Tuple
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
        # Single Article Tools
        find_article_by_description,
        get_article_full_content,
        get_article_summary,
        search_within_article,
        
        # Multiple Article Tools
        find_articles_by_topic,
        get_multiple_article_summaries,
        search_across_specific_articles,
        compare_articles_metadata,
        
        # Database-wide Tools
        get_database_overview,
        search_all_content,
        get_articles_by_category,
        get_articles_by_sentiment,
        extract_entities_across_database,
        get_trending_keywords,
    ]

# ==================== SINGLE ARTICLE TOOLS ====================

@tool
def find_article_by_description(description: str) -> Dict:
    """
    Finds a single article based on a description, title, or keywords.
    Returns the best matching article with its ID and metadata.
    Use this when the user is asking about a specific article.
    
    Args:
        description: Description, title, or keywords to identify the article
        
    Returns:
        The best matching article with metadata or error if not found
    """
    if not VECTOR_STORE:
        return {"error": "Vector store not initialized"}
    
    logger.info(f"Finding article by description: '{description}'")
    results = VECTOR_STORE.search_articles(description, k=1)
    
    if not results:
        return {"error": f"No article found matching: '{description}'"}
    
    # Return the complete search result directly
    return results[0]

@tool
def get_article_full_content(article_id: str) -> Dict:
    """
    Retrieves the complete content of an article including all chunks.
    Use this when you need the full text to answer detailed questions about a single article.
    
    Args:
        article_id: The unique ID of the article
        
    Returns:
        Full article content with metadata
    """
    if not VECTOR_STORE:
        return {"error": "Vector store not initialized"}
    
    logger.info(f"Getting full content for article ID: {article_id}")
    article = VECTOR_STORE.get_by_id(article_id)
    
    if not article:
        return {"error": f"Article with ID '{article_id}' not found"}
    
    return article

@tool
def get_article_summary(article_id: str) -> Dict:
    """
    Gets just the summary and key metadata of an article without full content.
    Efficient for quick overview of a single article.
    
    Args:
        article_id: The unique ID of the article
        
    Returns:
        Article summary and metadata
    """
    if not VECTOR_STORE:
        return {"error": "Vector store not initialized"}
    
    logger.info(f"Getting summary for article ID: {article_id}")
    article = VECTOR_STORE.get_by_id(article_id)
    
    if not article:
        return {"error": f"Article with ID '{article_id}' not found"}
    
    # Return article without full_content for efficiency
    summary_article = {k: v for k, v in article.items() if k != "full_content"}
    return summary_article

@tool
def search_within_article(article_id: str, query: str, max_chunks: int = 3) -> List[Dict]:
    """
    Searches for specific information within a single article.
    Returns the most relevant chunks from that article only.
    
    Args:
        article_id: The unique ID of the article to search within
        query: The search query
        max_chunks: Maximum number of chunks to return
        
    Returns:
        List of relevant chunks from the specified article
    """
    if not VECTOR_STORE:
        return {"error": "Vector store not initialized"}
    
    logger.info(f"Searching within article {article_id} for: '{query}'")
    
    # Search chunks with article_id filter
    chunks = VECTOR_STORE.search_chunks(
        query, 
        k=max_chunks, 
        filter_dict={"article_id": article_id}
    )
    
    return chunks

# ==================== MULTIPLE ARTICLE TOOLS ====================

@tool
def find_articles_by_topic(topic: str, max_articles: int = 5) -> List[Dict]:
    """
    Finds multiple articles related to a specific topic or theme.
    Returns article summaries with full content.
    Use this for comparing or analyzing multiple articles.
    
    Args:
        topic: The topic or theme to search for
        max_articles: Maximum number of articles to return
        
    Returns:
        List of article summaries with metadata
    """
    if not VECTOR_STORE:
        return {"error": "Vector store not initialized"}
    
    logger.info(f"Finding articles about topic: '{topic}'")
    # Return search results directly (already have the correct structure)
    return VECTOR_STORE.search_articles(topic, k=max_articles)

@tool
def get_multiple_article_summaries(article_ids: List[str]) -> List[Dict]:
    """
    Gets summaries for multiple specific articles at once.
    Efficient for comparing or extracting data from known articles.
    
    Args:
        article_ids: List of article IDs to retrieve
        
    Returns:
        List of article summaries
    """
    if not VECTOR_STORE:
        return {"error": "Vector store not initialized"}
    
    logger.info(f"Getting summaries for {len(article_ids)} articles")
    
    # Get all articles summaries efficiently (without full content)
    all_articles = VECTOR_STORE.get_all_articles()
    article_lookup = {a["id"]: a for a in all_articles}
    
    # Filter for requested IDs
    summaries = [article_lookup[aid] for aid in article_ids if aid in article_lookup]
    return summaries

@tool
def search_across_specific_articles(article_ids: List[str], query: str, chunks_per_article: int = 2) -> Dict:
    """
    Searches for information across specific articles only.
    Returns relevant chunks from each article for focused comparison.
    
    Args:
        article_ids: List of article IDs to search within
        query: The search query
        chunks_per_article: Maximum chunks to return per article
        
    Returns:
        Dictionary with results grouped by article
    """
    if not VECTOR_STORE:
        return {"error": "Vector store not initialized"}
    
    logger.info(f"Searching across {len(article_ids)} articles for: '{query}'")
    results = {}
    
    for article_id in article_ids:
        chunks = VECTOR_STORE.search_chunks(
            query,
            k=chunks_per_article,
            filter_dict={"article_id": article_id}
        )
        if chunks:
            results[article_id] = {
                "title": chunks[0].get("title"),
                "relevant_chunks": chunks
            }
    
    return results

@tool
def compare_articles_metadata(article_ids: List[str]) -> Dict:
    """
    Compares metadata of multiple articles for analysis.
    Useful for finding patterns, differences, or similarities.
    
    Args:
        article_ids: List of article IDs to compare
        
    Returns:
        Comparison data with categories, sentiments, and key entities
    """
    if not VECTOR_STORE:
        return {"error": "Vector store not initialized"}
    
    logger.info(f"Comparing metadata for {len(article_ids)} articles")
    comparison = {
        "articles": [],
        "categories": set(),
        "sentiments": [],
        "all_entities": set(),
        "all_keywords": set()
    }
    
    for article_id in article_ids:
        article = VECTOR_STORE.get_by_id(article_id)
        if article:
            # Use article data directly without reconstruction
            article_summary = {k: v for k, v in article.items() if k != "full_content"}
            comparison["articles"].append(article_summary)
            comparison["categories"].add(article["category"])
            comparison["sentiments"].append(article["sentiment"])
            comparison["all_entities"].update(article.get("entities", []))
            comparison["all_keywords"].update(article.get("keywords", []))
    
    # Convert sets to lists for JSON serialization
    comparison["categories"] = list(comparison["categories"])
    comparison["all_entities"] = list(comparison["all_entities"])
    comparison["all_keywords"] = list(comparison["all_keywords"])
    
    return comparison

# ==================== DATABASE-WIDE TOOLS ====================

@tool
def get_database_overview() -> Dict:
    """
    Provides a comprehensive overview of the entire article database.
    Use this for understanding the scope and content of the database.
    
    Returns:
        Database statistics and summary information
    """
    if not VECTOR_STORE:
        return {"error": "Vector store not initialized"}
    
    logger.info("Getting database overview")
    all_articles = VECTOR_STORE.get_all_articles()
    
    categories = {}
    sentiments = {"positive": 0, "negative": 0, "neutral": 0}
    total_entities = set()
    total_keywords = set()
    
    for article in all_articles:
        # Count categories
        cat = article.get("category", "uncategorized")
        categories[cat] = categories.get(cat, 0) + 1
        
        # Count sentiments
        sent = article.get("sentiment", "neutral")
        if sent in sentiments:
            sentiments[sent] += 1
            
        # Collect entities and keywords
        total_entities.update(article.get("entities", []))
        total_keywords.update(article.get("keywords", []))
    
    return {
        "total_articles": len(all_articles),
        "categories_distribution": categories,
        "sentiment_distribution": sentiments,
        "unique_entities_count": len(total_entities),
        "unique_keywords_count": len(total_keywords),
        "articles_list": [
            {"id": a["id"], "title": a["title"], "category": a.get("category")}
            for a in all_articles
        ]
    }

@tool
def search_all_content(query: str, max_results: int = 10) -> List[Dict]:
    """
    Searches across the entire database for relevant content.
    Returns both article summaries and relevant chunks.
    
    Args:
        query: The search query
        max_results: Maximum number of results to return
        
    Returns:
        Mixed results from articles and chunks
    """
    if not VECTOR_STORE:
        return {"error": "Vector store not initialized"}
    
    logger.info(f"Searching all content for: '{query}'")
    results = VECTOR_STORE.search_all(query, k=max_results)
    
    # Group results by type for better organization
    organized_results = {
        "articles": [],
        "chunks": []
    }
    
    for result in results:
        if result.get("type") == "article":
            organized_results["articles"].append({
                "id": result["id"],
                "title": result["title"],
                "summary": result["summary"],
                "similarity_score": result["similarity_score"]
            })
        elif result.get("type") == "chunk":
            organized_results["chunks"].append({
                "article_id": result["article_id"],
                "article_title": result["title"],
                "content": result["content"],
                "similarity_score": result["similarity_score"]
            })
    
    return organized_results

@tool
def get_articles_by_category(category: str) -> List[Dict]:
    """
    Retrieves all articles in a specific category.
    Useful for category-based analysis across the database.
    
    Args:
        category: The category to filter by
        
    Returns:
        List of articles in the specified category
    """
    if not VECTOR_STORE:
        return {"error": "Vector store not initialized"}
    
    logger.info(f"Getting articles in category: '{category}'")
    all_articles = VECTOR_STORE.get_all_articles()
    
    # Return complete article summaries for the category
    filtered_articles = [
        a for a in all_articles
        if a.get("category", "").lower() == category.lower()
    ]
    
    return filtered_articles

@tool
def get_articles_by_sentiment(sentiment: str) -> List[Dict]:
    """
    Retrieves all articles with a specific sentiment.
    Options: positive, negative, neutral.
    
    Args:
        sentiment: The sentiment to filter by (positive/negative/neutral)
        
    Returns:
        List of articles with the specified sentiment
    """
    if not VECTOR_STORE:
        return {"error": "Vector store not initialized"}
    
    logger.info(f"Getting articles with sentiment: '{sentiment}'")
    all_articles = VECTOR_STORE.get_all_articles()
    
    # Return complete article summaries for the sentiment
    filtered_articles = [
        a for a in all_articles
        if a.get("sentiment", "").lower() == sentiment.lower()
    ]
    
    return filtered_articles

@tool
def extract_entities_across_database(entity_type: Optional[str] = None) -> Dict:
    """
    Extracts and analyzes entities across the entire database.
    Can filter by entity type if specified.
    
    Args:
        entity_type: Optional entity type to filter (e.g., "person", "organization")
        
    Returns:
        Entity analysis with frequencies and article associations
    """
    if not VECTOR_STORE:
        return {"error": "Vector store not initialized"}
    
    logger.info(f"Extracting entities across database, type: {entity_type}")
    all_articles = VECTOR_STORE.get_all_articles()
    
    entity_map = {}
    
    for article in all_articles:
        entities = article.get("entities", [])
        for entity in entities:
            # Simple filtering if entity_type is specified
            if entity_type and entity_type.lower() not in entity.lower():
                continue
                
            if entity not in entity_map:
                entity_map[entity] = {
                    "count": 0,
                    "articles": []
                }
            entity_map[entity]["count"] += 1
            entity_map[entity]["articles"].append({
                "id": article["id"],
                "title": article["title"]
            })
    
    # Sort by frequency
    sorted_entities = sorted(
        entity_map.items(),
        key=lambda x: x[1]["count"],
        reverse=True
    )
    
    return {
        "total_unique_entities": len(entity_map),
        "top_entities": dict(sorted_entities[:20]),  # Top 20 entities
        "entity_type_filter": entity_type
    }

@tool
def get_trending_keywords() -> Dict:
    """
    Analyzes keywords across the database to identify trending topics.
    Returns the most frequent keywords and their article associations.
    
    Returns:
        Keyword frequency analysis
    """
    if not VECTOR_STORE:
        return {"error": "Vector store not initialized"}
    
    logger.info("Analyzing trending keywords")
    all_articles = VECTOR_STORE.get_all_articles()
    
    keyword_map = {}
    
    for article in all_articles:
        keywords = article.get("keywords", [])
        for keyword in keywords:
            if keyword not in keyword_map:
                keyword_map[keyword] = {
                    "count": 0,
                    "articles": []
                }
            keyword_map[keyword]["count"] += 1
            keyword_map[keyword]["articles"].append({
                "id": article["id"],
                "title": article["title"]
            })
    
    # Sort by frequency
    sorted_keywords = sorted(
        keyword_map.items(),
        key=lambda x: x[1]["count"],
        reverse=True
    )
    
    return {
        "total_unique_keywords": len(keyword_map),
        "trending_keywords": dict(sorted_keywords[:15]),  # Top 15 keywords
        "keyword_cloud": [k for k, _ in sorted_keywords[:30]]  # Top 30 for visualization
    }
