"""
Tools module for the article analysis agent.

This module provides various tools for searching, analyzing, and processing articles
including vector search, sentiment analysis, content analysis, and more.
"""

# Standard library imports
from typing import List, Dict, Optional, Any
from datetime import datetime
import json

# Third-party imports
from langchain.tools import tool
from pydantic import BaseModel, Field

# Local imports
from .vector_store import VectorStore
from .ingestion import ArticleProcessor
from .logger import logger

# Global instances
VECTOR_STORE: Optional[VectorStore] = None
PROCESSOR: Optional[ArticleProcessor] = None


def init_tools(vector_store_instance: VectorStore = None, ap: ArticleProcessor = None):
    """Initialize tools with dependencies"""
    global VECTOR_STORE, PROCESSOR
    VECTOR_STORE = vector_store_instance or VectorStore()
    PROCESSOR = ap or ArticleProcessor()
    logger.info("Tools initialized with VectorStore and ArticleProcessor")


def get_list_of_tools():
    """Get a list of all available tools for the agent."""
    return [
        # Search and retrieval tools
        search_articles_by_query,
        get_article_full_content,
        get_article_by_id,
        get_multiple_articles,
        list_all_articles,
        
        # Sentiment analysis tools
        get_articles_by_sentiment,
        analyze_sentiment_for_articles,
        compare_article_sentiments,
        
        # Category and topic tools
        get_articles_by_category,
        get_articles_by_keywords,
        get_trending_topics,
        
        # Entity analysis tools
        search_articles_by_entities,
        get_most_common_entities,
        analyze_entity_across_articles,
        
        # Comparison and similarity tools
        compare_articles,
        find_similar_articles,
        
        # Statistics and overview tools
        get_articles_statistics,
        get_category_distribution,
    ]


# ============= Search and Retrieval Tools =============

@tool
def search_articles_by_query(
    query: str, 
    max_results: int = 5,
    include_full_content: bool = False
) -> List[Dict]:
    """
    Search articles using semantic similarity.
    Use this as the primary search tool for general queries.
    
    Args:
        query: Search query (keywords, phrases, or questions)
        max_results: Maximum results to return (default: 5, max: 20)
        include_full_content: Whether to include full article content (default: False)
    
    Returns:
        List of articles with metadata and relevance scores
    """
    if not VECTOR_STORE:
        return []
    
    max_results = min(max_results, 20)  # Cap at 20 results
    search_results = VECTOR_STORE.search(query, k=max_results)
    
    formatted_results = []
    for result in search_results:
        article_data = {
            "id": result["id"],
            "title": result["title"],
            "url": result["url"],
            "summary": result["summary"],
            "category": result["category"],
            "keywords": result["keywords"],
            "entities": result["entities"],
            "sentiment": result["sentiment"],
            "relevance_score": round(result["similarity_score"], 3)
        }
        
        if include_full_content:
            article_data["content"] = result.get("full_content", "")
            
        formatted_results.append(article_data)
    
    logger.info(f"Search query '{query[:50]}...' returned {len(formatted_results)} results")
    return formatted_results


@tool
def get_article_full_content(article_url: str) -> Dict:
    """
    Get complete article content and metadata by URL.
    Use when you need the full text of a specific article.
    
    Args:
        article_url: Complete URL of the article
    
    Returns:
        Dict with full article content and all metadata
    """
    if not VECTOR_STORE:
        return {"error": "Vector store not initialized"}
    
    article_id = article_url.replace("https://", "").replace("/", "_")
    article = VECTOR_STORE.get_by_id(article_id)
    
    if article:
        return {
            "id": article["id"],
            "url": article["url"],
            "title": article["title"],
            "content": article["full_content"],
            "summary": article["summary"],
            "category": article["category"],
            "keywords": article["keywords"],
            "entities": article["entities"],
            "sentiment": article["sentiment"]
        }
    
    return {"error": f"Article not found: {article_url}"}


@tool
def get_article_by_id(article_id: str) -> Dict:
    """
    Get article by its ID (useful for follow-up queries).
    
    Args:
        article_id: Article ID from previous search results
    
    Returns:
        Complete article data
    """
    if not VECTOR_STORE:
        return {"error": "Vector store not initialized"}
    
    article = VECTOR_STORE.get_by_id(article_id)
    return article if article else {"error": f"Article not found: {article_id}"}


@tool
def get_multiple_articles(
    article_ids: List[str],
    include_content: bool = False
) -> List[Dict]:
    """
    Retrieve multiple articles in batch.
    Efficient for analyzing multiple articles at once.
    
    Args:
        article_ids: List of article IDs
        include_content: Whether to include full content
    
    Returns:
        List of article data
    """
    if not VECTOR_STORE:
        return []
    
    articles = []
    for article_id in article_ids:
        article = VECTOR_STORE.get_by_id(article_id)
        if article:
            if not include_content:
                article.pop("full_content", None)
            articles.append(article)
    
    logger.info(f"Retrieved {len(articles)}/{len(article_ids)} articles")
    return articles


@tool
def list_all_articles(
    output_format: str = "summary",
    sort_by: str = "title"
) -> Any:
    """
    List all articles in the database.
    
    Args:
        output_format: Format of output ("summary", "titles", "urls", "detailed")
        sort_by: Field to sort by ("title", "category", "sentiment")
    
    Returns:
        List of articles in requested format
    """
    if not VECTOR_STORE:
        return []
    
    articles = VECTOR_STORE.get_all_articles()
    
    if not articles:
        return []
    
    # Sort articles
    if sort_by == "sentiment":
        articles.sort(key=lambda x: x.get("sentiment", 0), reverse=True)
    elif sort_by == "category":
        articles.sort(key=lambda x: x.get("category", ""))
    else:  # default to title
        articles.sort(key=lambda x: x.get("title", ""))
    
    # Format output
    if output_format == "titles":
        return [a["title"] for a in articles]
    elif output_format == "urls":
        return [a["url"] for a in articles]
    elif output_format == "detailed":
        return articles
    else:  # summary format
        return [{
            "title": a["title"],
            "url": a["url"],
            "category": a["category"],
            "sentiment": round(a["sentiment"], 2)
        } for a in articles]


# ============= Sentiment Analysis Tools =============

@tool
def get_articles_by_sentiment(
    sentiment_type: str,
    threshold: float = 0.2,
    limit: int = 10
) -> List[Dict]:
    """
    Filter articles by sentiment.
    
    Args:
        sentiment_type: "positive", "negative", or "neutral"
        threshold: Sentiment threshold (default: 0.2)
        limit: Maximum articles to return
    
    Returns:
        List of articles matching sentiment criteria
    """
    if not VECTOR_STORE:
        return []
    
    articles = VECTOR_STORE.get_all_articles()
    filtered = []
    
    for article in articles:
        sentiment = article.get("sentiment", 0)
        
        if sentiment_type == "positive" and sentiment > threshold:
            filtered.append(article)
        elif sentiment_type == "negative" and sentiment < -threshold:
            filtered.append(article)
        elif sentiment_type == "neutral" and -threshold <= sentiment <= threshold:
            filtered.append(article)
    
    # Sort by sentiment strength
    filtered.sort(key=lambda x: abs(x.get("sentiment", 0)), reverse=True)
    
    return filtered[:limit]


@tool
def analyze_sentiment_for_articles(
    article_urls: List[str]
) -> Dict:
    """
    Analyze sentiment across multiple articles.
    
    Args:
        article_urls: List of article URLs to analyze
    
    Returns:
        Comprehensive sentiment analysis
    """
    if not VECTOR_STORE:
        return {"error": "Vector store not initialized"}
    
    sentiments = []
    article_sentiments = []
    
    for url in article_urls:
        article_id = url.replace("https://", "").replace("/", "_")
        article = VECTOR_STORE.get_by_id(article_id)
        
        if article:
            sentiment = article["sentiment"]
            sentiments.append(sentiment)
            article_sentiments.append({
                "title": article["title"][:60] + "...",
                "sentiment": round(sentiment, 3),
                "interpretation": _interpret_sentiment(sentiment)
            })
    
    if not sentiments:
        return {"error": "No articles found"}
    
    avg_sentiment = sum(sentiments) / len(sentiments)
    
    return {
        "average_sentiment": round(avg_sentiment, 3),
        "overall_interpretation": _interpret_sentiment(avg_sentiment),
        "sentiment_range": {
            "most_positive": round(max(sentiments), 3),
            "most_negative": round(min(sentiments), 3),
            "spread": round(max(sentiments) - min(sentiments), 3)
        },
        "distribution": {
            "positive": len([s for s in sentiments if s > 0.2]),
            "neutral": len([s for s in sentiments if -0.2 <= s <= 0.2]),
            "negative": len([s for s in sentiments if s < -0.2])
        },
        "articles": article_sentiments
    }


@tool
def compare_article_sentiments(
    group1_urls: List[str],
    group2_urls: List[str],
    group1_name: str = "Group 1",
    group2_name: str = "Group 2"
) -> Dict:
    """
    Compare sentiment between two groups of articles.
    
    Args:
        group1_urls: First group of article URLs
        group2_urls: Second group of article URLs
        group1_name: Name for first group
        group2_name: Name for second group
    
    Returns:
        Comparative sentiment analysis
    """
    if not VECTOR_STORE:
        return {"error": "Vector store not initialized"}
    
    def get_group_sentiment(urls):
        sentiments = []
        for url in urls:
            article_id = url.replace("https://", "").replace("/", "_")
            article = VECTOR_STORE.get_by_id(article_id)
            if article:
                sentiments.append(article["sentiment"])
        return sentiments
    
    group1_sentiments = get_group_sentiment(group1_urls)
    group2_sentiments = get_group_sentiment(group2_urls)
    
    if not group1_sentiments or not group2_sentiments:
        return {"error": "One or both groups have no valid articles"}
    
    avg1 = sum(group1_sentiments) / len(group1_sentiments)
    avg2 = sum(group2_sentiments) / len(group2_sentiments)
    
    return {
        group1_name: {
            "average_sentiment": round(avg1, 3),
            "interpretation": _interpret_sentiment(avg1),
            "article_count": len(group1_sentiments)
        },
        group2_name: {
            "average_sentiment": round(avg2, 3),
            "interpretation": _interpret_sentiment(avg2),
            "article_count": len(group2_sentiments)
        },
        "comparison": {
            "difference": round(avg1 - avg2, 3),
            "more_positive": group1_name if avg1 > avg2 else group2_name,
            "significant_difference": abs(avg1 - avg2) > 0.3
        }
    }


# ============= Category and Topic Tools =============

@tool
def get_articles_by_category(
    category: str,
    include_summary: bool = True
) -> List[Dict]:
    """
    Get all articles in a specific category.
    
    Args:
        category: Category name (technology, business, politics, other)
        include_summary: Whether to include article summaries
    
    Returns:
        List of articles in the category
    """
    if not VECTOR_STORE:
        return []
    
    all_articles = VECTOR_STORE.get_all_articles()
    filtered = [
        a for a in all_articles 
        if a.get("category", "").lower() == category.lower()
    ]
    
    if not include_summary:
        for article in filtered:
            article.pop("summary", None)
    
    return filtered


@tool
def get_articles_by_keywords(
    keywords: List[str],
    match_all: bool = False,
    max_results: int = 10
) -> List[Dict]:
    """
    Search articles by specific keywords.
    
    Args:
        keywords: List of keywords to search for
        match_all: If True, article must contain all keywords
        max_results: Maximum results to return
    
    Returns:
        List of matching articles with keyword highlights
    """
    if not VECTOR_STORE:
        return []
    
    all_articles = VECTOR_STORE.get_all_articles()
    matches = []
    
    keywords_lower = [k.lower() for k in keywords]
    
    for article in all_articles:
        article_keywords = [k.lower() for k in article.get("keywords", [])]
        
        if match_all:
            if all(kw in article_keywords for kw in keywords_lower):
                matches.append(article)
        else:
            if any(kw in article_keywords for kw in keywords_lower):
                matching_keywords = [
                    kw for kw in keywords 
                    if kw.lower() in article_keywords
                ]
                article["matched_keywords"] = matching_keywords
                matches.append(article)
    
    return matches[:max_results]


@tool
def get_trending_topics(
    top_n: int = 10,
    min_occurrences: int = 2
) -> Dict:
    """
    Identify most frequently discussed topics across all articles.
    
    Args:
        top_n: Number of top topics to return
        min_occurrences: Minimum occurrences to be considered trending
    
    Returns:
        Dict with trending topics and their statistics
    """
    if not VECTOR_STORE:
        return {"error": "Vector store not initialized"}
    
    all_articles = VECTOR_STORE.get_all_articles()
    
    if not all_articles:
        return {"error": "No articles in database"}
    
    # Count keyword occurrences
    keyword_counts = {}
    keyword_articles = {}
    
    for article in all_articles:
        for keyword in article.get("keywords", []):
            keyword_lower = keyword.lower()
            keyword_counts[keyword_lower] = keyword_counts.get(keyword_lower, 0) + 1
            
            if keyword_lower not in keyword_articles:
                keyword_articles[keyword_lower] = []
            keyword_articles[keyword_lower].append(article["title"][:50])
    
    # Filter by minimum occurrences and sort
    trending = [
        (kw, count) for kw, count in keyword_counts.items() 
        if count >= min_occurrences
    ]
    trending.sort(key=lambda x: x[1], reverse=True)
    
    return {
        "total_articles": len(all_articles),
        "trending_topics": [
            {
                "topic": topic,
                "occurrences": count,
                "percentage": round(count / len(all_articles) * 100, 1),
                "sample_articles": keyword_articles[topic][:3]
            }
            for topic, count in trending[:top_n]
        ]
    }


# ============= Entity Analysis Tools =============

@tool
def search_articles_by_entities(
    entities: List[str],
    match_all: bool = False,
    max_results: int = 10
) -> List[Dict]:
    """
    Find articles mentioning specific entities.
    
    Args:
        entities: List of entity names to search for
        match_all: If True, article must mention all entities
        max_results: Maximum results to return
    
    Returns:
        List of articles mentioning the entities
    """
    if not VECTOR_STORE:
        return []
    
    all_articles = VECTOR_STORE.get_all_articles()
    matches = []
    
    entities_lower = [e.lower() for e in entities]
    
    for article in all_articles:
        article_entities = [e.lower() for e in article.get("entities", [])]
        
        if match_all:
            if all(
                any(entity in art_entity for art_entity in article_entities)
                for entity in entities_lower
            ):
                matches.append(article)
        else:
            matching_entities = []
            for entity in entities:
                if any(entity.lower() in art_entity for art_entity in article_entities):
                    matching_entities.append(entity)
            
            if matching_entities:
                article["matched_entities"] = matching_entities
                matches.append(article)
    
    return matches[:max_results]


@tool
def get_most_common_entities(
    entity_type: str = "all",
    top_n: int = 10
) -> Dict:
    """
    Get the most commonly discussed entities across all articles.
    
    Args:
        entity_type: Filter by type ("all", "people", "organizations", "locations")
        top_n: Number of top entities to return
    
    Returns:
        Dict with most common entities and their statistics
    """
    if not VECTOR_STORE:
        return {"error": "Vector store not initialized"}
    
    articles = VECTOR_STORE.get_all_articles()
    
    if not articles:
        return {"total_articles": 0, "entities": []}
    
    # Count entity occurrences
    entity_counts = {}
    entity_sentiments = {}
    
    for article in articles:
        entities = article.get("entities", [])
        sentiment = article.get("sentiment", 0)
        
        for entity in entities:
            # Apply type filtering
            if entity_type != "all" and not _matches_entity_type(entity, entity_type):
                continue
            
            entity_counts[entity] = entity_counts.get(entity, 0) + 1
            
            if entity not in entity_sentiments:
                entity_sentiments[entity] = []
            entity_sentiments[entity].append(sentiment)
    
    # Sort by count
    sorted_entities = sorted(
        entity_counts.items(),
        key=lambda x: x[1],
        reverse=True
    )[:top_n]
    
    return {
        "total_articles": len(articles),
        "entity_type_filter": entity_type,
        "most_common_entities": [
            {
                "entity": entity,
                "occurrences": count,
                "percentage": round(count / len(articles) * 100, 1),
                "average_sentiment": round(
                    sum(entity_sentiments[entity]) / len(entity_sentiments[entity]), 3
                )
            }
            for entity, count in sorted_entities
        ]
    }


@tool
def analyze_entity_across_articles(
    entity_name: str,
    include_context: bool = False
) -> Dict:
    """
    Comprehensive analysis of how an entity is discussed across articles.
    
    Args:
        entity_name: Name of the entity to analyze
        include_context: Whether to include context snippets
    
    Returns:
        Detailed entity analysis
    """
    if not VECTOR_STORE:
        return {"error": "Vector store not initialized"}
    
    articles = VECTOR_STORE.get_all_articles()
    matching_articles = []
    
    for article in articles:
        entities = article.get("entities", [])
        if any(entity_name.lower() in entity.lower() for entity in entities):
            article_info = {
                "title": article["title"],
                "url": article["url"],
                "category": article["category"],
                "sentiment": article["sentiment"],
                "summary": article["summary"][:200] + "..."
            }
            
            if include_context:
                # Try to find context around entity mention
                content = article.get("full_content", "")
                if content and entity_name.lower() in content.lower():
                    # Find sentence containing entity
                    sentences = content.split(".")
                    for sentence in sentences:
                        if entity_name.lower() in sentence.lower():
                            article_info["context"] = sentence.strip()[:200]
                            break
            
            matching_articles.append(article_info)
    
    if not matching_articles:
        return {
            "entity": entity_name,
            "found": False,
            "message": "Entity not found in any articles"
        }
    
    # Calculate statistics
    sentiments = [a["sentiment"] for a in matching_articles]
    categories = {}
    for article in matching_articles:
        cat = article["category"]
        categories[cat] = categories.get(cat, 0) + 1
    
    return {
        "entity": entity_name,
        "found": True,
        "total_mentions": len(matching_articles),
        "sentiment_analysis": {
            "average": round(sum(sentiments) / len(sentiments), 3),
            "interpretation": _interpret_sentiment(sum(sentiments) / len(sentiments)),
            "most_positive": round(max(sentiments), 3),
            "most_negative": round(min(sentiments), 3)
        },
        "category_distribution": categories,
        "articles": matching_articles
    }


# ============= Comparison and Similarity Tools =============

@tool
def compare_articles(
    article_urls: List[str],
    comparison_aspects: List[str] = ["sentiment", "keywords", "entities", "category"]
) -> Dict:
    """
    Compare multiple articles across various aspects.
    
    Args:
        article_urls: List of article URLs to compare (2-5 articles)
        comparison_aspects: Aspects to compare
    
    Returns:
        Detailed comparison across requested aspects
    """
    if not VECTOR_STORE:
        return {"error": "Vector store not initialized"}
    
    if len(article_urls) < 2:
        return {"error": "Need at least 2 articles to compare"}
    if len(article_urls) > 5:
        return {"error": "Maximum 5 articles for comparison"}
    
    articles = []
    for url in article_urls:
        article_id = url.replace("https://", "").replace("/", "_")
        article = VECTOR_STORE.get_by_id(article_id)
        if article:
            articles.append(article)
    
    if len(articles) < 2:
        return {"error": "Could not find enough articles"}
    
    comparison = {
        "articles_compared": len(articles),
        "articles": [
            {
                "title": a["title"],
                "url": a["url"],
                "category": a["category"]
            }
            for a in articles
        ]
    }
    
    # Compare sentiment
    if "sentiment" in comparison_aspects:
        sentiments = [a["sentiment"] for a in articles]
        comparison["sentiment_comparison"] = {
            "values": {a["title"][:30]: round(a["sentiment"], 3) for a in articles},
            "most_positive": articles[sentiments.index(max(sentiments))]["title"],
            "most_negative": articles[sentiments.index(min(sentiments))]["title"],
            "range": round(max(sentiments) - min(sentiments), 3)
        }
    
    # Compare keywords
    if "keywords" in comparison_aspects:
        all_keywords = [set(a["keywords"]) for a in articles]
        common_keywords = set.intersection(*all_keywords) if all_keywords else set()
        unique_keywords = {
            a["title"][:30]: list(set(a["keywords"]) - common_keywords)
            for a in articles
        }
        comparison["keyword_comparison"] = {
            "common_keywords": list(common_keywords),
            "unique_keywords": unique_keywords,
            "total_unique": sum(len(v) for v in unique_keywords.values())
        }
    
    # Compare entities
    if "entities" in comparison_aspects:
        all_entities = [set(a["entities"]) for a in articles]
        common_entities = set.intersection(*all_entities) if all_entities else set()
        comparison["entity_comparison"] = {
            "common_entities": list(common_entities),
            "entity_counts": {a["title"][:30]: len(a["entities"]) for a in articles}
        }
    
    # Compare categories
    if "category" in comparison_aspects:
        categories = [a["category"] for a in articles]
        comparison["category_comparison"] = {
            "same_category": len(set(categories)) == 1,
            "categories": {a["title"][:30]: a["category"] for a in articles}
        }
    
    return comparison


@tool
def find_similar_articles(
    reference_url: str,
    max_results: int = 5,
    similarity_threshold: float = 0.5
) -> List[Dict]:
    """
    Find articles similar to a reference article.
    
    Args:
        reference_url: URL of the reference article
        max_results: Maximum similar articles to return
        similarity_threshold: Minimum similarity score (0.0-1.0)
    
    Returns:
        List of similar articles with similarity scores
    """
    if not VECTOR_STORE:
        return []
    
    # Get reference article
    article_id = reference_url.replace("https://", "").replace("/", "_")
    reference = VECTOR_STORE.get_by_id(article_id)
    
    if not reference:
        return []
    
    # Create search query from reference article
    search_query = f"{reference['title']} {reference['summary']} {' '.join(reference['keywords'])}"
    
    # Search for similar articles
    results = VECTOR_STORE.search(search_query, k=max_results + 1)  # +1 to exclude self
    
    # Filter out the reference article and low similarity
    similar = []
    for result in results:
        if result["url"] != reference_url and result["similarity_score"] >= similarity_threshold:
            similar.append({
                "title": result["title"],
                "url": result["url"],
                "summary": result["summary"],
                "category": result["category"],
                "similarity_score": round(result["similarity_score"], 3),
                "similarity_level": _get_similarity_level(result["similarity_score"])
            })
    
    return similar[:max_results]


# ============= Statistics and Overview Tools =============

@tool
def get_articles_statistics() -> Dict:
    """
    Get comprehensive statistics about the article database.
    
    Returns:
        Detailed statistics and insights
    """
    if not VECTOR_STORE:
        return {"error": "Vector store not initialized"}
    
    articles = VECTOR_STORE.get_all_articles()
    
    if not articles:
        return {"total_articles": 0, "message": "No articles in database"}
    
    # Calculate various statistics
    sentiments = [a["sentiment"] for a in articles]
    categories = {}
    all_keywords = []
    all_entities = []
    
    for article in articles:
        # Category counts
        cat = article.get("category", "other")
        categories[cat] = categories.get(cat, 0) + 1
        
        # Collect keywords and entities
        all_keywords.extend(article.get("keywords", []))
        all_entities.extend(article.get("entities", []))
    
    # Calculate unique counts
    unique_keywords = len(set(all_keywords))
    unique_entities = len(set(all_entities))
    
    return {
        "total_articles": len(articles),
        "category_distribution": categories,
        "sentiment_statistics": {
            "average": round(sum(sentiments) / len(sentiments), 3),
            "most_positive": round(max(sentiments), 3),
            "most_negative": round(min(sentiments), 3),
            "standard_deviation": round(_calculate_std_dev(sentiments), 3),
            "breakdown": {
                "positive": len([s for s in sentiments if s > 0.2]),
                "neutral": len([s for s in sentiments if -0.2 <= s <= 0.2]),
                "negative": len([s for s in sentiments if s < -0.2])
            }
        },
        "content_statistics": {
            "total_keywords": len(all_keywords),
            "unique_keywords": unique_keywords,
            "avg_keywords_per_article": round(len(all_keywords) / len(articles), 1),
            "total_entities": len(all_entities),
            "unique_entities": unique_entities,
            "avg_entities_per_article": round(len(all_entities) / len(articles), 1)
        },
        "insights": {
            "most_common_category": max(categories, key=categories.get),
            "sentiment_trend": _interpret_sentiment(sum(sentiments) / len(sentiments)),
            "content_diversity": "high" if unique_keywords > 100 else "medium" if unique_keywords > 50 else "low"
        }
    }


@tool
def get_category_distribution(include_examples: bool = True) -> Dict:
    """
    Get detailed category distribution with examples.
    
    Args:
        include_examples: Whether to include example articles
    
    Returns:
        Category distribution and statistics
    """
    if not VECTOR_STORE:
        return {"error": "Vector store not initialized"}
    
    articles = VECTOR_STORE.get_all_articles()
    
    if not articles:
        return {"total_articles": 0, "categories": {}}
    
    # Group by category
    category_groups = {}
    for article in articles:
        cat = article.get("category", "other")
        if cat not in category_groups:
            category_groups[cat] = {
                "articles": [],
                "sentiments": []
            }
        category_groups[cat]["articles"].append(article)
        category_groups[cat]["sentiments"].append(article["sentiment"])
    
    # Build response
    distribution = {}
    for category, data in category_groups.items():
        cat_info = {
            "count": len(data["articles"]),
            "percentage": round(len(data["articles"]) / len(articles) * 100, 1),
            "average_sentiment": round(
                sum(data["sentiments"]) / len(data["sentiments"]), 3
            )
        }
        
        if include_examples:
            cat_info["example_articles"] = [
                {
                    "title": a["title"],
                    "url": a["url"],
                    "sentiment": round(a["sentiment"], 2)
                }
                for a in data["articles"][:3]  # Top 3 examples
            ]
        
        distribution[category] = cat_info
    
    return {
        "total_articles": len(articles),
        "total_categories": len(category_groups),
        "distribution": distribution
    }


# ============= Helper Functions =============

def _interpret_sentiment(sentiment: float) -> str:
    """Interpret sentiment value as human-readable string"""
    if sentiment > 0.5:
        return "very positive"
    elif sentiment > 0.2:
        return "positive"
    elif sentiment < -0.5:
        return "very negative"
    elif sentiment < -0.2:
        return "negative"
    else:
        return "neutral"


def _get_similarity_level(score: float) -> str:
    """Convert similarity score to level"""
    if score >= 0.8:
        return "very high"
    elif score >= 0.6:
        return "high"
    elif score >= 0.4:
        return "medium"
    else:
        return "low"


def _matches_entity_type(entity: str, entity_type: str) -> bool:
    """Check if entity matches the specified type"""
    entity_lower = entity.lower()
    
    if entity_type == "people":
        # Simple heuristic for people
        org_indicators = ["corp", "inc", "ltd", "company", "tech", "bank", "group"]
        location_indicators = ["city", "country", "state", "county", "province"]
        return not any(ind in entity_lower for ind in org_indicators + location_indicators)
    
    elif entity_type == "organizations":
        org_indicators = ["corp", "inc", "ltd", "company", "tech", "bank", "group", "agency"]
        return any(ind in entity_lower for ind in org_indicators)
    
    elif entity_type == "locations":
        location_indicators = ["city", "country", "state", "county", "province", "region"]
        return any(ind in entity_lower for ind in location_indicators)
    
    return True  # Default to including if uncertain


def _calculate_std_dev(values: List[float]) -> float:
    """Calculate standard deviation"""
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return variance ** 0.5


# ============= Testing =============

if __name__ == "__main__":
    # Initialize tools
    vs = VectorStore()
    init_tools(vs)
    
    print("Tools module initialized successfully")
    print(f"Total tools available: {len(get_list_of_tools())}")
    
    # Test basic search
    print("\n=== Testing Search ===")
    results = search_articles_by_query.invoke({
        "query": "AI technology",
        "max_results": 3
    })
    for r in results:
        print(f"- {r['title'][:50]}... (relevance: {r['relevance_score']})")
    
    # Test statistics
    print("\n=== Testing Statistics ===")
    stats = get_articles_statistics.invoke({})
    print(f"Total articles: {stats.get('total_articles', 0)}")
    if "category_distribution" in stats:
        print(f"Categories: {stats['category_distribution']}")
