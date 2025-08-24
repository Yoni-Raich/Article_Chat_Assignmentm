# src/tools.py
from langchain.tools import tool
from typing import List, Dict, Optional
from vector_store import VectorStore
from ingestion import ArticleProcessor

# Initialize global instances
vector_store = None
processor = None

def init_tools(vs: VectorStore = None, ap: ArticleProcessor = None):
    """Initialize tools with dependencies"""
    global vector_store, processor
    vector_store = vs or VectorStore()
    processor = ap or ArticleProcessor()

def get_list_of_tools():
    return [
        search_articles,
        get_article_content,
        analyze_sentiment_batch,
        get_articles_by_category,
        compare_articles,
        find_most_similar_article
    ]

@tool
def search_articles(query: str, max_results: int = 5) -> List[Dict]:
    """
    Search articles using semantic similarity.
    
    Args:
        query: Search query (keywords, phrases, or questions)
        max_results: Maximum results to return (default: 5)
    
    Returns:
        List of articles with title, url, summary, category, and relevance score
    """
    if not vector_store:
        return []
    
    results = vector_store.search(query, k=max_results)
    
    # Simplify results for agent
    simplified = []
    for r in results:
        simplified.append({
            "title": r["title"],
            "url": r["url"],
            "summary": r["summary"],
            "category": r["category"],
            "relevance": r["similarity_score"]
        })
    
    return simplified

@tool
def get_article_content(article_url: str) -> str:
    """
    Get full article text content. use this when the user provides a specific article URL.
    
    Args:
        article_url: Complete URL from search results
    
    Returns:
        Formatted article with title and full content
    """
    if not vector_store:
        return "Article not found"
    
    # Convert URL to ID format
    article_id = article_url.replace("https://", "").replace("/", "_")
    
    article = vector_store.get_by_id(article_id)
    if article:
        return f"Title: {article['title']}\n\nContent: {article['full_content']}"
    
    return "Article not found in database"

@tool
def analyze_sentiment_batch(article_urls: List[str]) -> Dict:
    """
    Analyze sentiment across multiple articles.
    
    Args:
        article_urls: List of article URLs to analyze
    
    Returns:
        Dict with average_sentiment (-1.0 to 1.0), interpretation, and breakdown
    """
    if not vector_store:
        return {"error": "Vector store not initialized"}
    
    sentiments = []
    breakdown = {}
    
    for url in article_urls:
        article_id = url.replace("https://", "").replace("/", "_")
        article = vector_store.get_by_id(article_id)
        
        if article:
            sentiment = article["sentiment"]
            sentiments.append(sentiment)
            breakdown[article["title"][:50]] = sentiment
    
    if sentiments:
        avg_sentiment = sum(sentiments) / len(sentiments)
        return {
            "average_sentiment": avg_sentiment,
            "interpretation": "positive" if avg_sentiment > 0.2 else "negative" if avg_sentiment < -0.2 else "neutral",
            "breakdown": breakdown
        }
    
    return {"error": "No articles found"}

@tool
def get_articles_by_category(category: str) -> List[Dict]:
    """
    Get all articles in a specific category.
    
    Args:
        category: Category name (technology, business, politics, other)
    
    Returns:
        List of articles with complete metadata
    """
    if not vector_store:
        return []
    
    all_articles = vector_store.get_all_articles()
    filtered = [a for a in all_articles if a.get("category", "").lower() == category.lower()]
    
    return filtered

@tool
def compare_articles(url1: str, url2: str) -> Dict:
    """
    Compare two articles for similarities and differences.
    
    Args:
        url1: First article URL
        url2: Second article URL
    
    Returns:
        Dict with article details, common/unique keywords, and sentiment difference
    """
    if not vector_store:
        return {"error": "Vector store not initialized"}
    
    id1 = url1.replace("https://", "").replace("/", "_")
    id2 = url2.replace("https://", "").replace("/", "_")
    
    art1 = vector_store.get_by_id(id1)
    art2 = vector_store.get_by_id(id2)
    
    if not (art1 and art2):
        return {"error": "One or both articles not found"}
    
    # Find common and unique keywords
    keywords1 = set(art1["keywords"])
    keywords2 = set(art2["keywords"])
    
    return {
        "article1": {
            "title": art1["title"],
            "sentiment": art1["sentiment"],
            "category": art1["category"]
        },
        "article2": {
            "title": art2["title"],
            "sentiment": art2["sentiment"],
            "category": art2["category"]
        },
        "common_keywords": list(keywords1 & keywords2),
        "unique_to_article1": list(keywords1 - keywords2),
        "unique_to_article2": list(keywords2 - keywords1),
        "sentiment_difference": abs(art1["sentiment"] - art2["sentiment"])
    }

@tool
def get_all_articles_summary() -> Dict:
    """
    Get database overview and statistics.
    
    Returns:
        Dict with total_articles, categories breakdown, average_sentiment, and most_common_category
    """
    if not vector_store:
        return {"error": "Vector store not initialized"}
    
    articles = vector_store.get_all_articles()
    
    if not articles:
        return {"total": 0, "message": "No articles in database"}
    
    # Category breakdown
    categories = {}
    sentiments = []
    
    for article in articles:
        cat = article.get("category", "other")
        categories[cat] = categories.get(cat, 0) + 1
        sentiments.append(article.get("sentiment", 0))
    
    avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
    
    return {
        "total_articles": len(articles),
        "categories": categories,
        "average_sentiment": avg_sentiment,
        "most_common_category": max(categories, key=categories.get) if categories else None
    }

@tool
def find_most_similar_article(text: str, similarity_threshold: float = 0.5) -> Optional[Dict]:
    """
    Find the most similar article to given text. use this when the user only describes the article.
    
    Args:
        text: Text to match against
        similarity_threshold: Minimum similarity score (0.0-1.0, default: 0.5)
    
    Returns:
        Best matching article with similarity_score and confidence, or None
    """
    if not vector_store:
        return None
    
    # Search for the most similar article (k=1 to get only the best match)
    results = vector_store.search(text, k=1)
    
    if not results:
        return None
    
    best_match = results[0]
    similarity_score = best_match["similarity_score"]
    
    # Check if similarity is above threshold
    if similarity_score >= similarity_threshold:
        return {
            "article": {
                "title": best_match["title"],
                "url": best_match["url"],
                "summary": best_match["summary"],
                "category": best_match["category"],
                "keywords": best_match["keywords"],
                "sentiment": best_match["sentiment"]
            },
            "similarity_score": similarity_score,
            "confidence": "high" if similarity_score >= 0.8 else "medium" if similarity_score >= 0.6 else "low"
        }
    else:
        return None


if __name__ == "__main__":
    # Initialize
    vs = VectorStore()
    init_tools(vs)

    # Test search
    print("\n=== Testing Search ===")
    results = search_articles.invoke({"query": "AI technology"})
    for r in results:
        print(f"- {r['title'][:50]}... (relevance: {r['relevance']:.2f})")

    # Test new find_most_similar_article tool
    print("\n=== Testing Find Most Similar Article ===")
    test_texts = [
        "artificial intelligence and machine learning developments",
        "cyber security data breaches and hacking incidents", 
        "completely unrelated topic about cooking recipes"
    ]
    
    for text in test_texts:
        print(f"\nSearching for: '{text}'")
        result = find_most_similar_article.invoke({
            "text": text, 
            "similarity_threshold": 0.5
        })
        
        if result:
            print(f"✓ Found: {result['article']['title'][:60]}...")
            print(f"  Similarity: {result['similarity_score']:.3f} ({result['confidence']})")
            print(f"  Category: {result['article']['category']}")
        else:
            print("✗ No similar article found above threshold")

    # Test summary
    print("\n=== Database Summary ===")
    summary = get_all_articles_summary.invoke({})
    print(f"Total articles: {summary['total_articles']}")
    print(f"Categories: {summary['categories']}")

    # Test content fetch
    if results:
        print("\n=== Fetching First Article ===")
        content = get_article_content.invoke({"article_url": results[0]['url']})
        print(content[:200] + "...")