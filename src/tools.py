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

@tool
def search_articles(query: str, max_results: int = 5) -> List[Dict]:
    """
    Search for articles relevant to the query.
    Returns list of articles with title, summary, and metadata.
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
    Fetch the full content of a specific article by URL.
    Use this when you need detailed information from a specific article.
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
    Returns average sentiment and breakdown by article.
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
    Categories: technology, business, politics, other
    """
    if not vector_store:
        return []
    
    all_articles = vector_store.get_all_articles()
    filtered = [a for a in all_articles if a.get("category", "").lower() == category.lower()]
    
    return filtered

@tool
def compare_articles(url1: str, url2: str) -> Dict:
    """
    Compare two articles on various dimensions.
    Returns comparison of sentiment, keywords, and main topics.
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
    Get a summary of all articles in the database.
    Returns count by category and overall statistics.
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


if __name__ == "__main__":
    # Initialize
    vs = VectorStore()
    init_tools(vs)

    # Test search
    print("\n=== Testing Search ===")
    results = search_articles.invoke({"query": "AI technology"})
    for r in results:
        print(f"- {r['title'][:50]}... (relevance: {r['relevance']:.2f})")

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