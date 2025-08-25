# src/vector_store.py
import os
from typing import List, Dict, Optional
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.schema import Document
from models import Article
from logger import logger
import json

class VectorStore:
    def __init__(self, persist_directory: str = "./data/chroma_db", embedding_provider = None):
        """Initialize ChromaDB with Google embeddings"""
        self.embeddings = embedding_provider or GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-exp-03-07",
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        
        self.persist_directory = persist_directory
        
        # Initialize or load existing collection
        self.db = Chroma(
            collection_name="articles",
            embedding_function=self.embeddings,
            persist_directory=persist_directory
        )
        
        logger.info(f"VectorStore initialized. Current articles: {self.db._collection.count()}")
    
    def add_article(self, article: Article) -> bool:
        """Add single article to vector store"""
        try:
            # Create embedding text from title + summary + keywords
            embedding_text = f"{article.title} {article.metadata.summary} {' '.join(article.metadata.keywords)}"
            
            # Create document with full metadata
            doc = Document(
                page_content=embedding_text,
                metadata={
                    "id": article.id,
                    "url": article.url,
                    "title": article.title,
                    "full_content": article.content,  # Store full text!
                    "summary": article.metadata.summary,
                    "keywords": json.dumps(article.metadata.keywords),
                    "sentiment": article.metadata.sentiment,
                    "category": article.metadata.category,
                    "processed_at": str(article.processed_at)
                }
            )
            
            # Add to ChromaDB
            self.db.add_documents([doc], ids=[article.id])
            logger.info(f"Added article: {article.title[:50]}...")
            return True
            
        except Exception as e:
            logger.error(f"Error adding article {article.id}: {e}")
            return False
    
    def add_batch(self, articles: List[Article]) -> int:
        """Add multiple articles"""
        added = 0
        for article in articles:
            if self.add_article(article):
                added += 1
        
        logger.info(f"Added {added}/{len(articles)} articles to vector store")
        return added
    
    def search(self, query: str, k: int = 5, filter_dict: Optional[Dict] = None) -> List[Dict]:
        """Search for relevant articles"""
        try:
            # Perform similarity search
            results = self.db.similarity_search_with_score(
                query=query,
                k=k,
                filter=filter_dict
            )
            
            # Format results
            formatted_results = []
            for doc, score in results:
                formatted_results.append({
                    "id": doc.metadata.get("id"),
                    "url": doc.metadata.get("url"),
                    "title": doc.metadata.get("title"),
                    "summary": doc.metadata.get("summary"),
                    "full_content": doc.metadata.get("full_content"),
                    "keywords": json.loads(doc.metadata.get("keywords", "[]")),
                    "sentiment": doc.metadata.get("sentiment"),
                    "category": doc.metadata.get("category"),
                    "similarity_score": 1 - score  # Convert distance to similarity
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []
    
    def get_by_id(self, article_id: str) -> Optional[Dict]:
        """Get specific article by ID"""
        results = self.db.get(ids=[article_id])
        if results and results['documents']:
            metadata = results['metadatas'][0]
            return {
                "id": metadata.get("id"),
                "url": metadata.get("url"),
                "title": metadata.get("title"),
                "full_content": metadata.get("full_content"),
                "summary": metadata.get("summary"),
                "keywords": json.loads(metadata.get("keywords", "[]")),
                "sentiment": metadata.get("sentiment"),
                "category": metadata.get("category")
            }
        return None
    
    def get_all_articles(self) -> List[Dict]:
        """Get all articles metadata (without full content for efficiency)"""
        results = self.db.get()
        articles = []
        
        if results and results['metadatas']:
            for metadata in results['metadatas']:
                articles.append({
                    "id": metadata.get("id"),
                    "url": metadata.get("url"),
                    "title": metadata.get("title"),
                    "summary": metadata.get("summary"),
                    "category": metadata.get("category"),
                    "sentiment": metadata.get("sentiment")
                })
        
        return articles
    
    def article_exists(self, article_id: str) -> bool:
        """Check if article already exists"""
        result = self.db.get(ids=[article_id])
        return bool(result and result['documents'])
    
    def delete_article(self, article_id: str) -> bool:
        """Delete article from store"""
        try:
            self.db.delete(ids=[article_id])
            return True
        except:
            return False


if __name__ == "__main__":
    from ingestion import ArticleProcessor
    # Test URLs
    # test_urls = [
    #     "https://techcrunch.com/2025/07/26/astronomer-winks-at-viral-notoriety-with-temporary-spokesperson-gwyneth-paltrow/",
    #     "https://techcrunch.com/2025/07/26/allianz-life-says-majority-of-customers-personal-data-stolen-in-cyberattack/",
    #     "https://techcrunch.com/2025/07/27/itch-io-is-the-latest-marketplace-to-crack-down-on-adult-games/",
    #     "https://techcrunch.com/2025/07/26/tesla-vet-says-that-reviewing-real-products-not-mockups-is-the-key-to-staying-innovative/",
    #     "https://techcrunch.com/2025/07/25/meta-names-shengjia-zhao-as-chief-scientist-of-ai-superintelligence-unit/",
    #     "https://techcrunch.com/2025/07/26/dating-safety-app-tea-breached-exposing-72000-user-images/",
    #     "https://techcrunch.com/2025/07/25/sam-altman-warns-theres-no-legal-confidentiality-when-using-chatgpt-as-a-therapist/",
    #     "https://techcrunch.com/2025/07/25/intel-is-spinning-off-its-network-and-edge-group/",
    #     "https://techcrunch.com/2025/07/27/wizard-of-oz-blown-up-by-ai-for-giant-sphere-screen/",
    #     "https://techcrunch.com/2025/07/27/doge-has-built-an-ai-tool-to-slash-federal-regulations/",
    #     "https://edition.cnn.com/2025/07/27/business/us-china-trade-talks-stockholm-intl-hnk",
    #     "https://edition.cnn.com/2025/07/27/business/trump-us-eu-trade-deal",
    #     "https://edition.cnn.com/2025/07/27/business/eu-trade-deal",
    #     "https://edition.cnn.com/2025/07/26/tech/daydream-ai-online-shopping",
    #     "https://edition.cnn.com/2025/07/25/tech/meta-ai-superintelligence-team-who-its-hiring",
    #     "https://edition.cnn.com/2025/07/25/tech/sequoia-islamophobia-maguire-mamdani",
    #     "https://edition.cnn.com/2025/07/24/tech/intel-layoffs-15-percent-q2-earnings"
    # ]

    # # Process articles
    # processor = ArticleProcessor()
    # articles = processor.process_batch(test_urls)

    # Add to vector store
    store = VectorStore()
    #store.add_batch(articles)

    # Test search
    results = store.search(" Which article is more positive about the topic of AI regulation?", k=20)
    for r in results:
        logger.info(f"- {r['title']}: {r['similarity_score']:.2f}")