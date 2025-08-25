# src/vector_store.py
import os
from typing import List, Dict, Optional
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.schema import Document
from .models import Article
from .logger import logger
import json
import chromadb

class VectorStore:
    def __init__(self, persist_directory: str = "./data/chroma_db", embedding_provider = None, chroma_host: str = None, chroma_port: int = 8000):
        """Initialize ChromaDB with Google embeddings
        
        Args:
            persist_directory: Local directory for ChromaDB (when running locally)
            embedding_provider: Custom embedding provider (optional)
            chroma_host: ChromaDB host (for remote ChromaDB service)
            chroma_port: ChromaDB port (for remote ChromaDB service)
        """
        self.embeddings = embedding_provider or GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-exp-03-07",
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        
        # Check if we should use remote ChromaDB
        chroma_host = chroma_host or os.getenv("CHROMA_HOST")
        chroma_port = chroma_port or int(os.getenv("CHROMA_PORT", "8000"))
        
        if chroma_host:
            # Use remote ChromaDB service
            chroma_client = chromadb.HttpClient(
                host=chroma_host,
                port=chroma_port,
                ssl=False
            )
            
            self.db = Chroma(
                collection_name="articles",
                embedding_function=self.embeddings,
                client=chroma_client
            )
            logger.info(f"Connected to remote ChromaDB at {chroma_host}:{chroma_port}")
        else:
            # Use local ChromaDB with persistence
            self.persist_directory = persist_directory
            self.db = Chroma(
                collection_name="articles",
                embedding_function=self.embeddings,
                persist_directory=persist_directory
            )
            logger.info(f"Using local ChromaDB at {persist_directory}")
        
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
                    "entities": json.dumps(article.metadata.entities),
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
                    "entities": json.loads(doc.metadata.get("entities", "[]")),
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
                "entities": json.loads(metadata.get("entities", "[]")),
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
                    "sentiment": metadata.get("sentiment"),
                    "entities": json.loads(metadata.get("entities", "[]"))
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
