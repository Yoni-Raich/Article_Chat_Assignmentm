"""
Vector store module for article embeddings and semantic search.

This module provides functionality for storing and searching article embeddings
using ChromaDB and Google Generative AI embeddings.
"""

# Standard library imports
import os
import json
from typing import List, Dict, Optional

# Third-party imports
import chromadb
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.schema import Document

# Local imports
from .models import Article, Chunk
from .logger import logger

class VectorStore:
    """
    Vector store for article embeddings using ChromaDB.

    This class manages storing and searching article embeddings using ChromaDB
    and Google Generative AI embeddings for semantic search capabilities.
    """
    def __init__(
        self,
        persist_directory: str = "./data/chroma_db",
        embedding_provider = None,
        chroma_host: str = None,
        chroma_port: int = 8000
    ):
        """Initialize ChromaDB with Google embeddings

        Args:
            persist_directory: Local directory for ChromaDB (when running locally)
            embedding_provider: Custom embedding provider (optional)
            chroma_host: ChromaDB host (for remote ChromaDB service)
            chroma_port: ChromaDB port (for remote ChromaDB service)
        """
        self.embeddings = embedding_provider or GoogleGenerativeAIEmbeddings(
            model=os.getenv("EMBEDDING_MODEL_NAME", "gemini-embedding-exp-03-07"),
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
            logger.info("Connected to remote ChromaDB at %s:%s", chroma_host, chroma_port)
        else:
            # Use local ChromaDB with persistence
            self.persist_directory = persist_directory
            self.db = Chroma(
                collection_name="articles",
                embedding_function=self.embeddings,
                persist_directory=persist_directory
            )
            logger.info("Using local ChromaDB at %s", persist_directory)

        logger.info(
            "VectorStore initialized. Current articles: %s",
            self.db._collection.count()
        )

    def add_article_and_chunks(self, article: Article, chunks: List[Chunk]) -> bool:
        """Add a parent article and its content chunks to the vector store."""
        try:
            # 1. Create and store the parent article document (without embedding content)
            article_doc = Document(
                page_content=f"Article: {article.title}", # Minimal content
                metadata={
                    "doc_type": "article",
                    "id": article.id,
                    "url": article.url,
                    "title": article.title,
                    "full_content": article.content,
                    "summary": article.metadata.summary,
                    "keywords": json.dumps(article.metadata.keywords),
                    "entities": json.dumps(article.metadata.entities),
                    "sentiment": article.metadata.sentiment,
                    "category": article.metadata.category,
                    "processed_at": str(article.processed_at)
                }
            )
            self.db.add_documents([article_doc], ids=[article.id])
            logger.info("Stored parent article: %s", article.title)

            # 2. Create and store documents for each chunk (with embedding)
            chunk_docs = []
            for chunk in chunks:
                chunk_doc = Document(
                    page_content=chunk.content, # This is what gets embedded
                    metadata={
                        "doc_type": "chunk",
                        "chunk_id": chunk.id,
                        "article_id": chunk.article_id,
                        "chunk_index": chunk.index,
                        "title": article.title,
                        "url": article.url,
                    }
                )
                chunk_docs.append(chunk_doc)

            if chunk_docs:
                self.db.add_documents(chunk_docs, ids=[c.metadata['chunk_id'] for c in chunk_docs])
                logger.info("Stored %d chunks for article: %s", len(chunk_docs), article.title)

            return True

        except Exception as e:
            logger.error("Error adding article and chunks for %s: %s", article.id, e)
            return False

    def add_batch(self, processed_data: List[tuple[Article, List[Chunk]]]) -> int:
        """Add multiple articles and their chunks."""
        added_count = 0
        for article, chunks in processed_data:
            if self.add_article_and_chunks(article, chunks):
                added_count += 1
        logger.info("Successfully added %d articles and their chunks.", added_count)
        return added_count

    def search_chunks(self, query: str, k: int = 5, filter_dict: Optional[Dict] = None) -> List[Dict]:
        """Search for relevant article chunks."""
        try:
            # Add doc_type filter to existing filters
            search_filter = {"doc_type": "chunk"}
            if filter_dict:
                search_filter.update(filter_dict)

            # Perform similarity search on chunks
            results = self.db.similarity_search_with_score(
                query=query,
                k=k,
                filter=search_filter
            )

            # Format results from chunk metadata
            formatted_results = []
            for doc, score in results:
                formatted_results.append({
                    "chunk_id": doc.metadata.get("chunk_id"),
                    "article_id": doc.metadata.get("article_id"),
                    "content": doc.page_content,
                    "title": doc.metadata.get("title"),
                    "url": doc.metadata.get("url"),
                    "similarity_score": 1 - score  # Convert distance to similarity
                })
            return formatted_results

        except Exception as e:
            logger.error("Search error: %s", e)
            return []

    def get_by_id(self, article_id: str) -> Optional[Dict]:
        """Get specific article by ID"""
        results = self.db.get(ids=[article_id], where={"doc_type": "article"})
        if results and results['documents']:
            metadata = results['metadatas'][0]
            # Reconstruct the article dictionary from metadata
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
        results = self.db.get(where={"doc_type": "article"})
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
        """Check if an article (not a chunk) already exists."""
        result = self.db.get(ids=[article_id], where={"doc_type": "article"})
        return bool(result and result['ids'])

    def get_chunks_by_article_id(self, article_id: str) -> List[Dict]:
        """Get all chunks for a specific article."""
        results = self.db.get(where={"doc_type": "chunk", "article_id": article_id})
        chunks = []
        if results and results['documents']:
            for i, content in enumerate(results['documents']):
                metadata = results['metadatas'][i]
                chunks.append({
                    "chunk_id": metadata.get("chunk_id"),
                    "article_id": metadata.get("article_id"),
                    "content": content,
                    "index": metadata.get("chunk_index"),
                })
        # Sort chunks by index
        chunks.sort(key=lambda x: x.get('index', 0))
        return chunks

    def delete_article(self, article_id: str) -> bool:
        """Delete article from store"""
        try:
            # First, get all chunk IDs for the article
            chunk_ids_to_delete = [
                c['chunk_id'] for c in self.get_chunks_by_article_id(article_id)
            ]
            # Also delete the article document itself
            ids_to_delete = [article_id] + chunk_ids_to_delete
            if ids_to_delete:
                self.db.delete(ids=ids_to_delete)
                logger.info("Deleted article %s and its %d chunks.", article_id, len(chunk_ids_to_delete))
            return True
        except Exception as e:
            logger.error("Error deleting article %s: %s", article_id, e)
            return False
