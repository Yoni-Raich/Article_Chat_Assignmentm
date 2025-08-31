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

    def _create_article_document(self, article: Article) -> Document:
        """Create a document for the parent article with rich embedding content."""
        # Create rich content for embedding that includes title, summary, keywords, and entities
        embedding_content_parts = [
            f"Title: {article.title}",
            f"Summary: {article.metadata.summary}" if article.metadata.summary else "",
            f"Category: {article.metadata.category}" if article.metadata.category else "",
            f"Keywords: {', '.join(article.metadata.keywords)}" if article.metadata.keywords else "",
            f"Entities: {', '.join(article.metadata.entities)}" if article.metadata.entities else ""
        ]
        
        # Filter out empty parts and join with newlines
        embedding_content = "\n".join(part for part in embedding_content_parts if part)
        
        return Document(
            page_content=embedding_content,  # Rich content for better semantic search
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

    def _create_chunk_documents(self, article: Article, chunks: List[Chunk]) -> List[Document]:
        """Create documents for article chunks."""
        chunk_docs = []
        for chunk in chunks:
            chunk_doc = Document(
                page_content=chunk.content,  # This is what gets embedded
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
        return chunk_docs

    def add_article(self, article: Article) -> bool:
        """Add a parent article to the vector store."""
        try:
            article_doc = self._create_article_document(article)
            self.db.add_documents([article_doc], ids=[article.id])
            logger.info("Stored parent article: %s", article.title)
            return True
        except Exception as e:
            logger.error("Error adding article %s: %s", article.id, e)
            return False

    def add_chunks(self, article: Article, chunks: List[Chunk]) -> bool:
        """Add article chunks to the vector store."""
        try:
            chunk_docs = self._create_chunk_documents(article, chunks)
            if chunk_docs:
                self.db.add_documents(chunk_docs, ids=[c.metadata['chunk_id'] for c in chunk_docs])
                logger.info("Stored %d chunks for article: %s", len(chunk_docs), article.title)
            return True
        except Exception as e:
            logger.error("Error adding chunks for article %s: %s", article.id, e)
            return False

    def add_article_and_chunks(self, article: Article, chunks: List[Chunk]) -> bool:
        """Add a parent article and its content chunks to the vector store."""
        article_success = self.add_article(article)
        chunks_success = self.add_chunks(article, chunks)
        return article_success and chunks_success

    def add_batch(self, processed_data: List[tuple[Article, List[Chunk]]]) -> int:
        """Add multiple articles and their chunks."""
        added_count = 0
        for article, chunks in processed_data:
            if self.add_article_and_chunks(article, chunks):
                added_count += 1
        logger.info("Successfully added %d articles and their chunks.", added_count)
        return added_count

    def _build_search_filter(self, filter_dict: Optional[Dict] = None) -> Dict:
        """Build search filter with doc_type constraint."""
        search_filter = {"doc_type": "chunk"}
        if filter_dict:
            search_filter.update(filter_dict)
        return search_filter

    def _format_search_results(self, results: List[tuple]) -> List[Dict]:
        """Format search results from ChromaDB response."""
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

    def _build_article_search_filter(self, filter_dict: Optional[Dict] = None) -> Dict:
        """Build search filter for articles with doc_type constraint."""
        search_filter = {"doc_type": "article"}
        if filter_dict:
            search_filter.update(filter_dict)
        return search_filter

    def _format_article_search_results(self, results: List[tuple]) -> List[Dict]:
        """Format article search results from ChromaDB response."""
        formatted_results = []
        for doc, score in results:
            formatted_results.append({
                "id": doc.metadata.get("id"),
                "title": doc.metadata.get("title"),
                "url": doc.metadata.get("url"),
                "summary": doc.metadata.get("summary"),
                "category": doc.metadata.get("category"),
                "sentiment": doc.metadata.get("sentiment"),
                "keywords": json.loads(doc.metadata.get("keywords", "[]")),
                "entities": json.loads(doc.metadata.get("entities", "[]")),
                "similarity_score": 1 - score  # Convert distance to similarity
            })
        return formatted_results

    def search_articles(self, query: str, k: int = 5, filter_dict: Optional[Dict] = None) -> List[Dict]:
        """Search for relevant articles based on title, summary, keywords, and entities."""
        try:
            search_filter = self._build_article_search_filter(filter_dict)
            results = self.db.similarity_search_with_score(
                query=query,
                k=k,
                filter=search_filter
            )
            return self._format_article_search_results(results)
        except Exception as e:
            logger.error("Article search error: %s", e)
            return []

    def search_chunks(self, query: str, k: int = 5, filter_dict: Optional[Dict] = None) -> List[Dict]:
        """Search for relevant article chunks."""
        try:
            search_filter = self._build_search_filter(filter_dict)
            results = self.db.similarity_search_with_score(
                query=query,
                k=k,
                filter=search_filter
            )
            return self._format_search_results(results)
        except Exception as e:
            logger.error("Search error: %s", e)
            return []

    def _format_mixed_search_results(self, results: List[tuple]) -> List[Dict]:
        """Format mixed search results (both articles and chunks) from ChromaDB response."""
        formatted_results = []
        for doc, score in results:
            doc_type = doc.metadata.get("doc_type")
            similarity_score = 1 - score  # Convert distance to similarity
            
            if doc_type == "article":
                # Format as article result
                formatted_results.append({
                    "type": "article",
                    "id": doc.metadata.get("id"),
                    "title": doc.metadata.get("title"),
                    "url": doc.metadata.get("url"),
                    "summary": doc.metadata.get("summary"),
                    "category": doc.metadata.get("category"),
                    "sentiment": doc.metadata.get("sentiment"),
                    "keywords": json.loads(doc.metadata.get("keywords", "[]")),
                    "entities": json.loads(doc.metadata.get("entities", "[]")),
                    "similarity_score": similarity_score
                })
            elif doc_type == "chunk":
                # Format as chunk result
                formatted_results.append({
                    "type": "chunk",
                    "chunk_id": doc.metadata.get("chunk_id"),
                    "article_id": doc.metadata.get("article_id"),
                    "content": doc.page_content,
                    "title": doc.metadata.get("title"),
                    "url": doc.metadata.get("url"),
                    "similarity_score": similarity_score
                })
        return formatted_results

    def search_all(self, query: str, k: int = 10, filter_dict: Optional[Dict] = None) -> List[Dict]:
        """Free search across both articles and chunks for comprehensive results."""
        try:
            # Build filter without doc_type restriction to search everything
            search_filter = filter_dict.copy() if filter_dict else {}
            
            # Perform similarity search across all document types
            results = self.db.similarity_search_with_score(
                query=query,
                k=k,
                filter=search_filter if search_filter else None
            )
            
            # Format and return mixed results
            formatted_results = self._format_mixed_search_results(results)
            
            # Sort by similarity score (highest first)
            formatted_results.sort(key=lambda x: x['similarity_score'], reverse=True)
            
            logger.info(f"Free search returned {len(formatted_results)} results for query: '{query}'")
            return formatted_results
            
        except Exception as e:
            logger.error("Free search error: %s", e)
            return []

    def _reconstruct_article_from_metadata(self, metadata: Dict) -> Dict:
        """Reconstruct article dictionary from ChromaDB metadata."""
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

    def get_by_id(self, article_id: str) -> Optional[Dict]:
        """Get specific article by ID"""
        results = self.db.get(ids=[article_id], where={"doc_type": "article"})
        if results and results['documents']:
            metadata = results['metadatas'][0]
            return self._reconstruct_article_from_metadata(metadata)
        return None

    def _extract_article_summary(self, metadata: Dict) -> Dict:
        """Extract article summary from metadata (without full content)."""
        return {
            "id": metadata.get("id"),
            "url": metadata.get("url"),
            "title": metadata.get("title"),
            "summary": metadata.get("summary"),
            "category": metadata.get("category"),
            "sentiment": metadata.get("sentiment"),
            "entities": json.loads(metadata.get("entities", "[]"))
        }

    def get_all_articles(self) -> List[Dict]:
        """Get all articles metadata (without full content for efficiency)"""
        results = self.db.get(where={"doc_type": "article"})
        articles = []

        if results and results['metadatas']:
            for metadata in results['metadatas']:
                articles.append(self._extract_article_summary(metadata))
        return articles

    def article_exists(self, article_id: str) -> bool:
        """Check if an article (not a chunk) already exists."""
        result = self.db.get(ids=[article_id], where={"doc_type": "article"})
        return bool(result and result['ids'])

    def _format_chunk_data(self, content: str, metadata: Dict) -> Dict:
        """Format chunk data from ChromaDB response."""
        return {
            "chunk_id": metadata.get("chunk_id"),
            "article_id": metadata.get("article_id"),
            "content": content,
            "index": metadata.get("chunk_index"),
        }

    def _sort_chunks_by_index(self, chunks: List[Dict]) -> List[Dict]:
        """Sort chunks by their index."""
        chunks.sort(key=lambda x: x.get('index', 0))
        return chunks

    def get_chunks_by_article_id(self, article_id: str) -> List[Dict]:
        """Get all chunks for a specific article."""
        results = self.db.get(where={"doc_type": "chunk", "article_id": article_id})
        chunks = []
        if results and results['documents']:
            for i, content in enumerate(results['documents']):
                metadata = results['metadatas'][i]
                chunks.append(self._format_chunk_data(content, metadata))
        return self._sort_chunks_by_index(chunks)

    def _get_chunk_ids_for_article(self, article_id: str) -> List[str]:
        """Get all chunk IDs for a specific article."""
        return [c['chunk_id'] for c in self.get_chunks_by_article_id(article_id)]

    def _build_deletion_ids(self, article_id: str, chunk_ids: List[str]) -> List[str]:
        """Build list of all IDs to delete (article + chunks)."""
        return [article_id] + chunk_ids

    def delete_article(self, article_id: str) -> bool:
        """Delete article and all its chunks from store."""
        try:
            chunk_ids_to_delete = self._get_chunk_ids_for_article(article_id)
            ids_to_delete = self._build_deletion_ids(article_id, chunk_ids_to_delete)
            
            if ids_to_delete:
                self.db.delete(ids=ids_to_delete)
                logger.info("Deleted article %s and its %d chunks.", article_id, len(chunk_ids_to_delete))
            return True
        except Exception as e:
            logger.error("Error deleting article %s: %s", article_id, e)
            return False
