"""
Vector store module for article embeddings and semantic search.

This module provides functionality for storing and searching article embeddings
using ChromaDB and Google Generative AI embeddings.
"""

# Standard library imports
import json
import os
from typing import Dict, List, Optional, Tuple, Union

# Third-party imports
import chromadb
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Local imports
from .logger import logger
from .models import Article, Chunk

class VectorStore:
    """
    Vector store for article embeddings using ChromaDB.

    This class manages storing and searching article embeddings using ChromaDB
    and Google Generative AI embeddings for semantic search capabilities.
    
    Attributes:
        embeddings: The embedding provider for generating vector embeddings
        db: ChromaDB instance for vector storage and retrieval
        persist_directory: Local directory for ChromaDB persistence (if local)
    """

    def __init__(
        self,
        persist_directory: str = "./data/chroma_db",
        embedding_provider: Optional[GoogleGenerativeAIEmbeddings] = None,
        chroma_host: Optional[str] = None,
        chroma_port: int = 8000,
    ) -> None:
        """Initialize ChromaDB with Google embeddings.

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

    # ==================== STORAGE METHODS ====================

    def add_article(self, article: Article) -> bool:
        """Add a parent article to the vector store.
        
        Args:
            article: Article object to store
            
        Returns:
            True if successful, False otherwise
        """
        try:
            article_doc = self._create_article_document(article)
            self.db.add_documents([article_doc], ids=[article.id])
            logger.info("Stored parent article: %s", article.title)
            return True
        except Exception as e:
            logger.error("Error adding article %s: %s", article.id, e)
            return False

    def add_chunks(self, article: Article, chunks: List[Chunk]) -> bool:
        """Add article chunks to the vector store.
        
        Args:
            article: Parent article object
            chunks: List of chunk objects to store
            
        Returns:
            True if successful, False otherwise
        """
        try:
            chunk_docs = self._create_chunk_documents(article, chunks)
            if chunk_docs:
                chunk_ids = [c.metadata["chunk_id"] for c in chunk_docs]
                self.db.add_documents(chunk_docs, ids=chunk_ids)
                logger.info(
                    "Stored %d chunks for article: %s", len(chunk_docs), article.title
                )
            return True
        except Exception as e:
            logger.error("Error adding chunks for article %s: %s", article.id, e)
            return False

    def add_article_and_chunks(self, article: Article, chunks: List[Chunk]) -> bool:
        """Add a parent article and its content chunks to the vector store.
        
        Args:
            article: Article object to store
            chunks: List of chunk objects to store
            
        Returns:
            True if both article and chunks were stored successfully
        """
        article_success = self.add_article(article)
        chunks_success = self.add_chunks(article, chunks)
        return article_success and chunks_success

    def add_batch(self, processed_data: List[Tuple[Article, List[Chunk]]]) -> int:
        """Add multiple articles and their chunks in batch.
        
        Args:
            processed_data: List of (article, chunks) tuples
            
        Returns:
            Number of successfully added articles
        """
        added_count = 0
        for article, chunks in processed_data:
            if self.add_article_and_chunks(article, chunks):
                added_count += 1
        logger.info("Successfully added %d articles and their chunks.", added_count)
        return added_count

    # ==================== PRIVATE HELPER METHODS ====================
    
    def _build_search_filter(
        self, doc_type: str, filter_dict: Optional[Dict] = None
    ) -> Dict:
        """Build search filter with doc_type constraint.
        
        Args:
            doc_type: Type of document to filter for ('article', 'chunk', or None for all)
            filter_dict: Additional filters to apply
            
        Returns:
            Combined filter dictionary with proper ChromaDB operators
        """
        # Start with base filter
        filters = []
        
        # Add doc_type filter if specified
        if doc_type:
            filters.append({"doc_type": doc_type})
        
        # Add additional filters if provided
        if filter_dict:
            for key, value in filter_dict.items():
                filters.append({key: value})
        
        # Return appropriate filter structure
        if len(filters) == 0:
            return {}
        elif len(filters) == 1:
            return filters[0]
        else:
            # Use $and operator for multiple conditions
            return {"$and": filters}

    def _convert_distance_to_similarity(self, distance: float) -> float:
        """Convert ChromaDB distance score to similarity score."""
        return 1 - distance

    def _safe_json_loads(self, json_str: str, default: List = None) -> List:
        """Safely load JSON string with fallback to default."""
        if default is None:
            default = []
        try:
            return json.loads(json_str or "[]")
        except (json.JSONDecodeError, TypeError):
            return default

    def _format_chunk_result(self, doc: Document, score: float) -> Dict:
        """Format a single chunk search result."""
        return {
            "chunk_id": doc.metadata.get("chunk_id"),
            "article_id": doc.metadata.get("article_id"),
            "content": doc.page_content,
            "title": doc.metadata.get("title"),
            "url": doc.metadata.get("url"),
            "similarity_score": self._convert_distance_to_similarity(score),
        }

    def _format_article_result(self, doc: Document, score: float) -> Dict:
        """Format a single article search result."""
        return {
            "id": doc.metadata.get("id"),
            "title": doc.metadata.get("title"),
            "url": doc.metadata.get("url"),
            "summary": doc.metadata.get("summary"),
            "category": doc.metadata.get("category"),
            "sentiment": doc.metadata.get("sentiment"),
            "keywords": self._safe_json_loads(doc.metadata.get("keywords")),
            "entities": self._safe_json_loads(doc.metadata.get("entities")),
            "similarity_score": self._convert_distance_to_similarity(score),
        }

    def _format_search_results(
        self, results: List[Tuple], result_type: str
    ) -> List[Dict]:
        """Format search results from ChromaDB response.
        
        Args:
            results: List of (document, score) tuples from ChromaDB
            result_type: Type of results to format ('chunk', 'article', or 'mixed')
            
        Returns:
            List of formatted result dictionaries
        """
        formatted_results = []
        
        for doc, score in results:
            if result_type == "chunk":
                formatted_results.append(self._format_chunk_result(doc, score))
            elif result_type == "article":
                formatted_results.append(self._format_article_result(doc, score))
            elif result_type == "mixed":
                doc_type = doc.metadata.get("doc_type")
                if doc_type == "chunk":
                    result = self._format_chunk_result(doc, score)
                    result["type"] = "chunk"
                elif doc_type == "article":
                    result = self._format_article_result(doc, score)
                    result["type"] = "article"
                else:
                    continue  # Skip unknown document types
                formatted_results.append(result)
                
        return formatted_results

    # ==================== SEARCH METHODS ====================

    def search_articles(
        self, query: str, k: int = 5, filter_dict: Optional[Dict] = None
    ) -> List[Dict]:
        """Search for relevant articles based on title, summary, keywords, and entities.
        
        Args:
            query: Search query string
            k: Number of results to return
            filter_dict: Additional filters to apply
            
        Returns:
            List of article search results with metadata
        """
        try:
            search_filter = self._build_search_filter("article", filter_dict)
            results = self.db.similarity_search_with_score(
                query=query, k=k, filter=search_filter
            )
            return self._format_search_results(results, "article")
        except Exception as e:
            logger.error("Article search error: %s", e)
            return []

    def search_chunks(
        self, query: str, k: int = 5, filter_dict: Optional[Dict] = None
    ) -> List[Dict]:
        """Search for relevant article chunks.
        
        Args:
            query: Search query string
            k: Number of results to return
            filter_dict: Additional filters to apply
            
        Returns:
            List of chunk search results with content
        """
        try:
            search_filter = self._build_search_filter("chunk", filter_dict)
            
            # Handle empty filter case
            if not search_filter:
                search_filter = None
                
            results = self.db.similarity_search_with_score(
                query=query, k=k, filter=search_filter
            )
            return self._format_search_results(results, "chunk")
        except Exception as e:
            logger.error("Chunk search error: %s", e)
            return []

    def search_all(
        self, query: str, k: int = 10, filter_dict: Optional[Dict] = None
    ) -> List[Dict]:
        """Free search across both articles and chunks for comprehensive results.
        
        Args:
            query: Search query string
            k: Number of results to return
            filter_dict: Additional filters to apply (no doc_type restriction)
            
        Returns:
            List of mixed search results sorted by similarity score
        """
        try:
            # Build filter without doc_type restriction to search everything
            search_filter = filter_dict.copy() if filter_dict else {}

            # Perform similarity search across all document types
            results = self.db.similarity_search_with_score(
                query=query,
                k=k,
                filter=search_filter if search_filter else None,
            )

            # Format and return mixed results
            formatted_results = self._format_search_results(results, "mixed")

            # Sort by similarity score (highest first)
            formatted_results.sort(
                key=lambda x: x["similarity_score"], reverse=True
            )

            logger.info(
                "Free search returned %d results for query: '%s'",
                len(formatted_results),
                query,
            )
            return formatted_results

        except Exception as e:
            logger.error("Free search error: %s", e)
            return []

    def _reconstruct_article_from_metadata(self, metadata: Dict) -> Dict:
        """Reconstruct complete article dictionary from ChromaDB metadata."""
        return {
            "id": metadata.get("id"),
            "url": metadata.get("url"),
            "title": metadata.get("title"),
            "full_content": metadata.get("full_content"),
            "summary": metadata.get("summary"),
            "keywords": self._safe_json_loads(metadata.get("keywords")),
            "entities": self._safe_json_loads(metadata.get("entities")),
            "sentiment": metadata.get("sentiment"),
            "category": metadata.get("category"),
        }

    def _extract_article_summary(self, metadata: Dict) -> Dict:
        """Extract article summary from metadata (without full content)."""
        return {
            "id": metadata.get("id"),
            "url": metadata.get("url"),
            "title": metadata.get("title"),
            "summary": metadata.get("summary"),
            "category": metadata.get("category"),
            "sentiment": metadata.get("sentiment"),
            "keywords": self._safe_json_loads(metadata.get("keywords")),
            "entities": self._safe_json_loads(metadata.get("entities")),
        }

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
        return sorted(chunks, key=lambda x: x.get("index", 0))

    def _get_chunk_ids_for_article(self, article_id: str) -> List[str]:
        """Get all chunk IDs for a specific article."""
        return [c["chunk_id"] for c in self.get_chunks_by_article_id(article_id)]

    def _build_deletion_ids(self, article_id: str, chunk_ids: List[str]) -> List[str]:
        """Build list of all IDs to delete (article + chunks)."""
        return [article_id] + chunk_ids

    # ==================== RETRIEVAL METHODS ====================

    def get_by_id(self, article_id: str) -> Optional[Dict]:
        """Get specific article by ID.
        
        Args:
            article_id: Unique identifier for the article
            
        Returns:
            Complete article dictionary or None if not found
        """
        if not article_id or not article_id.strip():
            logger.warning("get_by_id called with empty article_id")
            return None
            
        try:
            results = self.db.get(ids=[article_id], where={"doc_type": "article"})
            if results and results["documents"]:
                metadata = results["metadatas"][0]
                return self._reconstruct_article_from_metadata(metadata)
            return None
        except Exception as e:
            logger.error("Error retrieving article %s: %s", article_id, e)
            return None

    def get_all_articles(self) -> List[Dict]:
        """Get all articles metadata (without full content for efficiency).
        
        Returns:
            List of article summaries without full content
        """
        try:
            results = self.db.get(where={"doc_type": "article"})
            articles = []

            if results and results["metadatas"]:
                for metadata in results["metadatas"]:
                    try:
                        articles.append(self._extract_article_summary(metadata))
                    except Exception as e:
                        logger.warning("Error processing article metadata: %s", e)
                        continue
            return articles
        except Exception as e:
            logger.error("Error retrieving all articles: %s", e)
            return []

    def article_exists(self, article_id: str) -> bool:
        """Check if an article already exists.
        
        Args:
            article_id: Unique identifier for the article
            
        Returns:
            True if article exists, False otherwise
        """
        result = self.db.get(ids=[article_id], where={"doc_type": "article"})
        return bool(result and result["ids"])

    def get_chunks_by_article_id(self, article_id: str) -> List[Dict]:
        """Get all chunks for a specific article.
        
        Args:
            article_id: Unique identifier for the article
            
        Returns:
            List of chunks sorted by index
        """
        if not article_id or not article_id.strip():
            logger.warning("get_chunks_by_article_id called with empty article_id")
            return []
            
        try:
            # Use proper ChromaDB filter format for multiple conditions
            where_filter = {"$and": [{"doc_type": "chunk"}, {"article_id": article_id}]}
            results = self.db.get(where=where_filter)
            chunks = []
            if results and results["documents"]:
                for i, content in enumerate(results["documents"]):
                    metadata = results["metadatas"][i]
                    chunks.append(self._format_chunk_data(content, metadata))
            return self._sort_chunks_by_index(chunks)
        except Exception as e:
            logger.error("Error retrieving chunks for article %s: %s", article_id, e)
            return []

    # ==================== DELETION METHODS ====================

    def delete_article(self, article_id: str) -> bool:
        """Delete article and all its chunks from store.
        
        Args:
            article_id: Unique identifier for the article to delete
            
        Returns:
            True if deletion was successful, False otherwise
        """
        try:
            chunk_ids_to_delete = self._get_chunk_ids_for_article(article_id)
            ids_to_delete = self._build_deletion_ids(article_id, chunk_ids_to_delete)

            if ids_to_delete:
                self.db.delete(ids=ids_to_delete)
                logger.info(
                    "Deleted article %s and its %d chunks.",
                    article_id,
                    len(chunk_ids_to_delete),
                )
            return True
        except Exception as e:
            logger.error("Error deleting article %s: %s", article_id, e)
            return False
