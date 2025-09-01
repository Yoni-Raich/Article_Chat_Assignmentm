"""
Comprehensive tests for all tools in the article analysis system.

This module tests all tools defined in src/tools.py to ensure they work correctly
with mocked VectorStore instances and handle various scenarios including
success cases, error cases, and edge cases.
"""

import os
import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime

# Add project root to Python path
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Local imports
from src.tools import (
    init_tools, get_list_of_tools,
    # Single Article Tools
    find_article_by_description, get_article_full_content, 
    get_article_summary, search_within_article,
    # Multiple Article Tools
    find_articles_by_topic, get_multiple_article_summaries,
    search_across_specific_articles, compare_articles_metadata,
    # Database-wide Tools
    get_database_overview, search_all_content,
    get_articles_by_category, get_articles_by_sentiment,
    extract_entities_across_database, get_trending_keywords
)


class TestToolsComprehensive(unittest.TestCase):
    """Comprehensive test suite for all article analysis tools."""

    def setUp(self):
        """Set up test fixtures with mock data."""
        self.mock_vector_store = MagicMock()
        
        # Sample article data for testing
        self.sample_article = {
            "id": "art1",
            "title": "Test Article About AI",
            "url": "https://example.com/ai-article",
            "summary": "This is a comprehensive article about artificial intelligence.",
            "category": "Technology",
            "keywords": ["AI", "machine learning", "technology"],
            "entities": ["OpenAI", "Google", "Microsoft"],
            "sentiment": "positive"
        }
        
        self.sample_article_2 = {
            "id": "art2", 
            "title": "Climate Change Research",
            "url": "https://example.com/climate",
            "summary": "Research findings on global climate change impacts.",
            "category": "Science",
            "keywords": ["climate", "environment", "research"],
            "entities": ["NASA", "IPCC", "United Nations"],
            "sentiment": "neutral"
        }
        
        self.sample_chunks = [
            {
                "chunk_id": "chunk1",
                "article_id": "art1",
                "content": "Artificial intelligence is transforming industries.",
                "title": "Test Article About AI",
                "url": "https://example.com/ai-article",
                "similarity_score": 0.95
            },
            {
                "chunk_id": "chunk2", 
                "article_id": "art1",
                "content": "Machine learning algorithms are becoming more sophisticated.",
                "title": "Test Article About AI",
                "url": "https://example.com/ai-article",
                "similarity_score": 0.88
            }
        ]
        
        # Initialize tools with mock vector store
        init_tools(self.mock_vector_store)

    def test_init_tools_and_get_list(self):
        """Test tool initialization and list retrieval."""
        tools = get_list_of_tools()
        
        # Verify we have all expected tools
        expected_tool_count = 14  # Total number of tools defined
        self.assertEqual(len(tools), expected_tool_count)
        
        # Verify specific tools are present
        tool_names = [tool.name for tool in tools]
        expected_tools = [
            "find_article_by_description", "get_article_full_content",
            "get_article_summary", "search_within_article",
            "find_articles_by_topic", "get_multiple_article_summaries",
            "search_across_specific_articles", "compare_articles_metadata",
            "get_database_overview", "search_all_content",
            "get_articles_by_category", "get_articles_by_sentiment",
            "extract_entities_across_database", "get_trending_keywords"
        ]
        
        for expected_tool in expected_tools:
            self.assertIn(expected_tool, tool_names)
        
        print("✅ Tool initialization and list retrieval works correctly")

    # ==================== SINGLE ARTICLE TOOLS TESTS ====================

    def test_find_article_by_description_success(self):
        """Test finding article by description - success case."""
        # Setup mock
        self.mock_vector_store.search_articles.return_value = [
            {**self.sample_article, "similarity_score": 0.95}
        ]
        
        # Execute
        result = find_article_by_description.invoke({"description": "AI technology"})
        
        # Verify
        self.assertIsInstance(result, dict)
        self.assertEqual(result["id"], "art1")
        self.assertEqual(result["title"], "Test Article About AI")
        self.assertIn("similarity_score", result)
        self.mock_vector_store.search_articles.assert_called_once_with("AI technology", k=1)
        
        print("✅ find_article_by_description works for success case")

    def test_find_article_by_description_not_found(self):
        """Test finding article by description - not found case."""
        # Setup mock
        self.mock_vector_store.search_articles.return_value = []
        
        # Execute
        result = find_article_by_description.invoke({"description": "nonexistent topic"})
        
        # Verify
        self.assertIn("error", result)
        self.assertIn("No article found", result["error"])
        
        print("✅ find_article_by_description handles not found case")

    def test_get_article_full_content_success(self):
        """Test getting full article content - success case."""
        # Setup mock with full_content included
        sample_article_with_content = self.sample_article.copy()
        sample_article_with_content["full_content"] = "This is the full article content..."
        self.mock_vector_store.get_by_id.return_value = sample_article_with_content
        
        # Execute
        result = get_article_full_content.invoke({"article_id": "art1"})
        
        # Verify - now returns the article directly
        self.assertIsInstance(result, dict)
        self.assertEqual(result["id"], "art1")
        self.assertIn("full_content", result)
        self.assertEqual(result["full_content"], "This is the full article content...")
        # No longer restructured into metadata
        self.assertEqual(result["title"], "Test Article About AI")
        
        print("✅ get_article_full_content works for success case")

    def test_get_article_full_content_not_found(self):
        """Test getting full article content - not found case."""
        # Setup mock
        self.mock_vector_store.get_by_id.return_value = None
        
        # Execute
        result = get_article_full_content.invoke({"article_id": "nonexistent"})
        
        # Verify
        self.assertIn("error", result)
        self.assertIn("not found", result["error"])
        
        print("✅ get_article_full_content handles not found case")

    def test_get_article_summary_success(self):
        """Test getting article summary - success case."""
        # Setup mock
        self.mock_vector_store.get_by_id.return_value = self.sample_article
        
        # Execute
        result = get_article_summary.invoke({"article_id": "art1"})
        
        # Verify
        self.assertIsInstance(result, dict)
        self.assertEqual(result["id"], "art1")
        self.assertEqual(result["title"], "Test Article About AI")
        self.assertEqual(result["summary"], "This is a comprehensive article about artificial intelligence.")
        self.assertIn("keywords", result)
        self.assertIn("entities", result)
        
        print("✅ get_article_summary works correctly")

    def test_search_within_article_success(self):
        """Test searching within specific article - success case."""
        # Setup mock
        self.mock_vector_store.search_chunks.return_value = self.sample_chunks[:2]
        
        # Execute
        result = search_within_article.invoke({
            "article_id": "art1", 
            "query": "machine learning",
            "max_chunks": 3
        })
        
        # Verify
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["chunk_id"], "chunk1")
        self.mock_vector_store.search_chunks.assert_called_once_with(
            "machine learning", k=3, filter_dict={"article_id": "art1"}
        )
        
        print("✅ search_within_article works correctly")

    # ==================== MULTIPLE ARTICLE TOOLS TESTS ====================

    def test_find_articles_by_topic_success(self):
        """Test finding multiple articles by topic."""
        # Setup mock
        self.mock_vector_store.search_articles.return_value = [
            {**self.sample_article, "similarity_score": 0.95},
            {**self.sample_article_2, "similarity_score": 0.82}
        ]
        
        # Execute
        result = find_articles_by_topic.invoke({"topic": "technology", "max_articles": 5})
        
        # Verify
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["id"], "art1")
        self.assertEqual(result[1]["id"], "art2")
        self.assertIn("similarity_score", result[0])
        
        print("✅ find_articles_by_topic works correctly")

    def test_get_multiple_article_summaries_success(self):
        """Test getting summaries for multiple articles."""
        # Setup mock for the optimized get_all_articles approach
        self.mock_vector_store.get_all_articles.return_value = [
            self.sample_article,   # id: "art1"
            self.sample_article_2  # id: "art2"
        ]
        
        # Execute
        result = get_multiple_article_summaries.invoke({"article_ids": ["art1", "art2"]})
        
        # Verify
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["id"], "art1")
        self.assertEqual(result[1]["id"], "art2")
        
        print("✅ get_multiple_article_summaries works correctly")

    def test_search_across_specific_articles_success(self):
        """Test searching across specific articles."""
        # Setup mock
        def mock_search_chunks(query, k, filter_dict):
            article_id = filter_dict.get("article_id")
            if article_id == "art1":
                return [self.sample_chunks[0]]
            elif article_id == "art2":
                return [{
                    "chunk_id": "chunk3",
                    "article_id": "art2", 
                    "content": "Climate research shows significant changes.",
                    "title": "Climate Change Research"
                }]
            return []
        
        self.mock_vector_store.search_chunks.side_effect = mock_search_chunks
        
        # Execute
        result = search_across_specific_articles.invoke({
            "article_ids": ["art1", "art2"],
            "query": "research findings",
            "chunks_per_article": 2
        })
        
        # Verify
        self.assertIsInstance(result, dict)
        self.assertIn("art1", result)
        self.assertIn("art2", result)
        self.assertIn("relevant_chunks", result["art1"])
        
        print("✅ search_across_specific_articles works correctly")

    def test_compare_articles_metadata_success(self):
        """Test comparing metadata of multiple articles."""
        # Setup mock
        def mock_get_by_id(article_id):
            if article_id == "art1":
                return self.sample_article
            elif article_id == "art2":
                return self.sample_article_2
            return None
        
        self.mock_vector_store.get_by_id.side_effect = mock_get_by_id
        
        # Execute
        result = compare_articles_metadata.invoke({"article_ids": ["art1", "art2"]})
        
        # Verify
        self.assertIsInstance(result, dict)
        self.assertIn("articles", result)
        self.assertIn("categories", result)
        self.assertIn("sentiments", result)
        self.assertEqual(len(result["articles"]), 2)
        self.assertIn("Technology", result["categories"])
        self.assertIn("Science", result["categories"])
        
        print("✅ compare_articles_metadata works correctly")

    # ==================== DATABASE-WIDE TOOLS TESTS ====================

    def test_get_database_overview_success(self):
        """Test getting database overview."""
        # Setup mock
        self.mock_vector_store.get_all_articles.return_value = [
            self.sample_article, self.sample_article_2
        ]
        
        # Execute
        result = get_database_overview.invoke({})
        
        # Verify
        self.assertIsInstance(result, dict)
        self.assertEqual(result["total_articles"], 2)
        self.assertIn("categories_distribution", result)
        self.assertIn("sentiment_distribution", result)
        self.assertIn("Technology", result["categories_distribution"])
        self.assertIn("Science", result["categories_distribution"])
        
        print("✅ get_database_overview works correctly")

    def test_search_all_content_success(self):
        """Test searching all content in database."""
        # Setup mock
        self.mock_vector_store.search_all.return_value = [
            {**self.sample_article, "type": "article", "similarity_score": 0.95},
            {**self.sample_chunks[0], "type": "chunk", "similarity_score": 0.88}
        ]
        
        # Execute
        result = search_all_content.invoke({"query": "AI technology", "max_results": 10})
        
        # Verify
        self.assertIsInstance(result, dict)
        self.assertIn("articles", result)
        self.assertIn("chunks", result)
        self.assertEqual(len(result["articles"]), 1)
        self.assertEqual(len(result["chunks"]), 1)
        
        print("✅ search_all_content works correctly")

    def test_get_articles_by_category_success(self):
        """Test getting articles by category."""
        # Setup mock
        self.mock_vector_store.get_all_articles.return_value = [
            self.sample_article, self.sample_article_2
        ]
        
        # Execute
        result = get_articles_by_category.invoke({"category": "Technology"})
        
        # Verify
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["id"], "art1")
        self.assertEqual(result[0]["title"], "Test Article About AI")
        
        print("✅ get_articles_by_category works correctly")

    def test_get_articles_by_sentiment_success(self):
        """Test getting articles by sentiment."""
        # Setup mock
        self.mock_vector_store.get_all_articles.return_value = [
            self.sample_article, self.sample_article_2
        ]
        
        # Execute
        result = get_articles_by_sentiment.invoke({"sentiment": "positive"})
        
        # Verify
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["id"], "art1")
        
        print("✅ get_articles_by_sentiment works correctly")

    def test_extract_entities_across_database_success(self):
        """Test extracting entities across database."""
        # Setup mock
        self.mock_vector_store.get_all_articles.return_value = [
            self.sample_article, self.sample_article_2
        ]
        
        # Execute
        result = extract_entities_across_database.invoke({})
        
        # Verify
        self.assertIsInstance(result, dict)
        self.assertIn("total_unique_entities", result)
        self.assertIn("top_entities", result)
        self.assertGreater(result["total_unique_entities"], 0)
        
        print("✅ extract_entities_across_database works correctly")

    def test_extract_entities_with_filter_success(self):
        """Test extracting entities with type filter."""
        # Setup mock
        self.mock_vector_store.get_all_articles.return_value = [
            self.sample_article, self.sample_article_2
        ]
        
        # Execute
        result = extract_entities_across_database.invoke({"entity_type": "organization"})
        
        # Verify
        self.assertIsInstance(result, dict)
        self.assertIn("entity_type_filter", result)
        self.assertEqual(result["entity_type_filter"], "organization")
        
        print("✅ extract_entities_across_database works with filters")

    def test_get_trending_keywords_success(self):
        """Test getting trending keywords."""
        # Setup mock
        self.mock_vector_store.get_all_articles.return_value = [
            self.sample_article, self.sample_article_2
        ]
        
        # Execute
        result = get_trending_keywords.invoke({})
        
        # Verify
        self.assertIsInstance(result, dict)
        self.assertIn("total_unique_keywords", result)
        self.assertIn("trending_keywords", result)
        self.assertIn("keyword_cloud", result)
        self.assertGreater(result["total_unique_keywords"], 0)
        
        print("✅ get_trending_keywords works correctly")

    # ==================== ERROR HANDLING TESTS ====================

    def test_tools_without_vector_store_initialization(self):
        """Test tools behavior when vector store is not initialized."""
        # Reset global vector store
        import src.tools
        src.tools.VECTOR_STORE = None
        
        # Test various tools
        result1 = find_article_by_description.invoke({"description": "test"})
        result2 = get_database_overview.invoke({})
        result3 = search_all_content.invoke({"query": "test"})
        
        # Verify all return error messages
        self.assertIn("error", result1)
        self.assertIn("Vector store not initialized", result1["error"])
        self.assertIn("error", result2)
        self.assertIn("error", result3)
        
        # Reinitialize for other tests
        init_tools(self.mock_vector_store)
        
        print("✅ Tools handle uninitialized vector store correctly")

    def test_tools_with_empty_database(self):
        """Test tools behavior with empty database."""
        # Setup mock for empty database
        self.mock_vector_store.get_all_articles.return_value = []
        self.mock_vector_store.search_articles.return_value = []
        self.mock_vector_store.search_all.return_value = []
        
        # Test various tools
        result1 = get_database_overview.invoke({})
        result2 = find_articles_by_topic.invoke({"topic": "test"})
        result3 = get_trending_keywords.invoke({})
        
        # Verify appropriate handling
        self.assertEqual(result1["total_articles"], 0)
        self.assertEqual(len(result2), 0)
        self.assertEqual(result3["total_unique_keywords"], 0)
        
        print("✅ Tools handle empty database correctly")

    def test_edge_cases_and_malformed_data(self):
        """Test tools with edge cases and malformed data."""
        # Test with article missing optional fields but with defaults
        incomplete_article = {
            "id": "art3",
            "title": "Incomplete Article",
            "url": "https://example.com/incomplete",
            "summary": "",  # Empty summary
            "keywords": [],  # Empty keywords
            "entities": [],  # Empty entities
            "sentiment": None,  # None sentiment
            "category": None   # None category
        }
        
        self.mock_vector_store.get_by_id.return_value = incomplete_article
        
        # Execute
        result = get_article_summary.invoke({"article_id": "art3"})
        
        # Verify graceful handling
        self.assertIsInstance(result, dict)
        self.assertEqual(result["id"], "art3")
        self.assertEqual(result["title"], "Incomplete Article")
        # Should handle missing/empty fields gracefully
        
        print("✅ Tools handle malformed data gracefully")

    def test_tools_with_completely_missing_keys(self):
        """Test tools behavior when article has completely missing required keys."""
        # Test with article missing required fields entirely
        broken_article = {
            "id": "art4",
            "title": "Broken Article"
            # Missing most required fields
        }
        
        self.mock_vector_store.get_by_id.return_value = broken_article
        
        # This should raise an error or handle gracefully
        try:
            result = get_article_summary.invoke({"article_id": "art4"})
            # If it doesn't raise an error, it should at least return the available data
            self.assertIsInstance(result, dict)
            self.assertEqual(result["id"], "art4")
        except KeyError:
            # This is expected behavior for completely malformed data
            pass
        
        print("✅ Tools handle completely missing keys appropriately")


if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(verbosity=2)