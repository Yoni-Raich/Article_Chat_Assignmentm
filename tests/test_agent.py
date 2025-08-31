"""
Tests for the LangGraph ReAct Agent.

This module contains tests for the agent's ability to initialize,
select tools, and respond to various queries using the new chunk-based tools.
"""

import os
import unittest
from unittest.mock import patch, MagicMock

# Third-party imports
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph

# Add project root to Python path
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Local imports
from src.agent import ArticleAnalysisAgent
from src.vector_store import VectorStore
from src.tools import get_list_of_tools, init_tools, search_article_chunks, get_article_details_by_id
from src.models import AgentState


@patch.dict(os.environ, {"GOOGLE_API_KEY": "test_api_key"})
@patch('src.tools.VectorStore')
class TestAgentAndTools(unittest.TestCase):
    """Test suite for the ReAct agent and its tools with a mocked VectorStore."""

    def test_search_article_chunks_tool(self, MockVectorStore):
        """Test if the search_article_chunks tool works correctly."""
        # Arrange
        mock_vs_instance = MockVectorStore.return_value
        mock_vs_instance.search_chunks.return_value = [
            {"chunk_id": "chunk1", "article_id": "art1", "content": "This is a test chunk."}
        ]
        init_tools(vector_store_instance=mock_vs_instance)

        # Act
        result = search_article_chunks.invoke({"query": "test"})

        # Assert
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['chunk_id'], 'chunk1')
        mock_vs_instance.search_chunks.assert_called_once_with("test", k=5)
        print("✅ Tool 'search_article_chunks' works as expected.")

    def test_get_article_details_by_id_tool(self, MockVectorStore):
        """Test if the get_article_details_by_id tool works correctly."""
        # Arrange
        mock_vs_instance = MockVectorStore.return_value
        mock_vs_instance.get_by_id.return_value = {
            "id": "art1", "title": "Test Article", "full_content": "This is the full content."
        }
        init_tools(vector_store_instance=mock_vs_instance)

        # Act
        result = get_article_details_by_id.invoke({"article_id": "art1"})

        # Assert
        self.assertIsInstance(result, dict)
        self.assertEqual(result['id'], 'art1')
        self.assertIn('full_content', result)
        mock_vs_instance.get_by_id.assert_called_once_with("art1")
        print("✅ Tool 'get_article_details_by_id' works as expected.")

    def test_get_article_details_not_found(self, MockVectorStore):
        """Test the get_article_details_by_id tool when an article is not found."""
        # Arrange
        mock_vs_instance = MockVectorStore.return_value
        mock_vs_instance.get_by_id.return_value = None  # Simulate not found
        init_tools(vector_store_instance=mock_vs_instance)

        # Act
        result = get_article_details_by_id.invoke({"article_id": "art_not_found"})

        # Assert
        self.assertIn("error", result)
        self.assertIn("not found", result["error"])
        mock_vs_instance.get_by_id.assert_called_once_with("art_not_found")
        print("✅ Tool 'get_article_details_by_id' handles not found errors.")


if __name__ == "__main__":
    unittest.main()
