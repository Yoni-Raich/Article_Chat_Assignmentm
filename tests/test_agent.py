"""
Tests for the article analysis agent.

This module contains unit tests for testing the agent functionality.
"""

import sys
import os
# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.prebuilt import create_react_agent
from src.tools import get_list_of_tools, init_tools

def test_agent_basic_functionality():
    """Test basic agent functionality with tools"""
    init_tools()
    tools = get_list_of_tools()

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

    pre_built_agent = create_react_agent(
        llm,
        tools=tools,
        prompt=SystemMessage(content=" You are an article analysis assistant and your task is to find the best tool to use to solve the user's query.")
    )

    messages = [HumanMessage(content="Find articles about artificial intelligence.")]
    messages = pre_built_agent.invoke({"messages": messages})
    for m in messages["messages"]:
        m.pretty_print()

if __name__ == "__main__":
    test_agent_basic_functionality()
