"""
Article Analysis Agent module using LangGraph and Google Generative AI.

This module provides an AI agent capable of analyzing articles using various tools
for content processing, sentiment analysis, and question answering.
"""


import os
import asyncio
from typing import List, Dict, Any
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

# Import your existing tools
from .tools import init_tools, get_list_of_tools
from .logger import logger

class ArticleAnalysisAgent:
    """
    Elegant LangGraph agent for article analysis with VectorDB integration
    """

    def __init__(self):
        # Initialize tools
        init_tools()
        self.tools = get_list_of_tools()

        # Initialize LLM
        self.llm = ChatGoogleGenerativeAI(
            model=os.getenv("LLM_MODEL_NAME", "gemini-2.5-flash"),
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0
        )

        # Enhanced system prompt with better query classification
        system_prompt = """
        You are an expert article analysis assistant with access to a vector database.
        
        QUERY CLASSIFICATION & STRATEGY:
        
        🎯 SINGLE ARTICLE QUERIES (specific article focus):
        Examples: "What is the main point of the AI ethics article?", "Summarize the climate change piece"
        Strategy:
        → find_article_by_description → get_article_summary → get_article_full_content (if needed)
        
        🔍 MULTI-ARTICLE QUERIES (comparative/thematic):
        Examples: "Compare articles about technology", "What do multiple sources say about X?"
        Strategy:
        → find_articles_by_topic → get_multiple_article_summaries → compare_articles_metadata
        
        📊 DATABASE-WIDE QUERIES (statistics/overview):
        Examples: "What topics are covered?", "Show me trending keywords", "How many positive articles?"
        Strategy:
        → get_database_overview → get_trending_keywords → get_articles_by_category/sentiment
        
        🔄 HYBRID QUERIES (mixed approach needed):
        Examples: "How does this specific article compare to others on the same topic?"
        Strategy: Combine approaches as needed
        
        EFFICIENCY PRINCIPLES:
        ✅ Always start with summaries before full content
        ✅ Use database overview for scope understanding
        ✅ Limit chunks to 3-5 per query unless specifically needed
        ✅ Stop when you have sufficient information to answer
        ❌ Don't load full content unless details are specifically requested
        ❌ Don't search all content when targeted tools exist
        
        TOOL SELECTION PRIORITY:
        1. Most specific tool first (single article > multi-article > database-wide)
        2. Metadata before content (summaries before full text)
        3. Targeted search before broad search
        
        Be accurate, complete, and efficient. Think step-by-step about the query type before selecting tools.
        """

        # Create the agent
        self.app = create_react_agent(
            model=self.llm,
            tools=self.tools,
            prompt=system_prompt,
            checkpointer=MemorySaver()
        )

    def _extract_tool_usage(self, messages: List) -> List[Dict[str, Any]]:
        """
        Extract tool usage information from messages using built-in LangChain parsing.
        
        Args:
            messages: List of messages from agent execution
            
        Returns:
            List of tool usage information
        """
        tool_usage = []
        
        for message in messages:
            # Use built-in tool_calls attribute from AIMessage
            if isinstance(message, AIMessage) and message.tool_calls:
                for tool_call in message.tool_calls:
                    tool_usage.append({
                        "tool_name": tool_call['name'],
                        "arguments": tool_call['args'],
                        "tool_call_id": tool_call['id'],
                        "status": "called"
                    })
            elif isinstance(message, ToolMessage):
                # Update status based on tool results
                for tool_info in tool_usage:
                    if tool_info.get("tool_call_id") == message.tool_call_id:
                        tool_info["result"] = message.content
                        tool_info["status"] = "success"
                        break
        
        return tool_usage

    def _log_tool_usage(self, tool_calls: List[Dict[str, Any]]) -> None:
        """
        Log tool usage information to the logger.
        
        Args:
            tool_calls: List of parsed tool call information
        """
        if not tool_calls:
            logger.info("No tools were used in this query")
            return
        
        logger.info("TOOLS USED:")
        for i, tool in enumerate(tool_calls, 1):
            status_text = "SUCCESS" if tool.get('status') == 'success' else "ERROR" if tool.get('status') == 'error' else "PENDING"
            logger.info(f"  {i}. [{status_text}] {tool['tool_name']}")
            
            # Log arguments (truncated if too long)
            args_str = str(tool['arguments'])
            if len(args_str) > 100:
                args_str = args_str[:100] + "..."
            logger.info(f"     Args: {args_str}")
            
            # Log result status
            if tool.get('status') == 'error':
                result = tool.get('result', 'Unknown error')
                if len(result) > 150:
                    result = result[:150] + "..."
                logger.error(f"     Error: {result}")
            elif tool.get('status') == 'success':
                logger.info(f"     Status: Success")
        
        # Summary
        successful = len([t for t in tool_calls if t.get('status') == 'success'])
        failed = len([t for t in tool_calls if t.get('status') == 'error'])
        logger.info(f"Tool Summary: {successful} successful, {failed} failed, {len(tool_calls)} total")

    def query(self, question: str, thread_id: str = "default") -> str:
        """Query the article analysis system"""
        config = {"configurable": {"thread_id": thread_id}}
        
        logger.info(f"Processing query: {question}")
        
        result = self.app.invoke(
            {"messages": [HumanMessage(content=question)]},
            config=config
        )
        
        # Extract and log tool usage
        tool_usage = self._extract_tool_usage(result["messages"])
        self._log_tool_usage(tool_usage)
        
        return result["messages"][-1].content

    def stream_query(self, question: str, thread_id: str = "default"):
        """Stream the response for real-time interaction"""
        config = {"configurable": {"thread_id": thread_id}}
        
        logger.info(f"Starting streaming query: {question}")
        
        all_messages = []
        for chunk in self.app.stream(
            {"messages": [HumanMessage(content=question)]},
            config=config
        ):
            # Collect messages for tool analysis
            if 'agent' in chunk and 'messages' in chunk['agent']:
                all_messages.extend(chunk['agent']['messages'])
            elif 'tools' in chunk and 'messages' in chunk['tools']:
                all_messages.extend(chunk['tools']['messages'])
                
            yield chunk
        
        # Extract and log tool usage after streaming is complete
        tool_usage = self._extract_tool_usage(all_messages)
        self._log_tool_usage(tool_usage)

# Usage example
if __name__ == "__main__":
    agent = ArticleAnalysisAgent()
    
    while True:
        user_input = input("Ask about articles (or 'exit'): ")
        if user_input.lower() == 'exit':
            break
            
        response = agent.query(user_input)
        print(f"Response: {response}")