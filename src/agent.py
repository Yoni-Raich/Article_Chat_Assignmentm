"""
Article Analysis Agent module using LangGraph and Google Generative AI.

This module provides an AI agent capable of analyzing articles using various tools
for content processing, sentiment analysis, and question answering.
"""

import os
import asyncio
from typing import Dict, Any
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

# Import your existing tools
from .tools import init_tools, get_list_of_tools

class ArticleAnalysisAgent:
    """
    Elegant LangGraph agent for article analysis with VectorDB integration
    """

    def __init__(self):
        # Ensure we have an event loop for async operations
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # No event loop in current thread, create one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        # Initialize tools with dependencies
        init_tools()
        self.tools = get_list_of_tools()

        # Initialize status tracking for UI - combine into single dict
        self.status = {
            "current": "Thinking...",
            "callback": None,
            "used_tools": []
        }

        # Initialize LLM with tools
        self.llm = ChatGoogleGenerativeAI(
            model=os.getenv("LLM_MODEL_NAME", "gemini-2.5-flash"),
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.1
        ).bind_tools(self.tools)

        # Create tool node
        self.tool_node = ToolNode(self.tools)

        # Build the graph
        self.graph = self._build_graph()

        # Add memory for stateful conversations
        memory = MemorySaver()
        self.app = self.graph.compile(checkpointer=memory)

        # Try to generate graph image, but don't fail if it doesn't work
        try:
            graph_image = self.app.get_graph().draw_mermaid_png()
            # Save to file
            with open("workflow_graph.png", "wb") as f:
                f.write(graph_image)
        except Exception as e:
            print(f"Warning: Could not generate workflow graph: {e}")

    def set_status_callback(self, callback):
        """Set a callback function to update status in UI"""
        self.status["callback"] = callback

    def _update_status(self, status):
        """Update current status and call callback if set"""
        self.status["current"] = status
        callback = self.status["callback"]
        if callback and callable(callback):
            callback(status)  # pylint: disable=not-callable

    def _tool_execution_wrapper(self, state: MessagesState) -> Dict[str, Any]:
        """Wrapper for tool execution with logging"""
        # Get the last message which should contain tool calls
        messages = state["messages"]
        if messages and hasattr(messages[-1], 'tool_calls') and messages[-1].tool_calls:
            # Track tools and update status with tool names
            tool_names = [tool_call['name'] for tool_call in messages[-1].tool_calls]
            self.status["used_tools"].extend(tool_names)  # Add to used tools list

            if len(tool_names) == 1:
                self._update_status(f"🔧 Running {tool_names[0]}...")
            else:
                self._update_status(f"🔧 Running {len(tool_names)} tools...")

            print("\n⚙️  Executing tools:")
            for i, tool_call in enumerate(messages[-1].tool_calls, 1):
                print(f"  {i}. Running {tool_call['name']}...")
            print()

        # Execute the tool node
        result = self.tool_node.invoke(state)

        # Print completion message and update status
        if messages and hasattr(messages[-1], 'tool_calls') and messages[-1].tool_calls:
            print("✅ Tool execution completed\n")
            self._update_status("🤔 Processing results...")

        return result

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""

        # Define the graph with MessagesState
        workflow = StateGraph(MessagesState)

        # Add nodes
        workflow.add_node("agent", self._call_agent)
        workflow.add_node("tools", self._tool_execution_wrapper)

        # Set entry point
        workflow.add_edge(START, "agent")

        # Add conditional routing after agent
        workflow.add_conditional_edges(
            "agent",
            tools_condition,  # Built-in condition checker
            {
                "tools": "tools",  # If tools are called, go to tools
                END: END,         # If no tools, end conversation
            }
        )

        # Return to agent after tool execution
        workflow.add_edge("tools", "agent")

        return workflow

    def _call_agent(self, state: MessagesState) -> Dict[str, Any]:
        """
        Main agent node - decides whether to use tools or respond directly
        """
        # Update status
        self._update_status("🤔 Thinking...")

        # Get the conversation history
        messages = state["messages"]

        # Add system message for article analysis context
        system_message = SystemMessage(content="""
        You are an expert article analysis assistant. You have access to a comprehensive vector database
        containing articles with the following capabilities:

        - Search articles by semantic similarity
        - Analyze sentiment across multiple articles
        - Compare articles and extract insights
        - Find articles by specific entities or categories
        - Extract summaries and key information

        Use the available tools when you need to:
        - Search for specific articles or topics
        - Get detailed content from articles
        - Perform sentiment analysis
        - Compare multiple articles
        - Find articles related to specific entities
        - Get statistics about the article database

        Always provide helpful, accurate, and well-structured responses based on the retrieved information.
        """)

        # Combine system message with user messages
        full_messages = [system_message] + messages

        # Call the LLM
        response = self.llm.invoke(full_messages)

        # Print tool calls if any
        if hasattr(response, 'tool_calls') and response.tool_calls:
            print("\n🔧 Tools being used:")
            for i, tool_call in enumerate(response.tool_calls, 1):
                print(f"  {i}. {tool_call['name']} - {tool_call.get('args', {})}")
            print()

        return {"messages": [response]}

    def query(self, question: str, thread_id: str = "default") -> str:
        """
        Simple interface to query the article analysis system

        Args:
            question: User's question about articles
            thread_id: Conversation thread identifier

        Returns:
            Agent's response
        """
        # Reset used tools for new query
        self.status["used_tools"] = []

        config = {"configurable": {"thread_id": thread_id}}

        # Execute the workflow
        result = self.app.invoke(
            {"messages": [HumanMessage(content=question)]},
            config=config
        )

        # Return the final response
        return result["messages"][-1].content

    def get_used_tools(self) -> list:
        """Get list of tools used in the last query"""
        return list(set(self.status["used_tools"]))  # Remove duplicates

    def stream_query(self, question: str, thread_id: str = "default"):
        """
        Stream the response for real-time interaction

        Args:
            question: User's question
            thread_id: Conversation thread identifier

        Yields:
            Updates from each node in the graph
        """
        config = {"configurable": {"thread_id": thread_id}}

        final_response = None
        for chunk in self.app.stream(
            {"messages": [HumanMessage(content=question)]},
            config=config
        ):
            # Store the final response
            if 'agent' in chunk and 'messages' in chunk['agent']:
                final_response = chunk['agent']['messages'][-1].content
            yield chunk

        return final_response

# Simple usage example
def main():
    """Example usage of the ArticleAnalysisAgent"""

    # Create the agent
    agent = ArticleAnalysisAgent()

    # Example queries that match your use cases

    print("🤖 Article Analysis Agent Ready!")
    print("=" * 50)
    while True:
        user_input = input("Enter your query (or 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break

        print(f"🔍 Processing query: {user_input}")

        # Stream the response and collect the final answer
        final_response = None
        for chunk in agent.stream_query(user_input):
            # The streaming shows tool executions in real-time
            if 'agent' in chunk and 'messages' in chunk['agent']:
                final_response = chunk['agent']['messages'][-1].content

        if final_response:
            print(f"💬 Response: {final_response}")
        else:
            print("❌ No response received")
        print("-" * 50)

if __name__ == "__main__":
    main()
