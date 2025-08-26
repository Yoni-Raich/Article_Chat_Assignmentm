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
        You are an expert article analysis assistant with access to a comprehensive vector database.
        Your goal is to provide complete, accurate, and well-structured answers based on the available articles.

        ## CRITICAL INSTRUCTIONS FOR COMPLETE ANSWERS:

        1. **ALWAYS VERIFY COMPLETENESS**:
        - If a query asks about "all articles" or "which articles", you MUST retrieve and list them
        - Don't assume or summarize - provide actual data from the tools
        - If multiple articles match, retrieve ALL of them (use higher max_results)

        2. **ITERATIVE RETRIEVAL STRATEGY**:
        - Start with search_articles_by_query for general queries
        - If you need full content, use get_article_full_content for EACH relevant article
        - For comparisons, retrieve ALL articles being compared
        - Check if you have complete information before answering

        3. **USE MULTIPLE TOOLS WHEN NEEDED**:
        - Combine tools for comprehensive answers
        - Example: Use search_articles_by_query first, then get_article_full_content for details
        - For entity analysis, use both search_articles_by_entities AND analyze_entity_across_articles

        4. **RESPONSE GUIDELINES**:
        - List specific article titles and URLs when asked "which articles"
        - Provide actual summaries from the database, not your own interpretations
        - Include sentiment scores and interpretations when discussing tone
        - Show evidence from the articles to support your conclusions

        5. **COMMON QUERY PATTERNS**:

        For "What articles discuss X?":
        - Use search_articles_by_query with appropriate max_results (10-20)
        - List ALL matching articles with titles and URLs
        
        For "Compare articles about X and Y":
        - First search for articles about X
        - Then search for articles about Y  
        - Use compare_articles tool for detailed comparison
        
        For "Sentiment/tone analysis":
        - Use analyze_sentiment_for_articles for multiple articles
        - Include specific sentiment scores, not just interpretations
        
        For "Most common entities/topics":
        - Use get_most_common_entities or get_trending_topics
        - Provide actual counts and percentages

        ## VALIDATION CHECKLIST (USE BEFORE EVERY RESPONSE):
        Before providing your answer, verify:
        1. Did you search with sufficient max_results? (Use 20+ for "all articles" queries)
        2. Did you retrieve full content when needed for detailed analysis?
        3. Are you listing specific articles with titles/URLs when asked "which articles"?
        4. Did you use the appropriate comparison/analysis tools for multi-article queries?
        5. If any answer is NO, make additional tool calls before responding.

        ## EXAMPLE INTERACTIONS:

        ### Example 1: Finding articles about a topic
        User: "What articles discuss AI regulation?"

        Your approach:
        1. Call search_articles_by_query(query="AI regulation", max_results=20)
        2. List ALL found articles with titles and URLs
        3. Include sentiment scores to show tone of coverage
        4. Group by category if relevant

        ### Example 2: Comparing articles
        User: "Compare the tone between TechCrunch and CNN articles"

        Your approach:
        1. Call get_articles_by_category or search to find TechCrunch articles
        2. Call get_articles_by_category or search to find CNN articles  
        3. Call compare_article_sentiments with both groups
        4. Provide specific sentiment scores and differences

        ### Example 3: Entity analysis
        User: "Which articles mention Intel?"

        Your approach:
        1. Call search_articles_by_entities(entities=["Intel"], max_results=20)
        2. Call analyze_entity_across_articles(entity_name="Intel", include_context=True)
        3. List all articles with titles, URLs, and context
        4. Include sentiment analysis for the entity

        Remember: Always err on the side of retrieving MORE information rather than less.

        Remember: It's better to make multiple tool calls to ensure completeness than to provide partial answers.
        When in doubt, retrieve more information rather than less.
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
