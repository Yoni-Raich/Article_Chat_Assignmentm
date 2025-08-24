# src/agent.py - Agent-based version with tools and memory
from typing import TypedDict, List, Dict, Annotated, Sequence
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent, ToolNode
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
import os
from models import QueryType
from logger import logger
from tools import (
    init_tools,
    search_articles,
    get_article_content,
    analyze_sentiment_batch,
    get_articles_by_category,
    compare_articles,
    get_all_articles_summary,
    find_most_similar_article
)

# State definition with message history
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    query_type: QueryType
    current_query: str
    final_answer: str
    sources: List[str]

class ArticleAgent:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.3
        )
        
        # Note: Memory is now handled by LangGraph state persistence
        # The 'messages' field in AgentState automatically accumulates conversation history
        
        # Tools for different query types
        init_tools()
        self.single_tools = [find_most_similar_article]
        self.multi_tools = [search_articles, compare_articles, analyze_sentiment_batch, find_most_similar_article]
        self.all_tools = [get_all_articles_summary, get_articles_by_category, search_articles, find_most_similar_article]
        
        # Build the main graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the main orchestrator graph"""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("classify", self.classify_query)
        workflow.add_node("single_agent", self.create_single_agent())
        workflow.add_node("multi_agent", self.create_multi_agent())
        workflow.add_node("all_agent", self.create_all_agent())
        workflow.add_node("synthesize", self.synthesize_answer)
        
        # Entry point
        workflow.add_edge(START, "classify")
        
        # Conditional routing based on classification
        workflow.add_conditional_edges(
            "classify",
            self.route_by_type,
            {
                "single": "single_agent",
                "multi": "multi_agent",
                "all": "all_agent"
            }
        )
        
        # All agents lead to synthesis
        workflow.add_edge("single_agent", "synthesize")
        workflow.add_edge("multi_agent", "synthesize")
        workflow.add_edge("all_agent", "synthesize")
        workflow.add_edge("synthesize", END)
        graph = workflow.compile()
        graph_image = graph.get_graph().draw_mermaid_png()

        # Save to file
        with open("workflow_graph.png", "wb") as f:
            f.write(graph_image)
        return graph

    def classify_query(self, state: AgentState) -> AgentState:
        """Classify query into single/multi/all"""
        # Get the last user message
        last_message = state["messages"][-1].content if state["messages"] else ""
        state["current_query"] = last_message
        
        # Configure the LLM with structured output using Pydantic model
        structured_llm = self.llm.with_structured_output(QueryType)
        
        # Build classification prompt with full conversation context
        classification_prompt = """
        Classify this query into ONE category based on the ENTIRE conversation context:
        - "single": Query about ONE specific article or topic requiring detailed info
        - "multi": Query comparing or analyzing 2-5 articles  
        - "all": Query about all articles, statistics, or broad analysis
        
        Consider the full conversation history to understand if this is:
        - A new question (classify based on the current query)
        - A follow-up question (classify based on what the user is asking about from the previous context)
        
        Provide:
        - type: one of "single", "multi", or "all"
        - confidence: how confident you are in this classification (0.0 to 1.0)
        - reasoning: brief explanation for your choice
        """
        
        # Create messages list with classification prompt + full conversation history
        classification_messages = [SystemMessage(content=classification_prompt)]
        classification_messages.extend(state["messages"])
        
        try:
            # Invoke the structured LLM
            response = structured_llm.invoke(classification_messages)
            query_type = response.type            
            # Validate the response
            if query_type not in ["single", "multi", "all"]:
                raise ValueError(f"Invalid query type: {query_type}")
                
        except Exception as e:
            print(f"Structured classification failed: {e}, falling back to keyword-based classification")
            # Fallback to keyword-based classification
            if any(word in last_message.lower() for word in ["all", "every", "total", "statistics"]):
                query_type = "all"
            elif any(word in last_message.lower() for word in ["compare", "difference", "between"]):
                query_type = "multi"
            else:
                query_type = "single"
        
        state["query_type"] = response
        print(f"Query classified as: {query_type}")
        return state
    
    def route_by_type(self, state: AgentState) -> str:
        """Route to appropriate agent"""
        return state["query_type"].type
    
    def create_single_agent(self):
        """Create agent for single article queries"""
        def single_agent_node(state: AgentState) -> AgentState:
            # Create a sub-agent with tools
            
            agent = create_react_agent(
                prompt=SystemMessage(content="""
                You are an article analysis assistant. When a user asks about a specific article:
                1. If they mention an article title or topic, use find_most_similar_article to locate it
                2. Always search for articles before providing answers
                3. Use the tools available to find and analyze the requested content

                Available tools:
                - find_most_similar_article: Use this to find articles by title, topic, or description
                """),
                model=self.llm,
                tools=self.single_tools
            )
            print(self.single_tools)
            # Run the agent
            result = agent.invoke({
                "messages": state["messages"]
            })
            print(result)
            # Update state with agent's messages
            state["messages"] = result["messages"]
            
            # Extract sources from tool calls
            sources = []
            for msg in result["messages"]:
                if hasattr(msg, 'tool_calls'):
                    for call in msg.tool_calls:
                        if 'article_url' in call.get('args', {}):
                            sources.append(call['args']['article_url'])
            
            state["sources"] = sources[:3]  # Keep top 3 sources
            return state
        
        return single_agent_node
    
    def create_multi_agent(self):
        """Create agent for multi-article analysis"""
        def multi_agent_node(state: AgentState) -> AgentState:
            agent = create_react_agent(
                self.llm,
                tools=self.multi_tools,
                prompt=SystemMessage(content="""
                You are comparing and analyzing multiple articles.
                Use search_articles to find relevant content, then use compare_articles or analyze_sentiment_batch.
                Focus on patterns, differences, and insights across articles.
                """)
            )
            
            result = agent.invoke({
                "messages": state["messages"]
            })
            
            state["messages"] = result["messages"]
            
            # Extract sources
            sources = []
            for msg in result["messages"]:
                if hasattr(msg, 'tool_calls'):
                    for call in msg.tool_calls:
                        args = call.get('args', {})
                        if 'article_urls' in args:
                            sources.extend(args['article_urls'])
                        elif 'url1' in args:
                            sources.extend([args['url1'], args.get('url2', '')])
            
            state["sources"] = list(set(sources))[:5]
            return state
        
        return multi_agent_node
    
    def create_all_agent(self):
        """Create agent for all-articles analysis"""
        def all_agent_node(state: AgentState) -> AgentState:
            agent = create_react_agent(
                self.llm,
                tools=self.all_tools,
                state_modifier=SystemMessage(content="""
                You are providing insights about the entire article collection.
                Use get_all_articles_summary for statistics, get_articles_by_category for filtering.
                Provide comprehensive overviews and trends.
                """)
            )
            
            result = agent.invoke({
                "messages": state["messages"]
            })
            
            state["messages"] = result["messages"]
            state["sources"] = []  # All articles queries don't have specific sources
            return state
        
        return all_agent_node
    
    def synthesize_answer(self, state: AgentState) -> AgentState:
        """Synthesize final answer from agent results"""
        # The last assistant message should contain the answer
        for msg in reversed(state["messages"]):
            if msg.type == "ai" and not hasattr(msg, 'tool_calls'):
                state["final_answer"] = msg.content
                break
        
        if not state["final_answer"]:
            # Fallback synthesis
            synthesis_prompt = f"""
            Based on the conversation and analysis, provide a final answer to:
            {state['current_query']}
            
            Context from tools and analysis:
            {self._extract_tool_results(state)}
            """
            
            response = self.llm.invoke([HumanMessage(content=synthesis_prompt)])
            state["final_answer"] = response.content
        
        # Note: LangGraph automatically persists state including conversation history
        # No manual memory saving needed
        
        return state
    
    def _get_history_context(self, state: AgentState) -> str:
        """Get relevant conversation history - simplified since full context is now passed to classification"""
        # Get recent messages from state (excluding current message)
        history = state["messages"][:-1][-4:] if len(state["messages"]) > 1 else []
        if history:
            return "\n".join([f"{msg.type}: {msg.content[:150]}..." for msg in history])
        return "No previous context"
    
    def _extract_tool_results(self, state: AgentState) -> str:
        """Extract tool results from messages"""
        results = []
        for msg in state["messages"]:
            if hasattr(msg, 'content') and msg.type == "tool":
                results.append(msg.content)
        return "\n".join(results[-3:]) if results else "No tool results"
    
    def process_query(self, query: str, conversation_history: List[BaseMessage] = None) -> Dict:
        """Main entry point
        
        Args:
            query: The user's query
            conversation_history: Optional list of previous messages for conversation continuity
        """
        # Build message history
        messages = conversation_history or []
        messages.append(HumanMessage(content=query))
        
        initial_state = {
            "messages": messages,
            "query_type": None,
            "current_query": query,
            "final_answer": "",
            "sources": []
        }
        
        # Run the graph
        result = self.graph.invoke(initial_state)
        
        return {
            "answer": result["final_answer"],
            "sources": result.get("sources", []),
            "query_type": result.get("query_type", "unknown"),
            "conversation_messages": result.get("messages", [])  # Return full conversation for continuity
        }
    
    def clear_memory(self):
        """Clear conversation memory - Note: In LangGraph, memory is handled per conversation thread"""
        # With LangGraph persistence, memory clearing is handled at the thread/conversation level
        # This method is kept for API compatibility but has no effect in the current implementation
        pass


if __name__ == "__main__":
    # Initialize
    agent = ArticleAgent()
    query = "please summarize the key points of the 'The Wizard of Oz' article"
    response = agent.process_query(query)
    print(response)
