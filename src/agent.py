# src/agent.py - Agent-based version with tools and memory
from typing import List, Dict
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
import os
from models import QueryType, AgentState
from logger import logger
from tools import (
    init_tools,
    search_articles,
    get_article_content,
    analyze_sentiment_batch,
    get_articles_by_category,
    compare_articles,
    get_all_articles_summary,
    find_most_similar_article,
    fetch_article_by_url,
    get_most_common_entities,
    get_entities_by_type,
    analyze_entity_sentiment,
    find_articles_by_entity,
    get_all_articles
)
from agents import create_single_agent, create_multi_agent, create_all_agent

class ArticleAgent:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.3
        )
        
        # Note: Memory is now handled by LangGraph state persistence
        # The 'messages' field in AgentState automatically accumulates conversation history
        
        # Tools for different query types
        init_tools()
        self.single_tools = [find_most_similar_article, fetch_article_by_url]
        self.multi_tools = [get_all_articles, search_articles, compare_articles, analyze_sentiment_batch, find_most_similar_article]
        self.all_tools = [get_all_articles_summary, get_articles_by_category, search_articles, find_most_similar_article, 
                         get_most_common_entities, get_entities_by_type, analyze_entity_sentiment, find_articles_by_entity, get_all_articles]
        
        # Build the main graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the main orchestrator graph"""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("classify", self.classify_query)
        workflow.add_node("single_agent", create_single_agent(self.llm, self.single_tools))
        workflow.add_node("multi_agent", create_multi_agent(self.llm, self.multi_tools))
        workflow.add_node("all_agent", create_all_agent(self.llm, self.all_tools))
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
    

    

    

    
    def synthesize_answer(self, state: AgentState) -> AgentState:
        """Synthesize final answer from agent results"""
        # The last assistant message should contain the answer
        for msg in reversed(state["messages"]):
            if msg.type == "ai" and not hasattr(msg, 'tool'):
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

if __name__ == "__main__":
    # Initialize
    agent = ArticleAgent()
    while True:
        query = input("Enter your query (or 'exit' to quit): ")
        if query.lower() == 'exit':
            break

        response = agent.process_query(query)
        print("="*50)
        print(response["answer"])
        print("="*50)
