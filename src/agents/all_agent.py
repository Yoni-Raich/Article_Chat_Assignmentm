# src/agents/all_agent.py
from typing import List
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from models import AgentState
from tools import get_all_articles_summary, get_articles_by_category, search_articles, find_most_similar_article


def create_all_agent(llm: ChatGoogleGenerativeAI, tools: List = None):
    """Create agent for all-articles analysis"""
    if tools is None:
        tools = [get_all_articles_summary, get_articles_by_category, search_articles, find_most_similar_article]
    
    def all_agent_node(state: AgentState) -> AgentState:
        agent = create_react_agent(
            model=llm,
            tools=tools,
            prompt=SystemMessage(content="""
            You are providing insights about the entire article collection.
            Use get_all_articles_summary for statistics, get_articles_by_category for filtering.
            Provide comprehensive overviews and trends.
            """)
        )
        
        result = agent.invoke({
            "messages": [HumanMessage(content=state["current_query"])]
        })
        for message in result["messages"]:
            # Check if it's a ToolMessage (which contains tool call results)
            if message.type == "tool":
                print("Tool Name:", message.name)
              
        state["messages"] = result["messages"]
        state["sources"] = []  # All articles queries don't have specific sources
        return state
    
    return all_agent_node