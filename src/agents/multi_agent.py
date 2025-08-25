# src/agents/multi_agent.py
from typing import List
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from models import AgentState
from tools import search_articles, compare_articles, analyze_sentiment_batch, find_most_similar_article


def create_multi_agent(llm: ChatGoogleGenerativeAI, tools: List = None):
    """Create agent for multi-article analysis"""
    if tools is None:
        tools = [search_articles, compare_articles, analyze_sentiment_batch, find_most_similar_article]
    
    def multi_agent_node(state: AgentState) -> AgentState:
        agent = create_react_agent(
            model=llm,
            tools=tools,
            prompt=SystemMessage(content="""
            You are an expert at answering questions that require retrieving and analyzing information from multiple articles.
            Use search_articles to find relevant content across the knowledge base, then analyze the information to provide comprehensive answers.
            You can compare articles when needed using compare_articles, analyze sentiment patterns with analyze_sentiment_batch, or find similar content.
            Focus on synthesizing information from multiple sources to provide thorough, well-researched responses.
            """)
        )
        
        result = agent.invoke({
            "messages": [HumanMessage(content=state["current_query"])]
        })
        
        state["messages"] = result["messages"]
        
        # Extract sources
        sources = []
        for message in result["messages"]:
            # Check if it's a ToolMessage (which contains tool call results)
            if message.type == "tool":
                print("Tool Name:", message.name)
            
                # Parse the tool result to extract URL
                try:
                    import json
                    tool_result = json.loads(message.content)
                    # Check if tool_result is not None and contains expected structure
                    if tool_result and "article" in tool_result and tool_result["article"] and "url" in tool_result["article"]:
                        sources.append(tool_result["article"]["url"])
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Error parsing tool result: {e}")        
        state["sources"] = list(set(sources))[:5]
        return state
    
    return multi_agent_node