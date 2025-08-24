# src/agents/single_agent.py
from typing import List
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from models import AgentState
from tools import find_most_similar_article, fetch_article_by_url


def create_single_agent(llm: ChatGoogleGenerativeAI, tools: List = None):
    """Create agent for single article queries"""
    if tools is None:
        tools = [fetch_article_by_url, find_most_similar_article]
    
    def single_agent_node(state: AgentState) -> AgentState:
        # Create a sub-agent with tools
        agent = create_react_agent(
            model=llm,
            tools=tools,
            prompt=SystemMessage(content="""
            You are an article analysis assistant
            and your task is to find the best tool to use to solve the user's query.
            When a user asks about a specific article:
            Use fetch_article_by_url if the user provides a specific URL
            Use find_most_similar_article in case the user provides free text
            if the user ask for article that is not found, explain the use that you didn't find the article in the Database
            and he always can use the User Interface to add more articles into the Database
            """)
        )
        
        # Run the agent
        result = agent.invoke({
            "messages": [HumanMessage(content=state["current_query"])]
        })
        
        # Update state with agent's messages
        state["messages"] = result["messages"]
        
        # Extract sources from tool calls
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
                    
        state["sources"] = sources[:3]  # Keep top 3 sources
        return state
    
    return single_agent_node