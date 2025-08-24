# src/agent.py
from typing import TypedDict, List, Dict, Literal
from langgraph.graph import StateGraph, END, START
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage
import os
import json
from tools import (
    search_articles,
    get_article_content,
    analyze_sentiment_batch,
    get_articles_by_category,
    compare_articles,
    get_all_articles_summary
)

# Agent State
class AgentState(TypedDict):
    query: str
    query_type: str
    relevant_articles: List[Dict]
    tool_results: Dict
    final_answer: str
    sources: List[str]

class ArticleAgent:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0
        )
        
        # Build the graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("classify_query", self.classify_query)
        workflow.add_node("search_relevant", self.search_relevant_articles)
        workflow.add_node("fetch_specific", self.fetch_specific_content)
        workflow.add_node("analyze_multiple", self.analyze_multiple_articles)
        workflow.add_node("generate_answer", self.generate_final_answer)
        
        # Add edges
        workflow.add_edge(START,"classify_query")
        
        # Conditional routing based on query type
        workflow.add_conditional_edges(
            "classify_query",
            self.route_query,
            {
                "search": "search_relevant",
                "specific": "fetch_specific",
                "comparison": "analyze_multiple",
                "summary": "generate_answer"
            }
        )
        
        # All paths lead to generate_answer
        workflow.add_edge("search_relevant", "generate_answer")
        workflow.add_edge("fetch_specific", "generate_answer")
        workflow.add_edge("analyze_multiple", "generate_answer")
        workflow.add_edge("generate_answer", END)
        graph = workflow.compile()

        # # TODO - for development please remove in production
        graph_image = graph.get_graph().draw_mermaid_png()
        with open("workflow_graph.png", "wb") as f:
            f.write(graph_image)
        print("Workflow graph saved as 'workflow_graph.png'")
        return graph
    
    def classify_query(self, state: AgentState) -> AgentState:
        """Classify the type of query"""
        # TODO - use llm with structured output!
        query = state["query"]
        
        classification_prompt = f"""
        Classify this query into ONE of these types:
        - "search": User wants to find articles about a topic
        - "specific": User asks about a specific article or wants detailed info
        - "comparison": User wants to compare articles or analyze multiple
        - "summary": User wants general statistics or overview
        
        Query: {query}
        
        Return ONLY the type word, nothing else.
        """
        
        response = self.llm.invoke([HumanMessage(content=classification_prompt)])
        query_type = response.content.strip().lower()
        
        # Validate
        if query_type not in ["search", "specific", "comparison", "summary"]:
            query_type = "search"  # Default
        
        state["query_type"] = query_type
        print(f"Query classified as: {query_type}")
        return state
    
    def route_query(self, state: AgentState) -> str:
        """Route based on query type"""
        return state["query_type"]
    
    def search_relevant_articles(self, state: AgentState) -> AgentState:
        """Search for relevant articles"""
        results = search_articles.invoke({"query": state["query"], "max_results": 5})
        
        state["relevant_articles"] = results
        state["sources"] = [r["url"] for r in results[:3]]  # Top 3 as sources
        
        print(f"Found {len(results)} relevant articles")
        return state
    
    def fetch_specific_content(self, state: AgentState) -> AgentState:
        """Fetch specific article content"""
        # First search to find the article
        results = search_articles.invoke({"query": state["query"], "max_results": 3})
        
        if results:
            # Get full content of most relevant
            content = get_article_content.invoke({"article_url": results[0]["url"]})
            state["tool_results"] = {"article_content": content}
            state["sources"] = [results[0]["url"]]
        else:
            state["tool_results"] = {"error": "No relevant article found"}
        
        return state
    
    def analyze_multiple_articles(self, state: AgentState) -> AgentState:
        """Analyze multiple articles for comparison or aggregate analysis"""
        # Search for relevant articles
        results = search_articles.invoke({"query": state["query"], "max_results": 5})
        
        if len(results) >= 2:
            # Get sentiment analysis
            urls = [r["url"] for r in results]
            sentiment_analysis = analyze_sentiment_batch.invoke({"article_urls": urls})
            
            # If it's a comparison, compare top 2
            if "compare" in state["query"].lower() and len(results) >= 2:
                comparison = compare_articles.invoke({
                    "url1": results[0]["url"],
                    "url2": results[1]["url"]
                })
                state["tool_results"] = {
                    "sentiment": sentiment_analysis,
                    "comparison": comparison
                }
            else:
                state["tool_results"] = {"sentiment": sentiment_analysis}
            
            state["sources"] = urls[:3]
        else:
            state["tool_results"] = {"error": "Not enough articles for analysis"}
        
        state["relevant_articles"] = results
        return state
    
    def generate_final_answer(self, state: AgentState) -> AgentState:
        """Generate the final answer based on gathered information"""
        
        # Build context
        context = f"""
        User Query: {state['query']}
        Query Type: {state.get('query_type', 'unknown')}
        
        Relevant Articles Found: {len(state.get('relevant_articles', []))}
        """
        
        if state.get("relevant_articles"):
            context += "\nArticles:\n"
            for article in state["relevant_articles"][:5]:
                context += f"- {article['title']}\n  Summary: {article['summary']}\n"
        
        if state.get("tool_results"):
            context += f"\nAnalysis Results:\n{json.dumps(state['tool_results'], indent=2)}"
        
        # Generate answer
        answer_prompt = f"""
        Based on the following context, provide a comprehensive answer to the user's query.
        Be specific and reference the articles when relevant.
        
        {context}
        
        Provide a clear, well-structured answer:
        """
        
        response = self.llm.invoke([
            SystemMessage(content="You are a helpful assistant that analyzes articles and provides insights."),
            HumanMessage(content=answer_prompt)
        ])
        
        state["final_answer"] = response.content
        return state
    
    def process_query(self, query: str) -> Dict:
        """Main entry point to process a query"""
        initial_state = {
            "query": query,
            "query_type": "",
            "relevant_articles": [],
            "tool_results": {},
            "final_answer": "",
            "sources": []
        }
        
        # Run the graph
        result = self.graph.invoke(initial_state)
        
        return {
            "answer": result["final_answer"],
            "sources": result.get("sources", []),
            "query_type": result.get("query_type", "unknown")
        }
    

if __name__ == "__main__":
    agent = ArticleAgent()

