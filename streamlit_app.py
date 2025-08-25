import streamlit as st
import os
import sys
import asyncio
import threading
from typing import Dict, Any

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.agent4 import ArticleAnalysisAgent
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def check_environment():
    """Check if the environment is properly set up"""
    issues = []
    
    # Check Google API key
    if not os.getenv("GOOGLE_API_KEY"):
        issues.append("❌ GOOGLE_API_KEY environment variable not found")
    
    # Check if data directory exists
    if not os.path.exists("data"):
        issues.append("❌ 'data' directory not found")
    
    # Check if source files exist
    if not os.path.exists("src/agent4.py"):
        issues.append("❌ 'src/agent4.py' not found")
    
    return issues

def initialize_agent():
    """Initialize the ArticleAnalysisAgent with error handling"""
    try:
        if "agent" not in st.session_state:
            with st.spinner("Initializing Article Analysis Agent..."):
                # Ensure we have an event loop for async operations
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    # No event loop in current thread, create one
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                st.session_state.agent = ArticleAnalysisAgent()
            st.success("Agent initialized successfully!")
        return st.session_state.agent
    except Exception as e:
        st.error(f"Failed to initialize agent: {str(e)}")
        st.error("Please check that your GOOGLE_API_KEY is set correctly and try refreshing the page.")
        return None

def main():
    st.set_page_config(
        page_title="Article Chat Assistant",
        page_icon="📰",
        layout="wide"
    )
    
    st.title("📰 Article Analysis Chat Assistant")
    st.markdown("Ask questions about your articles and get intelligent responses!")
    
    # Check environment before initializing agent
    env_issues = check_environment()
    if env_issues:
        st.error("⚠️ Environment setup issues detected:")
        for issue in env_issues:
            st.markdown(f"- {issue}")
        st.markdown("""
        **Please fix these issues and refresh the page:**
        1. Create a `.env` file with your Google API key
        2. Make sure you're in the correct directory
        3. Ensure all required files exist
        """)
        st.stop()
    
    # Initialize the agent
    agent = initialize_agent()
    
    if agent is None:
        st.warning("⚠️ Agent initialization failed. Please check the following:")
        st.markdown("""
        1. **Environment Variables**: Make sure you have a `.env` file with your Google API key:
           ```
           GOOGLE_API_KEY=your_api_key_here
           ```
        2. **Dependencies**: Ensure all required packages are installed:
           ```bash
           pip install -r requirements.txt
           ```
        3. **Database**: Check that your vector database exists in `data/chroma_db/`
        
        After fixing these issues, please refresh the page.
        """)
        st.stop()
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Initialize thread ID for conversation continuity
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = "streamlit_session"
    
    # Sidebar with information and controls
    with st.sidebar:
        st.header("🔧 Chat Controls")
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.session_state.thread_id = f"streamlit_session_{len(st.session_state.messages)}"
            st.rerun()
        
        st.markdown("---")
        st.info("🤖 This assistant can search, analyze, and discuss your article database using AI.")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about your articles..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            try:
                # Create a placeholder for the status
                status_placeholder = st.empty()
                
                # Use st.status for dynamic status updates
                with status_placeholder.container():
                    with st.status("🤔 Thinking...", expanded=False) as status:
                        
                        # Set up status callback for the agent
                        def update_status(new_status):
                            status.update(label=new_status)
                        
                        agent.set_status_callback(update_status)
                        
                        # Get response from agent
                        response = agent.query(prompt, st.session_state.thread_id)
                
                # Clear the status completely
                status_placeholder.empty()
                
                # Display the response
                message_placeholder.markdown(response)
                
                # Show tools that were used
                used_tools = agent.get_used_tools()
                if used_tools:
                    with st.expander(f"🔧 Tools used ({len(used_tools)})", expanded=False):
                        for tool in used_tools:
                            st.write(f"• {tool}")
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
                
            except Exception as e:
                error_message = f"❌ Sorry, I encountered an error: {str(e)}"
                message_placeholder.markdown(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})

if __name__ == "__main__":
    main()