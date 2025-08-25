# AI-Powered Article Analysis Chat System

This project is an advanced AI-powered chat system designed to analyze a collection of articles and answer complex questions about them. It leverages a sophisticated, tool-using AI agent built with LangGraph and Google's Gemini model.

## 1. Architecture Overview

This project is an AI-powered article analysis system designed to answer complex questions about a collection of articles. The architecture is built around a sophisticated, tool-using AI agent and is deployed as a containerized application.

The system consists of three main parts:
1.  **Dockerized Services**: The entire application runs within Docker containers, orchestrated by `docker-compose.yml`. This creates a consistent and reproducible environment.
    *   **FastAPI Application (`article-chat`)**: A Python-based web server that exposes a REST API for all functionalities, including chat, article ingestion, and system monitoring. It also serves a simple web interface.
    *   **ChromaDB (`chroma-db`)**: A dedicated vector database service used to store article embeddings and metadata, enabling efficient semantic search.

2.  **Data Flow**:
    *   **Ingestion**: When a new article is added via its URL, the `ArticleProcessor` fetches the content, extracts key information (summary, keywords, entities, sentiment) using an LLM, generates vector embeddings from the text, and stores everything in the ChromaDB database. The system can also self-initialize with a predefined list of articles on first startup.
    *   **Query**: When a user sends a query through the API, it is passed to the core AI agent for processing.

3.  **The AI Agent (`ArticleAnalysisAgent`)**: This is the core of the system, built using **LangGraph**. It's not just a simple LLM call; it's a stateful agent that can reason and execute multi-step logic to answer complex questions.
    *   **LangGraph Core**: The agent is implemented as a graph where nodes represent states (like "thinking" or "acting") and edges represent the transitions between them. This allows for robust and predictable execution flows.
    *   **ReAct (Reason + Act) Loop**: The agent operates on a "Reason and Act" principle. When it receives a query:
        1.  **Reason**: The Google Gemini LLM first "reasons" about the query and decides on a plan. It determines if it can answer directly or if it needs more information.
        2.  **Act**: If more information is needed, the LLM chooses one or more **tools** from its arsenal (e.g., `search_articles_by_query`, `analyze_sentiment_for_articles`). The agent then executes these tools. The tools interact with the ChromaDB database to fetch or analyze data.
        3.  **Observe & Repeat**: The output from the tools is fed back into the agent. The LLM then observes the results and returns to the "Reason" step, deciding if it now has enough information to answer the user's question or if it needs to use another tool. This loop continues until the agent is confident it can provide a complete and accurate answer.

This architecture creates a powerful and flexible system where the AI can dynamically access and analyze data to fulfill user requests, going far beyond simple keyword-based search.

## 2. Key Design Decisions

Several key design decisions were made to ensure the system is robust, scalable, and intelligent.

*   **LangGraph for Agent Architecture**: Instead of a simple LLM chain, we chose **LangGraph** to build the core agent. This was a deliberate decision to handle complex, multi-step queries. LangGraph allows us to define the agent's logic as a state machine (a graph), providing better control, observability, and the ability to implement cycles (like the ReAct loop). This makes the agent more powerful and easier to debug than a monolithic chain.

*   **Dedicated Vector Database (ChromaDB)**: We opted for a standalone **ChromaDB** service instead of an in-memory vector store. This decouples the data layer from the application logic, allowing each to be scaled or replaced independently. It also ensures data persistence, so the article embeddings don't have to be regenerated every time the application restarts.

*   **Modular, Tool-Based Agent**: The agent's capabilities are not hard-coded. They are defined by a collection of discrete, well-documented **tools** (e.g., for searching, sentiment analysis, etc.). The LLM is simply given this set of tools and intelligently decides which ones to use based on the user's query. This makes the system highly extensible; new capabilities can be added simply by creating new tools, without altering the agent's core reasoning logic.

*   **Google's Gemini Model**: The system utilizes a **Google Gemini** model as its reasoning engine. This model was chosen for its strong performance in tool-use and instruction-following, which are critical for the agent's ReAct loop to function effectively.

*   **Automatic Data Seeding**: The system is designed for immediate usability. On its first launch, it automatically checks if the database is empty and, if so, ingests a predefined list of articles. This ensures that there is content available for analysis right away without requiring manual setup.

*   **In-Memory Query Caching**: To enhance performance and reduce costs associated with frequent LLM API calls, a simple in-memory **cache** is implemented. It stores the results of recent queries, providing instantaneous responses for repeated questions.

## 3. Instructions to Run Locally with Docker

This project is fully containerized, and you can run the entire system with a single command after a one-time setup.

**Prerequisites:**
*   Docker and Docker Compose are installed on your machine.
*   You have a Google AI API key. You can get one from [Google AI Studio](https://aistudio.google.com/app/apikey).

**Step 1: Create the Environment File**

The application requires an environment file to store your API key.

1.  Find the `.env.example` file in the project root.
2.  Create a copy of this file and name it `.env`.
3.  Open the new `.env` file and replace `YOUR_GOOGLE_API_KEY` with your actual Google API key.

```
# .env file
GOOGLE_API_KEY=your_actual_api_key_here
```

**Step 2: Build and Run the Containers**

Open your terminal in the project's root directory and run the following command:

```bash
docker-compose up --build
```

This command will:
*   Build the Docker image for the `article-chat` FastAPI application.
*   Pull the Docker image for the `chroma-db` service.
*   Start both containers and connect them on a shared network.

**Step 3: Wait for Initialization**

On the first run, the system will automatically download and process a set of articles. You will see logs in your terminal indicating that articles are being initialized. Please wait for this process to complete. You should see a message like `✅ Article initialization complete!`.

**Step 4: Interact with the System**

Once the containers are running and initialized, you can access the system:

*   **Web UI**: Open your web browser and navigate to `http://localhost:8000`. You will find a simple chat interface to interact with the agent.
*   **API Docs**: The full OpenAPI documentation is available at `http://localhost:8000/docs`. You can explore and test all the available API endpoints from here.
*   **ChromaDB UI (Optional)**: The ChromaDB dashboard is accessible at `http://localhost:8001`.

**To Stop the System**

To stop the running containers, press `Ctrl + C` in your terminal. To remove the containers and the associated volume (which stores the database), you can run:

```bash
docker-compose down -v
```
