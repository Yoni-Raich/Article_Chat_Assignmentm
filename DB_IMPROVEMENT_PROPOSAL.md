# Proposal: Enhancing the Article Database for Advanced Agent Interaction

This document proposes a new architecture for the vector database to improve the capabilities of the ReAct agent, particularly for answering detailed questions that require deep content analysis.

## 1. Current Database Implementation

The current system stores articles in a ChromaDB vector store. The process is as follows:

1.  **Ingestion**: An article's content is fetched from a URL.
2.  **Metadata Extraction**: A Large Language Model (LLM) is used to generate a title, summary, keywords, and other metadata from the article's content (truncated to 5000 characters).
3.  **Embedding**: An embedding is created from a concatenated string of the **title, summary, and keywords**.
4.  **Storage**: The embedding is stored in ChromaDB. The full article content is stored in the document's metadata, but it is **not** used for semantic search.

### Limitations of the Current Approach

-   **Shallow Search**: Semantic search only operates on a summary level. The agent cannot find specific details, quotes, or arguments within the full text of the articles.
-   **Limited Cross-Article Comparison**: While the agent can compare articles based on their summaries and metadata, it cannot perform detailed comparisons of arguments or data points mentioned deep within the texts.
-   **Inefficient for "Cross Questions"**: The system is not optimized for questions that require finding specific information across multiple articles (e.g., "Which articles mention the financial results of NVIDIA?"). The current search would likely miss articles where "NVIDIA" is mentioned in the body but not in the summary or keywords.

## 2. Proposed Database Architecture: Content Chunking

To address these limitations, I propose a new database architecture based on **content chunking**.

### 2.1. The Chunking Strategy

Instead of creating a single embedding per article, we will split the full content of each article into smaller, overlapping text chunks.

-   **Chunk Size**: A fixed size, e.g., 512 or 1024 tokens.
-   **Chunk Overlap**: An overlap between chunks, e.g., 100 tokens, to ensure that semantic context is not lost at the boundaries of chunks.

Each chunk will be stored as a separate document in the vector database, with its own embedding.

### 2.2. Data Model

We will have two main types of documents in our database (or two separate collections):

1.  **Article Document (Parent)**: This document will represent the entire article and will contain:
    -   `article_id`: A unique identifier for the article.
    -   `url`: The source URL.
    -   `title`: The article title.
    -   `metadata`: The LLM-generated summary, keywords, sentiment, etc.
    -   **No embedding of its own, or an embedding of the summary for high-level searches.**

2.  **Chunk Document (Child)**: These documents will represent the individual chunks of an article's content.
    -   `chunk_id`: A unique identifier for the chunk (e.g., `{article_id}_chunk_{i}`).
    -   `article_id`: A foreign key linking back to the parent Article Document.
    -   `content`: The text of the chunk.
    -   `embedding`: An embedding of the `content`.
    -   `metadata`: Could include chunk-specific information, like its position in the article.

This parent-child relationship is crucial. It allows us to first find the most relevant chunks of text via semantic search, and then retrieve the full context of the parent article(s) to which those chunks belong.

## 3. New Agent Tools for the Enhanced Database

The ReAct agent will need a new, more powerful set of tools to interact with this chunked database structure.

### Tool 1: `search_article_chunks`

-   **Purpose**: To perform a semantic search over the collection of chunk documents.
-   **Input**: `query: str`, `k: int = 10`
-   **Output**: A list of the top `k` most relevant chunks, including their content, `chunk_id`, and parent `article_id`.
-   **Example**: `search_article_chunks("What were the main challenges for AI startups in the last year?")`

### Tool 2: `get_article_by_id`

-   **Purpose**: To retrieve the full details of a parent article.
-   **Input**: `article_id: str`
-   **Output**: The full Article Document, including title, URL, summary, and all other metadata. This tool would be used after `search_article_chunks` to get the context of the article from which a relevant chunk was found.
-   **Example**: `get_article_by_id("techcrunch_article_123")`

### Tool 3: `get_chunks_for_article`

-   **Purpose**: To retrieve all chunks belonging to a specific article, in order.
-   **Input**: `article_id: str`
-   **Output**: A list of all chunk documents for the given article. This is useful when the agent decides it needs to read the entire article to answer a question.
-   **Example**: `get_chunks_for_article("techcrunch_article_123")`

### Tool 4: `search_articles_by_metadata`

-   **Purpose**: To perform filtered searches on the Article Documents (the parents). This is useful for high-level questions.
-   **Input**: `filter_dict: dict` (e.g., `{"category": "technology", "sentiment_gt": 0.5}`)
-   **Output**: A list of articles matching the filter.
-   **Example**: `search_articles_by_metadata({"category": "business"})`

## 4. How This Solves the User's Request Types

This new architecture and toolset will allow the agent to handle different types of user queries much more effectively.

### 1. Questions about a Specific Article

-   **User Query**: "In the article about Tesla, what were the production numbers mentioned for the Model Y?"
-   **Agent's Process**:
    1.  Use `search_articles_by_metadata` to find the `article_id` for the Tesla article.
    2.  Use `get_chunks_for_article` to get all chunks for that article.
    3.  Perform a local search or use an LLM to scan the chunks for "Model Y" and "production numbers".
    4.  Synthesize the answer.

### 2. Questions about Multiple Articles

-   **User Query**: "Compare the sentiment of the articles about Google and Microsoft."
-   **Agent's Process**:
    1.  Use `search_articles_by_metadata` twice to get the parent articles for Google and Microsoft.
    2.  Compare the `sentiment` field from the metadata of the two articles.
    3.  Provide the comparison.

### 3. Questions about All Articles ("Cross Questions")

-   **User Query**: "Which articles discuss cybersecurity issues, and what are the main threats mentioned?"
-   **Agent's Process**:
    1.  Use `search_article_chunks` with the query "cybersecurity issues and threats".
    2.  The search will return relevant chunks from potentially multiple articles.
    3.  The agent can group the chunks by their parent `article_id`.
    4.  For each group, it can synthesize the threats mentioned.
    5.  It can then present a final answer, citing the different articles as sources.

This approach provides a much more robust and flexible system for interacting with the article database, enabling the agent to answer a wider range of questions with greater accuracy and detail.
