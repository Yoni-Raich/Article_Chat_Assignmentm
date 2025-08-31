"""
Article ingestion and processing module.

This module handles fetching, processing, and analyzing articles from URLs
using web scraping and AI-powered content analysis.
"""

# Standard library imports
import json
import os
from typing import List

# Third-party imports
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI

# Local imports
from .models import Article, ArticleMetadata, Chunk
from .logger import logger

class ArticleProcessor:
    """
    Article processor for fetching and analyzing web articles.

    This class handles web scraping, content extraction, and AI-powered analysis
    of articles from URLs.
    """
    def __init__(self, llm_provider=None):
        self.llm = llm_provider or ChatGoogleGenerativeAI(
            model=os.getenv("LLM_MODEL_NAME", "gemini-1.5-flash"),
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150
        )

    def fetch_article(self, article_url: str) -> str:
        """Fetch and extract text from URL"""
        try:
            response = requests.get(article_url, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')

            # Remove scripts and styles
            for script in soup(["script", "style"]):
                script.decompose()

            # Get text
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)

            # Get title
            title = soup.find('title').text if soup.find('title') else article_url

            return title, text

        except Exception as e:
            logger.error("Error fetching %s: %s", article_url, e)
            return article_url, ""

    def process_with_llm(self, title: str, content: str) -> ArticleMetadata:
        """Extract metadata using LLM"""
         # Configure the LLM with structured output using the Pydantic model
        structured_llm = self.llm.with_structured_output(ArticleMetadata)
        prompt = f"""
        Analyze this article and return JSON with:
        1. summary: 2-3 sentence summary
        2. keywords: list of 5-7 key topics
        3. entities: list of 5-10 important named entities (people, organizations, locations, products, brands, etc.)
        4. sentiment: float between -1 (negative) and 1 (positive)
        5. category: one of [technology, business, politics, other]

        For entities, extract proper nouns like:
        - People: CEO names, politicians, celebrities, experts quoted
        - Organizations: companies, government agencies, nonprofits
        - Locations: countries, cities, regions mentioned
        - Products: software, devices, services, brands
        - Events: conferences, incidents, campaigns

        Article Title: {title}
        Content: {content}

        Return ONLY valid JSON, no explanation.
        """

        try:
            # Use a smaller portion of content for LLM metadata extraction
            content_for_llm = content[:4000]
            # Invoke the structured LLM
            metadata = structured_llm.invoke(prompt.format(content=content_for_llm, title=title))
            return metadata
        except Exception as e:
            logger.warning("LLM extraction failed: %s, using fallback", e)
            # Fallback if LLM fails
            return ArticleMetadata(
                summary=title,
                keywords=["article"],
                entities=[],
                sentiment=0.0,
                category="other"
            )

    def _generate_article_id(self, article_url: str) -> str:
        """Generate a unique ID from article URL using hash"""
        import hashlib
        return hashlib.sha256(article_url.encode()).hexdigest()[:16]

    def _create_article_object(self, article_url: str, title: str, content: str, metadata: ArticleMetadata) -> Article:
        """Create Article object from processed data"""
        article_id = self._generate_article_id(article_url)
        return Article(
            id=article_id,
            url=article_url,
            title=title,
            content=content,
            metadata=metadata
        )

    def _create_chunks(self, article_id: str, content: str) -> List[Chunk]:
        """Split content into chunks and create Chunk objects"""
        text_chunks = self.text_splitter.split_text(content)
        chunks = []
        for i, text_chunk in enumerate(text_chunks):
            chunk_id = f"{article_id}_chunk_{i}"
            chunk = Chunk(
                id=chunk_id,
                article_id=article_id,
                content=text_chunk,
                index=i
            )
            chunks.append(chunk)
        return chunks

    def process_url(self, article_url: str) -> tuple[Article, List[Chunk]]:
        """Main processing pipeline"""
        logger.info("Processing: %s", article_url)

        # Fetch article content
        title, content = self.fetch_article(article_url)
        if not content:
            return None, []

        # Extract metadata using LLM
        metadata = self.process_with_llm(title, content)

        # Create article object
        article_obj = self._create_article_object(article_url, title, content, metadata)

        # Create content chunks
        chunks = self._create_chunks(article_obj.id, content)

        logger.info("Created %d chunks for article: %s", len(chunks), title)
        return article_obj, chunks

    def process_batch(self, urls: List[str]) -> list[tuple[Article, List[Chunk]]]:
        """Process multiple URLs"""
        processed_data = []
        for url in urls:
            article, chunks = self.process_url(url)
            if article and chunks:
                processed_data.append((article, chunks))
        return processed_data

    def save_to_file(self, articles: List[Article], chunks: List[List[Chunk]] = None):
        """Save articles and chunks to JSON files as backup"""
        # Save articles
        with open("articles.json", "w", encoding="utf-8") as f:
            json.dump(
                [art.model_dump() for art in articles],
                f,
                indent=2,
                default=str
            )
        logger.info("Saved %s articles", len(articles))
        
        # Save chunks if provided
        if chunks:
            # Flatten the list of chunk lists
            all_chunks = [chunk for chunk_list in chunks for chunk in chunk_list]
            with open("chunks.json", "w", encoding="utf-8") as f:
                json.dump(
                    [chunk.model_dump() for chunk in all_chunks],
                    f,
                    indent=2,
                    default=str
                )
            logger.info("Saved %s chunks", len(all_chunks))


if __name__ == "__main__":
    logger.info("Starting article processing...")
    ingestion = ArticleProcessor()
    TEST_URL = (
        "https://techcrunch.com/2025/07/26/"
        "astronomer-winks-at-viral-notoriety-with-temporary-spokesperson-gwyneth-paltrow/"
    )
    article, chunks = ingestion.process_url(TEST_URL)
    if article:
        logger.info("Processed article: %s", article.title)
        logger.info("Created %d chunks.", len(chunks))
    else:
        logger.warning("Failed to process article.")

    if article:
        ingestion.save_to_file([article], [chunks])
