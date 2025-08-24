import requests
from bs4 import BeautifulSoup
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import Dict, List
import json
import os
from datetime import datetime
from models import Article, ArticleMetadata
from logger import logger

class ArticleProcessor:
    def __init__(self, llm_provider = None):
        self.llm = llm_provider or ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
    
    def fetch_article(self, url: str) -> str:
        """Fetch and extract text from URL"""
        try:
            response = requests.get(url, timeout=10)
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
            title = soup.find('title').text if soup.find('title') else url
            
            return title, text[:5000]  # Limit text length
            
        except Exception as e:
            logger.error(f"Error fetching {url}: {e}")
            return url, ""
    
    def process_with_llm(self, title: str, content: str) -> Dict:
        """Extract metadata using LLM"""
         # Configure the LLM with structured output using Pydantic model
        structured_llm = self.llm.with_structured_output(ArticleMetadata)
        prompt = f"""
        Analyze this article and return JSON with:
        1. summary: 2-3 sentence summary
        2. keywords: list of 5-7 key topics
        3. sentiment: float between -1 (negative) and 1 (positive)
        4. category: one of [technology, business, politics, other]
        
        Article Title: {title}
        Content: {content}
        
        Return ONLY valid JSON, no explanation.
        """
    
        try:
            # Invoke the structured LLM
            metadata = structured_llm.invoke(prompt)
            return metadata.model_dump()
        except:
            # Fallback if LLM fails
            return {
                "summary": title,
                "keywords": ["article"],
                "sentiment": 0.0,
                "category": "other"
            }
    
    def process_url(self, url: str) -> Article:
        """Main processing pipeline"""
        logger.info(f"Processing: {url}")
        
        # Fetch article
        title, content = self.fetch_article(url)
        if not content:
            return None
        
        # Process with LLM
        metadata_dict = self.process_with_llm(title, content)
        
        # Create ArticleMetadata object
        metadata = ArticleMetadata(
            summary=metadata_dict.get("summary", title),
            keywords=metadata_dict.get("keywords", []),
            sentiment=metadata_dict.get("sentiment", 0.0),
            category=metadata_dict.get("category", "other")
        )
        
        # Create Article object
        article = Article(
            id=url.replace("https://", "").replace("/", "_"),
            url=url,
            title=title,
            content=content,
            metadata=metadata
        )
        
        return article
    
    def process_batch(self, urls: List[str]) -> List[Article]:
        """Process multiple URLs"""
        articles = []
        for url in urls:
            article = self.process_url(url)
            if article:
                articles.append(article)
                # Save progress
                self.save_to_file(articles)
        
        return articles
    
    def save_to_file(self, articles: List[Article]):
        """Save to JSON file as backup"""
        with open("data/articles.json", "w") as f:
            json.dump(
                [art.model_dump() for art in articles],
                f,
                indent=2,
                default=str
            )
        logger.info(f"Saved {len(articles)} articles")


if __name__ == "__main__":
    logger.info("Starting article processing...")
    ingestion = ArticleProcessor()
    url = "https://techcrunch.com/2025/07/26/astronomer-winks-at-viral-notoriety-with-temporary-spokesperson-gwyneth-paltrow/"
    article = ingestion.process_url(url)
    if article:
        logger.info(f"Processed article: {article.title}")
    else:
        logger.warning("Failed to process article.")

    ingestion.save_to_file([article])
