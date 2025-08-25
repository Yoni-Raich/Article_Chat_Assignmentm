#!/usr/bin/env python3
"""
Article Database Initialization Script

This script processes all 20 articles from the assignment and loads them into the vector database.
It includes parallel processing, progress tracking, and error handling for robust initialization.
"""

import sys
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple
import time
from tqdm import tqdm

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.ingestion import ArticleProcessor
from src.vector_store import VectorStore
from src.logger import logger

# All 20 article URLs from the assignment
ARTICLE_URLS = [
    # TechCrunch URLs
    "https://techcrunch.com/2025/07/26/astronomer-winks-at-viral-notoriety-with-temporary-spokesperson-gwyneth-paltrow/",
    "https://techcrunch.com/2025/07/26/allianz-life-says-majority-of-customers-personal-data-stolen-in-cyberattack/",
    "https://techcrunch.com/2025/07/27/itch-io-is-the-latest-marketplace-to-crack-down-on-adult-games/",
    "https://techcrunch.com/2025/07/26/tesla-vet-says-that-reviewing-real-products-not-mockups-is-the-key-to-staying-innovative/",
    "https://techcrunch.com/2025/07/25/meta-names-shengjia-zhao-as-chief-scientist-of-ai-superintelligence-unit/",
    "https://techcrunch.com/2025/07/26/dating-safety-app-tea-breached-exposing-72000-user-images/",
    "https://techcrunch.com/2025/07/25/sam-altman-warns-theres-no-legal-confidentiality-when-using-chatgpt-as-a-therapist/",
    "https://techcrunch.com/2025/07/25/intel-is-spinning-off-its-network-and-edge-group/",
    "https://techcrunch.com/2025/07/27/wizard-of-oz-blown-up-by-ai-for-giant-sphere-screen/",
    "https://techcrunch.com/2025/07/27/doge-has-built-an-ai-tool-to-slash-federal-regulations/",

    # CNN URLs
    "https://edition.cnn.com/2025/07/27/business/us-china-trade-talks-stockholm-intl-hnk",
    "https://edition.cnn.com/2025/07/27/business/trump-us-eu-trade-deal",
    "https://edition.cnn.com/2025/07/27/business/eu-trade-deal",
    "https://edition.cnn.com/2025/07/26/tech/daydream-ai-online-shopping",
    "https://edition.cnn.com/2025/07/25/tech/meta-ai-superintelligence-team-who-its-hiring",
    "https://edition.cnn.com/2025/07/25/tech/sequoia-islamophobia-maguire-mamdani",
    "https://edition.cnn.com/2025/07/24/tech/intel-layoffs-15-percent-q2-earnings"
]

# Note: The assignment lists 20 URLs but only provides 17. We have all the provided URLs above.


class ArticleInitializer:
    """
    Handles initialization of articles into the vector database with parallel processing,
    progress tracking, and comprehensive error handling.
    """

    def __init__(self, max_workers: int = 4, retry_attempts: int = 3):
        """
        Initialize the article processor.

        Args:
            max_workers: Maximum number of parallel workers
            retry_attempts: Number of retry attempts for failed URLs
        """
        self.max_workers = max_workers
        self.retry_attempts = retry_attempts
        self.processor = ArticleProcessor()
        self.vector_store = VectorStore()

        # Statistics tracking
        self.stats = {
            "total": 0,
            "successful": 0,
            "failed": 0,
            "skipped": 0,
            "start_time": None,
            "end_time": None
        }

    def check_existing_articles(self, urls: List[str]) -> Tuple[List[str], List[str]]:
        """
        Check which articles already exist in the database.

        Args:
            urls: List of URLs to check

        Returns:
            Tuple of (new_urls, existing_urls)
        """
        new_urls = []
        existing_urls = []

        logger.info("🔍 Checking for existing articles...")

        for url in urls:
            article_id = url.replace("https://", "").replace("/", "_")
            if self.vector_store.article_exists(article_id):
                existing_urls.append(url)
            else:
                new_urls.append(url)

        logger.info(f"Found {len(existing_urls)} existing articles, {len(new_urls)} new articles to process")
        return new_urls, existing_urls

    def process_single_article(self, url: str) -> Dict[str, any]:
        """
        Process a single article with retry logic.

        Args:
            url: Article URL to process

        Returns:
            Result dictionary with success status and details
        """
        for attempt in range(self.retry_attempts + 1):
            try:
                logger.info(f"📄 Processing: {url} (attempt {attempt + 1})")

                # Process the article
                article = self.processor.process_url(url)

                if not article:
                    return {
                        "url": url,
                        "success": False,
                        "error": "Failed to fetch or process article",
                        "attempts": attempt + 1
                    }

                # Add to vector store
                success = self.vector_store.add_article(article)

                if success:
                    return {
                        "url": url,
                        "success": True,
                        "title": article.title,
                        "attempts": attempt + 1
                    }
                else:
                    return {
                        "url": url,
                        "success": False,
                        "error": "Failed to add to vector store",
                        "attempts": attempt + 1
                    }

            except Exception as e:
                error_msg = f"Error processing article: {str(e)}"
                logger.warning(f"❌ Attempt {attempt + 1} failed for {url}: {error_msg}")

                if attempt == self.retry_attempts:
                    return {
                        "url": url,
                        "success": False,
                        "error": error_msg,
                        "attempts": attempt + 1
                    }

                # Wait before retry
                time.sleep(2 ** attempt)  # Exponential backoff

        return {
            "url": url,
            "success": False,
            "error": "Max retries exceeded",
            "attempts": self.retry_attempts + 1
        }

    def process_articles_parallel(self, urls: List[str]) -> List[Dict[str, any]]:
        """
        Process multiple articles in parallel with progress tracking.

        Args:
            urls: List of URLs to process

        Returns:
            List of result dictionaries
        """
        results = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_url = {
                executor.submit(self.process_single_article, url): url
                for url in urls
            }

            # Process completed tasks with progress bar
            with tqdm(total=len(urls), desc="Processing articles", unit="article") as pbar:
                for future in as_completed(future_to_url):
                    result = future.result()
                    results.append(result)

                    # Update progress bar
                    if result["success"]:
                        pbar.set_postfix({"✅": f"{result['title'][:30]}..."})
                        self.stats["successful"] += 1
                    else:
                        pbar.set_postfix({"❌": f"Failed: {result['url'].split('/')[-1]}"})
                        self.stats["failed"] += 1

                    pbar.update(1)

        return results

    def initialize_all_articles(self, urls: List[str] = None) -> Dict[str, any]:
        """
        Main method to initialize all articles.

        Args:
            urls: List of URLs to process (defaults to ARTICLE_URLS)

        Returns:
            Summary statistics
        """
        if urls is None:
            urls = ARTICLE_URLS

        self.stats["total"] = len(urls)
        self.stats["start_time"] = time.time()

        logger.info("🚀 Starting article database initialization...")
        logger.info(f"📊 Total articles to process: {len(urls)}")

        # Check for existing articles
        new_urls, existing_urls = self.check_existing_articles(urls)
        self.stats["skipped"] = len(existing_urls)

        if existing_urls:
            logger.info(f"⏭️  Skipping {len(existing_urls)} existing articles")

        if not new_urls:
            logger.info("✅ All articles already exist in database!")
            self.stats["end_time"] = time.time()
            return self.get_summary()

        # Process new articles
        logger.info(f"🔄 Processing {len(new_urls)} new articles with {self.max_workers} workers...")
        results = self.process_articles_parallel(new_urls)

        self.stats["end_time"] = time.time()

        # Log results
        self.log_results(results)

        return self.get_summary()

    def log_results(self, results: List[Dict[str, any]]):
        """Log detailed results of the processing."""
        logger.info("\n" + "="*60)
        logger.info("📊 PROCESSING RESULTS")
        logger.info("="*60)

        successful_articles = [r for r in results if r["success"]]
        failed_articles = [r for r in results if not r["success"]]

        if successful_articles:
            logger.info(f"✅ Successfully processed {len(successful_articles)} articles:")
            for result in successful_articles:
                logger.info(f"   • {result['title'][:60]}...")

        if failed_articles:
            logger.info(f"\n❌ Failed to process {len(failed_articles)} articles:")
            for result in failed_articles:
                logger.info(f"   • {result['url']}: {result['error']} (attempts: {result['attempts']})")

    def get_summary(self) -> Dict[str, any]:
        """Get processing summary statistics."""
        duration = self.stats["end_time"] - self.stats["start_time"] if self.stats["end_time"] else 0

        return {
            "total_urls": self.stats["total"],
            "successful": self.stats["successful"],
            "failed": self.stats["failed"],
            "skipped": self.stats["skipped"],
            "duration_seconds": round(duration, 2),
            "articles_per_minute": round((self.stats["successful"] / duration) * 60, 2) if duration > 0 else 0
        }


def main():
    """Main entry point for the script."""
    print("🚀 Article Database Initialization")
    print("=" * 50)

    # Initialize and run
    initializer = ArticleInitializer(max_workers=3, retry_attempts=2)

    try:
        summary = initializer.initialize_all_articles()

        # Print final summary
        print("\n🎉 INITIALIZATION COMPLETE!")
        print("=" * 50)
        print(f"📊 Total URLs: {summary['total_urls']}")
        print(f"✅ Successful: {summary['successful']}")
        print(f"❌ Failed: {summary['failed']}")
        print(f"⏭️  Skipped (existing): {summary['skipped']}")
        print(f"⏱️  Duration: {summary['duration_seconds']} seconds")
        print(f"📈 Rate: {summary['articles_per_minute']} articles/minute")

        if summary['failed'] > 0:
            print(f"\n⚠️  {summary['failed']} articles failed to process. Check logs for details.")
            return 1
        else:
            print(f"\n🎊 All articles processed successfully!")
            return 0

    except Exception as e:
        logger.error(f"💥 Initialization failed with error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
