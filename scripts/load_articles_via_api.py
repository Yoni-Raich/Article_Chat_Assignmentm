#!/usr/bin/env python3
"""
Load articles via API to the remote ChromaDB used by Docker.
This ensures articles are loaded into the same database that the Docker container uses.
"""

import requests
import time
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# All 17 article URLs from the assignment
ARTICLE_URLS = [
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
    "https://edition.cnn.com/2025/07/27/business/us-china-trade-talks-stockholm-intl-hnk",
    "https://edition.cnn.com/2025/07/27/business/trump-us-eu-trade-deal",
    "https://edition.cnn.com/2025/07/27/business/eu-trade-deal",
    "https://edition.cnn.com/2025/07/26/tech/daydream-ai-online-shopping",
    "https://edition.cnn.com/2025/07/25/tech/meta-ai-superintelligence-team-who-its-hiring",
    "https://edition.cnn.com/2025/07/25/tech/sequoia-islamophobia-maguire-mamdani",
    "https://edition.cnn.com/2025/07/24/tech/intel-layoffs-15-percent-q2-earnings"
]

API_BASE = "http://localhost:8000"

def check_server_status():
    """Check if the API server is running"""
    try:
        response = requests.get(f"{API_BASE}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def ingest_article(url: str, max_retries: int = 3) -> dict:
    """Ingest a single article via API"""
    for attempt in range(max_retries):
        try:
            response = requests.post(
                f"{API_BASE}/ingest",
                json={"url": url},
                timeout=60  # 60 seconds timeout for each article
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "url": url,
                    "success": True,
                    "message": result.get("message", "Success"),
                    "attempts": attempt + 1
                }
            else:
                error_msg = f"HTTP {response.status_code}: {response.text}"
                if attempt == max_retries - 1:
                    return {
                        "url": url,
                        "success": False,
                        "error": error_msg,
                        "attempts": attempt + 1
                    }
                
        except Exception as e:
            error_msg = str(e)
            if attempt == max_retries - 1:
                return {
                    "url": url,
                    "success": False,
                    "error": error_msg,
                    "attempts": attempt + 1
                }
        
        # Wait before retry
        time.sleep(2 ** attempt)
    
    return {
        "url": url,
        "success": False,
        "error": "Max retries exceeded",
        "attempts": max_retries
    }

def load_all_articles(max_workers: int = 3):
    """Load all articles via API with parallel processing"""
    print("🚀 Loading Articles via API")
    print("=" * 50)
    
    # Check server
    if not check_server_status():
        print("❌ API server is not running at http://localhost:8000")
        print("   Please make sure Docker is running: docker-compose up -d")
        return False
    
    print(f"✅ API server is running")
    print(f"📊 Loading {len(ARTICLE_URLS)} articles with {max_workers} workers...\n")
    
    results = []
    successful = 0
    failed = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_url = {
            executor.submit(ingest_article, url): url 
            for url in ARTICLE_URLS
        }
        
        # Process with progress bar
        with tqdm(total=len(ARTICLE_URLS), desc="Loading articles", unit="article") as pbar:
            for future in as_completed(future_to_url):
                result = future.result()
                results.append(result)
                
                if result["success"]:
                    successful += 1
                    pbar.set_postfix({"✅": "Success", "❌": f"{failed}"})
                else:
                    failed += 1
                    pbar.set_postfix({"✅": f"{successful}", "❌": "Failed"})
                
                pbar.update(1)
    
    # Print results
    print("\n" + "=" * 60)
    print("📊 LOADING RESULTS")
    print("=" * 60)
    
    successful_articles = [r for r in results if r["success"]]
    failed_articles = [r for r in results if not r["success"]]
    
    if successful_articles:
        print(f"✅ Successfully loaded {len(successful_articles)} articles:")
        for result in successful_articles:
            print(f"   • {result['url'].split('/')[-1]}")
    
    if failed_articles:
        print(f"\n❌ Failed to load {len(failed_articles)} articles:")
        for result in failed_articles:
            print(f"   • {result['url'].split('/')[-1]}: {result['error']}")
    
    print(f"\n🎉 SUMMARY:")
    print(f"   Total: {len(ARTICLE_URLS)}")
    print(f"   Successful: {successful}")
    print(f"   Failed: {failed}")
    
    return failed == 0

def main():
    """Main function"""
    try:
        success = load_all_articles()
        if success:
            print("\n🎊 All articles loaded successfully!")
            
            # Test the loaded articles
            print("\n🔍 Testing loaded articles...")
            response = requests.post(
                f"{API_BASE}/chat",
                json={"query": "How many articles are in the database?"}
            )
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Test response: {result['response'][:100]}...")
            
            return 0
        else:
            print("\n⚠️  Some articles failed to load. Check the errors above.")
            return 1
            
    except Exception as e:
        print(f"\n💥 Script failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())