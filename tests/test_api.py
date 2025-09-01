#!/usr/bin/env python3
"""
Test script for the FastAPI Article Chat service.

This script tests the main endpoints of the API to ensure everything works correctly.
"""

# Standard library imports
import json
import sys
import time
from typing import Dict, Any

# Third-party imports
import requests


class APITester:
    """
    API testing class for the Article Chat service.

    This class provides methods to test various API endpoints and functionality.
    """
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()

    def test_endpoint(self, method: str, endpoint: str, data: Dict[Any, Any] = None) -> Dict[Any, Any]:
        """Test a single API endpoint."""
        url = f"{self.base_url}{endpoint}"

        try:
            if method.upper() == "GET":
                response = self.session.get(url)
            elif method.upper() == "POST":
                response = self.session.post(url, json=data)
            else:
                raise ValueError(f"Unsupported method: {method}")

            print(f"🔍 Testing {method} {endpoint}")
            print(f"   Status: {response.status_code}")

            if response.status_code == 200:
                result = response.json()
                print(f"   ✅ Success: {json.dumps(result, indent=2)[:200]}...")
                return result
            else:
                print(f"   ❌ Error: {response.text}")
                return {"error": response.text, "status_code": response.status_code}

        except Exception as e:
            print(f"   💥 Exception: {str(e)}")
            return {"error": str(e)}

    def run_all_tests(self):
        """Run all API tests."""
        print("🚀 Starting API Tests...\n")

        # Test 1: Root endpoint
        print("=" * 50)
        print("TEST 1: Root Endpoint")
        self.test_endpoint("GET", "/")

        # Test 2: Health check
        print("\n" + "=" * 50)
        print("TEST 2: Health Check")
        health = self.test_endpoint("GET", "/health")

        # Wait for services to be ready
        if isinstance(health, dict) and not health.get("agent_ready", False):
            print("⏳ Waiting for services to initialize...")
            time.sleep(5)
            health = self.test_endpoint("GET", "/health")

        # Test 3: Chat endpoint
        print("\n" + "=" * 50)
        print("TEST 3: Chat Endpoint")
        chat_data = {
            "query": "What articles do you have about artificial intelligence?"
        }
        self.test_endpoint("POST", "/chat", chat_data)

        # Test 4: Ingest endpoint (with a test URL)
        print("\n" + "=" * 50)
        print("TEST 4: Ingest Endpoint")
        ingest_data = {
            "url": "https://techcrunch.com/2025/07/26/tesla-vet-says-that-reviewing-real-products-not-mockups-is-the-key-to-staying-innovative/"
        }
        self.test_endpoint("POST", "/ingest", ingest_data)

        # Test 5: Chat after ingestion
        print("\n" + "=" * 50)
        print("TEST 5: Chat After Ingestion")
        chat_data2 = {
            "query": "Tell me about Tesla and innovation"
        }
        self.test_endpoint("POST", "/chat", chat_data2)

        print("\n🎉 All tests completed!")


def check_server_running(url: str = "http://localhost:8000") -> bool:
    """Check if the server is running."""
    try:
        response = requests.get(f"{url}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def main():
    """Main test function."""
    base_url = "http://localhost:8000"

    print("🔧 Article Chat API Tester")
    print("=" * 50)

    # Check if server is running
    if not check_server_running(base_url):
        print(f"❌ Server not running at {base_url}")
        print("   Please start the server first:")
        print("   python api/main.py")
        sys.exit(1)

    print(f"✅ Server is running at {base_url}")

    # Run tests
    tester = APITester(base_url)
    tester.run_all_tests()


if __name__ == "__main__":
    main()
