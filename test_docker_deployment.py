#!/usr/bin/env python3
"""
Docker Deployment Test Script

This script tests the Docker deployment of the LLM API server.
Run this after starting the services with docker-compose up.

Usage: python test_docker_deployment.py [--local]
"""

import requests
import json
import time
import sys
import argparse
from typing import Dict, Any

class DockerDeploymentTester:
    """Test the Docker deployment of LLM API."""
    
    def __init__(self, api_base_url: str = "http://localhost:8000", qdrant_url: str = "http://localhost:6333", is_local: bool = False):
        self.api_base_url = api_base_url
        self.qdrant_url = qdrant_url
        self.is_local = is_local
        self.test_results = []
    
    def log_test(self, test_name: str, success: bool, message: str = ""):
        """Log test results."""
        status = "✅ PASS" if success else "❌ FAIL"
        self.test_results.append((test_name, success))
        print(f"{status} {test_name}")
        if message:
            print(f"    {message}")
    
    def wait_for_services(self, max_wait: int = 120) -> bool:
        """Wait for services to be ready."""
        if self.is_local:
            print("🔍 Testing local services...")
            # For local testing, check API health and RAG functionality (which uses in-memory Qdrant)
            return self.test_api_health() and self.test_rag_stats()
        
        print(f"⏳ Waiting for services to be ready (max {max_wait}s)...")
        
        for i in range(max_wait):
            try:
                # Test API health
                api_response = requests.get(f"{self.api_base_url}/health", timeout=2)
                if api_response.status_code != 200:
                    continue
                
                # Test Qdrant health  
                qdrant_response = requests.get(f"{self.qdrant_url}/health", timeout=2)
                if qdrant_response.status_code != 200:
                    continue
                
                print(f"\n✅ Services ready after {i+1}s")
                return True
                
            except requests.exceptions.RequestException:
                pass
            
            print(".", end="", flush=True)
            time.sleep(1)
        
        print(f"\n❌ Services not ready after {max_wait}s")
        return False
    
    def test_api_health(self) -> bool:
        """Test API server health endpoint."""
        try:
            response = requests.get(f"{self.api_base_url}/health", timeout=10)
            success = response.status_code == 200
            
            if success:
                data = response.json()
                self.log_test("API Health Check", True, f"Status: {data.get('status', 'unknown')}")
            else:
                self.log_test("API Health Check", False, f"HTTP {response.status_code}")
            
            return success
        except Exception as e:
            self.log_test("API Health Check", False, str(e))
            return False
    
    def test_qdrant_health(self) -> bool:
        """Test Qdrant database health endpoint."""
        try:
            response = requests.get(f"{self.qdrant_url}/health", timeout=10)
            success = response.status_code == 200
            
            if success:
                self.log_test("Qdrant Health Check", True, "Database accessible")
            else:
                self.log_test("Qdrant Health Check", False, f"HTTP {response.status_code}")
            
            return success
        except Exception as e:
            self.log_test("Qdrant Health Check", False, str(e))
            return False
    
    def test_chat_completion(self) -> bool:
        """Test basic chat completion functionality."""
        try:
            payload = {
                "model": "llama-2-7b-chat",
                "messages": [
                    {"role": "user", "content": "Say 'Hello Docker!' if you can read this."}
                ],
                "max_tokens": 50,
                "temperature": 0.1
            }
            
            response = requests.post(
                f"{self.api_base_url}/v1/chat/completions",
                json=payload,
                timeout=60
            )
            
            success = response.status_code == 200
            
            if success:
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                tokens_used = data["usage"]["total_tokens"]
                self.log_test("Chat Completion", True, f"Response: {content[:50]}... (Tokens: {tokens_used})")
            else:
                self.log_test("Chat Completion", False, f"HTTP {response.status_code}: {response.text}")
            
            return success
        except Exception as e:
            self.log_test("Chat Completion", False, str(e))
            return False
    
    def test_embeddings(self) -> bool:
        """Test embedding generation."""
        try:
            payload = {
                "input": ["Test embedding generation in Docker"],
                "model": "jina-embeddings-v3"
            }
            
            response = requests.post(
                f"{self.api_base_url}/v1/embeddings",
                json=payload,
                timeout=30
            )
            
            success = response.status_code == 200
            
            if success:
                data = response.json()
                embedding_dim = len(data["data"][0]["embedding"])
                tokens_used = data["usage"]["total_tokens"]
                self.log_test("Embeddings", True, f"Dimensions: {embedding_dim}, Tokens: {tokens_used}")
            else:
                self.log_test("Embeddings", False, f"HTTP {response.status_code}: {response.text}")
            
            return success
        except Exception as e:
            self.log_test("Embeddings", False, str(e))
            return False
    
    def test_rag_document_addition(self) -> bool:
        """Test adding documents to the RAG system."""
        try:
            test_docs = [
                "Docker is a containerization platform that allows developers to package applications and dependencies into lightweight containers.",
                "The LLM API server provides OpenAI-compatible endpoints for chat completions, embeddings, and RAG functionality.",
                "Qdrant is a vector database optimized for storing and searching high-dimensional vectors with metadata filtering."
            ]
            
            payload = {"texts": test_docs}
            
            response = requests.post(
                f"{self.api_base_url}/v1/rag/documents",
                json=payload,
                timeout=30
            )
            
            success = response.status_code == 200
            
            if success:
                data = response.json()
                count = data["count"]
                self.log_test("RAG Document Addition", True, f"Added {count} documents")
            else:
                self.log_test("RAG Document Addition", False, f"HTTP {response.status_code}: {response.text}")
            
            return success
        except Exception as e:
            self.log_test("RAG Document Addition", False, str(e))
            return False
    
    def test_rag_query(self) -> bool:
        """Test RAG document retrieval."""
        try:
            payload = {
                "query": "What is Docker used for?",
                "top_k": 2
            }
            
            response = requests.post(
                f"{self.api_base_url}/v1/rag/query",
                json=payload,
                timeout=30
            )
            
            success = response.status_code == 200
            
            if success:
                data = response.json()
                docs_retrieved = len(data.get("documents", data.get("context_documents", [])))
                documents = data.get("documents", data.get("context_documents", []))
                best_score = max([doc.get("score", doc.get("similarity", 0)) for doc in documents]) if documents else 0
                self.log_test("RAG Query", True, f"Retrieved {docs_retrieved} docs, best score: {best_score:.3f}")
            else:
                self.log_test("RAG Query", False, f"HTTP {response.status_code}: {response.text}")
            
            return success
        except Exception as e:
            self.log_test("RAG Query", False, str(e))
            return False
    
    def test_rag_chat_completion(self) -> bool:
        """Test end-to-end RAG chat completion."""
        try:
            payload = {
                "query": "Explain what Docker is and how it helps developers",
                "max_tokens": 150
            }
            
            response = requests.post(
                f"{self.api_base_url}/v1/rag/chat",
                json=payload,
                timeout=60
            )
            
            success = response.status_code == 200
            
            if success:
                data = response.json()
                answer = data["answer"]
                docs_retrieved = data["num_docs_retrieved"]
                tokens_used = data["tokens_used"]
                self.log_test("RAG Chat Completion", True, 
                           f"Answer: {answer[:100]}... (Docs: {docs_retrieved}, Tokens: {tokens_used})")
            else:
                self.log_test("RAG Chat Completion", False, f"HTTP {response.status_code}: {response.text}")
            
            return success
        except Exception as e:
            self.log_test("RAG Chat Completion", False, str(e))
            return False
    
    def test_rag_stats(self) -> bool:
        """Test RAG statistics endpoint."""
        try:
            response = requests.get(f"{self.api_base_url}/v1/rag/stats", timeout=10)
            success = response.status_code == 200
            
            if success:
                data = response.json()
                total_docs = data["total_documents"]
                dimensions = data["vector_dimensions"]
                self.log_test("RAG Statistics", True, f"Documents: {total_docs}, Dimensions: {dimensions}")
            else:
                self.log_test("RAG Statistics", False, f"HTTP {response.status_code}: {response.text}")
            
            return success
        except Exception as e:
            self.log_test("RAG Statistics", False, str(e))
            return False
    
    def run_all_tests(self) -> bool:
        """Run all tests and return overall success."""
        deployment_type = "Local" if self.is_local else "Docker"
        print(f"🐳 {deployment_type} Deployment Test Suite")
        print("=" * 60)
        
        # Wait for services
        if not self.wait_for_services():
            print("❌ Services not available. Make sure docker-compose is running." if not self.is_local else "❌ Local services not available.")
            return False
        
        print("\n🧪 Running Test Suite...")
        print("-" * 40)
        
        # Core functionality tests
        tests = [
            self.test_api_health,
            self.test_chat_completion,
            self.test_embeddings,
            self.test_rag_document_addition,
            self.test_rag_stats,
            self.test_rag_query,
            self.test_rag_chat_completion,
        ]
        
        # Add Qdrant health check only for Docker deployments
        if not self.is_local:
            tests.insert(1, self.test_qdrant_health)
        
        for test in tests:
            test()
            time.sleep(1)  # Brief pause between tests
        
        # Summary
        print("\n📊 Test Results Summary")
        print("-" * 40)
        
        passed = sum(1 for _, success in self.test_results if success)
        total = len(self.test_results)
        
        for test_name, success in self.test_results:
            status = "✅" if success else "❌"
            print(f"{status} {test_name}")
        
        print(f"\n🎯 Overall Result: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 All tests passed! Deployment is working correctly.")
            return True
        else:
            print("⚠️ Some tests failed. Check the logs above for details.")
            return False

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Test LLM API deployment")
    parser.add_argument("--local", action="store_true", help="Test local services instead of Docker")
    parser.add_argument("--api-url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--qdrant-url", default="http://localhost:6333", help="Qdrant URL")
    
    args = parser.parse_args()
    
    tester = DockerDeploymentTester(
        api_base_url=args.api_url,
        qdrant_url=args.qdrant_url,
        is_local=args.local
    )
    
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 