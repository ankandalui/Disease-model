#!/usr/bin/env python3
"""
Quick test for Lighthouse API connectivity
"""

import os
import requests
from dotenv import load_dotenv

def test_lighthouse_api():
    """Test if Lighthouse API key works"""
    load_dotenv()
    
    api_key = os.getenv('LIGHTHOUSE_API_KEY')
    
    if not api_key or api_key == 'your_lighthouse_api_key_here':
        print("❌ Please set your LIGHTHOUSE_API_KEY in the .env file")
        print("📋 Get your API key from: https://files.lighthouse.storage/")
        return False
    
    print(f"🔑 Testing API key: {api_key[:10]}...")
    
    try:
        headers = {'Authorization': f'Bearer {api_key}'}
        response = requests.get(
            "https://node.lighthouse.storage/api/v0/id",
            headers=headers,
            timeout=10
        )
        
        if response.status_code == 200:
            print("✅ Lighthouse API key is valid!")
            print("🎉 Blockchain storage is ready to use!")
            return True
        else:
            print(f"❌ API key validation failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Lighthouse API Key...\n")
    test_lighthouse_api()
