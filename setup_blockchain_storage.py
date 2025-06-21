#!/usr/bin/env python3
"""
Setup script for blockchain storage with Lighthouse API
"""

import os
import sys
import requests
from pymongo import MongoClient

def check_lighthouse_api_key():
    """Check if Lighthouse API key is configured"""
    api_key = os.getenv('LIGHTHOUSE_API_KEY')
    
    if not api_key or api_key == 'your_lighthouse_api_key_here':
        print("❌ Lighthouse API key not configured!")
        print("\n📋 How to get Lighthouse API key:")
        print("1. Go to https://files.lighthouse.storage/")
        print("2. Sign up or log in")
        print("3. Go to API Keys section")
        print("4. Create a new API key")
        print("5. Copy the key and update your .env file:")
        print("   LIGHTHOUSE_API_KEY=your_actual_api_key_here")
        return False
    else:
        print("✅ Lighthouse API key configured")
        return True

def test_lighthouse_connection():
    """Test connection to Lighthouse API"""
    api_key = os.getenv('LIGHTHOUSE_API_KEY')
    
    if not api_key or api_key == 'your_lighthouse_api_key_here':
        return False
        
    try:
        # Test with a simple ping
        headers = {'Authorization': f'Bearer {api_key}'}
        response = requests.get(
            "https://node.lighthouse.storage/api/v0/id",
            headers=headers,
            timeout=10
        )
        
        if response.status_code == 200:
            print("✅ Lighthouse API connection successful")
            return True
        else:
            print(f"❌ Lighthouse API error: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Lighthouse connection failed: {e}")
        return False

def check_mongodb():
    """Check MongoDB connection"""
    mongo_uri = os.getenv('MONGO_URI', 'mongodb://localhost:27017/')
    db_name = os.getenv('MONGO_DB_NAME', 'medai_local_dev')
    
    try:
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        client.server_info()  # Force connection
        
        # Test database access
        db = client[db_name]
        collections = db.list_collection_names()
        
        print(f"✅ MongoDB connected successfully")
        print(f"   Database: {db_name}")
        print(f"   Collections: {collections}")
        
        client.close()
        return True
        
    except Exception as e:
        print(f"❌ MongoDB connection failed: {e}")
        print("\n📋 How to install MongoDB locally:")
        print("1. Download MongoDB Community Server from https://www.mongodb.com/try/download/community")
        print("2. Install and start MongoDB service")
        print("3. Or use MongoDB Atlas (cloud): https://www.mongodb.com/atlas")
        print("   - Update MONGO_URI in .env with your Atlas connection string")
        return False

def test_image_storage():
    """Test complete image storage functionality"""
    print("\n🧪 Testing complete storage system...")
    
    # Import after environment is loaded
    from secure_storage import SecureImageStorage
    
    api_key = os.getenv('LIGHTHOUSE_API_KEY')
    mongo_uri = os.getenv('MONGO_URI', 'mongodb://localhost:27017/')
    db_name = os.getenv('MONGO_DB_NAME', 'medai_local_dev')
    
    storage = SecureImageStorage(api_key, mongo_uri, db_name)
    
    # Create a test image (small PNG)
    test_image_data = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\tpHYs\x00\x00\x0b\x13\x00\x00\x0b\x13\x01\x00\x9a\x9c\x18\x00\x00\x00\nIDATx\x9cc\xf8\x00\x00\x00\x01\x00\x01\x00\x00\x00\x00IEND\xaeB`\x82'
    
    result = storage.store_image(
        image_data=test_image_data,
        filename="test_image.png",
        metadata={"test": True, "purpose": "storage_test"}
    )
    
    if result['success']:
        print("✅ Image storage test successful!")
        print(f"   Image ID: {result['image_id']}")
        print(f"   Lighthouse Hash: {result['lighthouse_hash']}")
        print(f"   Gateway URL: {result['gateway_url']}")
        
        # Test retrieval
        if result['image_id']:
            info = storage.get_image_info(result['image_id'])
            if info:
                print("✅ Image retrieval test successful!")
            else:
                print("❌ Image retrieval test failed!")
                
        return True
    else:
        print(f"❌ Image storage test failed: {result['error']}")
        return False

def main():
    print("🔧 Setting up Blockchain Storage with Lighthouse API\n")
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    all_good = True
    
    # Check Lighthouse API key
    if not check_lighthouse_api_key():
        all_good = False
    
    # Test Lighthouse connection
    if not test_lighthouse_connection():
        all_good = False
    
    # Check MongoDB
    if not check_mongodb():
        all_good = False
    
    # Run complete test if everything is configured
    if all_good:
        if test_image_storage():
            print("\n🎉 Blockchain storage setup complete!")
            print("Your application is ready to store images securely on IPFS!")
        else:
            print("\n❌ Storage test failed - check your configuration")
    else:
        print("\n⚠️ Please fix the issues above before using blockchain storage")

if __name__ == "__main__":
    main()
