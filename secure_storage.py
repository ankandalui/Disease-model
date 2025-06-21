import os
import json
import hashlib
import base64
from datetime import datetime
from typing import Optional, Dict, Any
import requests
from pymongo import MongoClient
from bson import ObjectId
import io
from PIL import Image


class SecureImageStorage:
    """
    Secure image storage service using Lighthouse blockchain and MongoDB
    """
    
    def __init__(self, lighthouse_api_key: str, mongo_uri: str, db_name: str = "medai_secure"):
        self.lighthouse_api_key = lighthouse_api_key
        self.lighthouse_base_url = "https://node.lighthouse.storage"
        self.lighthouse_enabled = lighthouse_api_key and lighthouse_api_key != "lighthouse_api_key_placeholder"
        
        # MongoDB setup
        try:
            self.mongo_client = MongoClient(mongo_uri)
            self.db = self.mongo_client[db_name]
            self.images_collection = self.db.images
            self.predictions_collection = self.db.predictions
            
            # Create indexes for better performance
            self.images_collection.create_index("image_hash", unique=True)
            self.images_collection.create_index("lighthouse_hash")
            self.predictions_collection.create_index("image_id")
            
            self.mongo_enabled = True
            print(f"MongoDB connected successfully to {db_name}")
        except Exception as e:
            print(f"MongoDB connection failed: {e}")
            self.mongo_enabled = False
            self.mongo_client = None
        
        if not self.lighthouse_enabled:
            print("Warning: Lighthouse API key not configured - blockchain storage disabled")
        if not self.mongo_enabled:
            print("Warning: MongoDB not available - metadata storage disabled")        
    def _generate_image_hash(self, image_data: bytes) -> str:
        """Generate SHA-256 hash of image data"""
        return hashlib.sha256(image_data).hexdigest()
    
    def _upload_to_lighthouse(self, image_data: bytes, filename: str) -> Dict[str, Any]:
        """
        Upload image to Lighthouse IPFS storage
        """
        if not self.lighthouse_enabled:
            return {
                'success': False,
                'error': 'Lighthouse storage is disabled - API key not configured',
                'hash': None
            }
            
        try:
            # Prepare the file for upload
            files = {
                'file': (filename, io.BytesIO(image_data), 'image/jpeg')
            }
            
            headers = {
                'Authorization': f'Bearer {self.lighthouse_api_key}'
            }
            
            # Upload to Lighthouse
            response = requests.post(
                f"{self.lighthouse_base_url}/api/v0/add",
                files=files,
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    'success': True,
                    'lighthouse_hash': result.get('Hash'),
                    'lighthouse_name': result.get('Name'),
                    'lighthouse_size': result.get('Size'),
                    'gateway_url': f"https://gateway.lighthouse.storage/ipfs/{result.get('Hash')}"
                }
            else:
                return {
                    'success': False,
                    'error': f'Lighthouse upload failed: {response.status_code}',
                    'details': response.text
                }                
        except Exception as e:
            return {
                'success': False,
                'error': f'Lighthouse upload error: {str(e)}'
            }
    
    def store_image(self, image_data: bytes, filename: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Securely store image using both Lighthouse and MongoDB
        """
        if not self.lighthouse_enabled and not self.mongo_enabled:
            return {
                'success': False,
                'error': 'All storage services are disabled',
                'image_id': None,
                'lighthouse_hash': None,
                'gateway_url': None
            }
            
        try:
            # Generate image hash for deduplication
            image_hash = self._generate_image_hash(image_data)
            
            # Check if image already exists (only if MongoDB is enabled)
            if self.mongo_enabled:
                existing_image = self.images_collection.find_one({"image_hash": image_hash})
                if existing_image:
                    return {
                        'success': True,
                        'image_id': str(existing_image['_id']),
                        'lighthouse_hash': existing_image.get('lighthouse_hash'),
                        'gateway_url': existing_image.get('gateway_url'),
                        'duplicate': True,
                        'message': 'Image already exists in secure storage'
                    }
            
            # Upload to Lighthouse (if enabled)
            lighthouse_result = {'success': True, 'hash': None, 'gateway_url': None}
            if self.lighthouse_enabled:
                lighthouse_result = self._upload_to_lighthouse(image_data, filename)
                
                if not lighthouse_result['success'] and not self.mongo_enabled:
                    # If both Lighthouse fails and MongoDB is disabled, return error
                    return lighthouse_result
            
            # Create document for MongoDB
            image_doc = {
                'image_hash': image_hash,
                'lighthouse_hash': lighthouse_result['lighthouse_hash'],
                'lighthouse_name': lighthouse_result['lighthouse_name'],
                'lighthouse_size': lighthouse_result['lighthouse_size'],
                'gateway_url': lighthouse_result['gateway_url'],
                'original_filename': filename,
                'upload_timestamp': datetime.utcnow(),
                'metadata': metadata or {},
                'file_size': len(image_data),
                'storage_type': 'lighthouse_ipfs'
            }
            
            # Store metadata in MongoDB
            result = self.images_collection.insert_one(image_doc)
            
            return {
                'success': True,
                'image_id': str(result.inserted_id),
                'lighthouse_hash': lighthouse_result['lighthouse_hash'],
                'gateway_url': lighthouse_result['gateway_url'],
                'duplicate': False,
                'message': 'Image stored securely on blockchain and database'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Storage failed: {str(e)}'
            }
    
    def get_image_info(self, image_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve image information from MongoDB
        """
        try:
            image_doc = self.images_collection.find_one({"_id": ObjectId(image_id)})
            if image_doc:
                image_doc['_id'] = str(image_doc['_id'])
                return image_doc
            return None
        except Exception as e:
            print(f"Error retrieving image info: {e}")
            return None
    
    def store_prediction_result(self, image_id: str, prediction_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Store AI prediction results linked to the secure image
        """
        try:
            prediction_doc = {
                'image_id': image_id,
                'prediction': prediction_data.get('prediction'),
                'confidence': prediction_data.get('confidence'),
                'model_version': prediction_data.get('model_version', 'v1.0'),
                'prediction_timestamp': datetime.utcnow(),
                'additional_data': prediction_data
            }
            
            result = self.predictions_collection.insert_one(prediction_doc)
            
            return {
                'success': True,
                'prediction_id': str(result.inserted_id),
                'message': 'Prediction result stored securely'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Prediction storage failed: {str(e)}'
            }
    
    def get_prediction_results(self, image_id: str) -> list:
        """
        Retrieve all prediction results for a specific image
        """
        try:
            predictions = list(self.predictions_collection.find({"image_id": image_id}))
            for pred in predictions:
                pred['_id'] = str(pred['_id'])
            return predictions
        except Exception as e:
            print(f"Error retrieving predictions: {e}")
            return []
    
    def delete_image(self, image_id: str) -> Dict[str, Any]:
        """
        Delete image metadata from MongoDB (Note: IPFS data remains distributed)
        """
        try:
            # Delete associated predictions first
            self.predictions_collection.delete_many({"image_id": image_id})
            
            # Delete image document
            result = self.images_collection.delete_one({"_id": ObjectId(image_id)})
            
            if result.deleted_count > 0:
                return {
                    'success': True,
                    'message': 'Image metadata deleted from database'
                }
            else:
                return {
                    'success': False,
                    'error': 'Image not found'
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': f'Deletion failed: {str(e)}'
            }
    
    def get_image_statistics(self) -> Dict[str, Any]:
        """
        Get storage statistics
        """
        try:
            total_images = self.images_collection.count_documents({})
            total_predictions = self.predictions_collection.count_documents({})
            
            # Calculate total storage used
            pipeline = [
                {"$group": {"_id": None, "total_size": {"$sum": "$file_size"}}}
            ]
            size_result = list(self.images_collection.aggregate(pipeline))
            total_size = size_result[0]['total_size'] if size_result else 0
            
            return {
                'total_images': total_images,
                'total_predictions': total_predictions,
                'total_storage_bytes': total_size,
                'total_storage_mb': round(total_size / (1024 * 1024), 2)
            }
            
        except Exception as e:
            return {
                'error': f'Statistics retrieval failed: {str(e)}'
            }
    
    def close_connection(self):
        """Close MongoDB connection"""
        if self.mongo_client:
            self.mongo_client.close()
