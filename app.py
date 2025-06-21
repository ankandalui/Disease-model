from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import base64
from PIL import Image
import io
from dotenv import load_dotenv
from secure_storage import SecureImageStorage
import hashlib
from datetime import datetime

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Enable CORS for all routes (get origins from environment)
cors_origins = os.getenv('CORS_ORIGINS', 'http://localhost:3000,http://127.0.0.1:3000').split(',')
CORS(app, origins=cors_origins)

# Configuration
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['MAX_CONTENT_LENGTH'] = int(os.getenv('MAX_FILE_SIZE_MB', 16)) * 1024 * 1024  # Default 16MB
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'dev-secret-key')

# Initialize Secure Storage
try:
    storage = SecureImageStorage(
        lighthouse_api_key=os.getenv('LIGHTHOUSE_API_KEY'),
        mongo_uri=os.getenv('MONGO_URI', 'mongodb://localhost:27017/'),
        db_name=os.getenv('MONGO_DB_NAME', 'medai_secure_storage')
    )
    print("Secure storage initialized successfully!")
except Exception as e:
    print(f"Warning: Secure storage initialization failed: {e}")
    storage = None

# Load the model
try:
    # Try loading with compile=False to avoid compatibility issues
    model = load_model('best_model.keras', compile=False)
    
    # Manually compile the model for inference
    if model is not None:
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print("Model loaded and compiled successfully!")
    
except Exception as e:
    print(f"Error loading model with compile=False: {e}")
    try:
        # Fallback: try with custom_objects and safe_mode
        model = load_model('best_model.keras', compile=False, safe_mode=False)
        if model is not None:
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            print("Model loaded successfully with safe_mode=False and compiled!")
    except Exception as e2:
        print(f"Error loading model with safe_mode=False: {e2}")
        model = None

# Class names
class_names = [
    'Eczema',
    'Warts Molluscum and other Viral Infections',
    'Melanoma',
    'Atopic Dermatitis',
    'Basal Cell Carcinoma (BCC)',
    'Melanocytic Nevi (NV)',
    'Benign Keratosis-like Lesions (BKL)',
    'Psoriasis pictures Lichen Planus and related diseases',
    'Seborrheic Keratoses and other Benign Tumors',
    'Tinea Ringworm Candidiasis and other Fungal Infections'
]

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_image(img_data):
    """
    Predict from image data (either file path or PIL Image)
    """
    if isinstance(img_data, str):  # file path
        img = image.load_img(img_data, target_size=(224, 224))
    else:  # PIL Image
        img = img_data.resize((224, 224))
    
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize
    
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = float(np.max(predictions[0]))
    
    return class_names[predicted_class], confidence

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'storage_available': storage is not None,
        'version': os.getenv('MODEL_VERSION', 'v1.0'),
        'message': 'Flask API is running',
        'timestamp': datetime.utcnow().isoformat()
    })

@app.route('/api/classes', methods=['GET'])
def get_classes():
    """Get available class names"""
    return jsonify({
        'classes': class_names,
        'total_classes': len(class_names)
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Main prediction endpoint with secure storage"""
    try:
        if model is None:
            return jsonify({
                'error': 'Model not loaded',
                'message': 'The ML model could not be loaded'
            }), 500

        image_data = None
        filename = None
        metadata = {
            'request_timestamp': datetime.utcnow().isoformat(),
            'client_ip': request.remote_addr,
            'user_agent': request.headers.get('User-Agent', 'Unknown')
        }

        # Check if request has file or base64 data
        if 'file' in request.files:
            # Handle file upload
            file = request.files['file']
            
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            
            if not allowed_file(file.filename):
                return jsonify({
                    'error': 'Invalid file type',
                    'allowed_types': list(ALLOWED_EXTENSIONS)
                }), 400
            
            # Read file data into memory
            filename = secure_filename(file.filename)
            image_data = file.read()
            
            # Create PIL Image for prediction
            img = Image.open(io.BytesIO(image_data))
            if img.mode != 'RGB':
                img = img.convert('RGB')
                
        elif 'image' in request.json:
            # Handle base64 image data
            try:
                # Remove data URL prefix if present
                image_b64 = request.json['image']
                if image_b64.startswith('data:image'):
                    image_b64 = image_b64.split(',')[1]
                
                # Decode base64
                image_data = base64.b64decode(image_b64)
                img = Image.open(io.BytesIO(image_data))
                
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                filename = f"prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                
            except Exception as e:
                return jsonify({
                    'error': 'Invalid image data',
                    'message': str(e)
                }), 400
        else:
            return jsonify({
                'error': 'No image provided',
                'message': 'Please provide either a file or base64 image data'
            }), 400

        # Make prediction
        prediction, confidence = predict_image(img)
        
        # Prepare response data
        prediction_data = {
            'prediction': prediction,
            'confidence': round(confidence * 100, 2),
            'model_version': os.getenv('MODEL_VERSION', 'v1.0'),
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Store image and prediction securely if storage is available
        storage_result = None
        if storage and image_data:
            try:
                # Store image securely
                storage_result = storage.store_image(
                    image_data=image_data,
                    filename=filename,
                    metadata=metadata
                )
                
                if storage_result['success']:
                    # Store prediction result
                    prediction_storage = storage.store_prediction_result(
                        image_id=storage_result['image_id'],
                        prediction_data=prediction_data
                    )
                    
                    # Add storage info to response
                    prediction_data.update({
                        'storage_info': {
                            'image_id': storage_result['image_id'],
                            'lighthouse_hash': storage_result['lighthouse_hash'],
                            'gateway_url': storage_result['gateway_url'],
                            'duplicate': storage_result.get('duplicate', False),
                            'prediction_id': prediction_storage.get('prediction_id') if prediction_storage['success'] else None
                        }
                    })
                else:
                    print(f"Storage failed: {storage_result}")
                    
            except Exception as e:
                print(f"Secure storage error: {e}")
                # Continue with prediction even if storage fails

        return jsonify({
            'success': True,
            'prediction': prediction,
            'confidence': round(confidence * 100, 2),
            'storage_secure': storage_result['success'] if storage_result else False,
            'storage_info': prediction_data.get('storage_info'),
            'timestamp': prediction_data['timestamp']
        })

    except Exception as e:
        return jsonify({
            'error': 'Prediction failed',
            'message': str(e)
        }), 500

@app.route('/api/predict-batch', methods=['POST'])
def predict_batch():
    """Batch prediction endpoint for multiple images with secure storage"""
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500

        files = request.files.getlist('files')
        
        if not files:
            return jsonify({'error': 'No files provided'}), 400

        results = []
        
        for file in files:
            if file.filename != '' and allowed_file(file.filename):
                try:
                    # Read file data
                    filename = secure_filename(file.filename)
                    image_data = file.read()
                    
                    # Create PIL Image for prediction
                    img = Image.open(io.BytesIO(image_data))
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # Make prediction
                    prediction, confidence = predict_image(img)
                    
                    # Prepare result
                    result = {
                        'filename': file.filename,
                        'prediction': prediction,
                        'confidence': round(confidence * 100, 2),
                        'status': 'success',
                        'timestamp': datetime.utcnow().isoformat()
                    }
                    
                    # Store securely if storage is available
                    if storage:
                        try:
                            metadata = {
                                'batch_request': True,
                                'original_filename': file.filename,
                                'request_timestamp': datetime.utcnow().isoformat()
                            }
                            
                            storage_result = storage.store_image(
                                image_data=image_data,
                                filename=filename,
                                metadata=metadata
                            )
                            
                            if storage_result['success']:
                                prediction_data = {
                                    'prediction': prediction,
                                    'confidence': round(confidence * 100, 2),
                                    'model_version': os.getenv('MODEL_VERSION', 'v1.0'),
                                    'batch_processing': True
                                }
                                
                                prediction_storage = storage.store_prediction_result(
                                    image_id=storage_result['image_id'],
                                    prediction_data=prediction_data
                                )
                                
                                result['storage_info'] = {
                                    'image_id': storage_result['image_id'],
                                    'lighthouse_hash': storage_result['lighthouse_hash'],
                                    'prediction_id': prediction_storage.get('prediction_id') if prediction_storage['success'] else None
                                }
                                result['storage_secure'] = True
                            else:
                                result['storage_secure'] = False
                                result['storage_error'] = storage_result.get('error')
                        except Exception as e:
                            result['storage_secure'] = False
                            result['storage_error'] = str(e)
                    
                    results.append(result)
                        
                except Exception as e:
                    results.append({
                        'filename': file.filename,
                        'error': str(e),
                        'status': 'failed'
                    })
        
        return jsonify({
            'success': True,
            'results': results,
            'total_processed': len(results),
            'secure_storage_enabled': storage is not None
        })

    except Exception as e:
        return jsonify({
            'error': 'Batch prediction failed',
            'message': str(e)
        }), 500

@app.route('/api/storage/stats', methods=['GET'])
def get_storage_stats():
    """Get storage statistics"""
    if not storage:
        return jsonify({
            'error': 'Secure storage not available'
        }), 503
    
    try:
        stats = storage.get_image_statistics()
        return jsonify({
            'success': True,
            'statistics': stats
        })
    except Exception as e:
        return jsonify({
            'error': 'Failed to retrieve statistics',
            'message': str(e)
        }), 500

@app.route('/api/storage/image/<image_id>', methods=['GET'])
def get_image_info(image_id):
    """Get information about a stored image"""
    if not storage:
        return jsonify({
            'error': 'Secure storage not available'
        }), 503
    
    try:
        image_info = storage.get_image_info(image_id)
        if image_info:
            predictions = storage.get_prediction_results(image_id)
            image_info['predictions'] = predictions
            return jsonify({
                'success': True,
                'image_info': image_info
            })
        else:
            return jsonify({
                'error': 'Image not found'
            }), 404
    except Exception as e:
        return jsonify({
            'error': 'Failed to retrieve image info',
            'message': str(e)
        }), 500

@app.route('/api/storage/image/<image_id>', methods=['DELETE'])
def delete_image(image_id):
    """Delete image metadata (IPFS data remains distributed)"""
    if not storage:
        return jsonify({
            'error': 'Secure storage not available'
        }), 503
    
    try:
        result = storage.delete_image(image_id)
        return jsonify(result)
    except Exception as e:
        return jsonify({
            'error': 'Failed to delete image',
            'message': str(e)
        }), 500

@app.errorhandler(413)
def too_large(e):
    return jsonify({
        'error': 'File too large',
        'message': f'Maximum file size is {app.config["MAX_CONTENT_LENGTH"] // (1024*1024)}MB'
    }), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({
        'error': 'Endpoint not found',
        'message': 'Please check the API documentation'
    }), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({
        'error': 'Internal server error',
        'message': 'Something went wrong on the server'
    }), 500

if __name__ == '__main__':
    print("Starting Flask API server with secure storage...")
    print("Available endpoints:")
    print("  GET  /api/health - Health check")
    print("  GET  /api/classes - Get class names")
    print("  POST /api/predict - Single image prediction with secure storage")
    print("  POST /api/predict-batch - Batch image prediction")
    print("  GET  /api/storage/stats - Get storage statistics")
    print("  GET  /api/storage/image/<id> - Get image information")
    print("  DELETE /api/storage/image/<id> - Delete image metadata")
    
    print(f"\nSecure Storage Status: {'Enabled' if storage else 'Disabled'}")
    if storage:
        print("Images will be stored on Lighthouse blockchain and MongoDB")
    else:
        print("Warning: Secure storage not available - check configuration")
    
    # Use PORT environment variable for Render deployment
    port = int(os.getenv('PORT', 5000))
    debug_mode = os.getenv('FLASK_ENV', 'production') == 'development'
    app.run(host='0.0.0.0', port=port, debug=debug_mode)