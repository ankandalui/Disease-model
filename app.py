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
import google.generativeai as genai

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

# Initialize Secure Storage only if explicitly enabled
enable_storage = os.getenv('ENABLE_BLOCKCHAIN_STORAGE', 'false').lower() == 'true'

if enable_storage:
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
else:
    storage = None
    print("Secure storage disabled for local development")

# Load the model with multiple fallback strategies
model = None
model_error = None

print("🔍 Starting model loading diagnostics...")

# Check environment and file system first
try:
    import os
    import tensorflow as tf
    print(f"📊 TensorFlow version: {tf.__version__}")
    print(f"📊 Python version: {os.sys.version}")
    print(f"📂 Current directory: {os.getcwd()}")
    print(f"📂 Directory contents: {os.listdir('.')}")
    
    if os.path.exists('best_model.keras'):
        file_size = os.path.getsize('best_model.keras')
        print(f"✅ Model file found: best_model.keras ({file_size:,} bytes)")
    else:
        print("❌ Model file 'best_model.keras' not found!")
        print("📂 Looking for alternative model files...")
        model_files = [f for f in os.listdir('.') if f.endswith(('.keras', '.h5', '.pb'))]
        print(f"📂 Found model files: {model_files}")
        
except Exception as e:
    print(f"❌ Environment check failed: {e}")

print("\n🔄 Attempting to load model...")

# Strategy 1: Load with compile=False
try:
    print("📝 Strategy 1: Loading with compile=False...")
    model = load_model('best_model.keras', compile=False)
    if model is not None:
        print("📝 Manually compiling model...")
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print("✅ Model loaded and compiled successfully!")
        
        # Test prediction to ensure model works
        import numpy as np
        test_input = np.random.random((1, 224, 224, 3))
        test_pred = model.predict(test_input, verbose=0)
        print(f"✅ Model test prediction successful! Output shape: {test_pred.shape}")
        
except Exception as e:
    print(f"❌ Strategy 1 failed: {e}")
    model_error = str(e)
    model = None

# Strategy 2: Load with safe_mode=False if Strategy 1 fails
if model is None:
    try:
        print("📝 Strategy 2: Loading with safe_mode=False...")
        model = load_model('best_model.keras', compile=False, safe_mode=False)
        if model is not None:
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            print("✅ Model loaded with safe_mode=False and compiled!")
    except Exception as e:
        print(f"❌ Strategy 2 failed: {e}")
        model_error = str(e)

# Strategy 3: Try loading as H5 format if available
if model is None:
    try:
        print("📝 Strategy 3: Looking for H5 format...")
        import os
        if os.path.exists('best_model.h5'):
            model = load_model('best_model.h5')
            print("✅ Model loaded from H5 format!")
        else:
            print("❌ No H5 model file found")
    except Exception as e:
        print(f"❌ Strategy 3 (H5) failed: {e}")
        model_error = str(e)

# Strategy 4: Try with custom objects (in case of custom layers)
if model is None:
    try:
        print("📝 Strategy 4: Loading with custom_objects...")
        custom_objects = {'tf': tf}  # Add TensorFlow to custom objects
        model = load_model('best_model.keras', custom_objects=custom_objects, compile=False)
        if model is not None:
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            print("✅ Model loaded with custom_objects!")
    except Exception as e:
        print(f"❌ Strategy 4 failed: {e}")
        model_error = str(e)

# Final status
if model is None:
    print("\n🚨 CRITICAL: Model could not be loaded with any strategy")
    print(f"🔧 Final error: {model_error}")
    print("🔧 The API will run but predictions will return errors")
else:
    print(f"\n🎉 SUCCESS: Model loaded successfully!")
    try:
        print(f"📊 Model summary: {model.count_params():,} parameters")
        # Test the model with a dummy prediction to ensure it works
        import numpy as np
        test_input = np.random.random((1, 224, 224, 3))
        test_prediction = model.predict(test_input, verbose=0)
        print(f"✅ Model test successful! Output shape: {test_prediction.shape}")
    except Exception as e:
        print(f"⚠️ Model loaded but test prediction failed: {e}")

print("\n" + "="*50)

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
    # Always use Keras preprocessing for consistency with HTML Flask app
    if isinstance(img_data, str):  # file path
        img = image.load_img(img_data, target_size=(224, 224))
    else:  # PIL Image - save temporarily and reload with Keras
        # Save PIL image temporarily
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            img_data.save(tmp_file.name, 'JPEG')
            tmp_path = tmp_file.name
        
        # Load with Keras preprocessing (same as HTML Flask app)
        img = image.load_img(tmp_path, target_size=(224, 224))
        
        # Clean up temporary file
        import os
        os.unlink(tmp_path)
    
    # Use identical preprocessing as HTML Flask app
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize
    
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = float(np.max(predictions[0]))
    
    return class_names[predicted_class], confidence

# Configure Gemini API
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

def validate_skin_image_with_gemini(image_data: bytes) -> dict:
    """
    Use Gemini to validate if the image contains skin/skin disease content
    """
    try:
        gemini_api_key = os.getenv('GEMINI_API_KEY')
        if not gemini_api_key or gemini_api_key == 'your_gemini_api_key_here':
            return {
                'is_valid': True,  # Allow all images if Gemini is not configured
                'message': 'Gemini validation disabled - API key not configured',
                'confidence': 1.0
            }
        
        # Convert image data to PIL Image for Gemini
        img = Image.open(io.BytesIO(image_data))
        
        # Create Gemini model
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Prompt for skin disease validation
        prompt = """
        Please analyze this image and determine if it shows:
        1. Human skin (any body part)
        2. Skin conditions, lesions, rashes, or diseases
        3. Medical/dermatological content
        
        Respond with ONLY one of these:
        - "VALID_SKIN" if the image shows skin or skin conditions
        - "NOT_SKIN" if the image does not show skin or medical content
        - "UNCLEAR" if you cannot determine clearly
        
        Be strict - only allow clear skin/medical images.
        """
        
        # Generate content with the image
        response = model.generate_content([prompt, img])
        
        result_text = response.text.strip().upper()
        
        if "VALID_SKIN" in result_text:
            return {
                'is_valid': True,
                'message': 'Gemini confirmed: Image contains skin/medical content',
                'confidence': 0.9,
                'gemini_response': result_text
            }
        elif "NOT_SKIN" in result_text:
            return {
                'is_valid': False,
                'message': 'Please upload a skin or skin disease image. The uploaded image does not appear to contain skin or medical dermatological content.',
                'confidence': 0.9,
                'gemini_response': result_text
            }
        else:
            return {
                'is_valid': False,
                'message': 'Please upload a clear skin or skin disease image. We could not clearly identify skin-related content in the uploaded image.',
                'confidence': 0.5,
                'gemini_response': result_text
            }
        
    except Exception as e:
        print(f"Gemini validation error: {e}")
        return {
            'is_valid': True,  # Allow prediction to proceed if Gemini fails
            'message': f'Gemini validation failed: {str(e)}',
            'confidence': 0.0
        }

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'model_error': model_error if model is None else None,
        'storage_available': storage is not None,
        'version': os.getenv('MODEL_VERSION', 'v1.0'),
        'message': 'Flask API is running',
        'tensorflow_version': __import__('tensorflow').__version__,
        'python_version': __import__('sys').version,
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
                'message': f'The ML model could not be loaded. Error: {model_error}',
                'success': False,
                'model_status': 'unavailable'
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
            }), 400        # Validate image with Gemini before prediction
        validation_result = validate_skin_image_with_gemini(image_data)
        if not validation_result['is_valid']:
            return jsonify({
                'error': 'Invalid image type',
                'message': validation_result['message'],
                'instructions': 'Please upload an image showing human skin or skin conditions such as rashes, lesions, moles, or other dermatological conditions.',
                'accepted_formats': 'JPG, JPEG, PNG',
                'validation_confidence': validation_result['confidence'],
                'success': False
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
            'timestamp': prediction_data['timestamp'],
            'validation': {
                'gemini_validated': True,
                'validation_message': validation_result['message'],
                'validation_confidence': validation_result['confidence']
            }
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