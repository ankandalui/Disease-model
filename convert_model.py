#!/usr/bin/env python3
"""
Model Format Converter
This script can help convert between different Keras model formats
"""

import os
import sys

def convert_model():
    """Convert keras model to different formats for compatibility"""
    try:
        import tensorflow as tf
        print(f"TensorFlow version: {tf.__version__}")
        
        # Try to load the original model
        if os.path.exists('best_model.keras'):
            print("Found best_model.keras")
            
            try:
                # Load with minimal options
                model = tf.keras.models.load_model('best_model.keras', compile=False)
                print("✅ Model loaded successfully!")
                
                # Save in H5 format as backup
                model.save('best_model.h5')
                print("✅ Saved as best_model.h5")
                
                # Save in SavedModel format
                model.save('saved_model_dir')
                print("✅ Saved as SavedModel format")
                
                # Test prediction
                import numpy as np
                test_input = np.random.random((1, 224, 224, 3))
                prediction = model.predict(test_input, verbose=0)
                print(f"✅ Test prediction successful! Shape: {prediction.shape}")
                
                return True
                
            except Exception as e:
                print(f"❌ Model conversion failed: {e}")
                return False
        else:
            print("❌ best_model.keras not found")
            return False
            
    except ImportError as e:
        print(f"❌ TensorFlow import failed: {e}")
        return False

if __name__ == "__main__":
    print("🔄 Starting model conversion...")
    success = convert_model()
    
    if success:
        print("🎉 Model conversion completed successfully!")
    else:
        print("💥 Model conversion failed!")
        sys.exit(1)
