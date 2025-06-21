# Skin Disease Classification API

A Flask-based API for skin disease classification using machine learning, with secure storage capabilities.

## Features

- **Machine Learning**: Classifies 10 different types of skin diseases
- **Secure Storage**: Stores images on Lighthouse blockchain and metadata in MongoDB
- **RESTful API**: Clean API endpoints for integration
- **CORS Support**: Configured for cross-origin requests

## API Endpoints

- `GET /api/health` - Health check
- `GET /api/classes` - Get available disease classes
- `POST /api/predict` - Single image prediction
- `POST /api/predict-batch` - Batch image prediction
- `GET /api/storage/stats` - Storage statistics
- `GET /api/storage/image/<id>` - Get image information
- `DELETE /api/storage/image/<id>` - Delete image metadata

## Deployment on Render

### Prerequisites

1. A Render account
2. MongoDB Atlas database (or other MongoDB instance)
3. Lighthouse API key (optional, for blockchain storage)

### Environment Variables

Set these environment variables in your Render dashboard:

```
LIGHTHOUSE_API_KEY=your_lighthouse_api_key
MONGO_URI=your_mongodb_connection_string
MONGO_DB_NAME=medai_secure_storage
FLASK_SECRET_KEY=your_secret_key
MAX_FILE_SIZE_MB=16
CORS_ORIGINS=https://your-frontend-domain.com
FLASK_ENV=production
```

### Deployment Steps

1. **Connect Repository**: Link your GitHub repository to Render
2. **Create Web Service**: Choose "Web Service" and select your repository
3. **Configure Build**:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120`
4. **Set Environment Variables**: Add all required environment variables
5. **Deploy**: Click "Create Web Service"

### File Structure

```
skin/
├── app.py                 # Main Flask application
├── secure_storage.py      # Secure storage implementation
├── best_model.keras       # ML model file
├── requirements.txt       # Python dependencies
├── Procfile              # Render deployment config
├── .env                  # Environment variables (local)
├── .env.example          # Environment template
└── README.md            # This file
```

## Supported Disease Classes

1. Eczema
2. Warts Molluscum and other Viral Infections
3. Melanoma
4. Atopic Dermatitis
5. Basal Cell Carcinoma (BCC)
6. Melanocytic Nevi (NV)
7. Benign Keratosis-like Lesions (BKL)
8. Psoriasis pictures Lichen Planus and related diseases
9. Seborrheic Keratoses and other Benign Tumors
10. Tinea Ringworm Candidiasis and other Fungal Infections

## Usage

### Single Image Prediction

```bash
curl -X POST https://your-app.render.com/api/predict \
  -F "file=@image.jpg" \
  -F "metadata={\"patient_id\":\"123\",\"age\":30}"
```

### Health Check

```bash
curl https://your-app.render.com/api/health
```

## Security Features

- Secure image storage on Lighthouse blockchain
- Metadata backup in MongoDB
- Image hash verification
- Environment-based configuration
- CORS protection

## Note

This application is for educational and research purposes only. Always consult with healthcare professionals for medical diagnosis and treatment.
