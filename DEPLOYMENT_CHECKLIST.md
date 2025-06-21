# Deployment Checklist for Render

## Pre-deployment Steps

- [x] Cleaned up unused files and directories
- [x] Added Procfile for Render
- [x] Updated requirements.txt with gunicorn
- [x] Modified app.py for production (PORT env var, debug mode)
- [x] Created .env.example for documentation
- [x] Added comprehensive README.md
- [x] Created .gitignore file
- [x] Enhanced health check endpoint

## Files Ready for Deployment

```
skin/
├── app.py                 # Main Flask application (production-ready)
├── secure_storage.py      # Secure storage implementation
├── best_model.keras       # ML model file (19MB - ensure this uploads)
├── requirements.txt       # Python dependencies with gunicorn
├── Procfile              # Render deployment config
├── .env                  # Local environment (excluded from git)
├── .env.example          # Environment template
├── .gitignore           # Git ignore rules
├── README.md            # Documentation
└── render-env-vars.txt  # Environment variables for Render
```

## Render Setup Steps

1. **Create Web Service**

   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120`

2. **Environment Variables** (copy from render-env-vars.txt)

   ```
   LIGHTHOUSE_API_KEY=7b146688.fc67f2808630406ba32be48e442b7f7f
   MONGO_URI=mongodb+srv://sutapak2903:mTSMDqu5lv94qEae@cluster0.gxohtww.mongodb.net/medai?retryWrites=true&w=majority&appName=MedAI
   MONGO_DB_NAME=medai_secure_storage
   FLASK_SECRET_KEY=prod-secret-key-change-this-in-production-12345
   CORS_ORIGINS=http://localhost:3000,https://your-frontend-domain.onrender.com
   ```

3. **Important Notes**
   - Model file (best_model.keras) is ~19MB - ensure it uploads successfully
   - First deployment may take 5-10 minutes due to TensorFlow installation
   - Health check available at: `https://your-app.onrender.com/api/health`
   - Update CORS_ORIGINS with your actual frontend domain

## Test Endpoints After Deployment

- `GET /api/health` - Should return healthy status
- `GET /api/classes` - Should return list of 10 disease classes
- `POST /api/predict` - Test with a sample image

## Security Considerations

- [x] Removed sensitive data from .env (use Render environment variables)
- [x] Added proper CORS configuration
- [x] Configured for production mode
- [x] Added .gitignore to exclude sensitive files
