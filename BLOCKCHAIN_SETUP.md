# Blockchain Storage Setup Instructions

## 🔗 Setting up Lighthouse API for IPFS Storage

### Step 1: Get Lighthouse API Key

1. Go to [Lighthouse Storage](https://files.lighthouse.storage/)
2. Sign up or log in with your account
3. Navigate to the **API Keys** section
4. Click **Create New API Key**
5. Copy the generated API key

### Step 2: Update Environment Variables

Edit your `.env` file and replace the placeholder:

```env
LIGHTHOUSE_API_KEY=your_actual_lighthouse_api_key_here
```

### Step 3: Set up MongoDB (Optional but Recommended)

#### Option A: Local MongoDB

1. Download [MongoDB Community Server](https://www.mongodb.com/try/download/community)
2. Install and start the MongoDB service
3. Keep the default settings in `.env`:
   ```env
   MONGO_URI=mongodb://localhost:27017/
   MONGO_DB_NAME=medai_local_dev
   ```

#### Option B: MongoDB Atlas (Cloud)

1. Sign up at [MongoDB Atlas](https://www.mongodb.com/atlas)
2. Create a free cluster
3. Get your connection string
4. Update your `.env` file:
   ```env
   MONGO_URI=mongodb+srv://username:password@cluster.mongodb.net/
   MONGO_DB_NAME=medai_production
   ```

### Step 4: Test Your Setup

Run the setup script to verify everything works:

```bash
python setup_blockchain_storage.py
```

This will:

- ✅ Check your Lighthouse API key
- ✅ Test IPFS connection
- ✅ Verify MongoDB connection
- ✅ Run a complete storage test

### Step 5: Start Your Flask API

```bash
python app.py
```

## 🎯 What This Enables

With blockchain storage enabled, your application will:

1. **🔒 Secure Storage**: Images are stored on IPFS (decentralized, immutable)
2. **🗂️ Metadata Tracking**: MongoDB stores image metadata and prediction results
3. **🔗 Permanent Links**: Each image gets a permanent IPFS link
4. **📊 Analytics**: Track all predictions and their results
5. **🔄 Deduplication**: Avoid storing duplicate images

## 📋 Features in Your API Response

When blockchain storage is enabled, your API will return:

```json
{
  "success": true,
  "prediction": "Eczema",
  "confidence": 0.95,
  "storage_info": {
    "image_id": "507f1f77bcf86cd799439011",
    "lighthouse_hash": "QmXxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
    "gateway_url": "https://gateway.lighthouse.storage/ipfs/QmXxxxxx"
  },
  "ipfs": {
    "hash": "QmXxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
    "url": "https://gateway.lighthouse.storage/ipfs/QmXxxxxx",
    "lighthouse_url": "https://gateway.lighthouse.storage/ipfs/QmXxxxxx"
  }
}
```

## 🚨 Important Notes

- **API Key Security**: Never commit your real API key to version control
- **Rate Limits**: Lighthouse has rate limits on free accounts
- **Storage Costs**: IPFS storage may have costs for large amounts of data
- **Privacy**: Images stored on IPFS are publicly accessible via their hash

## 🔧 Troubleshooting

### "API key not configured" error

- Make sure you've replaced `your_lighthouse_api_key_here` with your actual key
- Restart your Flask application after updating the `.env` file

### MongoDB connection errors

- Ensure MongoDB is running locally, OR
- Use MongoDB Atlas with the correct connection string

### IPFS upload failures

- Check your internet connection
- Verify your Lighthouse API key is valid
- Check if you've exceeded your rate limit

## 🆓 Free Tier Limitations

**Lighthouse Free Tier:**

- 100 MB storage
- Rate limits apply

**MongoDB Atlas Free Tier:**

- 512 MB storage
- Suitable for thousands of image metadata records

For production use, consider upgrading to paid plans for better performance and higher limits.
