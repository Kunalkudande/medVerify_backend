from fastapi import FastAPI, File, UploadFile
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import cv2
import shutil
import uvicorn
from pathlib import Path
import requests
import os

# ✅ Initialize FastAPI
app = FastAPI()

# ✅ Google Drive File ID
FILE_ID = "1S5rjYgyOANUt62DxFJVjeNNAG8g_F2Un"
MODEL_PATH = "medical_deepfake_cnn.keras"

# ✅ Function to download model from Google Drive & verify integrity
def download_model():
    if not Path(MODEL_PATH).exists() or os.path.getsize(MODEL_PATH) < 500000:  # Check if file exists and is at least 500KB
        print("🔄 Downloading model from Google Drive...")
        
        # Google Drive direct download link handler
        session = requests.Session()
        url = f"https://drive.google.com/uc?export=download&id={FILE_ID}"
        response = session.get(url, stream=True)

        # Handle Google Drive confirmation page
        for key, value in response.cookies.items():
            if "download_warning" in key:
                url = f"https://drive.google.com/uc?export=download&id={FILE_ID}&confirm={value}"
                response = session.get(url, stream=True)

        if response.status_code == 200:
            with open(MODEL_PATH, "wb") as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            print("✅ Model downloaded successfully!")

            # ✅ Verify model integrity
            try:
                test_model = tf.keras.models.load_model(MODEL_PATH)
                print("✅ Model integrity verified!")
                return True  # Model is valid
            except Exception as e:
                print(f"❌ Model loading test failed: {e}")
                os.remove(MODEL_PATH)  # Delete the corrupt file
                return False  # Prevent loading a broken model
        else:
            print("❌ Failed to download model!")
            return False  # Prevents loading a broken model

    return True  # Download successful

# ✅ Lazy Loading: Load model only when needed
model = None

def get_model():
    global model
    if model is None:
        success = download_model()
        if success:  # Only load if model downloaded successfully
            try:
                model = tf.keras.models.load_model(MODEL_PATH)
                print("✅ Model loaded successfully!")
            except Exception as e:
                print(f"❌ Model loading failed: {e}")
                model = None  # Prevent crashing if model fails
    return model

# ✅ Class labels
class_labels = ["False-Benign (FB)", "False-Malicious (FM)", "True-Benign (TB)", "True-Malicious (TM)"]

# ✅ Function to preprocess and predict image
def predict_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    model = get_model()  # Load model only when needed
    if model is None:
        return {"error": "Model failed to load. Please check server logs."}

    predictions = model.predict(img)
    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions)

    return {"prediction": class_labels[predicted_class], "confidence": round(float(confidence), 2)}

# ✅ API endpoint to receive and predict image
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    temp_file = Path(f"temp_{file.filename}")

    # Save uploaded file
    with temp_file.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Predict the image
    result = predict_image(str(temp_file))

    # Remove the temporary file
    temp_file.unlink()

    return result

# ✅ Root endpoint for testing
@app.get("/")
def home():
    return {"message": "Welcome to MedVerify API! Use /predict/ to upload an image."}

# ✅ Run the server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
