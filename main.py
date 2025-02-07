from fastapi import FastAPI, File, UploadFile
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import cv2
import shutil
import uvicorn
from pathlib import Path

# ✅ Initialize FastAPI
app = FastAPI()

# ✅ Google Drive File ID
FILE_ID = "1S5rjYgyOANUt62DxFJVjeNNAG8g_F2Un"
MODEL_PATH = "medical_deepfake_cnn.keras"

# ✅ Function to download model from Google Drive
def download_model():
    if not Path(MODEL_PATH).exists():  # Download only if the model is missing
        print("🔄 Downloading model from Google Drive...")
        url = f"https://drive.google.com/uc?export=download&id={FILE_ID}"
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(MODEL_PATH, "wb") as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            print("✅ Model downloaded successfully!")
        else:
            print("❌ Failed to download model!")

# ✅ Lazy Loading: Load model only when needed
model = None
def get_model():
    global model
    if model is None:
        download_model()  # Ensure model is downloaded
        model = tf.keras.models.load_model(MODEL_PATH)
    return model

# ✅ Class labels
class_labels = ["False-Benign (FB)", "False-Malicious (FM)", "True-Benign (TB)", "True-Malicious (TM)"]

# ✅ Function to preprocess and predict image
def predict_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)  # Ensure 3 channels
    img = cv2.resize(img, (224, 224))  # Resize
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    model = get_model()  # Load model only when needed
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

# ✅ Run the server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
