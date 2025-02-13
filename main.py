from fastapi import FastAPI, File, UploadFile
import numpy as np
import tensorflow as tf
import cv2
import shutil
from pathlib import Path
from prometheus_fastapi_instrumentator import Instrumentator
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI app
app = FastAPI()

# Enable Prometheus metrics collection for monitoring
Instrumentator().instrument(app).expose(app)

# CORS Middleware - Allow frontend requests from React app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the path where the model is stored
MODEL_PATH = "medical_deepfake_cnn.keras"

# Ensure the model file exists before starting the server
if not Path(MODEL_PATH).exists():
    raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found. Please add it to the correct directory.")

# Load the model at startup
print("Loading model...")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None  # Prevent application crash if the model fails to load

# Define class labels for prediction output
class_labels = ["False-Benign (FB)", "False-Malicious (FM)", "True-Benign (TB)", "True-Malicious (TM)"]

def predict_image(img_path):
    """
    Loads an image, preprocesses it, and makes a prediction using the model.

    Args:
        img_path (str): Path to the image file.

    Returns:
        dict: Predicted class label and confidence score.
    """
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (224, 224))  # Resize to match model input size
    img = img / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    if model is None:
        return {"error": "Model failed to load. Check server logs for details."}

    predictions = model.predict(img)
    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions)

    return {"prediction": class_labels[predicted_class], "confidence": round(float(confidence), 2)}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """
    API endpoint to receive an image file, process it, and return the prediction.

    Args:
        file (UploadFile): Uploaded image file.

    Returns:
        dict: Prediction result containing class label and confidence score.
    """
    temp_dir = Path("temp/")
    temp_dir.mkdir(exist_ok=True)  # Ensure the temporary directory exists

    temp_file = temp_dir / file.filename  # Define file path

    # Save the uploaded file
    with temp_file.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Perform prediction
    result = predict_image(str(temp_file))

    # Remove the temporary file after prediction
    temp_file.unlink()

    return result

@app.get("/")
def home():
    """
    Root endpoint for API testing.
    """
    return {"message": "Welcome to MedVerify API. Use /predict/ to upload an image for prediction."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
