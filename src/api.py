import io
import os
import sys
import yaml
import torch
import cv2
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from contextlib import asynccontextmanager
from PIL import Image
from fastapi.middleware.cors import CORSMiddleware

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.lightning_module import DiseaseModule

# Global variables
model = None
config = None
label_map = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_config():
    with open('configs/default.yaml', 'r') as f:
        return yaml.safe_load(f)

def load_labels(data_path):
    labels_path = os.path.join(data_path, 'labels.csv')
    if os.path.exists(labels_path):
        df = pd.read_csv(labels_path)
        return dict(zip(df['label_idx'], df['class_name']))
    return None

def preprocess_image(image_bytes, img_size):
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image = np.array(image)
        if len(image.shape) == 2: # Grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4: # RGBA
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            
        transform = A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
        
        augmented = transform(image=image)
        return augmented['image'].unsqueeze(0)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model on startup
    global model, config, label_map
    try:
        config = load_config()
        checkpoint_path = config.get('best_checkpoint_path', 'weights/best.ckpt')
        
        if os.path.exists(checkpoint_path):
            print(f"Loading model from {checkpoint_path}...")
            model = DiseaseModule.load_from_checkpoint(checkpoint_path, config=config)
            model.eval()
            model.to(device)
            print("Model loaded successfully.")
        else:
            print(f"Warning: Checkpoint {checkpoint_path} not found. Model will not work.")
            
        label_map = load_labels(config['data_path'])
        
    except Exception as e:
        print(f"Error loading model: {e}")
        
    yield
    # Cleanup if needed

app = FastAPI(title="Ethiopia Plant Disease API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    print(f"DEBUG: {request.method} {request.url}")
    response = await call_next(request)
    print(f"DEBUG: Status {response.status_code}")
    return response

@app.get("/")
def read_root():
    return {"message": "Welcome to Ethiopia Plant Disease Classification API"}

@app.get("/health")
def health_check():
    if model is None:
        return {"status": "unhealthy", "reason": "Model not loaded"}
    return {"status": "healthy"}

@app.get("/classes")
def get_classes():
    if label_map:
        return {"classes": list(label_map.values())}
    return {"classes": []}

@app.post("/predict")
async def predict_disease(file: UploadFile = File(...)):
    print(f"Received prediction request: {file.filename}")
    if model is None:
        print("Error: Model is not loaded!")
        raise HTTPException(status_code=503, detail="Model not loaded")
        
    try:
        contents = await file.read()
        print(f"File size: {len(contents)} bytes")
        
        input_tensor = preprocess_image(contents, config['img_size'])
        input_tensor = input_tensor.to(device)
        print("Image preprocessed successfully.")
        
        with torch.no_grad():
            logits = model(input_tensor)
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            
        confidence, predicted_idx = torch.max(probabilities, 1)
        idx = predicted_idx.item()
        conf = confidence.item()
        
        label_name = label_map[idx] if label_map else str(idx)
        print(f"Prediction: {label_name} ({conf:.2f})")
        
        # Get all probabilities
        probs_dict = {}
        if label_map:
            for i, prob in enumerate(probabilities[0]):
                probs_dict[label_map[i]] = float(prob)
                
        return {
            "disease": label_name,
            "confidence": float(conf),
            "probabilities": probs_dict
        }
    except Exception as e:
        print(f"CRITICAL ERROR during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
