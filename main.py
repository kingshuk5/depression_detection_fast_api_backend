import io
import os
import cv2
import joblib
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense

app = FastAPI(title="Facial Expression Hybrid Detector API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CLASS_NAMES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# Rebuild exact CNN structure using the Keras Functional API
def build_feature_extractor_network():
    input_layer = Input(shape=(48, 48, 1))
    
    x = Conv2D(64, kernel_size=(3, 3), activation='relu')(input_layer)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)
    
    x = Conv2D(128, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)
    
    x = Conv2D(256, kernel_size=(3, 3), activation='relu', name='last_conv_layer')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)
    
    x = Flatten(name='flatten_layer')(x)
    
    feature_layer_output = Dense(128, activation='relu', name='feature_layer')(x)
    
    x = Dropout(0.5)(feature_layer_output)
    final_output = Dense(7, activation='softmax')(x)
    
    model = Model(inputs=input_layer, outputs=final_output)
    return model


ADABOOST_PATH = "adaboost_emotion_classifier.pkl"
WEIGHTS_PATH = "emotion_cnn.weights.h5"

try:
    print("Building CNN architecture...")
    base_cnn = build_feature_extractor_network()
    
    base_cnn.load_weights(WEIGHTS_PATH)
    
    feature_extractor = Model(
        inputs=base_cnn.input, 
        outputs=base_cnn.get_layer('feature_layer').output
    )
    print("✅ CNN Structure rebuilt and weights loaded successfully!")
    
    adaboost_model = joblib.load(ADABOOST_PATH)
    print("✅ AdaBoost Classifier loaded successfully!")

except Exception as e:
    print(f"❌ Error loading models/weights: {e}")
    adaboost_model = None
    feature_extractor = None

def preprocess_frame(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    frame = np.array(image)
    
    if len(frame.shape) == 3 and frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2GRAY)
    elif len(frame.shape) == 3 and frame.shape[2] == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
    IMAGE_SIZE = 48
    frame = cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE))
    
    frame = frame.astype("float32") / 255.0
    
    frame = np.expand_dims(frame, axis=-1)
    
    frame = np.expand_dims(frame, axis=0)
    
    return frame


#  WebSocket Streaming Endpoint
@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Frontend stream link opened!")
    
    try:
        while True:
            data = await websocket.receive_bytes()
            
            if adaboost_model is None or feature_extractor is None:
                await websocket.send_json({"error": "Backend Error: Models uninitialized."})
                continue
            
            try:
                processed_image = preprocess_frame(data)
                
                extracted_features = feature_extractor.predict(processed_image, verbose=0)
                
                prediction = adaboost_model.predict(extracted_features)
                
                predicted_idx = int(prediction[0])
                detected_emotion = CLASS_NAMES[predicted_idx]
                
                await websocket.send_json({
                    "expression": detected_emotion,
                    "class_index": predicted_idx
                })
                
            except Exception as inference_error:
                await websocket.send_json({"error": f"Inference pipeline crash: {str(inference_error)}"})
                
    except WebSocketDisconnect:
        print("Frontend stream link closed.")