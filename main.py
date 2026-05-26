import io
import os
import cv2
import joblib
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

# Suppress noisy TensorFlow startup logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense

# 1. Initialize the FastAPI app
app = FastAPI(title="Facial Expression Hybrid Detector API")

# 2. Configure CORS so your local HTML testing file can talk to the server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Class names ordered alphabetically matching your dataset structure
CLASS_NAMES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# 3. Rebuild your exact CNN structure using modern Keras 3 layers
# 3. Rebuild your exact CNN structure using the Keras Functional API
def build_feature_extractor_network():
    # Explicitly define the input tensor shape
    input_layer = Input(shape=(48, 48, 1))
    
    # Chain the layers together exactly like your Cell 3 architecture
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
    
    # This is the 128-node target layer we need for AdaBoost features
    feature_layer_output = Dense(128, activation='relu', name='feature_layer')(x)
    
    x = Dropout(0.5)(feature_layer_output)
    final_output = Dense(7, activation='softmax')(x)
    
    # Construct the final concrete functional model
    model = Model(inputs=input_layer, outputs=final_output)
    return model


# 4. Global Model Initialization on Server Startup
ADABOOST_PATH = "adaboost_emotion_classifier.pkl"
WEIGHTS_PATH = "emotion_cnn.weights.h5"

try:
    # STEP A: Reconstruct the CNN structure first
    print("Building CNN architecture...")
    base_cnn = build_feature_extractor_network()
    
    # STEP B: Load the saved weights into the built structure
    base_cnn.load_weights(WEIGHTS_PATH)
    
    # STEP C: Slice the network to output directly from the 128-node 'feature_layer'
    feature_extractor = Model(
        inputs=base_cnn.input, 
        outputs=base_cnn.get_layer('feature_layer').output
    )
    print("✅ CNN Structure rebuilt and weights loaded successfully!")
    
    # STEP D: Load the AdaBoost classifier last
    adaboost_model = joblib.load(ADABOOST_PATH)
    print("✅ AdaBoost Classifier loaded successfully!")

except Exception as e:
    print(f"❌ Error loading models/weights: {e}")
    adaboost_model = None
    feature_extractor = None

# 5. Preprocessing pipeline matching Cell 2 of your training notebook
def preprocess_frame(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    frame = np.array(image)
    
    # Step A: Convert from color formats (RGB/RGBA) to Grayscale
    if len(frame.shape) == 3 and frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2GRAY)
    elif len(frame.shape) == 3 and frame.shape[2] == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
    # Step B: Resize to 48x48
    IMAGE_SIZE = 48
    frame = cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE))
    
    # Step C: Normalize pixel values to [0.0, 1.0]
    frame = frame.astype("float32") / 255.0
    
    # Step D: Add channel dimension -> shape becomes (48, 48, 1)
    frame = np.expand_dims(frame, axis=-1)
    
    # Step E: Add batch dimension -> shape becomes (1, 48, 48, 1)
    frame = np.expand_dims(frame, axis=0)
    
    return frame


# 6. WebSocket Streaming Endpoint
@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Frontend stream link opened!")
    
    try:
        while True:
            # Receive binary frame packets from index.html canvas setup
            data = await websocket.receive_bytes()
            
            if adaboost_model is None or feature_extractor is None:
                await websocket.send_json({"error": "Backend Error: Models uninitialized."})
                continue
            
            try:
                # 1. Format the image bytes to matrix shape (1, 48, 48, 1)
                processed_image = preprocess_frame(data)
                
                # 2. Extract the 128 numerical features from the CNN 'feature_layer'
                extracted_features = feature_extractor.predict(processed_image, verbose=0)
                
                # 3. Fire those 128 deep learning features straight into the AdaBoost classifier
                prediction = adaboost_model.predict(extracted_features)
                
                # 4. Map the integer class prediction back to the string emotion text
                predicted_idx = int(prediction[0])
                detected_emotion = CLASS_NAMES[predicted_idx]
                
                # 5. Emit payload back to index.html frontend script
                await websocket.send_json({
                    "expression": detected_emotion,
                    "class_index": predicted_idx
                })
                
            except Exception as inference_error:
                await websocket.send_json({"error": f"Inference pipeline crash: {str(inference_error)}"})
                
    except WebSocketDisconnect:
        print("Frontend stream link closed.")