import os

# --- RENDER MEMORY OPTIMIZATION ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import tflite_runtime.interpreter as tflite

import httpx
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv
import uvicorn
import joblib
import pandas as pd
import numpy as np
import json
import io

from tensorflow.keras.preprocessing import image
from PIL import Image

# --- LOAD ENVIRONMENT VARIABLES ---
load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
WEATHER_API_KEY = os.environ.get("WEATHER_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is not set. Please add it to your .env file or Render environment variables.")

if not WEATHER_API_KEY:
    raise RuntimeError("WEATHER_API_KEY is not set. Please add it to your .env file or Render environment variables.")

# --- LOAD MODELS ---

# Lazy loading for TFLite model
_interpreter = None
_crop_model = None
_scaler = None
_label_encoder = None

def get_interpreter():
    global _interpreter
    if _interpreter is None:
        _interpreter = tflite.Interpreter(model_path="model_optimized.tflite")
        _interpreter.allocate_tensors()
    return _interpreter


def get_crop_model():
    global _crop_model, _scaler, _label_encoder
    if _crop_model is None:
        _crop_model = joblib.load("best_model.pkl")
        _scaler = joblib.load("scaler.pkl")
        _label_encoder = joblib.load("label_encoder.pkl")
    return _crop_model, _scaler, _label_encoder

with open("disease_labels.json", "r") as f:
    disease_labels = json.load(f)

app = FastAPI()

# Static files serve করার জন্য
app.mount("/static", StaticFiles(directory="."), name="static")

@app.get("/")
async def read_index():
    return FileResponse("index.html")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Groq(api_key=GROQ_API_KEY)

# --- MODELS ---
class FarmerProfile(BaseModel):
    name: str = "Farmer"
    age: str = "Unknown"
    location: str = "India"
    soil: str = "General"
    crop: str = "Crops"
    size: str = "Unknown"
    irrigation: str = "General"

class SensorData(BaseModel):
    temperature: float
    humidity: float

class PredictionInput(BaseModel):
    n: float
    p: float
    k: float
    ph: float

class ChatRequest(BaseModel):
    message: str
    language: str = "English"

# --- GLOBAL STORAGE ---
current_farmer_data = FarmerProfile()
latest_sensors = SensorData(temperature=0.0, humidity=0.0)

# 1. ESP32 ENDPOINT
@app.post("/update-sensors")
async def update_sensors(data: SensorData):
    global latest_sensors
    latest_sensors = data
    print(f"📡 Hardware Update -> Temp: {data.temperature}°C, Hum: {data.humidity}%")
    return {"status": "success"}

# 2. PROFILE ENDPOINT
@app.post("/save-profile")
async def save_profile(profile: FarmerProfile):
    global current_farmer_data
    current_farmer_data = profile
    print(f"✅ Profile saved: {profile.name}")
    return {"status": "success"}

# 3. DISEASE PREDICTION ENDPOINT
@app.post("/predict-disease")
async def predict_disease(file: UploadFile = File(...)):
    interp = get_interpreter()
    input_details = interp.get_input_details()
    output_details = interp.get_output_details()

    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        interp.set_tensor(input_details[0]['index'], img_array)
        interp.invoke()
        prediction = interp.get_tensor(output_details[0]['index'])
        class_idx = np.argmax(prediction)
        confidence = np.max(prediction)
        disease_name = disease_labels[str(class_idx)]

        treatment_prompt = f"""
        The crop disease identified is '{disease_name}'. 
        Provide a concise, 3-step organic treatment guide for an Indian small-scale farmer. 
        Keep it practical and easy to follow.
        """

        ai_res = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": treatment_prompt}]
        )

        return {
            "disease": disease_name,
            "confidence": f"{confidence*100:.2f}%",
            "treatment": ai_res.choices[0].message.content
        }
    except Exception as e:
        print(f"Disease Prediction Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 4. CROP RECOMMENDATION ENDPOINT
@app.post("/predict-crop")
async def predict_crop(manual: PredictionInput):
    model, scaler, label_encoder = get_crop_model()
    try:
        current_temp = latest_sensors.temperature if latest_sensors.temperature != 0 else 25.0
        current_hum = latest_sensors.humidity if latest_sensors.humidity != 0 else 50.0

        features = np.array([[
            manual.n, manual.p, manual.k, current_temp, current_hum, manual.ph, 100.0
        ]])

        scaled_features = scaler.transform(features)
        prediction_id = model.predict(scaled_features)[0]
        crop_name = label_encoder.inverse_transform([prediction_id])[0]

        advisory_prompt = f"""
Act as a Senior Agricultural Consultant. Provide a cultivation guide for {crop_name.upper()}.
YOU MUST format your response as a simple list where each line contains exactly one '|' character.
DO NOT use markdown table syntax.
Example Format:
Time to Harvest | 90-120 days
Organic Manures | 2-3 tons of Vermicompost per acre
Water Management | Drip irrigation recommended
Disease Control | Use Neem cake or organic pesticides
"""

        explanation = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a precise agricultural assistant. Output only lines separated by | for data table parsing."},
                {"role": "user", "content": advisory_prompt}
            ],
            temperature=0.2
        )

        return {
            "recommendation": crop_name.upper(),
            "reasoning": explanation.choices[0].message.content,
            "sensors_used": {"temp": current_temp, "hum": current_hum}
        }

    except Exception as e:
        print(f"ML Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 5. CHAT ENDPOINT
@app.post("/ask")
async def ask_bot(request: ChatRequest):
    try:
        weather_url = f"https://api.openweathermap.org/data/2.5/weather?q={current_farmer_data.location}&appid={WEATHER_API_KEY}&units=metric"
        weather_info = "Weather unavailable."
        alert = ""

        async with httpx.AsyncClient() as client_http:
            w_res = await client_http.get(weather_url)
            if w_res.status_code == 200:
                wd = w_res.json()
                temp = wd['main']['temp']
                desc = wd['weather'][0]['description'].lower()
                weather_info = f"{temp}°C, {desc}"

                if "rain" in desc or "storm" in desc:
                    alert = "⚠️ Heavy rain expected. Postpone your irrigation to save water."

        context = (
            f"Farmer: {current_farmer_data.name}, Crop: {current_farmer_data.crop}, "
            f"Soil: {current_farmer_data.soil}, Location: {current_farmer_data.location}, "
            f"Live Weather: {weather_info}, Sensor Temp: {latest_sensors.temperature}°C"
        )

        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": f"You are KrishiMitra AI. Context: {context}. Respond in {request.language}."},
                {"role": "user", "content": request.message}
            ]
        )

        return {
            "response": completion.choices[0].message.content,
            "alert": alert
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)
