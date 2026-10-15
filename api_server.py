import os
import sys
import torch
import numpy as np
import scipy.special
import cv2
import requests
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, Form
import uvicorn
import shutil
import tempfile
from gtts import gTTS
import base64

# Add bsl1k to path so we can import its modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'bsl1k'))

from models import InceptionI3d
from demo.demo import load_vocab, prepare_input, sliding_windows
# Import helper functions from your existing script
from offline_pipeline_app import load_rgb_video_cv2, load_model_device_agnostic

app = FastAPI(title="ASL Translator API")

# Global variables for model
slr_model = None
device = None
word_data = None

@app.on_event("startup")
async def startup_event():
    global slr_model, device, word_data
    print("Loading I3D SLR model... this may take a moment.")
    checkpoint_url = "https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/wlasl_i3d_pbsl1k/model.pth.tar"
    checkpoint_path = Path("bsl1k_weights/model.pth.tar")
    word_data_url = "https://www.robots.ox.ac.uk/~vgg/research/bsl1k/demo-data/wlasl/info/info.pkl"
    word_data_path = Path("bsl1k_weights/info.pkl")
    
    # Ensure dirs exist
    os.makedirs("bsl1k_weights", exist_ok=True)
    
    slr_model, device = load_model_device_agnostic(
        checkpoint_path=checkpoint_path,
        checkpoint_url=checkpoint_url,
        num_classes=2000,
        num_in_frames=64,
    )
    word_data = load_vocab(word_data_pkl_path=word_data_path, word_data_pkl_url=word_data_url)
    print("Model Loaded and Ready for inference ✓")

@app.post("/translate_video")
async def translate_video(file: UploadFile = File(...), target_language: str = Form("English")):
    # Save the uploaded file temporarily
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    try:
        with open(temp_file.name, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # 1. Preprocessing
        print("Step 1: Preprocessing video frames...")
        rgb_orig = load_rgb_video_cv2(Path(temp_file.name))
        rgb_input = prepare_input(rgb_orig)
        
        stride = 8 
        rgb_slides, _ = sliding_windows(rgb=rgb_input, stride=stride, num_in_frames=64)
        
        # 2. SLR Prediction
        print(f"Step 2: Predicting signs across {rgb_slides.shape[0]} windows...")
        raw_predictions = []
        with torch.no_grad():
            for i in range(rgb_slides.shape[0]):
                inp = rgb_slides[i:i+1].to(device)
                out = slr_model(inp)
                logits = out["logits"].cpu().numpy()[0]
                probs = scipy.special.softmax(logits)
                top_idx = np.argmax(probs)
                word = word_data["words"][top_idx]
                conf = probs[top_idx]
                
                if conf > 0.45: 
                    raw_predictions.append(word)

        # Noise reduction logic
        stable_predictions = []
        count = 1
        for i in range(1, len(raw_predictions)):
            if raw_predictions[i] == raw_predictions[i-1]:
                count += 1
            else:
                if count >= 3: 
                    stable_predictions.append(raw_predictions[i-1])
                count = 1
        if len(raw_predictions) > 0 and count >= 3:
            stable_predictions.append(raw_predictions[-1])

        predicted_glosses = []
        for word in stable_predictions:
            if len(predicted_glosses) == 0 or predicted_glosses[-1] != word:
                predicted_glosses.append(word)
        
        if not predicted_glosses:
            predicted_glosses = ["unknown"]
            
        gloss_str = " ➔ ".join(predicted_glosses)
        print(f"Predicted Glosses: {gloss_str}")
        
        # 3. Gemini LLM API
        print("Step 3: Reconstructing language using Gemini API...")
        api_key = os.getenv("GEMINI_API_KEY")
        final_sentence = " ".join(predicted_glosses).capitalize() + "."
        
        if api_key:
            try:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
                prompt = (
                    f"You are an expert Sign Language Translator. "
                    f"I will give you a sequence of raw, disjointed sign language words that were predicted by a vision model. "
                    f"These words might contain duplicates, have poor grammar, or be slightly out of order. "
                    f"Your job is to infer the intended meaning and output ONLY a single, natural, grammatically correct sentence in {target_language}. "
                    f"If the language is Hindi or Marathi, output the sentence in the respective native script (Devanagari). "
                    f"Do not explain yourself. Do not add quotes.\n\n"
                    f"Raw Words: {' '.join(predicted_glosses)}\n\n"
                    f"Natural Sentence:"
                )
                headers = {'Content-Type': 'application/json'}
                data = {"contents": [{"parts": [{"text": prompt}]}]}
                
                response = requests.post(url, headers=headers, json=data)
                response_json = response.json()
                final_sentence = response_json['candidates'][0]['content']['parts'][0]['text'].strip().replace('"', '')
            except Exception as e:
                print(f"Gemini API Error: {str(e)}")
        else:
            print("WARNING: No GEMINI_API_KEY found in environment.")
        
        # 4. Audio Generation using gTTS
        lang_code = 'en'
        if target_language == "Hindi":
            lang_code = 'hi'
        elif target_language == "Marathi":
            lang_code = 'mr'
            
        tts = gTTS(text=final_sentence, lang=lang_code, slow=False)
        audio_temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(audio_temp_file.name)
        
        # Read audio file and convert to base64 to send it to Android app
        with open(audio_temp_file.name, "rb") as audio_file:
            encoded_audio = base64.b64encode(audio_file.read()).decode('utf-8')
            
        # Clean up audio temp file
        os.remove(audio_temp_file.name)
            
        return {
            "glosses": gloss_str,
            "sentence": final_sentence,
            "audio_base64": encoded_audio
        }
        
    finally:
        # Clean up video temp file
        os.remove(temp_file.name)

if __name__ == "__main__":
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)
