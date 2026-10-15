import tensorflow as tf
import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import pandas as pd
import os
import pyttsx3
import threading
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import warnings
warnings.filterwarnings('ignore')

# --- 1. SETTINGS & LABELS ---
NUM_CLASSES = 300 
MODEL_PATH = f"best_hybrid_{NUM_CLASSES}.keras"

import re
try:
    train_df = pd.read_csv('ASL_Citizen/splits/train.csv')
    raw_top_words = sorted(train_df['Gloss'].value_counts().head(NUM_CLASSES).index.tolist())
    
    def clean_word(word):
        # Remove any numbers
        w = re.sub(r'\d+', '', word)
        # Replace hyphens with spaces
        w = w.replace('-', ' ').replace('_', ' ')
        # Return clean lowercase word
        return w.strip().lower()
        
    top_words = [clean_word(w) for w in raw_top_words]
except Exception as e:
    print("Could not load train.csv. Ensure it is in ASL_Citizen/splits/")
    exit()

if not os.path.exists(MODEL_PATH):
    print(f"ERROR: Keras Model {MODEL_PATH} not found in root folder!")
    exit()

print(f"Loading Keras Model ({NUM_CLASSES} Classes)...")
model = tf.keras.models.load_model(MODEL_PATH)

# --- 2. OFFLINE AI (LLM & TTS) ---
print("Loading Offline LLM Engine...")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
llm_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

engine = pyttsx3.init()
engine.setProperty('rate', 150)

is_speaking = False
generated_sentence = "Thumbs Up to Speak | Open Palm to Clear"

def generate_and_speak(word_list):
    global is_speaking, generated_sentence
    is_speaking = True
    raw_text = " ".join(word_list)
    # Highly Optimized Offline Prompt
    prompt = f"Translate the following words into a natural, meaningful English sentence: {raw_text}"
    inputs = tokenizer(prompt, return_tensors="pt")
    
    try:
        with torch.no_grad(): 
            outputs = llm_model.generate(
                **inputs, 
                max_length=40,
                no_repeat_ngram_size=2,
                early_stopping=True,
                num_beams=3
            )
        generated_sentence = tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        print(f"\n[AI ERROR]: {e}\n")
        generated_sentence = "AI Generation Error."
        
    print(f"[AI Output] {generated_sentence}")
    engine.say(generated_sentence)
    engine.runAndWait()
    is_speaking = False

# --- 3. MEDIAPIPE TASKS SETUP ---
# Exact match to your local_feature_extractor.py to ensure 100% accurate coordinates!
hand_detector = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path='hand_landmarker.task'), num_hands=2))
pose_detector = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path='pose_landmarker_lite.task')))

sequence = deque(maxlen=50)
word_buffer = []

def extract_keypoints(mp_img):
    hf = np.zeros(126)
    h_res = hand_detector.detect(mp_img)
    if h_res.hand_landmarks:
        for i, h_lm in enumerate(h_res.hand_landmarks[:2]):
            for j, lm in enumerate(h_lm): hf[(i*63)+(j*3):(i*63)+(j*3)+3] = [lm.x, lm.y, lm.z]
            
    pf = np.zeros(99)
    p_res = pose_detector.detect(mp_img)
    if p_res.pose_landmarks:
        for j, lm in enumerate(p_res.pose_landmarks[0]): pf[j*3:(j*3)+3] = [lm.x, lm.y, lm.z]
        
    return np.concatenate([hf, pf]), h_res.hand_landmarks

# --- THE MAGIC: UI GESTURE RECOGNIZERS ---
def is_thumbs_up(hand_landmarks_list):
    if not hand_landmarks_list: return False
    lm = hand_landmarks_list[0] 
    thumb_tip = lm[4].y
    index_tip = lm[8].y
    pinky_tip = lm[20].y
    return (thumb_tip < index_tip - 0.15) and (thumb_tip < pinky_tip - 0.15)

def is_open_palm(hand_landmarks_list):
    if not hand_landmarks_list: return False
    lm = hand_landmarks_list[0]
    # Check if all fingers are extended (tips higher than lower joints)
    fingers_open = True
    for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)]:
        if lm[tip].y > lm[pip].y:
            fingers_open = False
            break
    return fingers_open

# --- 4. MAIN WEBCAM LOOP ---
cap = cv2.VideoCapture(0)
# Use 640x480 so the camera captures frames as fast as possible
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
print("Starting Webcam Interface...")

# Optimization: Variables for sliding window stabilization
frame_counter = 0
last_predictions = deque(maxlen=5) 
current_live_word = "..."

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
        
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    
    keypoints, h_landmarks = extract_keypoints(mp_img)
    
    image = frame.copy()
    frame_counter += 1
    
    # 1. CHECK UI GESTURES (Thumbs Up = Speak | Open Palm = Clear)
    trigger_speak = False
    trigger_clear = False
    if h_landmarks:
        if is_thumbs_up(h_landmarks):
            trigger_speak = True
        elif is_open_palm(h_landmarks):
            trigger_clear = True
            
    if trigger_clear:
        word_buffer = []
        sequence.clear()
        last_predictions.clear()
            
    if trigger_speak and len(word_buffer) > 0 and not is_speaking:
        threading.Thread(target=generate_and_speak, args=(word_buffer.copy(),)).start()
        word_buffer = [] # Clear buffer automatically after triggering sentence
        sequence.clear()
        last_predictions.clear()
    
    # 2. EXTRACT ASL FEATURES (Sliding Window for High Speed)
    sequence.append(keypoints)
    
    if len(sequence) == 50 and frame_counter % 3 == 0: 
        seq_array = np.array(sequence)
        
        # Predict directly on RAW coordinates
        input_data = seq_array.reshape(1, 50, 225)
        res = model.predict(input_data, verbose=0)[0]
        predicted_idx = np.argmax(res)
        confidence = res[predicted_idx]
        
        # Loosened Criteria: 50% confidence is enough
        if confidence > 0.50: 
            word = top_words[predicted_idx]
            current_live_word = f"{word} ({confidence:.2f})"
            last_predictions.append(word)
            
            # Loosened Criteria: Only need to see it 2 times instead of 4
            if last_predictions.count(word) >= 2:
                if len(word_buffer) == 0 or word_buffer[-1] != word:
                    word_buffer.append(word)
                last_predictions.clear() 
                sequence.clear() 
        else:
            current_live_word = "..."
    

            
    # --- 3. DRAW UI ---
    h, w, c = image.shape
    
    # Draw a professional "Signing Area" Box in the center
    cv2.rectangle(image, (int(w*0.2), int(h*0.2)), (int(w*0.8), int(h*0.8)), (0, 255, 0), 2)
    cv2.putText(image, "Signing Window", (int(w*0.2), int(h*0.2)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Top Banner: Live Prediction + Buffer
    cv2.rectangle(image, (0,0), (w, 70), (245, 117, 16), -1)
    cv2.putText(image, f"Live: {current_live_word}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(image, f"Buffer: {' '.join(word_buffer)}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Bottom Banner: AI Generated Sentence
    cv2.rectangle(image, (0, h-60), (w, h), (16, 117, 245), -1)
    cv2.putText(image, f"AI: {generated_sentence}", (10, h-25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
    cv2.imshow('Ultimate Fast ASL Translator', image)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()
