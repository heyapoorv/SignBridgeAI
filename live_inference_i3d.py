import cv2
import numpy as np
import torch
import scipy.special
from collections import deque
import pyttsx3
import threading
import os
import sys
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import warnings
warnings.filterwarnings('ignore')

# Add bsl1k to path so we can import its modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'bsl1k'))
from models import InceptionI3d
from demo.demo import load_vocab, fetch_from_url
from utils.imutils import resize_generic, im_to_numpy
from utils.transforms import color_normalize
from utils.misc import to_torch

# --- 1. SETTINGS & LABELS ---
NUM_CLASSES = 2000 
NUM_IN_FRAMES = 64
STRIDE = 8
CONFIDENCE_THRESHOLD = 0.35 # Slightly lower for WLASL 2000 classes

print(f"Loading I3D Keras Model (WLASL {NUM_CLASSES} Classes)...")

def load_model_device_agnostic(checkpoint_path: Path, checkpoint_url: str, num_classes: int, num_in_frames: int):
    fetch_from_url(url=checkpoint_url, dest_path=checkpoint_path)
    model = InceptionI3d(
        num_classes=num_classes,
        spatiotemporal_squeeze=True,
        final_endpoint="Logits",
        name="inception_i3d",
        in_channels=3,
        dropout_keep_prob=0.5,
        num_in_frames=num_in_frames,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["state_dict"]
    
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace("module.", "") if k.startswith("module.") else k
        new_state_dict[name] = v
        
    model.load_state_dict(new_state_dict)
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = model.to(device)
    model.eval()
    return model, device

checkpoint_url = "https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/wlasl_i3d_pbsl1k/model.pth.tar"
checkpoint_path = Path("bsl1k_weights/model.pth.tar")
word_data_url = "https://www.robots.ox.ac.uk/~vgg/research/bsl1k/demo-data/wlasl/info/info.pkl"
word_data_path = Path("bsl1k_weights/info.pkl")

model, device = load_model_device_agnostic(checkpoint_path, checkpoint_url, NUM_CLASSES, NUM_IN_FRAMES)
word_data = load_vocab(word_data_pkl_path=word_data_path, word_data_pkl_url=word_data_url)
top_words = word_data["words"]

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

# --- 3. MEDIAPIPE TASKS SETUP FOR UI GESTURES ---
hand_detector = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path='hand_landmarker.task'), num_hands=2))

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
    fingers_open = True
    for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)]:
        if lm[tip].y > lm[pip].y:
            fingers_open = False
            break
    return fingers_open

# --- 4. PREPARE I3D INPUT LOGIC ---
import torch.nn.functional as F

def process_frames_for_i3d(frames_list):
    """ Super fast GPU-accelerated frame processing. """
    # Stack list of (C, H, W) to (N, C, H, W) and move to device
    stacked = torch.stack(frames_list).to(device)
    
    # Resize to 256x256 using fast hardware interpolation
    resized = F.interpolate(stacked, size=(256, 256), mode='bilinear', align_corners=False)
    
    # Center crop to 224x224
    ulx = (256 - 224) // 2
    uly = (256 - 224) // 2
    cropped = resized[:, :, uly:uly+224, ulx:ulx+224]
    
    # Color normalize
    mean = torch.tensor([0.5, 0.5, 0.5], device=device).view(1, 3, 1, 1)
    std = torch.tensor([1.0, 1.0, 1.0], device=device).view(1, 3, 1, 1)
    normalized = (cropped - mean) / std
    
    # I3D expects (Batch, Channels, Frames, Height, Width) -> (1, 3, N, 224, 224)
    final_input = normalized.permute(1, 0, 2, 3).unsqueeze(0)
    
    return final_input


# --- 5. MAIN WEBCAM LOOP ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
print("Starting Webcam Interface with I3D Model...")

frame_sequence = deque(maxlen=NUM_IN_FRAMES)
word_buffer = []
last_predictions = deque(maxlen=3)
current_live_word = "..."
frame_counter = 0

# Variables for background inference
inference_thread = None
inference_result = None

def run_i3d_inference(frames_to_process):
    global inference_result
    try:
        inp = process_frames_for_i3d(frames_to_process)
        with torch.no_grad():
            out = model(inp)
            raw_scores = out["logits"].cpu().numpy()[0]
            prob_scores = scipy.special.softmax(raw_scores)
            
            predicted_idx = np.argmax(prob_scores)
            confidence = prob_scores[predicted_idx]
            inference_result = (predicted_idx, confidence)
    except Exception as e:
        print("Inference Error:", e)
        inference_result = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
        
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # --- MEDIA-PIPE UI LOGIC ---
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    h_res = hand_detector.detect(mp_img)
    
    trigger_speak = False
    trigger_clear = False
    if h_res.hand_landmarks:
        if is_thumbs_up(h_res.hand_landmarks):
            trigger_speak = True
        elif is_open_palm(h_res.hand_landmarks):
            trigger_clear = True
            
    if trigger_clear:
        word_buffer = []
        frame_sequence.clear()
        last_predictions.clear()
            
    if trigger_speak and len(word_buffer) > 0 and not is_speaking:
        threading.Thread(target=generate_and_speak, args=(word_buffer.copy(),)).start()
        word_buffer = []
        frame_sequence.clear()
        last_predictions.clear()
    
    # --- I3D FRAME EXTRACTION LOGIC ---
    # Store frame as float tensor [0,1] with shape (C, H, W)
    rgb_t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    frame_sequence.append(rgb_t)
    
    frame_counter += 1
    
    # If buffer is full, and we are not currently running inference, spawn a thread!
    if len(frame_sequence) == NUM_IN_FRAMES and frame_counter % STRIDE == 0: 
        if inference_thread is None or not inference_thread.is_alive():
            # Snapshot the sequence to process in background
            frames_copy = list(frame_sequence)
            inference_thread = threading.Thread(target=run_i3d_inference, args=(frames_copy,))
            inference_thread.start()
            
    # Process the result from the inference thread if it's ready
    if inference_result is not None:
        predicted_idx, confidence = inference_result
        inference_result = None # Reset
        
        if confidence > CONFIDENCE_THRESHOLD: 
            word = top_words[predicted_idx]
            current_live_word = f"{word} ({confidence:.2f})"
            last_predictions.append(word)
            
            # Lowered the requirement to 1 since 3D CNN inference runs less frequently than Keras
            if last_predictions.count(word) >= 1:
                if len(word_buffer) == 0 or word_buffer[-1] != word:
                    word_buffer.append(word)
                last_predictions.clear() 
                # We DO NOT clear frame_sequence. 
                # This ensures the camera buffer is always full, allowing fluid transition to the next word!
        else:
            current_live_word = "..."
            
    # --- DRAW UI ---
    image = frame.copy()
    h, w, c = image.shape
    
    cv2.rectangle(image, (int(w*0.2), int(h*0.2)), (int(w*0.8), int(h*0.8)), (0, 255, 0), 2)
    cv2.putText(image, "Signing Window (Keep hands in view)", (int(w*0.2), int(h*0.2)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Top Banner
    cv2.rectangle(image, (0,0), (w, 70), (245, 117, 16), -1)
    cv2.putText(image, f"Live: {current_live_word}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(image, f"Buffer: {' '.join(word_buffer)}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Bottom Banner
    cv2.rectangle(image, (0, h-60), (w, h), (16, 117, 245), -1)
    cv2.putText(image, f"AI: {generated_sentence}", (10, h-25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
    cv2.imshow('Ultimate Fast ASL Translator (WLASL I3D)', image)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()
