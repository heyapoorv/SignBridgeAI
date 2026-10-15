import customtkinter as ctk
import tkinter as tk
from PIL import Image, ImageTk
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from collections import deque
import pandas as pd
import os
import pyttsx3
import threading
import torch
import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import warnings
warnings.filterwarnings('ignore')

# --- GLOBALS & SETTINGS ---
NUM_CLASSES = 300 
MODEL_PATH = f"best_hybrid_{NUM_CLASSES}.keras"
CONFIDENCE_THRESHOLD = 0.70  # Model must be 70% sure (prevents jitter)
STABILITY_FRAMES = 3         # Must predict the same word 3 times to accept it

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class ASLTranslatorApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Ultimate Fast ASL Translator - AI Powered")
        self.geometry("1100x700")
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        # --- AI Variables ---
        self.sequence = deque(maxlen=50)
        self.word_buffer = []
        self.last_predictions = deque(maxlen=5) 
        self.current_live_word = "Waiting for signs..."
        self.generated_sentence = "Thumbs Up to Speak | Open Palm to Clear"
        self.is_speaking = False
        self.frame_counter = 0

        # --- UI THEME COLORS ---
        self.bg_color = "#0B0C10"          # Deep rich black/blue
        self.panel_color = "#1F2833"       # Sleek dark gray/blue
        self.accent_primary = "#66FCF1"    # Neon Cyan
        self.accent_secondary = "#45A29E"  # Muted Cyan
        self.text_main = "#C5C6C7"         # Crisp gray/white
        self.text_highlight = "#FFFFFF"
        self.ai_color = "#F4A261"          # Warm orange for AI output
        
        self.configure(fg_color=self.bg_color)

        # --- UI LAYOUT ---
        # Main Container
        self.main_container = ctk.CTkFrame(self, fg_color=self.bg_color)
        self.main_container.pack(fill="both", expand=True, padx=15, pady=15)

        # Top Header
        self.header = ctk.CTkFrame(self.main_container, fg_color=self.panel_color, corner_radius=10, height=60)
        self.header.pack(fill="x", pady=(0, 15))
        self.header.pack_propagate(False)
        
        title = ctk.CTkLabel(self.header, text="✨ ASL-Citizen : Real-Time AI Translator ✨", 
                             font=ctk.CTkFont(family="Segoe UI", size=26, weight="bold"), 
                             text_color=self.accent_primary)
        title.pack(side="left", padx=20, pady=10)
        
        self.status_lbl = ctk.CTkLabel(self.header, text="⏳ Booting AI Engines...", 
                                       font=ctk.CTkFont(family="Segoe UI", size=16, weight="bold"), 
                                       text_color=self.ai_color)
        self.status_lbl.pack(side="right", padx=20, pady=10)

        # Body Container
        self.body = ctk.CTkFrame(self.main_container, fg_color="transparent")
        self.body.pack(fill="both", expand=True)

        # Left Frame: Camera
        self.left_frame = ctk.CTkFrame(self.body, fg_color=self.panel_color, corner_radius=15)
        self.left_frame.pack(side="left", fill="both", expand=True, padx=(0, 15))
        
        self.cam_label = ctk.CTkLabel(self.left_frame, text="Warming up Camera...", font=ctk.CTkFont(size=20), text_color=self.text_main)
        self.cam_label.pack(fill="both", expand=True, padx=10, pady=10)

        # Right Frame: Dashboard
        self.right_frame = ctk.CTkFrame(self.body, width=420, fg_color=self.bg_color)
        self.right_frame.pack(side="right", fill="y")
        self.right_frame.pack_propagate(False)

        # Widget 1: Live Prediction
        self.live_frame = ctk.CTkFrame(self.right_frame, fg_color=self.panel_color, corner_radius=15)
        self.live_frame.pack(fill="x", pady=(0, 15))
        ctk.CTkLabel(self.live_frame, text="🟢 LIVE PREDICTION", text_color=self.text_main, font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="w", padx=20, pady=(15, 0))
        self.live_lbl = ctk.CTkLabel(self.live_frame, text="...", font=ctk.CTkFont(family="Consolas", size=32, weight="bold"), text_color=self.accent_primary)
        self.live_lbl.pack(anchor="w", padx=20, pady=(5, 20))

        # Widget 2: Word Buffer
        self.buf_frame = ctk.CTkFrame(self.right_frame, fg_color=self.panel_color, corner_radius=15)
        self.buf_frame.pack(fill="x", pady=(0, 15))
        ctk.CTkLabel(self.buf_frame, text="📝 BUFFERED SIGNS", text_color=self.text_main, font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="w", padx=20, pady=(15, 0))
        self.buf_lbl = ctk.CTkTextbox(self.buf_frame, height=90, font=ctk.CTkFont(size=18, weight="bold"), state="disabled", wrap="word", fg_color="transparent", text_color=self.text_highlight)
        self.buf_lbl.pack(fill="x", padx=15, pady=(5, 15))

        # Widget 3: AI Translation Sentence
        self.ai_frame = ctk.CTkFrame(self.right_frame, fg_color="#18202A", corner_radius=15, border_width=2, border_color=self.ai_color)
        self.ai_frame.pack(fill="x", pady=(0, 15))
        ctk.CTkLabel(self.ai_frame, text="🤖 LLM TRANSLATION", text_color=self.ai_color, font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="w", padx=20, pady=(15, 0))
        self.ai_lbl = ctk.CTkTextbox(self.ai_frame, height=130, font=ctk.CTkFont(size=22, weight="bold", slant="italic"), state="disabled", wrap="word", fg_color="transparent", text_color=self.text_highlight)
        self.ai_lbl.pack(fill="x", padx=15, pady=(5, 15))

        # Widget 4: Instructions / Controls
        self.inst_frame = ctk.CTkFrame(self.right_frame, fg_color=self.panel_color, corner_radius=15)
        self.inst_frame.pack(fill="x", side="bottom")
        ctk.CTkLabel(self.inst_frame, text="GESTURE CONTROLS", text_color=self.text_main, font=ctk.CTkFont(size=12, weight="bold")).pack(anchor="w", padx=20, pady=(10, 0))
        ctk.CTkLabel(self.inst_frame, text="👍 Thumbs Up = Speak Sentence", font=ctk.CTkFont(size=16), text_color=self.text_highlight).pack(anchor="w", padx=20, pady=(5, 0))
        ctk.CTkLabel(self.inst_frame, text="✋ Open Palm = Clear Buffer", font=ctk.CTkFont(size=16), text_color=self.text_highlight).pack(anchor="w", padx=20, pady=(0, 15))

        # Start Background Loaders
        threading.Thread(target=self.init_ai_models, daemon=True).start()

    def init_ai_models(self):
        # 1. Load Labels
        try:
            train_df = pd.read_csv('ASL_Citizen/splits/train.csv')
            raw_top_words = sorted(train_df['Gloss'].value_counts().head(NUM_CLASSES).index.tolist())
            def clean_word(word):
                w = re.sub(r'\d+', '', word)
                return w.replace('-', ' ').replace('_', ' ').strip().lower()
            self.top_words = [clean_word(w) for w in raw_top_words]
        except Exception:
            print("Could not load train.csv. Ensure it is in ASL_Citizen/splits/")
            os._exit(1)

        # 2. Load ASL Model
        print(f"Loading Keras Model ({NUM_CLASSES} Classes)...")
        self.model = tf.keras.models.load_model(MODEL_PATH)

        # 3. Load Offline LLM & TTS
        print("Loading Offline LLM Engine...")
        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
        self.llm_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)

        # 4. MediaPipe
        self.hand_detector = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path='hand_landmarker.task'), num_hands=2))
        self.pose_detector = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path='pose_landmarker_lite.task')))

        # Update UI & Start Camera
        self.update_ai_label(self.generated_sentence)
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
        self.update_frame()

    def is_thumbs_up(self, hand_landmarks_list):
        if not hand_landmarks_list: return False
        lm = hand_landmarks_list[0] 
        thumb_tip, index_tip, pinky_tip = lm[4].y, lm[8].y, lm[20].y
        return (thumb_tip < index_tip - 0.15) and (thumb_tip < pinky_tip - 0.15)

    def is_open_palm(self, hand_landmarks_list):
        if not hand_landmarks_list: return False
        lm = hand_landmarks_list[0]
        for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)]:
            if lm[tip].y > lm[pip].y: return False
        return True

    def extract_keypoints(self, mp_img):
        hf = np.zeros(126)
        h_res = self.hand_detector.detect(mp_img)
        if h_res.hand_landmarks:
            for i, h_lm in enumerate(h_res.hand_landmarks[:2]):
                for j, lm in enumerate(h_lm): hf[(i*63)+(j*3):(i*63)+(j*3)+3] = [lm.x, lm.y, lm.z]
                
        pf = np.zeros(99)
        p_res = self.pose_detector.detect(mp_img)
        if p_res.pose_landmarks:
            for j, lm in enumerate(p_res.pose_landmarks[0]): pf[j*3:(j*3)+3] = [lm.x, lm.y, lm.z]
            
        return np.concatenate([hf, pf]), h_res.hand_landmarks

    def generate_and_speak(self, word_list):
        self.is_speaking = True
        self.update_ai_label("Generating sentence...")
        raw_text = " ".join(word_list)
        # Improved prompt for better grammatical understanding
        prompt = f"Rephrase these disjointed words into a single grammatically correct, natural English sentence: {raw_text}"
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        try:
            with torch.no_grad(): 
                outputs = self.llm_model.generate(**inputs, max_length=50, no_repeat_ngram_size=2, early_stopping=True, num_beams=4)
            self.generated_sentence = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Capitalize and add period if missing
            if self.generated_sentence:
                self.generated_sentence = self.generated_sentence[0].upper() + self.generated_sentence[1:]
                if not self.generated_sentence.endswith('.'): self.generated_sentence += '.'
        except Exception:
            self.generated_sentence = "AI Generation Error."
            
        self.update_ai_label(self.generated_sentence)
        self.engine.say(self.generated_sentence)
        self.engine.runAndWait()
        self.is_speaking = False

    def update_buffer_label(self):
        self.buf_lbl.configure(state="normal")
        self.buf_lbl.delete("0.0", "end")
        self.buf_lbl.insert("0.0", " ".join(self.word_buffer))
        self.buf_lbl.configure(state="disabled")

    def update_ai_label(self, text):
        self.ai_lbl.configure(state="normal")
        self.ai_lbl.delete("0.0", "end")
        self.ai_lbl.insert("0.0", text)
        self.ai_lbl.configure(state="disabled")

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            keypoints, h_landmarks = self.extract_keypoints(mp_img)
            self.frame_counter += 1

            # 1. CHECK UI GESTURES
            trigger_speak = False
            trigger_clear = False
            if h_landmarks:
                if self.is_thumbs_up(h_landmarks): trigger_speak = True
                elif self.is_open_palm(h_landmarks): trigger_clear = True
                    
            if trigger_clear:
                self.word_buffer = []
                self.sequence.clear()
                self.last_predictions.clear()
                self.update_buffer_label()
                self.update_ai_label("Buffer Cleared.")
                    
            # Speaking generates the sentence and clears the buffer
            if trigger_speak and len(self.word_buffer) > 0 and not self.is_speaking:
                threading.Thread(target=self.generate_and_speak, args=(self.word_buffer.copy(),)).start()
                self.word_buffer = []
                self.sequence.clear()
                self.last_predictions.clear()
                self.update_buffer_label()
            
            # 2. EXTRACT ASL FEATURES
            self.sequence.append(keypoints)
            if len(self.sequence) == 50 and self.frame_counter % 3 == 0: 
                seq_array = np.array(self.sequence)
                input_data = seq_array.reshape(1, 50, 225)
                res = self.model.predict(input_data, verbose=0)[0]
                predicted_idx = np.argmax(res)
                confidence = res[predicted_idx]
                
                if confidence > CONFIDENCE_THRESHOLD: 
                    word = self.top_words[predicted_idx]
                    self.live_lbl.configure(text=f"{word} ({confidence*100:.0f}%)", text_color="#66FCF1")
                    self.last_predictions.append(word)
                    
                    if self.last_predictions.count(word) >= STABILITY_FRAMES:
                        if len(self.word_buffer) == 0 or self.word_buffer[-1] != word:
                            self.word_buffer.append(word)
                            self.update_buffer_label()
                        self.last_predictions.clear() 
                        self.sequence.clear() 
                else:
                    self.live_lbl.configure(text="...", text_color=self.text_main)

            # 3. DRAW SKELETON ON FRAME (Optional, can just draw a box)
            h, w, c = frame.shape
            cv2.rectangle(frame, (int(w*0.1), int(h*0.1)), (int(w*0.9), int(h*0.9)), (0, 255, 0), 2)
            cv2.putText(frame, "Signing Window", (int(w*0.1), int(h*0.1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Display Image in CustomTkinter
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(800, 600))
            self.cam_label.configure(image=ctk_img, text="")

        # Loop
        self.after(15, self.update_frame)

    def on_closing(self):
        if hasattr(self, 'cap'):
            self.cap.release()
        self.destroy()

if __name__ == "__main__":
    app = ASLTranslatorApp()
    app.mainloop()
