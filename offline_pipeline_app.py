import os
import sys
import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from pathlib import Path
import threading
import torch
import numpy as np
import scipy.special
import cv2
import requests
import json
from gtts import gTTS
import tempfile
import pygame

# Add bsl1k to path so we can import its modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'bsl1k'))

from models import InceptionI3d
from demo.demo import load_vocab, prepare_input, sliding_windows, fetch_from_url

def load_rgb_video_cv2(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    rgb = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_t = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        rgb.append(rgb_t)
    cap.release()
    if len(rgb) == 0:
        raise ValueError(f"Could not read any frames from {video_path}")
    rgb = torch.stack(rgb).permute(1, 0, 2, 3)
    return rgb

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

# Set customtkinter appearance
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class WebcamRecorder(ctk.CTkToplevel):
    def __init__(self, parent, output_path, on_done_callback):
        super().__init__(parent)
        self.title("Webcam Recording Studio")
        self.geometry("750x650")
        self.grab_set()  # Make window modal
        
        self.output_path = output_path
        self.on_done_callback = on_done_callback
        
        self.cap = cv2.VideoCapture(0)
        self.is_recording = False
        self.recorded_frames = []
        
        self.fps = 30
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if self.cap.isOpened() else 640
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if self.cap.isOpened() else 480
        
        # UI Setup
        self.status_label = ctk.CTkLabel(self, text="Status: Ready", text_color="#66FCF1", font=ctk.CTkFont(size=16, weight="bold"))
        self.status_label.pack(pady=10)
        
        self.vid_label = ctk.CTkLabel(self, text="")
        self.vid_label.pack(pady=10, fill="both", expand=True)
        
        self.btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.btn_frame.pack(pady=20)
        
        self.start_btn = ctk.CTkButton(self.btn_frame, text="▶ Start Recording", fg_color="#F4A261", hover_color="#E76F51", command=self.start_record)
        self.start_btn.grid(row=0, column=0, padx=10)
        
        self.stop_btn = ctk.CTkButton(self.btn_frame, text="⏹ Stop", fg_color="#E63946", hover_color="#D62828", state="disabled", command=self.stop_record)
        self.stop_btn.grid(row=0, column=1, padx=10)
        
        self.rerecord_btn = ctk.CTkButton(self.btn_frame, text="🔄 Rerecord", state="disabled", command=self.rerecord)
        self.rerecord_btn.grid(row=0, column=2, padx=10)
        
        self.done_btn = ctk.CTkButton(self.btn_frame, text="✅ Done", fg_color="#2A9D8F", hover_color="#21867A", state="disabled", command=self.finish)
        self.done_btn.grid(row=0, column=3, padx=10)
        
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.update_feed()

    def update_feed(self):
        if not self.cap.isOpened():
            return
            
        ret, frame = self.cap.read()
        if ret:
            if self.is_recording:
                self.recorded_frames.append(frame)
                # Draw a recording indicator
                cv2.circle(frame, (30, 30), 10, (0, 0, 255), -1)
                
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_frame)
            ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(640, 480))
            self.vid_label.configure(image=ctk_img)
            
        self.after(30, self.update_feed)

    def start_record(self):
        self.is_recording = True
        self.recorded_frames = []
        self.status_label.configure(text="Status: 🔴 Recording...", text_color="#E63946")
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.rerecord_btn.configure(state="disabled")
        self.done_btn.configure(state="disabled")

    def stop_record(self):
        self.is_recording = False
        self.status_label.configure(text="Status: ⏸ Recording Stopped", text_color="#F4A261")
        self.stop_btn.configure(state="disabled")
        self.start_btn.configure(state="disabled") # Can't start again without rerecording
        self.rerecord_btn.configure(state="normal")
        self.done_btn.configure(state="normal")

    def rerecord(self):
        self.recorded_frames = []
        self.status_label.configure(text="Status: Ready", text_color="#66FCF1")
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.rerecord_btn.configure(state="disabled")
        self.done_btn.configure(state="disabled")

    def finish(self):
        if self.recorded_frames:
            self.status_label.configure(text="Status: Saving video...", text_color="#2A9D8F")
            self.update()
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(self.output_path, fourcc, self.fps, (self.frame_width, self.frame_height))
            for f in self.recorded_frames:
                out.write(f)
            out.release()
            
            self.cap.release()
            self.on_done_callback(True)
            self.destroy()
        else:
            self.on_close()

    def on_close(self):
        self.cap.release()
        self.on_done_callback(False)
        self.destroy()


class OfflinePipelineApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("Offline Sign Language Pipeline - Pro Edition")
        self.geometry("800x700")
        
        # --- UI THEME COLORS ---
        self.bg_color = "#0B0C10"          
        self.panel_color = "#1F2833"       
        self.accent_primary = "#66FCF1"    
        self.text_main = "#C5C6C7"         
        self.text_highlight = "#FFFFFF"
        self.ai_color = "#F4A261"          
        
        self.configure(fg_color=self.bg_color)
        
        self.video_path = None
        self.slr_model = None
        self.device = None
        self.word_data = None
        
        # Gemini API doesn't need to be initialized here, we'll just use requests later
        
        pygame.mixer.init()
        
        self.setup_ui()
        
        # Load model in background
        self.log("Initializing application... loading I3D SLR model in background.")
        threading.Thread(target=self.load_slr_model, daemon=True).start()

    def setup_ui(self):
        # Header
        self.header = ctk.CTkFrame(self, fg_color=self.panel_color, corner_radius=10, height=60)
        self.header.pack(fill="x", padx=15, pady=15)
        self.header.pack_propagate(False)
        
        title = ctk.CTkLabel(self.header, text="✨ ASL Offline Translator ✨", 
                             font=ctk.CTkFont(family="Segoe UI", size=24, weight="bold"), 
                             text_color=self.accent_primary)
        title.pack(side="left", padx=20, pady=10)
        
        self.status_lbl = ctk.CTkLabel(self.header, text="⏳ Booting...", 
                                       font=ctk.CTkFont(family="Segoe UI", size=14, weight="bold"), 
                                       text_color=self.ai_color)
        self.status_lbl.pack(side="right", padx=20, pady=10)
        
        # Main Body
        self.body = ctk.CTkFrame(self, fg_color="transparent")
        self.body.pack(fill="both", expand=True, padx=15, pady=(0, 15))
        
        # Step 1: Input controls
        self.input_frame = ctk.CTkFrame(self.body, fg_color=self.panel_color, corner_radius=15)
        self.input_frame.pack(fill="x", pady=(0, 15))
        
        ctk.CTkLabel(self.input_frame, text="STEP 1: PROVIDE VIDEO", text_color=self.text_main, font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="w", padx=20, pady=(15, 5))
        
        btn_frame = ctk.CTkFrame(self.input_frame, fg_color="transparent")
        btn_frame.pack(pady=10)
        
        self.upload_btn = ctk.CTkButton(btn_frame, text="📁 Upload Video (.mp4)", command=self.upload_video, font=ctk.CTkFont(weight="bold"), fg_color="#45A29E", hover_color="#66FCF1")
        self.upload_btn.grid(row=0, column=0, padx=10)
        
        self.record_btn = ctk.CTkButton(btn_frame, text="🎥 Record Webcam", command=self.record_video, font=ctk.CTkFont(weight="bold"), fg_color="#F4A261", hover_color="#E76F51")
        self.record_btn.grid(row=0, column=1, padx=10)
        
        self.video_label = ctk.CTkLabel(self.input_frame, text="No video selected", text_color="gray")
        self.video_label.pack(pady=(0, 15))

        # Step 2: Pipeline Runner & Language Selection
        self.run_frame = ctk.CTkFrame(self.body, fg_color=self.panel_color, corner_radius=15)
        self.run_frame.pack(fill="x", pady=(0, 15))
        
        lang_frame = ctk.CTkFrame(self.run_frame, fg_color="transparent")
        lang_frame.pack(pady=(15, 5))
        
        ctk.CTkLabel(lang_frame, text="Target Language:", text_color=self.text_main, font=ctk.CTkFont(size=14, weight="bold")).pack(side="left", padx=10)
        self.language_var = ctk.StringVar(value="English")
        self.language_menu = ctk.CTkOptionMenu(lang_frame, values=["English", "Hindi", "Marathi"], variable=self.language_var, fg_color="#45A29E", button_color="#2A9D8F", button_hover_color="#21867A")
        self.language_menu.pack(side="left", padx=10)
        
        self.process_btn = ctk.CTkButton(self.run_frame, text="⚡ RUN AI PIPELINE", command=self.run_pipeline, font=ctk.CTkFont(size=16, weight="bold"), height=40, state="disabled")
        self.process_btn.pack(pady=15, padx=20, fill="x")

        # Step 3: Outputs
        self.out_frame = ctk.CTkFrame(self.body, fg_color="#18202A", corner_radius=15, border_width=2, border_color=self.accent_primary)
        self.out_frame.pack(fill="both", expand=True)
        
        ctk.CTkLabel(self.out_frame, text="1. PREDICTED GLOSSES (SLR MODEL)", text_color=self.text_main, font=ctk.CTkFont(size=12, weight="bold")).pack(anchor="w", padx=20, pady=(15, 0))
        self.gloss_text = ctk.CTkTextbox(self.out_frame, height=50, font=ctk.CTkFont(size=16, weight="bold"), fg_color="transparent", text_color=self.accent_primary, state="disabled")
        self.gloss_text.pack(fill="x", padx=15, pady=(5, 15))
        
        ctk.CTkLabel(self.out_frame, text="2. RECONSTRUCTED SENTENCE (LLM)", text_color=self.ai_color, font=ctk.CTkFont(size=12, weight="bold")).pack(anchor="w", padx=20, pady=(5, 0))
        self.sentence_text = ctk.CTkTextbox(self.out_frame, height=80, font=ctk.CTkFont(size=22, weight="bold", slant="italic"), fg_color="transparent", text_color=self.text_highlight, state="disabled")
        self.sentence_text.pack(fill="x", padx=15, pady=(5, 15))
        
        self.play_audio_btn = ctk.CTkButton(self.out_frame, text="🔊 Play Audio", command=self.play_audio, font=ctk.CTkFont(weight="bold"), fg_color="#F4A261", hover_color="#E76F51", state="disabled")
        self.play_audio_btn.pack(pady=(0, 15))

    def log(self, message):
        self.status_lbl.configure(text=message)
        self.update()

    def update_textbox(self, widget, text):
        widget.configure(state="normal")
        widget.delete("0.0", "end")
        widget.insert("0.0", text)
        widget.configure(state="disabled")
    def load_slr_model(self):
        try:
            self.log("Loading I3D SLR model... this may take a moment.")
            checkpoint_url = "https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/wlasl_i3d_pbsl1k/model.pth.tar"
            checkpoint_path = Path("bsl1k_weights/model.pth.tar")
            word_data_url = "https://www.robots.ox.ac.uk/~vgg/research/bsl1k/demo-data/wlasl/info/info.pkl"
            word_data_path = Path("bsl1k_weights/info.pkl")
            
            self.slr_model, self.device = load_model_device_agnostic(
                checkpoint_path=checkpoint_path,
                checkpoint_url=checkpoint_url,
                num_classes=2000,
                num_in_frames=64,
            )
            self.word_data = load_vocab(word_data_pkl_path=word_data_path, word_data_pkl_url=word_data_url)
            self.log("Ready for inference ✓")
        except Exception as e:
            self.log(f"Error loading model")
            print(e)

    def upload_video(self):
        file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mkv")])
        if file_path:
            self.video_path = file_path
            self.video_label.configure(text=f"Selected: {os.path.basename(file_path)}", text_color=self.accent_primary)
            self.process_btn.configure(state="normal")

    def record_video(self):
        output_path = os.path.join(os.getcwd(), "recorded_video.mp4")
        
        def on_record_done(success):
            if success:
                self.video_path = output_path
                self.video_label.configure(text=f"Selected: recorded_video.mp4", text_color=self.accent_primary)
                self.process_btn.configure(state="normal")
                self.log("Recording saved ✓")
            else:
                self.log("Recording cancelled.")
                
        recorder = WebcamRecorder(self, output_path, on_record_done)

    def run_pipeline(self):
        if not self.slr_model:
            messagebox.showwarning("Wait", "Model is still loading. Please wait.")
            return
            
        self.update_textbox(self.gloss_text, "")
        self.update_textbox(self.sentence_text, "")
        self.play_audio_btn.configure(state="disabled")
        
        threading.Thread(target=self._pipeline_thread, daemon=True).start()
        
    def _pipeline_thread(self):
        try:
            # 1. Preprocessing
            self.log("Step 1: Preprocessing video frames...")
            rgb_orig = load_rgb_video_cv2(Path(self.video_path))
            rgb_input = prepare_input(rgb_orig)
            
            stride = 8 
            rgb_slides, _ = sliding_windows(rgb=rgb_input, stride=stride, num_in_frames=64)
            
            # 2. SLR Prediction (Sequence of Glosses)
            self.log(f"Step 2: Predicting signs across {rgb_slides.shape[0]} windows...")
            
            raw_predictions = []
            with torch.no_grad():
                for i in range(rgb_slides.shape[0]):
                    inp = rgb_slides[i:i+1].to(self.device)
                    out = self.slr_model(inp)
                    logits = out["logits"].cpu().numpy()[0]
                    probs = scipy.special.softmax(logits)
                    top_idx = np.argmax(probs)
                    word = self.word_data["words"][top_idx]
                    conf = probs[top_idx]
                    
                    # Debug print to console so we can see what's happening
                    print(f"Window {i} | Predicted: {word:<15} | Confidence: {conf*100:.1f}%")
                    
                    if conf > 0.45: 
                        raw_predictions.append(word)

            # ELIMINATE NOISE (Must appear in at least 3 consecutive windows) + ELIMINATE DUPLICATES
            stable_predictions = []
            count = 1
            for i in range(1, len(raw_predictions)):
                if raw_predictions[i] == raw_predictions[i-1]:
                    count += 1
                else:
                    if count >= 3: # Word must be stable for at least 3 overlapping windows
                        stable_predictions.append(raw_predictions[i-1])
                    count = 1
            
            # Check the last sequence
            if len(raw_predictions) > 0 and count >= 3:
                stable_predictions.append(raw_predictions[-1])

            predicted_glosses = []
            for word in stable_predictions:
                if len(predicted_glosses) == 0 or predicted_glosses[-1] != word:
                    predicted_glosses.append(word)
            
            if not predicted_glosses:
                predicted_glosses = ["unknown"]
                
            gloss_str = " ➔ ".join(predicted_glosses)
            self.update_textbox(self.gloss_text, gloss_str)
            
            # 3. Language Reconstruction
            self.log("Step 3: Reconstructing language using Gemini API...")
            try:
                api_key = os.getenv("GEMINI_API_KEY")
                url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
                
                target_lang = self.language_var.get()
                
                prompt = (
                    f"You are an expert Sign Language Translator. "
                    f"I will give you a sequence of raw, disjointed sign language words that were predicted by a vision model. "
                    f"These words might contain duplicates, have poor grammar, or be slightly out of order. "
                    f"Your job is to infer the intended meaning and output ONLY a single, natural, grammatically correct sentence in {target_lang}. "
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
                self.log(f"Gemini API Error: {str(e)}")
                final_sentence = " ".join(predicted_glosses).capitalize() + "."
                
            self.update_textbox(self.sentence_text, final_sentence)
            
            # 4. Ready for TTS
            self.log("Pipeline complete! ✓")
            self.play_audio_btn.configure(state="normal")
            
        except Exception as e:
            self.log(f"Error: Pipeline failed.")
            print(e)
            
    def play_audio(self):
        sentence = self.sentence_text.get("0.0", "end").strip()
        if sentence:
            try:
                lang = self.language_var.get()
                lang_code = 'en'
                if lang == "Hindi":
                    lang_code = 'hi'
                elif lang == "Marathi":
                    lang_code = 'mr'
                    
                tts = gTTS(text=sentence, lang=lang_code, slow=False)
                
                # Create a temporary file
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
                temp_file_path = temp_file.name
                temp_file.close()
                
                tts.save(temp_file_path)
                
                # Play the audio using pygame
                pygame.mixer.music.load(temp_file_path)
                pygame.mixer.music.play()
                
            except Exception as e:
                self.log(f"TTS Error: {str(e)}")
                print(e)

if __name__ == "__main__":
    app = OfflinePipelineApp()
    app.mainloop()
