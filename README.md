# 🤟 SignBridgeAI — ASL Sign Language Translator

> A real-time, end-to-end **American Sign Language (ASL) recognition and translation** system.  
> Video input → I3D Deep Learning Model → Gemini LLM → Natural Language sentence + Audio output.

---

## 📌 Project Overview

SignBridgeAI converts ASL sign language videos into natural spoken sentences.  
The pipeline has **3 core stages**:

```
Video (webcam / upload)
        ↓
[Stage 1] I3D Model (BSL1K) → Predicted Sign Glosses
        ↓
[Stage 2] Gemini 2.5 Flash LLM → Natural Language Sentence
        ↓
[Stage 3] gTTS → Audio Output (English / Hindi / Marathi)
```

The system is exposed as both:
- A **Desktop GUI App** (offline, local use)
- A **REST API Server** (for Android app integration)

---

## 🗂️ File Guide — What Does What?

> ⚠️ **For teammates doing diagrams:** Focus ONLY on the `🟢 ACTIVE` files listed below.  
> Files marked `🔴 ARCHIVED` are old experiments — ignore them.

---

### 🟢 ACTIVE — Core Production Files

| File | Role | What to diagram |
|------|------|-----------------|
| **`api_server.py`** | 🌐 **FastAPI REST Backend** — Main server used by Android app | Class: `FastAPI app`, endpoints `/translate_video`. Sequence: Upload → Predict → Gemini → Audio |
| **`offline_pipeline_app.py`** | 🖥️ **Desktop GUI App** — Local desktop tool (CustomTkinter) | Class: `OfflinePipelineApp`, `WebcamRecorder`. Activity: Record/Upload → Pipeline → Play Audio |
| **`local_feature_extractor.py`** | 🔧 **Data Preprocessing Script** — Extracts MediaPipe hand+pose keypoints from ASL Citizen dataset and saves as `.npy` | Used during training only. Class: `process_single_video`, `init_worker` |
| **`live_inference.py`** | 🎯 **Live Webcam Inference** — Runs the lightweight Keras+MediaPipe model on live webcam feed | Class: uses Keras model + MediaPipe landmarks + offline Flan-T5 LLM |
| **`requirements.txt`** | 📦 Python dependency list | — |

---

### 🔴 ARCHIVED / EXPERIMENTAL — Do Not Diagram

| File | Why it's archived |
|------|-------------------|
| `live_inference_gui.py` | Old GUI version of live_inference — superseded by `offline_pipeline_app.py` |
| `live_inference_i3d.py` | Experimental: tested I3D model for live feed — too slow, dropped |
| `live_inference_phonssm.py` | Experimental: tested PhonSSM model — not integrated |
| `test_bsl1k_offline.py` | One-off test script for BSL1K model validation |
| `test_gemini.py` | One-off test script for Gemini API connection check |
| `find_good_classes.py` | Utility: finds best ASL classes by frequency in dataset |
| `join_videos.py` | Utility: joins test video clips together |
| `Antigravity/generate_notebook.py` | Auto-generates Kaggle notebook — dev utility |
| `Antigravity/test_landmark_extraction.py` | Early test for MediaPipe landmark extraction |
| `kaggle_project/` | Kaggle-specific notebook — not part of main app |

---

## 🏗️ System Architecture (High Level)

```
┌─────────────────────────────────────────────────────┐
│                  Android App (Client)                │
│          Sends .mp4 video via HTTP POST              │
└─────────────────────┬───────────────────────────────┘
                      │ REST API (port 8000)
┌─────────────────────▼───────────────────────────────┐
│               api_server.py  (FastAPI)               │
│  ┌──────────────────────────────────────────────┐   │
│  │  1. load_rgb_video_cv2()  ← Video Decode     │   │
│  │  2. I3D Model (bsl1k)     ← Sign Recognition │   │
│  │  3. Gemini 2.5 Flash API  ← NLP Translation  │   │
│  │  4. gTTS                  ← Audio Generation  │   │
│  └──────────────────────────────────────────────┘   │
│               offline_pipeline_app.py (Desktop GUI)  │
│  ┌──────────────────────────────────────────────┐   │
│  │  Same pipeline as above but with             │   │
│  │  CustomTkinter UI + Webcam recording         │   │
│  └──────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
         ↑
┌────────┴──────────────────────────────────────────┐
│            bsl1k/  (submodule - BSL1K repo)       │
│   Contains InceptionI3d model architecture code   │
└───────────────────────────────────────────────────┘
         ↑ Training Pipeline (separate, offline)
┌────────┴──────────────────────────────────────────┐
│        local_feature_extractor.py                 │
│   Reads ASL Citizen dataset (archive zip)         │
│   Extracts MediaPipe landmarks → .npy files       │
│   Training happens in Antigravity/ notebooks      │
└───────────────────────────────────────────────────┘
```

---

## 🧩 Key Classes & Components (for Class Diagram)

### `api_server.py`
- **FastAPI `app`** instance
- Global state: `slr_model`, `device`, `word_data`
- Events: `startup_event()` — loads I3D model on boot
- Endpoint: `POST /translate_video(file, target_language)` → `{glosses, sentence, audio_base64}`

### `offline_pipeline_app.py`
- **`OfflinePipelineApp(ctk.CTk)`** — main window
  - Attributes: `video_path`, `slr_model`, `device`, `word_data`, `language_var`
  - Methods: `load_slr_model()`, `upload_video()`, `record_video()`, `run_pipeline()`, `_pipeline_thread()`, `play_audio()`
- **`WebcamRecorder(ctk.CTkToplevel)`** — modal webcam window
  - Attributes: `cap`, `is_recording`, `recorded_frames`, `output_path`
  - Methods: `start_record()`, `stop_record()`, `rerecord()`, `finish()`

### Shared Utility Functions (defined in `offline_pipeline_app.py`, reused in `api_server.py`)
- `load_rgb_video_cv2(video_path)` → torch tensor of video frames
- `load_model_device_agnostic(checkpoint_path, checkpoint_url, num_classes, num_in_frames)` → model, device

---

## 🔑 Environment Variables

Create a `.env` file in the project root:
```
GEMINI_API_KEY=your_google_gemini_api_key_here
```

---

## 🚀 How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Desktop App (offline pipeline)
```bash
python offline_pipeline_app.py
```

### 3. API Server (for Android app)
```bash
python api_server.py
```
Server starts at `http://0.0.0.0:8000`  
API Docs at `http://localhost:8000/docs`

### 4. Live Webcam (lightweight Keras model)
```bash
python live_inference.py
```

---

## 📦 Tech Stack

| Layer | Technology |
|-------|------------|
| Sign Recognition Model | I3D (InceptionI3D) — BSL1K pretrained on WLASL 2000 classes |
| Feature Extraction | MediaPipe (Hand + Pose Landmarks) |
| Language Reconstruction | Google Gemini 2.5 Flash API |
| Text-to-Speech | gTTS (Google Text-to-Speech) |
| REST API | FastAPI + Uvicorn |
| Desktop GUI | CustomTkinter |
| Training Notebooks | Kaggle / Antigravity (local) |
| Dataset | ASL Citizen (community-sourced, ~83k videos) |

---

## 👥 Team Division of Work

| Person | Diagrams |
|--------|----------|
| **Apoorv** (You) | System Architecture, Class Diagram, Object Diagram, SRS/Design Docs |
| **Omkar G** | Use Case Diagram, Sequence Diagram, Activity Diagram, State Chart, DFD (Level 0, 1), CFD |

> 📝 Use **draw.io (diagrams.net)** — free, browser-based, and supports full UML notation.

---

## ⚠️ Large Files Not in Repo

The following files are **too large for GitHub** and must be downloaded separately:
- `bsl1k_weights/model.pth.tar` (109 MB) — downloaded automatically on first run
- `bsl1k_weights/info.pkl` — downloaded automatically on first run
- Dataset (`archive(8).zip`) — ASL Citizen dataset from Kaggle
- Trained Keras models (`*.keras`, `*.npy`) — generated after running training notebooks
