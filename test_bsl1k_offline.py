import os
import sys
import torch
import numpy as np
import scipy.special
from pathlib import Path
import cv2
import math

# Add bsl1k to path so we can import its modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'bsl1k'))

from models import InceptionI3d
from demo.demo import load_vocab, prepare_input, sliding_windows, fetch_from_url

def load_rgb_video_cv2(video_path: Path):
    """Fallback video loader using only OpenCV to avoid ffmpeg dependency."""
    cap = cv2.VideoCapture(str(video_path))
    rgb = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # OpenCV reads BGR, we need RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Convert to tensor and normalize to [0, 1] (similar to im_to_torch)
        rgb_t = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        rgb.append(rgb_t)
    cap.release()
    # Stack to (C, T, H, W) -> (3, nframes, H, W)
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
    
    # Their checkpoint is saved from DataParallel, which prepends 'module.' to keys.
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["state_dict"]
    
    # Remove 'module.' prefix if we are loading into a non-DataParallel model
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

def predict_video(video_path, top_k=3):
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return

    # Configurations (WLASL pretrained model)
    checkpoint_url = "https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/wlasl_i3d_pbsl1k/model.pth.tar"
    checkpoint_path = Path("bsl1k_weights/model.pth.tar")
    word_data_url = "https://www.robots.ox.ac.uk/~vgg/research/bsl1k/demo-data/wlasl/info/info.pkl"
    word_data_path = Path("bsl1k_weights/info.pkl")
    
    num_classes = 2000 # WLASL classes
    num_in_frames = 64
    stride = 8
    
    print(f"Loading Model (will download weights if missing)...")
    model, device = load_model_device_agnostic(
        checkpoint_path=checkpoint_path,
        checkpoint_url=checkpoint_url,
        num_classes=num_classes,
        num_in_frames=num_in_frames,
    )
    
    print(f"Loading Vocabulary...")
    word_data = load_vocab(
        word_data_pkl_path=word_data_path,
        word_data_pkl_url=word_data_url,
    )
    
    print(f"Processing Video: {video_path}")
    rgb_orig = load_rgb_video_cv2(video_path=Path(video_path))
    
    rgb_input = prepare_input(rgb_orig)
    rgb_slides, t_mid = sliding_windows(rgb=rgb_input, stride=stride, num_in_frames=num_in_frames)
    
    print(f"Running Inference on {rgb_slides.shape[0]} clips...")
    
    raw_scores = []
    with torch.no_grad():
        for i in range(rgb_slides.shape[0]):
            inp = rgb_slides[i:i+1].to(device)
            out = model(inp)
            raw_scores.append(out["logits"].cpu().numpy()[0])
            
    raw_scores = np.array(raw_scores)
    prob_scores = scipy.special.softmax(raw_scores, axis=1)
    
    # Average probabilities across all clips (simple consensus)
    avg_probs = np.mean(prob_scores, axis=0)
    
    top_indices = np.argsort(avg_probs)[::-1][:top_k]
    
    print("\n" + "=" * 40)
    print("Top Predictions for entire video")
    print("=" * 40)
    for idx in top_indices:
        word = word_data["words"][idx]
        conf = avg_probs[idx]
        print(f" -> {word:<15}: {conf*100:.2f}%")
    print("=" * 40 + "\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_bsl1k_offline.py <path_to_video.mp4>")
        sys.exit(1)
        
    video_path = sys.argv[1]
    predict_video(video_path)
