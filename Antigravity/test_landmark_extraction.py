import cv2
import mediapipe as mp
import numpy as np
import os

# Setup paths
PROJECT_DIR = os.getcwd()
DATASET_DIR = os.path.join(PROJECT_DIR, 'WLASL-complete-curated')
VIDEOS_DIR = os.path.join(DATASET_DIR, 'videos')
LANDMARKS_DIR = os.path.join(PROJECT_DIR, 'landmarks')
os.makedirs(LANDMARKS_DIR, exist_ok=True)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

SEQ_LEN = 50
NUM_FEATURES = 126

def process_single_frame(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    frame_features = np.zeros(NUM_FEATURES, dtype=np.float32)
    
    if results.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks[:2]):
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
            wrist = landmarks[0]
            normalized_landmarks = landmarks - wrist
            
            start_idx = i * 63
            end_idx = start_idx + 63
            frame_features[start_idx:end_idx] = normalized_landmarks.flatten()
            
    return frame_features

def process_video_to_sequence(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return None
        
    frames_features = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        features = process_single_frame(frame)
        frames_features.append(features)
    cap.release()
    
    frames_features = np.array(frames_features)
    num_frames = len(frames_features)
    print(f"-> Extracted raw frames: {num_frames}")
    
    if num_frames == 0:
        return np.zeros((SEQ_LEN, NUM_FEATURES), dtype=np.float32)
    
    if num_frames > SEQ_LEN:
        indices = np.linspace(0, num_frames - 1, SEQ_LEN, dtype=int)
        final_sequence = frames_features[indices]
    elif num_frames < SEQ_LEN:
        pad_length = SEQ_LEN - num_frames
        padding = np.zeros((pad_length, NUM_FEATURES), dtype=np.float32)
        final_sequence = np.vstack((frames_features, padding))
    else:
        final_sequence = frames_features
        
    return final_sequence

# Grab the first video from the dataset to test
video_files = [f for f in os.listdir(VIDEOS_DIR) if f.endswith('.mp4')]
if not video_files:
    print("No videos found in", VIDEOS_DIR)
    exit(1)

test_video = video_files[0]
VIDEO_PATH = os.path.join(VIDEOS_DIR, test_video)
VIDEO_ID = os.path.splitext(test_video)[0]

print(f"Testing on Video: {VIDEO_PATH}")
print("\nProcessing...")
sequence = process_video_to_sequence(VIDEO_PATH)

if sequence is not None:
    print(f"-> Final sequence shape: {sequence.shape}")
    output_path = os.path.join(LANDMARKS_DIR, f"{VIDEO_ID}.npy")
    np.save(output_path, sequence)
    print(f"-> Saved to {output_path}")

    reloaded_sequence = np.load(output_path)
    print(f"-> Reloaded shape: {reloaded_sequence.shape}")
    print(f"-> Reloaded exactly matches saved: {np.array_equal(sequence, reloaded_sequence)}")
else:
    print("Failed to process video.")
