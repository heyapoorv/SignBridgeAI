import os
import cv2
import zipfile
import numpy as np
import pandas as pd
import mediapipe as mp
from tqdm import tqdm
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from concurrent.futures import ProcessPoolExecutor, as_completed
import uuid

# Global variables for worker processes
worker_hand_detector = None
worker_pose_detector = None
worker_zip_ref = None
MAX_FRAMES = 50
ZIP_PATH = "archive(8).zip"
OUTPUT_DIR = "asl_features_500"

def init_worker(zip_path):
    global worker_hand_detector, worker_pose_detector, worker_zip_ref
    
    # Init mediapipe
    worker_hand_detector = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path='hand_landmarker.task'), num_hands=2))
    worker_pose_detector = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path='pose_landmarker_lite.task')))
        
    worker_zip_ref = zipfile.ZipFile(zip_path, 'r')

def process_single_video(item):
    video_filename, label, split_name = item
    global worker_hand_detector, worker_pose_detector, worker_zip_ref
    
    feat_path = os.path.join(OUTPUT_DIR, split_name, f"{video_filename}.npy")
    if os.path.exists(feat_path):
        try:
            return np.load(feat_path), label
        except:
            pass # recompute if corrupted

    zip_video_path = f"ASL_Citizen/videos/{video_filename}"
    try:
        video_bytes = worker_zip_ref.read(zip_video_path)
        temp_path = f"temp_video_{uuid.uuid4().hex}.mp4"
        with open(temp_path, "wb") as f:
            f.write(video_bytes)
    except KeyError:
        return None, label # Video not found in zip

    cap = cv2.VideoCapture(temp_path)
    feats = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        
        hf = np.zeros(126)
        h_res = worker_hand_detector.detect(mp_img)
        if h_res.hand_landmarks:
            for i, h_lm in enumerate(h_res.hand_landmarks[:2]):
                for j, lm in enumerate(h_lm): hf[(i*63)+(j*3):(i*63)+(j*3)+3] = [lm.x, lm.y, lm.z]
                
        pf = np.zeros(99)
        p_res = worker_pose_detector.detect(mp_img)
        if p_res.pose_landmarks:
            for j, lm in enumerate(p_res.pose_landmarks[0]): pf[j*3:(j*3)+3] = [lm.x, lm.y, lm.z]
            
        feats.append(np.concatenate([hf, pf]))
    cap.release()
    try:
        os.remove(temp_path) # Clean up temp file
    except:
        pass
    
    seq = np.array(feats)
    if len(seq) == 0: 
        final_seq = np.zeros((MAX_FRAMES, 225))
    elif len(seq) >= MAX_FRAMES: 
        final_seq = seq[:MAX_FRAMES, :]
    else:
        final_seq = np.vstack((seq, np.zeros((MAX_FRAMES - len(seq), 225))))
        
    np.save(feat_path, final_seq)
    return final_seq, label

def main():
    zip_path = ZIP_PATH
    output_dir = OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(zip_path):
        print(f"ERROR: Could not find the dataset at {zip_path}")
        return

    print("Extracting metadata CSVs from the zip...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extract('ASL_Citizen/splits/train.csv', '.')
        z.extract('ASL_Citizen/splits/val.csv', '.')
        z.extract('ASL_Citizen/splits/test.csv', '.')

    train_df = pd.read_csv('ASL_Citizen/splits/train.csv')
    val_df = pd.read_csv('ASL_Citizen/splits/val.csv')
    test_df = pd.read_csv('ASL_Citizen/splits/test.csv')

    top_500_words = train_df['Gloss'].value_counts().head(500).index.tolist()
    label_map = {word: idx for idx, word in enumerate(sorted(top_500_words))}

    train_500 = train_df[train_df['Gloss'].isin(top_500_words)]
    val_500 = val_df[val_df['Gloss'].isin(top_500_words)]
    test_500 = test_df[test_df['Gloss'].isin(top_500_words)]

    def process_df_parallel(df, name):
        X, y = [], []
        print(f"\nProcessing {name} Set ({len(df)} videos) in parallel...")
        
        split_dir = os.path.join(output_dir, name)
        os.makedirs(split_dir, exist_ok=True)
        
        items = [(row['Video file'], label_map[row['Gloss']], name) for _, row in df.iterrows()]
        
        # Determine number of workers. Default to cpu_count.
        num_workers = min(32, os.cpu_count() or 4)
        print(f"Using {num_workers} CPU cores simultaneously!")
        
        with ProcessPoolExecutor(max_workers=num_workers, initializer=init_worker, initargs=(zip_path,)) as executor:
            futures = [executor.submit(process_single_video, item) for item in items]
            for future in tqdm(as_completed(futures), total=len(items)):
                res, lbl = future.result()
                if res is not None:
                    X.append(res)
                    y.append(lbl)
                    
        return np.array(X), np.array(y)

    print("\nStarting massive feature extraction utilizing MULTIPLE CPU CORES. This will be much faster!")
    
    X_train, y_train = process_df_parallel(train_500, "Train")
    print("Saving Train set...")
    np.save(os.path.join(output_dir, 'X_train_500.npy'), X_train)
    np.save(os.path.join(output_dir, 'y_train_500.npy'), y_train)

    X_val, y_val = process_df_parallel(val_500, "Validation")
    print("Saving Validation set...")
    np.save(os.path.join(output_dir, 'X_val_500.npy'), X_val)
    np.save(os.path.join(output_dir, 'y_val_500.npy'), y_val)

    X_test, y_test = process_df_parallel(test_500, "Test")
    print("Saving Test set...")
    np.save(os.path.join(output_dir, 'X_test_500.npy'), X_test)
    np.save(os.path.join(output_dir, 'y_test_500.npy'), y_test)

    print(f"\n✅ Complete! All features saved in the {output_dir} folder.")

if __name__ == "__main__":
    # Windows requires this for multiprocessing
    from multiprocessing import freeze_support
    freeze_support()
    main()
