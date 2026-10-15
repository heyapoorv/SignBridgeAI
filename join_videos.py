import cv2
import sys
import os

def join_videos(video_paths, output_path="joined_test_video.mp4"):
    print(f"Joining {len(video_paths)} videos...")
    
    # Open the first video to get dimensions and fps
    first_cap = cv2.VideoCapture(video_paths[0])
    if not first_cap.isOpened():
        print(f"Error: Could not open {video_paths[0]}")
        return
        
    width = int(first_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(first_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(first_cap.get(cv2.CAP_PROP_FPS))
    if fps == 0: fps = 30
    first_cap.release()
    
    # Initialize writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for vp in video_paths:
        if not os.path.exists(vp):
            print(f"Warning: File {vp} not found. Skipping.")
            continue
            
        print(f"Processing: {os.path.basename(vp)}")
        cap = cv2.VideoCapture(vp)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Resize if necessary to match the first video's dimensions
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height))
                
            out.write(frame)
        cap.release()
        
    out.release()
    print(f"\nSuccess! Saved joined video to: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python join_videos.py output.mp4 video1.mp4 video2.mp4 ...")
    else:
        out_name = sys.argv[1]
        video_files = sys.argv[2:]
        join_videos(video_files, out_name)
