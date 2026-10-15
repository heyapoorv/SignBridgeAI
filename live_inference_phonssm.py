import cv2
import mediapipe as mp
import numpy as np
import torch
import sys
import os
import pyttsx3
import threading
from collections import deque

# 1. Add the PhonSSM folder to Python path so we can import the model
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'PhonSSM'))
from models.phonssm import PhonSSM, PhonSSMConfig

# 2. Text-to-Speech Engine
engine = pyttsx3.init()
def speak(text):
    # Run speech in a separate thread so it doesn't freeze the webcam
    def run_speech():
        engine.say(text)
        engine.runAndWait()
    threading.Thread(target=run_speech).start()

# 3. Model Configuration & Loading
print("Loading SOTA PhonSSM PyTorch Model...")
config = PhonSSMConfig(
    num_signs=100, 
    temperature=1.0, 
    input_mode="pose_hands",
    num_landmarks=75
)
model = PhonSSM(config)
model.eval()

# TODO: After training in Colab, save the weights with: 
# torch.save(model.state_dict(), 'best_phonssm_weights.pt')
# Then download that file and update this path!
WEIGHTS_PATH = os.path.join(current_dir, 'best_phonssm_weights.pt')
if os.path.exists(WEIGHTS_PATH):
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=torch.device('cpu')))
    print("Weights loaded successfully!")
else:
    print("WARNING: Weights file not found. Running with random weights for testing.")

# Actual 100 words in the correct order!
ACTIONS = ['book', 'drink', 'computer', 'before', 'chair', 'go', 'clothes', 'who', 'candy', 'cousin', 'deaf', 'fine', 'help', 'no', 'thin', 'walk', 'year', 'yes', 'all', 'black', 'cool', 'finish', 'hot', 'like', 'many', 'mother', 'now', 'orange', 'table', 'thanksgiving', 'what', 'woman', 'bed', 'blue', 'bowling', 'can', 'dog', 'family', 'fish', 'graduate', 'hat', 'hearing', 'kiss', 'language', 'later', 'man', 'shirt', 'study', 'tall', 'white', 'wrong', 'accident', 'apple', 'bird', 'change', 'color', 'corn', 'cow', 'dance', 'dark', 'doctor', 'eat', 'enjoy', 'forget', 'give', 'last', 'meet', 'pink', 'pizza', 'play', 'school', 'secretary', 'short', 'time', 'want', 'work', 'africa', 'basketball', 'birthday', 'brown', 'but', 'cheat', 'city', 'cook', 'decide', 'full', 'how', 'jacket', 'letter', 'medicine', 'need', 'paint', 'paper', 'pull', 'purple', 'right', 'same', 'son', 'tell', 'thursday']

# 4. MediaPipe Setup
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def extract_75_landmarks(results):
    """
    Extracts exactly 75 landmarks (33 Pose + 21 Left Hand + 21 Right Hand)
    Returns shape: (75, 3)
    """
    # 1. Pose (33 landmarks)
    if results.pose_landmarks:
        pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark])
    else:
        pose = np.zeros((33, 3))
        
    # 2. Left Hand (21 landmarks)
    if results.left_hand_landmarks:
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark])
    else:
        lh = np.zeros((21, 3))
        
    # 3. Right Hand (21 landmarks)
    if results.right_hand_landmarks:
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark])
    else:
        rh = np.zeros((21, 3))
        
    # Concatenate to exactly 75 landmarks
    return np.concatenate([pose, lh, rh], axis=0)

# 5. Live Webcam Loop
cap = cv2.VideoCapture(0)
sequence = deque(maxlen=30)  # PhonSSM expects exactly 30 frames
current_prediction = "Waiting for gesture..."
last_spoken = ""

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # Recolor and process with MediaPipe
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = holistic.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Draw skeleton on screen
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        
        # Extract the 75 landmarks
        landmarks = extract_75_landmarks(results)
        sequence.append(landmarks)
        
        # Once we have 30 frames, make a prediction!
        if len(sequence) == 30:
            # Convert to PyTorch Tensor -> Shape: (1, 30, 75, 3)
            X_tensor = torch.FloatTensor(np.array(sequence)).unsqueeze(0)
            
            with torch.no_grad():
                outputs = model(X_tensor)
                prediction_idx = outputs['logits'].argmax(dim=1).item()
                
                # Check if we have enough actions mapped
                if prediction_idx < len(ACTIONS):
                    predicted_word = ACTIONS[prediction_idx]
                    current_prediction = f"Sign: {predicted_word.upper()}"
                    
                    # Speak the word out loud if it's new!
                    if predicted_word != last_spoken:
                        speak(predicted_word)
                        last_spoken = predicted_word

        # Display UI
        cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, current_prediction, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        cv2.imshow('BE Project - Real-Time Sign Language', image)
        
        # Press 'q' to quit
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
