import json

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# SignBridge AI - End-to-End ML Pipeline\n",
                "\n",
                "This notebook implements the core machine learning pipeline for the SignBridge AI project, an offline real-time word-level Sign Language to Speech Translation system."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Setup & Configuration"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "import json\n",
                "import cv2\n",
                "import numpy as np\n",
                "import pandas as pd\n",
                "import mediapipe as mp\n",
                "import torch\n",
                "import torch.nn as nn\n",
                "from torch.utils.data import Dataset, DataLoader\n",
                "from sklearn.metrics import classification_report, confusion_matrix\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "\n",
                "# --- CONFIGURATION ---\n",
                "class Config:\n",
                "    DATA_DIR = './WLASL-complete-curated/videos'\n",
                "    JSON_INDEX = './WLASL-complete-curated/WLASL_custom_split.json'\n",
                "    PROCESSED_DATA_DIR = './ml/data/processed'\n",
                "    \n",
                "    NUM_CLASSES = 100\n",
                "    SEQUENCE_LENGTH = 30\n",
                "    FEATURE_DIM = 126  # 63 Left + 63 Right\n",
                "    \n",
                "    BATCH_SIZE = 32\n",
                "    HIDDEN_SIZE = 128\n",
                "    NUM_LAYERS = 2\n",
                "    DROPOUT = 0.3\n",
                "    LEARNING_RATE = 0.001\n",
                "    NUM_EPOCHS = 50\n",
                "    \n",
                "    SEED = 42\n",
                "\n",
                "config = Config()\n",
                "\n",
                "# Reproducibility\n",
                "torch.manual_seed(config.SEED)\n",
                "np.random.seed(config.SEED)\n",
                "\n",
                "os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Phase 3: Dataset Preprocessing\n",
                "\n",
                "Here we define functions to extract 126-D landmarks from videos and process the entire dataset."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "mp_hands = mp.solutions.hands\n",
                "hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)\n",
                "\n",
                "def extract_126d_features(image_rgb):\n",
                "    left_hand = np.zeros(63)\n",
                "    right_hand = np.zeros(63)\n",
                "    \n",
                "    results = hands.process(image_rgb)\n",
                "    \n",
                "    if results.multi_hand_landmarks:\n",
                "        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):\n",
                "            handedness = results.multi_handedness[hand_idx].classification[0].label\n",
                "            hand_points = []\n",
                "            for landmark in hand_landmarks.landmark:\n",
                "                hand_points.extend([landmark.x, landmark.y, landmark.z])\n",
                "            \n",
                "            if handedness == 'Left':\n",
                "                left_hand = np.array(hand_points)\n",
                "            elif handedness == 'Right':\n",
                "                right_hand = np.array(hand_points)\n",
                "                \n",
                "    return np.concatenate([left_hand, right_hand])\n",
                "\n",
                "def process_video(video_path, max_seq_len=30):\n",
                "    cap = cv2.VideoCapture(video_path)\n",
                "    frames_features = []\n",
                "    \n",
                "    while cap.isOpened():\n",
                "        ret, frame = cap.read()\n",
                "        if not ret:\n",
                "            break\n",
                "        # MediaPipe needs RGB\n",
                "        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)\n",
                "        features = extract_126d_features(frame_rgb)\n",
                "        frames_features.append(features)\n",
                "        \n",
                "    cap.release()\n",
                "    \n",
                "    if len(frames_features) == 0:\n",
                "        return np.zeros((max_seq_len, config.FEATURE_DIM))\n",
                "        \n",
                "    # Temporal sampling/padding to fixed max_seq_len\n",
                "    seq = np.array(frames_features)\n",
                "    seq_len = seq.shape[0]\n",
                "    \n",
                "    if seq_len > max_seq_len:\n",
                "        # Sample evenly\n",
                "        indices = np.linspace(0, seq_len - 1, max_seq_len, dtype=int)\n",
                "        seq = seq[indices]\n",
                "    elif seq_len < max_seq_len:\n",
                "        # Zero pad at the end\n",
                "        padding = np.zeros((max_seq_len - seq_len, config.FEATURE_DIM))\n",
                "        seq = np.vstack((seq, padding))\n",
                "        \n",
                "    return seq"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Dataset Generation (Run Once)\n",
                "Iterate over dataset index, extract features, and save as `.npy` files."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def generate_dataset():\n",
                "    # This is a placeholder for the actual dataset generation logic.\n",
                "    # It parses WLASL_custom_split.json, runs process_video() on each, \n",
                "    # and saves train_X.npy, train_y.npy, etc. in PROCESSED_DATA_DIR.\n",
                "    # You should run this once and then load from disk.\n",
                "    print(\"Dataset Generation block - Implement JSON parsing here based on exact index format.\")\n",
                "    pass\n",
                "\n",
                "# generate_dataset()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. PyTorch Data Pipeline"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "class SignLanguageDataset(Dataset):\n",
                "    def __init__(self, X_path, y_path):\n",
                "        # Use dummy data for initialization if files don't exist yet\n",
                "        if os.path.exists(X_path) and os.path.exists(y_path):\n",
                "            self.X = torch.FloatTensor(np.load(X_path))\n",
                "            self.y = torch.LongTensor(np.load(y_path))\n",
                "        else:\n",
                "            print(f\"Warning: {X_path} not found. Using random dummy data for illustration.\")\n",
                "            self.X = torch.rand((100, config.SEQUENCE_LENGTH, config.FEATURE_DIM))\n",
                "            self.y = torch.randint(0, config.NUM_CLASSES, (100,))\n",
                "            \n",
                "    def __len__(self):\n",
                "        return len(self.X)\n",
                "        \n",
                "    def __getitem__(self, idx):\n",
                "        return self.X[idx], self.y[idx]\n",
                "\n",
                "# Load DataLoaders\n",
                "train_dataset = SignLanguageDataset(f'{config.PROCESSED_DATA_DIR}/train_X.npy', f'{config.PROCESSED_DATA_DIR}/train_y.npy')\n",
                "val_dataset = SignLanguageDataset(f'{config.PROCESSED_DATA_DIR}/val_X.npy', f'{config.PROCESSED_DATA_DIR}/val_y.npy')\n",
                "test_dataset = SignLanguageDataset(f'{config.PROCESSED_DATA_DIR}/test_X.npy', f'{config.PROCESSED_DATA_DIR}/test_y.npy')\n",
                "\n",
                "train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)\n",
                "val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)\n",
                "test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Phase 4: Model Architecture\n",
                "Baseline LSTM Model"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "class SignLanguageLSTM(nn.Module):\n",
                "    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout):\n",
                "        super(SignLanguageLSTM, self).__init__()\n",
                "        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, \n",
                "                            batch_first=True, dropout=dropout, bidirectional=False)\n",
                "        self.fc = nn.Linear(hidden_size, num_classes)\n",
                "        \n",
                "    def forward(self, x):\n",
                "        # x shape: (batch_size, seq_len, input_size)\n",
                "        out, (h_n, c_n) = self.lstm(x)\n",
                "        # Get the output of the last time step\n",
                "        out = out[:, -1, :]\n",
                "        out = self.fc(out)\n",
                "        return out\n",
                "\n",
                "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
                "model = SignLanguageLSTM(\n",
                "    input_size=config.FEATURE_DIM, \n",
                "    hidden_size=config.HIDDEN_SIZE, \n",
                "    num_layers=config.NUM_LAYERS, \n",
                "    num_classes=config.NUM_CLASSES, \n",
                "    dropout=config.DROPOUT\n",
                ").to(device)\n",
                "\n",
                "print(model)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. Training & Validation"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "criterion = nn.CrossEntropyLoss()\n",
                "optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)\n",
                "\n",
                "train_losses, val_losses = [], []\n",
                "train_accs, val_accs = [], []\n",
                "best_val_acc = 0.0\n",
                "\n",
                "for epoch in range(config.NUM_EPOCHS):\n",
                "    # --- TRAINING ---\n",
                "    model.train()\n",
                "    running_loss, correct, total = 0.0, 0, 0\n",
                "    for X_batch, y_batch in train_loader:\n",
                "        X_batch, y_batch = X_batch.to(device), y_batch.to(device)\n",
                "        \n",
                "        optimizer.zero_grad()\n",
                "        outputs = model(X_batch)\n",
                "        loss = criterion(outputs, y_batch)\n",
                "        loss.backward()\n",
                "        optimizer.step()\n",
                "        \n",
                "        running_loss += loss.item()\n",
                "        _, predicted = torch.max(outputs.data, 1)\n",
                "        total += y_batch.size(0)\n",
                "        correct += (predicted == y_batch).sum().item()\n",
                "        \n",
                "    train_losses.append(running_loss / len(train_loader))\n",
                "    train_accs.append(100 * correct / total)\n",
                "    \n",
                "    # --- VALIDATION ---\n",
                "    model.eval()\n",
                "    val_loss, correct, total = 0.0, 0, 0\n",
                "    with torch.no_grad():\n",
                "        for X_batch, y_batch in val_loader:\n",
                "            X_batch, y_batch = X_batch.to(device), y_batch.to(device)\n",
                "            outputs = model(X_batch)\n",
                "            loss = criterion(outputs, y_batch)\n",
                "            \n",
                "            val_loss += loss.item()\n",
                "            _, predicted = torch.max(outputs.data, 1)\n",
                "            total += y_batch.size(0)\n",
                "            correct += (predicted == y_batch).sum().item()\n",
                "            \n",
                "    val_losses.append(val_loss / len(val_loader))\n",
                "    val_acc = 100 * correct / total\n",
                "    val_accs.append(val_acc)\n",
                "    \n",
                "    print(f'Epoch [{epoch+1}/{config.NUM_EPOCHS}] | Train Loss: {train_losses[-1]:.4f} | Train Acc: {train_accs[-1]:.2f}% | Val Loss: {val_losses[-1]:.4f} | Val Acc: {val_acc:.2f}%')\n",
                "    \n",
                "    if val_acc > best_val_acc:\n",
                "        best_val_acc = val_acc\n",
                "        torch.save(model.state_dict(), 'best_model.pt')\n",
                "        print(\"--> Saved new best model!\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 6. Evaluation\n",
                "Evaluate the best model on the unseen Test set."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "model.load_state_dict(torch.load('best_model.pt'))\n",
                "model.eval()\n",
                "\n",
                "all_preds = []\n",
                "all_labels = []\n",
                "\n",
                "with torch.no_grad():\n",
                "    for X_batch, y_batch in test_loader:\n",
                "        X_batch = X_batch.to(device)\n",
                "        outputs = model(X_batch)\n",
                "        _, predicted = torch.max(outputs.data, 1)\n",
                "        \n",
                "        all_preds.extend(predicted.cpu().numpy())\n",
                "        all_labels.extend(y_batch.numpy())\n",
                "\n",
                "print(\"Classification Report:\")\n",
                "print(classification_report(all_labels, all_preds))\n",
                "\n",
                "# Confusion Matrix\n",
                "cm = confusion_matrix(all_labels, all_preds)\n",
                "plt.figure(figsize=(10, 8))\n",
                "sns.heatmap(cm, annot=False, cmap='Blues')\n",
                "plt.xlabel('Predicted')\n",
                "plt.ylabel('True')\n",
                "plt.title('Confusion Matrix')\n",
                "plt.show()"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.11.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open("SignBridge_EndToEnd_ML.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=4)
print("Successfully generated SignBridge_EndToEnd_ML.ipynb")
