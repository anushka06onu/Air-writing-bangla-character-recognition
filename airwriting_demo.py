from checkpoint_utils import load_checkpoint, decode_character
import cv2
import numpy as np
import mediapipe as mp
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# ----------------------
# 1️⃣ Setup MediaPipe Hand Tracking
# ----------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# ----------------------
# 2️⃣ Load Trained ResNet18
# ----------------------
num_classes = 84  # Bangla letters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pretrained ResNet18
model = models.resnet18()
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
model.fc = nn.Linear(model.fc.in_features, num_classes)
idx_to_folder = load_checkpoint(model, "bangla_resnet18.pth", "class_to_idx.json", num_classes, device)
model.to(device)
model.eval()

bangla_letters = [
    'অ', 'আ', 'ই', 'ঈ', 'উ', 'ঊ', 'ঋ', 'এ', 'ঐ', 'ও', 'ঔ',
    'ক', 'খ', 'গ', 'ঘ', 'ঙ', 'চ', 'ছ', 'জ', 'ঝ', 'ঞ',
    'ট', 'ঠ', 'ড', 'ঢ', 'ণ', 'ত', 'থ', 'দ', 'ধ', 'ন',
    'প', 'ফ', 'ব', 'ভ', 'ম', 'য', 'র', 'ল', 'শ', 'ষ',
    'স', 'হ', 'ড়', 'ঢ়', 'য়', 'ৎ', 'ং', 'ঃ', 'ঁ',
    '০', '১', '২', '৩', '৪', '৫', '৬', '৭', '৮', '৯',
    'ক্ষ', 'ব্দ', 'ঙ্গ', 'স্ক', 'স্ফ', 'স্হ', 'চ্ছ', 'ক্ত', 'স্ন', 'ষ্ণ',
    'ম্প', 'ক্ষ', 'প্ত', 'ম্ব', 'ণ্ড', 'দ্ভ', 'ত্থ', 'ষ্ঠ', 'ল্প', 'ষ্প', 'ন্দ', 'ন্ধ', 'ম্ম', 'ণ্ঠ'
]

import os
import json

# ----------------------
# Dynamic Mapping Setup to prevent index mismatch
# ----------------------
def get_predicted_character(pred_idx, letters):
    return decode_character(pred_idx, letters, idx_to_folder)


# ----------------------
# 3️⃣ Setup Drawing Canvas
# ----------------------
canvas_size = 256
canvas = np.ones((canvas_size, canvas_size), dtype=np.uint8) * 255
prev_x, prev_y = None, None

# ----------------------
# 4️⃣ Image Transform
# ----------------------
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28,28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ----------------------
# 5️⃣ Start Webcam
# ----------------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # Track fingertip
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            lm = handLms.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            h, w, _ = frame.shape
            cx = max(0, min(canvas_size - 1, int(lm.x * canvas_size)))
            cy = max(0, min(canvas_size - 1, int(lm.y * canvas_size)))
            if prev_x is None:
                prev_x, prev_y = cx, cy
            cv2.line(canvas, (prev_x, prev_y), (cx, cy), 0, 5)
            prev_x, prev_y = cx, cy

    else:
        prev_x, prev_y = None, None

    # Show current trajectory
    cv2.imshow("Canvas", canvas)
    cv2.imshow("Webcam", frame)

    key = cv2.waitKey(1) & 0xFF

    # Press 'c' to clear canvas
    if key == ord('c'):
        canvas = np.ones((canvas_size, canvas_size), dtype=np.uint8) * 255
        prev_x, prev_y = None, None

    # Press 'p' to predict
    if key == ord('p'):
        img = canvas  # Match the single-channel 28x28 training input.
        img_tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(img_tensor)
            pred_idx = torch.argmax(output, 1).item()
        
        predicted_letter = get_predicted_character(pred_idx, bangla_letters)
        print(f"Predicted index: {pred_idx} -> Bangla Character: {predicted_letter}")
        cv2.putText(frame, f"Predicted: {predicted_letter}", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)
        cv2.imshow("Webcam", frame)

    # Press 'q' to quit
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
