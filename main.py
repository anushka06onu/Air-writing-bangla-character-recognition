import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from cvzone.HandTrackingModule import HandDetector

# -------------------------------
# Bangla letters (dataset order)
# -------------------------------
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

# -------------------------------
# Dynamic Mapping Setup to prevent index mismatch
# -------------------------------
class_to_idx = None
idx_to_folder = None

if os.path.exists("class_to_idx.json"):
    with open("class_to_idx.json", "r") as f:
        try:
            class_to_idx = json.load(f)
            idx_to_folder = {int(v): k for k, v in class_to_idx.items()}
            print("✅ Loaded index mapping from 'class_to_idx.json'")
        except Exception as e:
            print("⚠️ Warning: Could not parse class_to_idx.json.")

def get_predicted_character(pred_idx, letters):
    if idx_to_folder is not None and pred_idx in idx_to_folder:
        folder_name = idx_to_folder[pred_idx]
        if folder_name.isdigit():
            letter_idx = int(folder_name) - 1
            if 0 <= letter_idx < len(letters):
                return letters[letter_idx]
    if 0 <= pred_idx < len(letters):
        return letters[pred_idx]
    return "Unknown"

# -------------------------------
# Load model
# -------------------------------
model = models.resnet18(weights=None)
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
model.fc = nn.Linear(model.fc.in_features, len(bangla_letters))

# Try both filenames
model_path = "bangla_resnet18 (1).pth"
if not os.path.exists(model_path):
    model_path = "bangla_resnet18.pth"

if os.path.exists(model_path):
    print(f"📦 Loading model weights from '{model_path}'...")
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
else:
    print("\n" + "="*80)
    print("❌ FATAL ERROR: Model weights file not found!")
    print("Because no model weights were loaded, the application cannot perform predictions.")
    print("Please resolve this using one of the following two options:")
    print("1. If you have the pre-trained weights file 'bangla_resnet18 (1).pth' on your computer,")
    print("   copy it into this folder: /Users/fatehahossainanushka/Desktop/Air-writing-bangla-character-recognition/")
    print("2. Or, train a new model from scratch:")
    print("   - Download 'BanglaLekha-Isolated.zip' from https://data.mendeley.com/datasets/hf6sf8zrkc/2")
    print("   - Place it in this folder and run: python train.py")
    print("="*80 + "\n")
    import sys
    sys.exit(1)

model.eval()

# -------------------------------
# Camera & Hand Detector
# -------------------------------
cap = cv2.VideoCapture(0)
detector = HandDetector(detectionCon=0.8, maxHands=1)

canvas = None
xp, yp = 0, 0
drawing = False

# -------------------------------
# Main loop
# -------------------------------
while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)

    if canvas is None:
        canvas = np.ones_like(frame) * 255  # white canvas

    hands, img = detector.findHands(frame, flipType=False)

    if hands:
        hand = hands[0]
        fingers = detector.fingersUp(hand)

        # index fingertip
        cx, cy = hand['lmList'][8][0], hand['lmList'][8][1]

        # draw only if index finger up
        if fingers == [0, 1, 0, 0, 0]:
            if not drawing:
                drawing = True
                xp, yp = cx, cy

            cv2.line(canvas, (xp, yp), (cx, cy), (0, 0, 0), 12)
            xp, yp = cx, cy
        else:
            drawing = False
            xp, yp = 0, 0

    # -------------------------------
    # Merge canvas with frame
    # -------------------------------
    gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_canvas, 250, 255, cv2.THRESH_BINARY_INV)

    mask_inv = cv2.bitwise_not(mask)
    frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
    canvas_fg = cv2.bitwise_and(canvas, canvas, mask=mask)

    combined = cv2.add(frame_bg, canvas_fg)

    cv2.imshow("Air Writing Bangla", combined)

    # -------------------------------
    # Controls
    # -------------------------------
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    elif key == ord('c'):
        canvas[:] = 255

    elif key == ord('p'):
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)

        # invert
        gray = cv2.bitwise_not(gray)

        # find drawing
        coords = cv2.findNonZero(gray)

        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            cropped = gray[y:y+h, x:x+w]

            # 🔥 thicken strokes
            kernel = np.ones((3, 3), np.uint8)
            cropped = cv2.dilate(cropped, kernel, iterations=1)

            # 🔥 smooth
            cropped = cv2.GaussianBlur(cropped, (5, 5), 0)

            # 🔥 resize with aspect ratio
            h, w = cropped.shape

            if h > w:
                new_h = 28
                new_w = int(w * (28 / h))
            else:
                new_w = 28
                new_h = int(h * (28 / w))

            resized = cv2.resize(cropped, (new_w, new_h))

            # 🔥 center in 28x28
            final_img = np.zeros((28, 28), dtype=np.uint8)

            x_offset = (28 - new_w) // 2
            y_offset = (28 - new_h) // 2

            final_img[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

            # normalize
            normalized = final_img / 255.0

            tensor = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

            with torch.no_grad():
                outputs = model(tensor)
                _, pred_idx_tensor = torch.max(outputs, 1)
                pred_idx = int(pred_idx_tensor.item())

            predicted_letter = get_predicted_character(pred_idx, bangla_letters)

            print("Predicted index:", pred_idx, "-> Character:", predicted_letter)

            cv2.putText(combined, f"Prediction: {predicted_letter}", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)

            cv2.imshow("Air Writing Bangla", combined)
            cv2.waitKey(2000)

        else:
            print("Nothing drawn!")


cap.release()
cv2.destroyAllWindows()
