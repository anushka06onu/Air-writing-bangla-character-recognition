import cv2
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np

# -------------------------------
# Bangla letters
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
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
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
# Load test image
# -------------------------------
img = cv2.imread("test.png", cv2.IMREAD_GRAYSCALE)  # <-- CHANGE THIS

# Resize to 28x28
img = cv2.resize(img, (28, 28))

# Normalize
img = img / 255.0

# Convert to tensor
tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

# -------------------------------
# Predict
# -------------------------------
with torch.no_grad():
    outputs = model(tensor)
    _, pred_tensor = torch.max(outputs, 1)
    pred_idx = int(pred_tensor.item())

predicted_letter = get_predicted_character(pred_idx, bangla_letters)
print("Predicted index:", pred_idx, "-> Character:", predicted_letter)