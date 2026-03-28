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

# -------------------------------
# Load model
# -------------------------------
model = models.resnet18(weights=None)
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
model.fc = nn.Linear(model.fc.in_features, len(bangla_letters))

model.load_state_dict(torch.load("bangla_resnet18 (1).pth", map_location='cpu'))
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
    _, pred = torch.max(outputs, 1)

print("Predicted:", bangla_letters[int(pred)])