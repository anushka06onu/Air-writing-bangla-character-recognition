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

# -------------------------------
# Load model
# -------------------------------
model = models.resnet18(weights=None)
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
model.fc = nn.Linear(model.fc.in_features, len(bangla_letters))

state_dict = torch.load("bangla_resnet18 (1).pth", map_location=torch.device('cpu'))
model.load_state_dict(state_dict)
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
                _, pred_idx = torch.max(outputs, 1)

            predicted_letter = bangla_letters[int(pred_idx)]

            print("Predicted:", predicted_letter)

            cv2.putText(combined, f"Prediction: {predicted_letter}", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)

            cv2.imshow("Air Writing Bangla", combined)
            cv2.waitKey(2000)

        else:
            print("Nothing drawn!")


cap.release()
cv2.destroyAllWindows()
