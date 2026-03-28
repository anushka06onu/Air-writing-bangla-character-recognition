# Air-Writing Bangla Character Recognition

I built this as a small uni project to recognize Bangla characters written in the air with a webcam. It uses a ResNet18 I trained on the BanglaLekha-Isolated dataset.

## What I used
- PyTorch for the ResNet18 model.
- OpenCV for the camera feed and image ops.
- cvzone HandTracking (built on Google MediaPipe) to track my index fingertip.

## How it works (quick run-down)
1. The camera feed is flipped like a mirror.
2. When my index finger is up, a black stroke is drawn on a white canvas.
3. Press `p` to crop the drawing, thicken + smooth it, resize to 28x28, and send it through the model.
4. The predicted Bangla letter is printed and overlaid on the video.
5. Controls: `c` = clear canvas, `q` = quit.

## Setup
1. Create/activate a Python 3.10+ environment.
2. Install deps:
   ```bash
   pip install torch torchvision opencv-python cvzone numpy
   ```
3. Download or place the trained weights file as `bangla_resnet18.pth` (or update the path in `main.py`). I kept the weights out of git because they’re large.

## Run it
```bash
python main.py
```
A window named "Air Writing Bangla" should open. Raise your index finger to draw; hit `p` to see the predicted character.

## Notes
- The class list is in `class_to_idx.json`; the Bangla label order matches the model head.
- If the camera doesn’t open, check `cv2.VideoCapture(0)` in `main.py` and switch to the right camera index.
- For smoother strokes, you can tweak the dilation kernel or Gaussian blur in the preprocessing section.
