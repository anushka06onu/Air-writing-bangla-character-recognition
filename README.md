# Air-Writing Bangla Character Recognition

A professional, real-time computer vision application that enables users to write Bangla characters in the air using a webcam and recognize them using a deep learning model.

## 🚀 Overview

This project leverages state-of-the-art computer vision and deep learning techniques to recognize hand-drawn Bangla characters. By tracking the user's index finger in real-time, the system allows "air-writing" on a virtual canvas. The drawn character is then preprocessed and classified by a ResNet18 convolutional neural network.

## 🛠️ Technologies Used

- **Python**: The core programming language.
- **PyTorch**: Used for loading the ResNet18 model and performing inference.
- **OpenCV**: Used for video capture, image processing, and rendering the user interface.
- **MediaPipe / cvzone**: Used for robust, real-time hand detection and finger tracking.

## 🧠 How It Works

The system operates in a multi-step pipeline:

### 1. Hand Tracking & Gesture Recognition
- The application captures video from the webcam and flips it horizontally to act as a mirror.
- It uses the `cvzone` HandTracking module (powered by Google MediaPipe) to detect hands and track specific landmarks.
- **Gesture Control**: The system only draws when the **index finger is up** and all other fingers are down (Gesture: `[0, 1, 0, 0, 0]`). This prevents accidental drawing.

### 2. Virtual Canvas
- A white canvas of the same size as the video frame is maintained in memory.
- As the user moves their finger, black lines are drawn on this canvas.
- The canvas is blended with the live camera feed using bitwise operations, creating an augmented reality overlay.

### 3. Image Preprocessing
When the user triggers a prediction (by pressing `p`), the drawing undergoes several preprocessing steps to match the training data format:
- **Cropping**: The bounding box of the drawing is detected, and the character is cropped to remove empty space.
- **Dilation**: A morphological operation (dilation) is applied to thicken the strokes, making them more prominent.
- **Smoothing**: Gaussian blur is applied to smooth out jagged edges from the drawing.
- **Resizing & Centering**: The image is resized to fit within a $28 \times 28$ pixel grid while maintaining its aspect ratio, and then centered on a black background (as expected by the model).

### 4. Classification
- The preprocessed $28 \times 28$ image is converted into a PyTorch tensor.
- It is passed through a **ResNet18** model (modified for single-channel input and 84 output classes).
- The model outputs the predicted class, which is mapped to the corresponding Bangla character and displayed on the screen.

## 📋 Supported Characters
The model supports **84 classes**, including:
- Vowels (অ, আ, ই, etc.)
- Consonants (ক, খ, গ, etc.)
- Digits (০, ১, ২, etc.)
- Common Compound Characters (ক্ষ, ন্দ, ম্প, etc.)

## ⚙️ Setup and Installation

### Prerequisites
- Python 3.10 or higher.
- A working webcam.

### Installation Steps

1. **Clone the repository** (or navigate to the project directory).
2. **Install the required dependencies**:
   ```bash
   pip install torch torchvision opencv-python cvzone numpy
   ```
3. **Model Weights**:
   Ensure you have the trained model weights file named `bangla_resnet18 (1).pth` (or update the filename in `main.py`) in the project root directory. Note: The weights are typically kept out of version control due to their size.

## 🎮 Usage

Run the main application script:

```bash
python main.py
```

### Controls
- **Draw**: Raise your index finger and move it in front of the camera.
- **`p`**: Predict the drawn character.
- **`c`**: Clear the canvas and start over.
- **`q`**: Quit the application.

## 📁 Project Structure

- `main.py`: The primary application script using `cvzone` and ResNet18 (28x28 input).
- `airwriting_demo.py`: An alternative implementation using raw MediaPipe (64x64 input).
- `test_model.py`: A script to test the model on a static image (`test.png`).
- `class_to_idx.json`: JSON file mapping class indices to Bangla characters.
- `extract_classes.py`: Helper script to inspect checkpoint contents.

---
*Note: This project was trained on the BanglaLekha-Isolated dataset.*
