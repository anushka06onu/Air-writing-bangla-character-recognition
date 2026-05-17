import os
import zipfile
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
import json

# ----------------------------------------------------
# 1. Config & Hyperparameters
# ----------------------------------------------------
DATASET_ZIP = "BanglaLekha-Isolated.zip"
DATASET_DIR = "BanglaLekha-Isolated/Images"
MODEL_SAVE_PATH = "bangla_resnet18.pth"
CLASS_MAP_PATH = "class_to_idx.json"

BATCH_SIZE = 128
EPOCHS = 15
LEARNING_RATE = 0.001
VAL_SPLIT = 0.15
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# ----------------------------------------------------
# 2. Extract Dataset if Zip exists
# ----------------------------------------------------
if not os.path.exists(DATASET_DIR):
    if os.path.exists(DATASET_ZIP):
        print(f"📦 Extracting {DATASET_ZIP} to {DATASET_DIR}...")
        with zipfile.ZipFile(DATASET_ZIP, 'r') as zip_ref:
            zip_ref.extractall(DATASET_DIR)
        print("✅ Extraction complete!")
    else:
        print(f"❌ Error: Neither directory '{DATASET_DIR}' nor '{DATASET_ZIP}' was found.")
        print("Please download 'BanglaLekha-Isolated.zip' from:")
        print("https://data.mendeley.com/datasets/hf6sf8zrkc/2")
        print("and place it in this directory, then run train.py again.")
        exit(1)

# ----------------------------------------------------
# 3. Custom Dataset Folder for Numerical Sorting
# ----------------------------------------------------
class NumericalImageFolder(ImageFolder):
    """
    Manually overrides class finding to sort class folder names numerically.
    This ensures that folder '1' -> index 0, '2' -> index 1, ... '84' -> index 83,
    aligning perfectly with the natural order of bangla_letters.
    """
    def _find_classes(self, dir):
        # Find numeric directories (1 to 84)
        classes = [d.name for d in os.scandir(dir) if d.is_dir() and d.name.isdigit()]
        if not classes:
            raise RuntimeError(f"No numeric directories (like 1, 2, ..., 84) found in {dir}")
        # Sort numerically
        classes.sort(key=int)
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

# ----------------------------------------------------
# 4. Data Preparation & Augmentation
# ----------------------------------------------------
# Define robust training and validation transforms
train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    # Data Augmentation to simulate varying hand movements, rotations, and scales
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

val_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load dataset
print("🔍 Loading dataset from:", DATASET_DIR)
full_dataset = NumericalImageFolder(DATASET_DIR)

# Split into Train and Validation sets
val_size = int(len(full_dataset) * VAL_SPLIT)
train_size = len(full_dataset) - val_size
train_subset, val_subset = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

# Apply different transforms for train vs val
class transformed_dataset(torch.utils.data.Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform
    def __getitem__(self, index):
        x, y = self.subset[index]
        return self.transform(x), y
    def __len__(self):
        return len(self.subset)

train_dataset = transformed_dataset(train_subset, train_transform)
val_dataset = transformed_dataset(val_subset, val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

print(f"📊 Dataset successfully loaded!")
print(f"   - Total Images: {len(full_dataset)}")
print(f"   - Training Images: {train_size}")
print(f"   - Validation Images: {val_size}")
print(f"   - Number of Classes: {len(full_dataset.classes)}")

# Save class_to_idx mapping to JSON
with open(CLASS_MAP_PATH, "w") as f:
    json.dump(full_dataset.class_to_idx, f, indent=4)
print(f"💾 Class-to-Index mapping saved to '{CLASS_MAP_PATH}'")

# ----------------------------------------------------
# 5. Model Architecture (ResNet18)
# ----------------------------------------------------
model = models.resnet18(weights=None)
# Modify conv1 to accept single-channel grayscale input
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
# Modify fully connected layer to output len(classes) = 84
model.fc = nn.Linear(model.fc.in_features, len(full_dataset.classes))
model = model.to(DEVICE)

# ----------------------------------------------------
# 6. Training Configuration
# ----------------------------------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

# ----------------------------------------------------
# 7. Training & Validation Loop
# ----------------------------------------------------
best_val_acc = 0.0
print(f"🚀 Training starting on device: {DEVICE}")

for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct_train += (preds == labels).sum().item()
        total_train += labels.size(0)
        
    epoch_loss = running_loss / train_size
    epoch_acc = (correct_train / total_train) * 100
    
    # Validation Loop
    model.eval()
    val_loss = 0.0
    correct_val = 0
    total_val = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct_val += (preds == labels).sum().item()
            total_val += labels.size(0)
            
    epoch_val_loss = val_loss / val_size
    epoch_val_acc = (correct_val / total_val) * 100
    
    scheduler.step(epoch_val_loss)
    
    print(f"Epoch [{epoch}/{EPOCHS}] - Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}% | Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%")
    
    # Save the best model
    if epoch_val_acc > best_val_acc:
        best_val_acc = epoch_val_acc
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"   🏆 New best validation accuracy: {best_val_acc:.2f}%. Model saved to {MODEL_SAVE_PATH}!")

print(f"🎉 Training complete! Best validation accuracy: {best_val_acc:.2f}%")
