import torch

# Path to your model
checkpoint_path = "bangla_resnet18 (1).pth"

# Load the checkpoint
checkpoint = torch.load(checkpoint_path, map_location="cpu")

# Check what's inside
print("Keys in checkpoint:", checkpoint.keys())

# If 'classes' is saved in the checkpoint
if 'classes' in checkpoint:
    classes = checkpoint['classes']
    print("Classes found in checkpoint:")
    print(classes)
else:
    print("No 'classes' found in checkpoint. You'll need the dataset to get them.")