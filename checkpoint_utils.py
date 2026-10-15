"""Validate the class map paired with a model instead of guessing label order."""
import json
from pathlib import Path


def validate_mapping(mapping, expected_classes):
    if not isinstance(mapping, dict) or len(mapping) != expected_classes:
        raise ValueError("Checkpoint needs a complete matching class-to-index map.")
    if any(not isinstance(k, str) or not k.isdigit() or not 1 <= int(k) <= expected_classes for k in mapping):
        raise ValueError("Class folders must be the expected numeric labels.")
    if any(type(v) is not int for v in mapping.values()) or set(mapping.values()) != set(range(expected_classes)):
        raise ValueError("Class indices must be unique and contiguous.")
    if len({int(k) for k in mapping}) != expected_classes:
        raise ValueError("Class folder numbers must be unique.")
    return {index: folder for folder, index in mapping.items()}


def decode_character(index, letters, index_to_folder):
    if index not in index_to_folder:
        raise ValueError("Prediction is outside the checkpoint class mapping.")
    letter_index = int(index_to_folder[index]) - 1
    if not 0 <= letter_index < len(letters):
        raise ValueError("Class map and character table disagree.")
    return letters[letter_index]


def load_checkpoint(model, model_path, mapping_path, expected_classes, device):
    import torch
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
        mapping = checkpoint.get("class_to_idx")
    else:
        state = checkpoint
        path = Path(mapping_path)
        if not path.is_file():
            raise ValueError("Legacy weights require the exact class_to_idx.json from their training run.")
        mapping = json.loads(path.read_text(encoding="utf-8"))
    index_to_folder = validate_mapping(mapping, expected_classes)
    model.load_state_dict(state)
    return index_to_folder
