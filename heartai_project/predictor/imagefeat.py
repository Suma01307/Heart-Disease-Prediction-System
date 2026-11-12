"""Image feature extraction shared by training (ml_training/train_image.py) and inference."""
import numpy as np
from PIL import Image

IMG_SIZE = 64


def extract_features(file_or_path) -> np.ndarray:
    """Grayscale -> 64x64 pixels (scaled 0-1) + 32-bin intensity histogram + edge-energy summary."""
    img = Image.open(file_or_path).convert("L").resize((IMG_SIZE, IMG_SIZE))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    hist, _ = np.histogram(arr, bins=32, range=(0, 1), density=True)
    gx = np.abs(np.diff(arr, axis=1)).mean()
    gy = np.abs(np.diff(arr, axis=0)).mean()
    return np.concatenate([arr.ravel(), hist, [gx, gy, arr.mean(), arr.std()]]).astype(np.float32)
