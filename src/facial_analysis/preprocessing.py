"""
Image Preprocessing for Facial Expression Recognition (FER).

FER-2013 Specification:
- Input: Face crop (BGR or Grayscale)
- Grayscale conversion
- Histogram equalization / CLAHE for illumination invariance
- Resize to 48 x 48 pixels
- Pixel normalization to [0.0, 1.0]
"""

import cv2
import numpy as np
from typing import Tuple, Optional


def preprocess_face_image(
    face_crop: np.ndarray,
    target_size: Tuple[int, int] = (48, 48),
    apply_clahe: bool = True,
) -> np.ndarray:
    """
    Transform a cropped facial image into FER-2013 model input tensor format.

    Args:
        face_crop: BGR or single-channel cropped face image
        target_size: Tuple (width, height), default (48, 48)
        apply_clahe: Whether to apply CLAHE to balance face illumination

    Returns:
        np.ndarray of shape (1, 48, 48, 1) normalized to [0.0, 1.0]
    """
    if face_crop is None or face_crop.size == 0:
        raise ValueError("Invalid or empty face crop provided for preprocessing.")

    # 1. Convert to grayscale if 3 channels
    if len(face_crop.shape) == 3 and face_crop.shape[2] == 3:
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    elif len(face_crop.shape) == 3 and face_crop.shape[2] == 1:
        gray = face_crop.squeeze(axis=-1)
    else:
        gray = face_crop.copy()

    # 2. Illumination normalization using CLAHE
    if apply_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        gray = clahe.apply(gray)

    # 3. Resize to 48x48
    resized = cv2.resize(gray, target_size, interpolation=cv2.INTER_AREA)

    # 4. Normalize pixel values to [0.0, 1.0]
    normalized = resized.astype("float32") / 255.0

    # 5. Expand dims to (1, 48, 48, 1)
    tensor = np.expand_dims(normalized, axis=(0, -1))
    return tensor


def visualize_preprocessed_face(face_tensor: np.ndarray) -> np.ndarray:
    """
    Convert (1, 48, 48, 1) normalized tensor back to displayable 8-bit image.
    """
    img = (face_tensor[0, :, :, 0] * 255.0).astype(np.uint8)
    return img

