"""
Facial Recognition & Identity Verification Module.

Extracts normalized spatial feature descriptors from face crops to identify registered users.
Features:
- Illumination-invariant LBP (Local Binary Patterns) + Multi-region Spatial Pyramid Descriptor
- Cosine similarity matching against enrolled face signatures
- Serialization to models/enrolled_faces.json
"""

import os
import json
# pyrefly: ignore[missing-import]
import cv2
# pyrefly: ignore[missing-import]
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ENROLLED_FACES_FILE = os.path.join(BASE_DIR, "models", "enrolled_faces.json")


def compute_lbp(image_gray: np.ndarray) -> np.ndarray:
    """Compute basic 8-neighbor Local Binary Pattern for a grayscale image."""
    h, w = image_gray.shape
    lbp = np.zeros((h - 2, w - 2), dtype=np.uint8)
    center = image_gray[1:-1, 1:-1]

    # 8 neighbors
    lbp |= (image_gray[0:-2, 0:-2] >= center).astype(np.uint8) << 7
    lbp |= (image_gray[0:-2, 1:-1] >= center).astype(np.uint8) << 6
    lbp |= (image_gray[0:-2, 2:] >= center).astype(np.uint8) << 5
    lbp |= (image_gray[1:-1, 2:] >= center).astype(np.uint8) << 4
    lbp |= (image_gray[2:, 2:] >= center).astype(np.uint8) << 3
    lbp |= (image_gray[2:, 1:-1] >= center).astype(np.uint8) << 2
    lbp |= (image_gray[2:, 0:-2] >= center).astype(np.uint8) << 1
    lbp |= (image_gray[1:-1, 0:-2] >= center).astype(np.uint8) << 0

    return lbp


def extract_face_descriptor(face_crop: np.ndarray, grid_size: Tuple[int, int] = (4, 4)) -> np.ndarray:
    """
    Extract a normalized 128-dimensional spatial LBP histogram descriptor from a face crop.
    """
    if face_crop is None or face_crop.size == 0:
        return np.zeros(128, dtype=np.float32)

    # Convert to grayscale
    if len(face_crop.shape) == 3:
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = face_crop.copy()

    # Normalize to 96x96
    resized = cv2.resize(gray, (96, 96), interpolation=cv2.INTER_AREA)

    # Illumination equalization via CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    equalized = clahe.apply(resized)

    # Compute LBP
    lbp = compute_lbp(equalized)

    # Divide into grid (e.g. 4x4 = 16 cells)
    gh, gw = grid_size
    cell_h = lbp.shape[0] // gh
    cell_w = lbp.shape[1] // gw

    histograms = []
    # 8-bin histogram per cell (16 cells * 8 bins = 128 dimensions)
    for r in range(gh):
        for c in range(gw):
            cell = lbp[r * cell_h : (r + 1) * cell_h, c * cell_w : (c + 1) * cell_w]
            hist, _ = np.histogram(cell, bins=8, range=(0, 256), density=True)
            histograms.extend(hist)

    descriptor = np.array(histograms, dtype=np.float32)
    # L2 normalize descriptor
    norm = np.linalg.norm(descriptor)
    if norm > 0:
        descriptor /= norm

    return descriptor


class FaceRecognizer:
    """
    Identity recognition engine that matches real-time faces against enrolled profiles.
    """

    def __init__(self, storage_path: str = ENROLLED_FACES_FILE, similarity_threshold: float = 0.82):
        self.storage_path = storage_path
        self.similarity_threshold = similarity_threshold
        self.profiles: Dict[str, List[float]] = {}
        self.load_profiles()

    def load_profiles(self) -> None:
        """Load enrolled profiles from JSON."""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    self.profiles = json.load(f)
            except Exception as e:
                print(f"[FaceRecognizer] Error loading profiles: {e}")
                self.profiles = {}
        else:
            self.profiles = {}

    def save_profiles(self) -> None:
        """Persist enrolled profiles to disk."""
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(self.profiles, f, indent=2)

    def enroll(self, name: str, face_crop: np.ndarray) -> bool:
        """Enroll a new identity with a face crop."""
        desc = extract_face_descriptor(face_crop)
        if np.all(desc == 0):
            return False
        clean_name = name.strip()
        self.profiles[clean_name] = desc.tolist()
        self.save_profiles()
        return True

    def identify(self, face_crop: np.ndarray) -> Tuple[str, float]:
        """
        Identify face against enrolled profiles using cosine similarity.
        Returns:
            Tuple (name, similarity_score)
            If no match exceeds threshold: ("Guest / Unknown", max_score)
        """
        if not self.profiles:
            return "Unregistered User", 0.0

        query_desc = extract_face_descriptor(face_crop)
        if np.all(query_desc == 0):
            return "Unknown", 0.0

        best_name = "Unknown"
        best_score = -1.0

        for name, enrolled_vec in self.profiles.items():
            enrolled_arr = np.array(enrolled_vec, dtype=np.float32)
            # Cosine similarity between unit vectors is dot product
            sim = float(np.dot(query_desc, enrolled_arr))
            if sim > best_score:
                best_score = sim
                best_name = name

        if best_score >= self.similarity_threshold:
            return best_name, round(best_score, 4)
        else:
            return f"Guest ({round(best_score*100):.0f}% match to {best_name})", round(best_score, 4)

    def delete_profile(self, name: str) -> bool:
        """Remove an enrolled identity."""
        if name in self.profiles:
            del self.profiles[name]
            self.save_profiles()
            return True
        return False

    def list_enrolled(self) -> List[str]:
        """List all currently enrolled names."""
        return list(self.profiles.keys())


# Global singleton instance
_global_face_recognizer: Optional[FaceRecognizer] = None


def get_face_recognizer() -> FaceRecognizer:
    global _global_face_recognizer
    if _global_face_recognizer is None:
        _global_face_recognizer = FaceRecognizer()
    return _global_face_recognizer

