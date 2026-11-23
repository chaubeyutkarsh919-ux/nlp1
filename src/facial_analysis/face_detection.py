"""
Face Detection and Landmark Extraction Module.

Integrates MediaPipe Face Landmarker / Face Detection with OpenCV Haar Cascade fallback.
Provides bounding box detection, landmark extraction, and face region cropping with safety margins.
"""

import os
import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Any, Union


class FaceDetector:
    """
    Robust Face Detector with dual backends:
    1. MediaPipe (primary, if available)
    2. OpenCV Haar Cascade (fallback, zero extra dependencies)
    """

    def __init__(self, confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold
        self.mediapipe_available = False
        self.mp_face_detection = None
        self.mp_detector = None
        self.haar_cascade = None

        # Try initializing MediaPipe
        try:
            import mediapipe as mp
            self.mp_face_detection = mp.solutions.face_detection
            self.mp_detector = self.mp_face_detection.FaceDetection(
                model_selection=1, min_detection_confidence=confidence_threshold
            )
            self.mediapipe_available = True
        except (ImportError, Exception):
            self.mediapipe_available = False

        # Load OpenCV Haar Cascade fallback
        cascade_path = os.path.join(
            cv2.data.haarcascades, "haarcascade_frontalface_default.xml"
        )
        if os.path.exists(cascade_path):
            self.haar_cascade = cv2.CascadeClassifier(cascade_path)

    def load_image(self, image_input: Union[str, bytes, np.ndarray]) -> np.ndarray:
        """Helper to load image from file path, raw bytes, or numpy array."""
        if isinstance(image_input, np.ndarray):
            return image_input
        elif isinstance(image_input, (bytes, bytearray)):
            nparr = np.frombuffer(image_input, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Could not decode image from provided bytes.")
            return img
        elif isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Image not found at path: {image_input}")
            img = cv2.imread(image_input)
            if img is None:
                raise ValueError(f"Could not read image from {image_input}")
            return img
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

    def detect_face(
        self,
        image_input: Union[str, bytes, np.ndarray],
        margin_ratio: float = 0.15,
    ) -> Dict[str, Any]:
        """
        Detect face in the provided image and extract cropped region.

        Args:
            image_input: File path, bytes, or numpy BGR image
            margin_ratio: Margin to expand bounding box around face (default: 0.15 = 15%)

        Returns:
            Dict containing:
            - face_detected: bool
            - bounding_box: [x, y, w, h] (in original image coordinates)
            - face_crop: np.ndarray BGR cropped face (or None)
            - detector_used: 'mediapipe' | 'opencv_haarcascade' | 'none'
            - original_shape: (height, width, channels)
        """
        img = self.load_image(image_input)
        h, w = img.shape[:2]

        # 1. Try MediaPipe if available
        if self.mediapipe_available and self.mp_detector:
            try:
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = self.mp_detector.process(rgb_img)
                if results.detections:
                    best_detection = max(
                        results.detections, key=lambda d: d.score[0]
                    )
                    bb = best_detection.location_data.relative_bounding_box
                    x = int(bb.xmin * w)
                    y = int(bb.ymin * h)
                    bw = int(bb.width * w)
                    bh = int(bb.height * h)

                    # Expand with margin
                    mx = int(bw * margin_ratio)
                    my = int(bh * margin_ratio)
                    x1 = max(0, x - mx)
                    y1 = max(0, y - my)
                    x2 = min(w, x + bw + mx)
                    y2 = min(h, y + bh + my)

                    crop = img[y1:y2, x1:x2]
                    return {
                        "face_detected": True,
                        "bounding_box": [x1, y1, x2 - x1, y2 - y1],
                        "face_crop": crop,
                        "detector_used": "mediapipe",
                        "original_shape": img.shape,
                        "detection_confidence": float(best_detection.score[0]),
                    }
            except Exception:
                pass  # Fallback to Haar Cascade

        # 2. OpenCV Haar Cascade Fallback
        if self.haar_cascade is not None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Equalize histogram for contrast invariance
            gray = cv2.equalizeHist(gray)
            faces = self.haar_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
            )

            if len(faces) > 0:
                # Pick the largest face (w * h)
                best_face = max(faces, key=lambda f: f[2] * f[3])
                x, y, bw, bh = best_face

                # Expand with margin
                mx = int(bw * margin_ratio)
                my = int(bh * margin_ratio)
                x1 = max(0, x - mx)
                y1 = max(0, y - my)
                x2 = min(w, x + bw + mx)
                y2 = min(h, y + bh + my)

                crop = img[y1:y2, x1:x2]
                return {
                    "face_detected": True,
                    "bounding_box": [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
                    "face_crop": crop,
                    "detector_used": "opencv_haarcascade",
                    "original_shape": img.shape,
                    "detection_confidence": 0.85,
                }

        # 3. No face detected
        return {
            "face_detected": False,
            "bounding_box": None,
            "face_crop": None,
            "detector_used": "none",
            "original_shape": img.shape,
            "detection_confidence": 0.0,
        }


# Global detector instance
_global_face_detector: Optional[FaceDetector] = None


def get_face_detector() -> FaceDetector:
    """Singleton getter for face detector."""
    global _global_face_detector
    if _global_face_detector is None:
        _global_face_detector = FaceDetector()
    return _global_face_detector


def detect_and_crop_face(
    image_input: Union[str, bytes, np.ndarray], margin_ratio: float = 0.15
) -> Dict[str, Any]:
    """Convenience function to detect and crop face."""
    detector = get_face_detector()
    return detector.detect_face(image_input, margin_ratio=margin_ratio)

