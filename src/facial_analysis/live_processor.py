"""
Real-Time Live Video Stream Processor for Face & Expression Recognition.

Provides:
- High-speed frame processing with temporal exponential smoothing
- Dual Face Recognition (Identity) + Facial Expression Recognition (Emotion)
- High-tech futuristic HUD rendering (corner reticles, status badges, live emotion bars)
"""

import cv2
import numpy as np
from typing import Dict, Any, Tuple, Optional

from src.facial_analysis.face_detection import get_face_detector
from src.facial_analysis.predict import get_facial_classifier
from src.facial_analysis.face_recognizer import get_face_recognizer
from src.facial_analysis.train_cnn import EMOTION_LABELS


class LiveVideoProcessor:
    """
    Stateful processor for continuous video streams.
    Applies exponential moving average (EMA) smoothing over emotion predictions
    to prevent jitter between frames.
    """

    def __init__(self, smoothing_alpha: float = 0.35):
        self.detector = get_face_detector()
        self.classifier = get_facial_classifier()
        self.recognizer = get_face_recognizer()
        self.smoothing_alpha = smoothing_alpha  # Weight of new frame vs history

        # Rolling state
        self.prev_probs: Optional[np.ndarray] = None
        self.prev_bbox: Optional[Tuple[int, int, int, int]] = None
        self.tracked_identity: str = "Scanning..."
        self.tracked_expression: str = "Neutral"

    def process_frame(
        self,
        frame: np.ndarray,
        draw_hud: bool = True,
        show_emotion_chart: bool = True,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Process a single live video frame:
        1. Detect face
        2. Recognize identity
        3. Classify expression with temporal smoothing
        4. Render HUD overlay

        Returns:
            Tuple of (annotated_frame, metadata_dict)
        """
        annotated = frame.copy()
        h, w = frame.shape[:2]

        det_result = self.detector.detect_face(frame, margin_ratio=0.18)

        metadata: Dict[str, Any] = {
            "face_detected": False,
            "identity": "No Face Detected",
            "expression": "None",
            "confidence": 0.0,
            "probabilities": {},
            "bounding_box": None,
        }

        if not det_result["face_detected"]:
            # Decay previous state
            if draw_hud:
                self._draw_no_face_hud(annotated)
            return annotated, metadata

        # Face detected
        bbox = det_result["bounding_box"]
        x, y, bw, bh = bbox
        face_crop = det_result["face_crop"]

        # Smooth bounding box
        if self.prev_bbox is not None:
            px, py, pbw, pbh = self.prev_bbox
            x = int(0.7 * px + 0.3 * x)
            y = int(0.7 * py + 0.3 * y)
            bw = int(0.7 * pbw + 0.3 * bw)
            bh = int(0.7 * pbh + 0.3 * bh)
        self.prev_bbox = (x, y, bw, bh)

        # 1. Identity Recognition
        identity_name, id_conf = self.recognizer.identify(face_crop)
        self.tracked_identity = identity_name

        # 2. Expression Classification
        pred_dict = self.classifier.predict(face_crop)
        raw_probs = np.array(
            [pred_dict["probabilities"][label.lower()] for label in EMOTION_LABELS],
            dtype=np.float32,
        )

        # Apply temporal smoothing
        if self.prev_probs is None:
            smoothed_probs = raw_probs
        else:
            smoothed_probs = (
                self.smoothing_alpha * raw_probs
                + (1.0 - self.smoothing_alpha) * self.prev_probs
            )
        self.prev_probs = smoothed_probs

        best_idx = int(np.argmax(smoothed_probs))
        expr_label = EMOTION_LABELS[best_idx]
        expr_conf = float(smoothed_probs[best_idx])
        self.tracked_expression = expr_label

        prob_map = {
            EMOTION_LABELS[i].lower(): round(float(smoothed_probs[i]), 3)
            for i in range(len(EMOTION_LABELS))
        }

        metadata = {
            "face_detected": True,
            "identity": identity_name,
            "identity_confidence": id_conf,
            "expression": expr_label,
            "confidence": expr_conf,
            "probabilities": prob_map,
            "bounding_box": [x, y, bw, bh],
            "face_crop": face_crop,
        }

        if draw_hud:
            self._draw_live_hud(
                annotated, x, y, bw, bh, identity_name, expr_label, expr_conf, smoothed_probs, show_emotion_chart
            )

        return annotated, metadata

    def _draw_live_hud(
        self,
        img: np.ndarray,
        x: int,
        y: int,
        bw: int,
        bh: int,
        identity: str,
        expression: str,
        confidence: float,
        probs: np.ndarray,
        show_chart: bool,
    ) -> None:
        """Render high-tech cyber HUD on live camera feed."""
        h, w = img.shape[:2]

        # Expression color palette
        color_map = {
            "Happy": (34, 197, 94),     # Neon Green
            "Sad": (239, 68, 68),       # Red
            "Angry": (185, 28, 28),     # Dark Red
            "Surprise": (245, 158, 11), # Amber / Orange
            "Fear": (147, 51, 234),     # Purple
            "Disgust": (168, 85, 247),  # Magenta
            "Neutral": (14, 165, 233),  # Cyan
        }
        accent_color = color_map.get(expression, (14, 165, 233))

        # 1. Bounding box with corner reticles
        corner_len = min(24, int(bw * 0.2))
        thickness = 2
        # Main faint rectangle
        cv2.rectangle(img, (x, y), (x + bw, y + bh), (60, 60, 60), 1)

        # 4 Accent Corners
        # Top-Left
        cv2.line(img, (x, y), (x + corner_len, y), accent_color, thickness + 1)
        cv2.line(img, (x, y), (x, y + corner_len), accent_color, thickness + 1)
        # Top-Right
        cv2.line(img, (x + bw, y), (x + bw - corner_len, y), accent_color, thickness + 1)
        cv2.line(img, (x + bw, y), (x + bw, y + corner_len), accent_color, thickness + 1)
        # Bottom-Left
        cv2.line(img, (x, y + bh), (x + corner_len, y + bh), accent_color, thickness + 1)
        cv2.line(img, (x, y + bh), (x, y + bh - corner_len), accent_color, thickness + 1)
        # Bottom-Right
        cv2.line(img, (x + bw, y + bh), (x + bw - corner_len, y + bh), accent_color, thickness + 1)
        cv2.line(img, (x + bw, y + bh), (x + bw, y + bh - corner_len), accent_color, thickness + 1)

        # 2. Identity Header Tag
        id_text = f"ID: {identity}"
        cv2.rectangle(img, (x, max(0, y - 26)), (x + len(id_text) * 11 + 10, max(0, y)), (20, 20, 20), -1)
        cv2.putText(
            img, id_text, (x + 5, max(15, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA
        )

        # 3. Expression Footer Tag
        expr_text = f"{expression.upper()} ({confidence * 100:.0f}%)"
        tag_y = min(h - 5, y + bh + 24)
        cv2.rectangle(img, (x, y + bh), (x + len(expr_text) * 11 + 12, tag_y), accent_color, -1)
        cv2.putText(
            img, expr_text, (x + 6, tag_y - 7),
            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 2, cv2.LINE_AA
        )

        # 4. Live Emotion Probability Bars (Side Panel)
        if show_chart and w >= 480:
            panel_x = w - 170
            panel_y = 20
            panel_w = 155
            panel_h = len(EMOTION_LABELS) * 24 + 30

            # Semi-transparent dark overlay for HUD panel
            overlay = img.copy()
            cv2.rectangle(overlay, (panel_x - 10, panel_y - 15), (panel_x + panel_w, panel_y + panel_h), (15, 23, 42), -1)
            cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)

            cv2.putText(
                img, "LIVE EMOTIONS", (panel_x, panel_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA
            )

            for i, label in enumerate(EMOTION_LABELS):
                bar_y = panel_y + 22 + i * 22
                prob_val = float(probs[i])
                bar_width = int(prob_val * 75)

                is_top = (label == expression)
                bar_color = accent_color if is_top else (100, 116, 139)

                # Label
                cv2.putText(
                    img, label[:4], (panel_x, bar_y + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255) if is_top else (160, 160, 160), 1
                )
                # Background track
                cv2.rectangle(img, (panel_x + 35, bar_y + 2), (panel_x + 115, bar_y + 11), (40, 50, 70), -1)
                # Active Bar
                if bar_width > 0:
                    cv2.rectangle(img, (panel_x + 35, bar_y + 2), (panel_x + 35 + bar_width, bar_y + 11), bar_color, -1)
                # Percentage
                cv2.putText(
                    img, f"{int(prob_val*100)}%", (panel_x + 120, bar_y + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, (200, 200, 200), 1
                )

    def _draw_no_face_hud(self, img: np.ndarray) -> None:
        """Draw HUD message when no face is in view."""
        h, w = img.shape[:2]
        msg = "SCANNING FOR FACE..."
        cv2.putText(
            img, msg, (int(w / 2) - 100, int(h / 2)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 180, 255), 2, cv2.LINE_AA
        )

