"""
Standalone Real-Time Live Webcam Application.

Run with:
    python3 -m src.facial_analysis.live_cam

Hotkeys:
    [q] or [ESC] : Quit live stream
    [r]         : Register / Enroll currently detected face with your name
    [s]         : Capture snapshot & run Multimodal Sentiment consistency check
"""

import sys
import os
import cv2
import time
import numpy as np

# Ensure workspace is on sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, BASE_DIR)

from src.facial_analysis.live_processor import LiveVideoProcessor
from src.facial_analysis.face_recognizer import get_face_recognizer
from src.text_analysis.predict import predict_sentiment
from src.fusion.fusion_engine import fuse_predictions


def start_live_webcam(camera_index: int = 0):
    print("=" * 65)
    print("  🎥 STARTING LIVE REAL-TIME FACIAL & EXPRESSION RECOGNITION")
    print("=" * 65)
    print("  Controls:")
    print("    [q] or [ESC] : Exit camera window")
    print("    [r]         : Enroll your face with your name")
    print("    [s]         : Run live Multimodal check (Text + Live Face)")
    print("=" * 65)

    processor = LiveVideoProcessor(smoothing_alpha=0.35)
    recognizer = get_face_recognizer()

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[ERROR] Could not open camera {camera_index}. Trying fallback camera 1...")
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            print("[ERROR] No webcam device available. Please check macOS Camera permissions.")
            return

    # Set frame dimensions
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    fps_count = 0
    fps_start_time = time.time()
    current_fps = 0.0

    last_metadata = {}

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[WARN] Blank frame received. Retrying...")
                time.sleep(0.05)
                continue

            # Mirror frame horizontally for natural selfie view
            frame = cv2.flip(frame, 1)

            # Process frame
            annotated_frame, metadata = processor.process_frame(
                frame, draw_hud=True, show_emotion_chart=True
            )
            last_metadata = metadata

            # Calculate FPS
            fps_count += 1
            if time.time() - fps_start_time >= 1.0:
                current_fps = fps_count / (time.time() - fps_start_time)
                fps_count = 0
                fps_start_time = time.time()

            # Render FPS in top-left
            cv2.putText(
                annotated_frame, f"FPS: {current_fps:.1f}", (15, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA
            )

            # Display window
            cv2.imshow("Live Facial & Expression Recognition HUD", annotated_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:  # 'q' or ESC
                print("\n[INFO] Exiting live camera session.")
                break

            elif key == ord("r"):  # Register face
                if metadata.get("face_detected") and metadata.get("face_crop") is not None:
                    # Freeze frame to ask for name in console
                    print("\n" + "-" * 50)
                    print("👤 FACE ENROLLMENT / REGISTRATION")
                    user_name = input("Enter name for this face (e.g. Utkarsh): ").strip()
                    if user_name:
                        success = recognizer.enroll(user_name, metadata["face_crop"])
                        if success:
                            print(f"[✓] Successfully enrolled '{user_name}'!")
                        else:
                            print("[X] Failed to extract facial signature.")
                    print("-" * 50)
                else:
                    print("\n[WARN] No face clearly detected in frame. Please look directly at the camera.")

            elif key == ord("s"):  # Snapshot multimodal check
                if metadata.get("face_detected"):
                    print("\n" + "-" * 50)
                    print("🔀 LIVE MULTIMODAL SENTIMENT CHECK")
                    user_text = input("Enter sentence to compare with your live face: ").strip()
                    if user_text:
                        text_res = predict_sentiment(user_text)
                        # Adapt metadata to format expected by fusion
                        facial_res = {
                            "expression": metadata["expression"].lower(),
                            "confidence": metadata["confidence"],
                            "probabilities": metadata["probabilities"],
                            "face_detected": True,
                            "detector_used": "live_webcam",
                        }
                        fused = fuse_predictions(text_res, facial_res)
                        print(f"\nText Sentiment:     {text_res['sentiment'].upper()} ({text_res['confidence']*100:.1f}%)")
                        print(f"Live Face Expression: {metadata['expression'].upper()} ({metadata['confidence']*100:.1f}%)")
                        print(f"Identity:             {metadata['identity']}")
                        print(f"Multimodal Alignment: {fused['comparison'].upper()}")
                        print(f"Summary:              {fused.get('comparison_summary')}")
                        print(f"Details:              {fused.get('details')}")
                    print("-" * 50)

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    start_live_webcam()

