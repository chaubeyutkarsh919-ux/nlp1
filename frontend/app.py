"""
Interactive Streamlit Dashboard for Multimodal Sentiment Analysis & Live Facial Recognition.

Features:
- Multimodal Sentiment & Expression Analyzer (Late Fusion)
- Live Real-Time Camera Stream with Face Recognition & Expression HUD
- Face Enrollment / Identity Registration
- Text Sentiment Playground with Negation Handling
- Facial Expression Recognition Studio
- Model Performance Metrics & Responsible AI Framework
"""

import sys
import os
import json
import time
# pyrefly: ignore[missing-import]
import cv2
# pyrefly: ignore[missing-import]
import numpy as np
# pyrefly: ignore[missing-import]
from PIL import Image
# pyrefly: ignore[missing-import]
import streamlit as st

# Add project root to sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)

from src.text_analysis.predict import predict_sentiment
from src.facial_analysis.predict import predict_expression
from src.facial_analysis.preprocessing import preprocess_face_image
from src.facial_analysis.face_recognizer import get_face_recognizer
from src.facial_analysis.live_processor import LiveVideoProcessor
from src.facial_analysis.train_cnn import EMOTION_LABELS
from src.fusion.fusion_engine import fuse_predictions, RESPONSIBLE_AI_DISCLAIMER

st.set_page_config(
    page_title="Multimodal Sentiment Analysis & Live Face Recognition",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        background: linear-gradient(90deg, #1E88E5, #7E57C2, #00E676);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .subtitle {
        color: #666;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .badge-consistent {
        background-color: #e8f5e9;
        color: #2e7d32;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
    }
    .badge-discrepant {
        background-color: #ffebee;
        color: #c62828;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
    }
    .badge-neutral {
        background-color: #ede7f6;
        color: #512da8;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
    }
    .disclaimer-box {
        background-color: #fff8e1;
        border-left: 4px solid #ffb300;
        padding: 0.9rem;
        border-radius: 4px;
        font-size: 0.88rem;
        color: #5d4037;
        margin-top: 1rem;
    }
    .live-hud-card {
        background-color: #0f172a;
        color: #f8fafc;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #1e293b;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar Navigation
st.sidebar.title("🧠 Navigation")
app_mode = st.sidebar.radio(
    "Select Mode",
    [
        "🚀 Multimodal Analyzer",
        "🎥 Live Face & Expression Recognition",
        "📝 Text Sentiment Playground",
        "👤 Facial Expression Studio",
        "📊 Model Metrics & Ethics",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🛡️ Privacy & Ethics")
st.sidebar.info(
    "**Voluntary Input Only**: Facial processing runs purely locally in-memory. "
    "Expressions are physical cues, not psychological certainty."
)


def draw_bounding_box(image_np: np.ndarray, bbox: list) -> np.ndarray:
    """Draw bounding box on face image."""
    img_copy = image_np.copy()
    if bbox:
        x, y, w, h = bbox
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 3)
    return img_copy


# ==============================================================================
# MODE 1: MULTIMODAL ANALYZER
# ==============================================================================
if app_mode == "🚀 Multimodal Analyzer":
    st.markdown('<div class="main-title">Multimodal Sentiment & Expression Analyzer</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Combining NLP Text Sentiment and Computer Vision Facial Expression Recognition</div>', unsafe_allow_html=True)

    col_input_left, col_input_right = st.columns([1.1, 0.9])

    with col_input_left:
        st.markdown("#### 1. Text Input")
        sample_texts = [
            "I am having a wonderful day!",
            "A terrible waste of time and money, completely frustrating.",
            "This is not a bad film; in fact, I enjoyed it quite a lot!",
            "I'm feeling calm and ready for the week.",
        ]
        chosen_sample = st.selectbox("Quick-fill sample sentence (optional):", ["-- Choose a sample or type below --"] + sample_texts)
        default_val = "" if chosen_sample.startswith("--") else chosen_sample
        user_text = st.text_area("Enter your thoughts, review, or statement:", value=default_val, height=120, placeholder="e.g. Today was such a productive and joyful day!")

    with col_input_right:
        st.markdown("#### 2. Voluntary Facial Input")
        use_face = st.checkbox("Enable Voluntary Facial Input", value=True)
        img_bytes = None
        img_np = None

        if use_face:
            input_source = st.radio("Input Source:", ["Upload Image File", "Camera Snapshot"], horizontal=True)
            if input_source == "Upload Image File":
                uploaded_file = st.file_uploader("Upload a face image (JPG, PNG)", type=["jpg", "jpeg", "png"])
                if uploaded_file is not None:
                    img_bytes = uploaded_file.getvalue()
                    pil_img = Image.open(uploaded_file)
                    img_np = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                    st.image(uploaded_file, caption="Uploaded Image", width=220)
            else:
                cam_pic = st.camera_input("Take a photo")
                if cam_pic is not None:
                    img_bytes = cam_pic.getvalue()
                    pil_img = Image.open(cam_pic)
                    img_np = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    st.markdown("---")
    analyze_btn = st.button("🔍 Run Multimodal Analysis", type="primary", use_container_width=True)

    if analyze_btn:
        if not user_text.strip():
            st.warning("Please enter some text in the text input box to begin.")
        else:
            with st.spinner("Processing modalities..."):
                # 1. Text Analysis
                text_res = predict_sentiment(user_text)

                # 2. Facial Analysis
                face_res = None
                if use_face and img_bytes is not None:
                    face_res = predict_expression(img_bytes)

                # 3. Multimodal Fusion
                fused = fuse_predictions(text_res, face_res)

            st.markdown("### 📋 Analysis Results")
            res_col1, res_col2, res_col3 = st.columns([1, 1, 1.2])

            # Text Card
            with res_col1:
                st.markdown("#### 📝 Text Sentiment")
                sent = text_res["sentiment"].capitalize()
                conf = text_res["confidence"]
                color = "green" if sent == "Positive" else ("red" if sent == "Negative" else "gray")
                st.markdown(f"**Predicted Sentiment:** <span style='font-size:1.3rem; color:{color}; font-weight:bold;'>{sent}</span>", unsafe_allow_html=True)
                st.progress(conf)
                st.write(f"**Confidence:** `{conf * 100:.1f}%`")
                st.write("**Probabilities:**")
                st.json(text_res["probabilities"])

            # Facial Card
            with res_col2:
                st.markdown("#### 👤 Facial Expression")
                if face_res and face_res.get("face_detected"):
                    expr = face_res["expression"].capitalize()
                    fconf = face_res["confidence"]
                    st.markdown(f"**Visible Expression:** <span style='font-size:1.3rem; color:#1E88E5; font-weight:bold;'>{expr}</span>", unsafe_allow_html=True)
                    st.progress(fconf)
                    st.write(f"**Confidence:** `{fconf * 100:.1f}%`")
                    st.write(f"**Detector Used:** `{face_res.get('detector_used', 'N/A')}`")

                    # Show bounding box on face
                    if img_np is not None and face_res.get("bounding_box"):
                        annotated = draw_bounding_box(img_np, face_res["bounding_box"])
                        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Detected Face Region", width=180)
                elif use_face and img_bytes is not None:
                    st.warning("⚠️ No face was detected in the provided image.")
                else:
                    st.info("ℹ️ Facial modality disabled (Text-Only Mode).")

            # Fusion Card
            with res_col3:
                st.markdown("#### 🔀 Multimodal Fusion")
                comp_status = fused.get("comparison", "text_only")

                if comp_status == "consistent":
                    badge = '<span class="badge-consistent">✓ CONSISTENT</span>'
                elif comp_status == "discrepant":
                    badge = '<span class="badge-discrepant">⚠️ DISCREPANT</span>'
                elif comp_status == "context_dependent":
                    badge = '<span class="badge-neutral">ℹ️ CONTEXT-DEPENDENT</span>'
                else:
                    badge = '<span class="badge-neutral">📝 TEXT ONLY</span>'

                st.markdown(f"**Consistency Assessment:** {badge}", unsafe_allow_html=True)
                st.markdown(f"**Summary:** {fused.get('comparison_summary', '')}")
                st.write(f"**Details:** {fused.get('details', '')}")
                if "alignment_score" in fused:
                    st.write(f"**Alignment Index:** `{fused['alignment_score'] * 100:.1f}%`")

                st.markdown(f'<div class="disclaimer-box">{RESPONSIBLE_AI_DISCLAIMER}</div>', unsafe_allow_html=True)


# ==============================================================================
# MODE 2: LIVE FACE & EXPRESSION RECOGNITION (NEW!)
# ==============================================================================
elif app_mode == "🎥 Live Face & Expression Recognition":
    st.markdown('<div class="main-title">Live Facial Recognition & Expression Stream 🎥</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Access your live facial expression in real time, identify registered faces, and evaluate live emotional congruence.</div>',
        unsafe_allow_html=True,
    )

    recognizer = get_face_recognizer()

    # Tabs for Live Stream vs Face Enrollment
    tab_stream, tab_enroll, tab_desktop = st.tabs([
        "📹 Live Webcam Feed", "👤 Enroll Face Identity", "🖥️ Standalone 60 FPS Desktop App"
    ])

    with tab_stream:
        st.markdown("### Real-Time Live Camera Feed")

        col_live_cfg1, col_live_cfg2 = st.columns([1.2, 0.8])
        with col_live_cfg1:
            live_sentence = st.text_input(
                "💬 Optional live text to compare against your facial expression in real-time:",
                value="I am having a wonderful day!",
            )
        with col_live_cfg2:
            smoothing_val = st.slider("Temporal Smoothing (EMA Alpha):", 0.1, 0.9, 0.35, 0.05, help="Higher values react faster; lower values stabilize predictions.")

        run_live = st.toggle("▶️ Activate Continuous Live Camera Feed", value=False)

        col_video, col_hud = st.columns([1.2, 0.8])
        video_placeholder = col_video.empty()
        hud_placeholder = col_hud.empty()

        if run_live:
            processor = LiveVideoProcessor(smoothing_alpha=smoothing_val)
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                cap = cv2.VideoCapture(1)

            if not cap.isOpened():
                st.error("❌ Could not connect to local webcam. Please check macOS camera permissions or use Camera Snapshot below.")
            else:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

                # Process frames in loop until toggled off
                frame_count = 0
                try:
                    while run_live:
                        ret, frame = cap.read()
                        if not ret or frame is None:
                            time.sleep(0.04)
                            continue

                        # Mirror frame
                        frame = cv2.flip(frame, 1)

                        # Process with live HUD
                        annotated, meta = processor.process_frame(
                            frame, draw_hud=True, show_emotion_chart=True
                        )

                        # Render frame to Streamlit
                        rgb_frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                        video_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

                        # Render live metrics in HUD side panel
                        with hud_placeholder.container():
                            st.markdown("#### 📊 Live HUD Status")
                            if meta.get("face_detected"):
                                st.success(f"👤 **Identity:** {meta.get('identity', 'Unknown')}")
                                expr = meta.get('expression', 'Neutral')
                                conf = meta.get('confidence', 0.0)
                                st.metric("Live Expression", f"{expr.upper()}", f"{conf*100:.0f}% confidence")

                                # Compare with live text
                                if live_sentence.strip():
                                    t_res = predict_sentiment(live_sentence)
                                    f_res = {
                                        "expression": expr.lower(),
                                        "confidence": conf,
                                        "probabilities": meta.get("probabilities", {}),
                                        "face_detected": True,
                                    }
                                    fused = fuse_predictions(t_res, f_res)
                                    status = fused.get("comparison", "neutral")
                                    if status == "consistent":
                                        st.markdown('<span class="badge-consistent">✓ CONGRUENT</span> Text matches facial cues.', unsafe_allow_html=True)
                                    elif status == "discrepant":
                                        st.markdown('<span class="badge-discrepant">⚠️ DISCREPANT</span> Text disagrees with facial expression.', unsafe_allow_html=True)
                                    else:
                                        st.markdown('<span class="badge-neutral">ℹ️ CONTEXT-DEPENDENT</span>', unsafe_allow_html=True)

                                st.write("**Real-Time Probabilities:**")
                                st.bar_chart(meta.get("probabilities", {}))
                            else:
                                st.info("🔍 Searching for face in video stream...")

                        frame_count += 1
                        time.sleep(0.03)  # ~30 FPS loop rate
                finally:
                    cap.release()
        else:
            video_placeholder.info("Click '▶️ Activate Continuous Live Camera Feed' above, or capture a single frame below:")
            live_snapshot = st.camera_input("Quick Live Frame Capture")
            if live_snapshot is not None:
                img_bytes = live_snapshot.getvalue()
                pil_img = Image.open(live_snapshot)
                frame_np = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

                proc = LiveVideoProcessor()
                annotated, meta = proc.process_frame(frame_np, draw_hud=True, show_emotion_chart=True)

                col_s1, col_s2 = st.columns([1, 1])
                with col_s1:
                    st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Analyzed Live Snapshot", use_container_width=True)
                with col_s2:
                    st.markdown("#### 🎯 Snapshot Analysis")
                    if meta.get("face_detected"):
                        st.write(f"**Identity:** `{meta.get('identity')}`")
                        st.write(f"**Expression:** `{meta.get('expression')}` ({meta.get('confidence')*100:.1f}%)")
                        st.bar_chart(meta.get("probabilities", {}))
                    else:
                        st.warning("No face detected in snapshot.")

    with tab_enroll:
        st.markdown("### 👤 Enroll Face Identity")
        st.write("Register your face signature so the live recognition system can identify you by name in real-time.")

        enroll_col1, enroll_col2 = st.columns([1, 1])
        with enroll_col1:
            enroll_name = st.text_input("Enter your name (e.g. Utkarsh):")
            enroll_cam = st.camera_input("Capture Enrollment Photo")

            if st.button("💾 Enroll / Register Face", type="primary"):
                if not enroll_name.strip():
                    st.warning("Please provide a name before enrolling.")
                elif enroll_cam is None:
                    st.warning("Please take a photo using the camera above.")
                else:
                    pil_img = Image.open(enroll_cam)
                    frame_np = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                    from src.facial_analysis.face_detection import get_face_detector
                    det = get_face_detector().detect_face(frame_np)

                    if det.get("face_detected") and det.get("face_crop") is not None:
                        ok = recognizer.enroll(enroll_name.strip(), det["face_crop"])
                        if ok:
                            st.success(f"🎉 Successfully enrolled '{enroll_name.strip()}' into facial recognition database!")
                        else:
                            st.error("Failed to compute descriptor from face.")
                    else:
                        st.error("No face detected in the photo. Please look directly into the camera.")

        with enroll_col2:
            st.markdown("#### Currently Enrolled Profiles")
            enrolled_list = recognizer.list_enrolled()
            if enrolled_list:
                for name in enrolled_list:
                    st.write(f"• **{name}**")
                if st.button("Clear All Profiles"):
                    for name in enrolled_list:
                        recognizer.delete_profile(name)
                    st.rerun()
            else:
                st.info("No face profiles enrolled yet. Capture your photo to enroll your profile.")

    with tab_desktop:
        st.markdown("### 🖥️ Native 60 FPS Desktop HUD Window")
        st.write("For ultra-smooth, zero-latency live facial and expression recognition, launch the native OpenCV desktop application:")

        st.code("python3 -m src.facial_analysis.live_cam", language="bash")

        st.markdown(
            """
            **Desktop Hotkeys:**
            - <kbd>q</kbd> or <kbd>ESC</kbd>: Exit camera window cleanly
            - <kbd>r</kbd>: Register / enroll your face directly from the live feed
            - <kbd>s</kbd>: Capture live snapshot & run text sentiment consistency comparison
            """
        )


# ==============================================================================
# MODE 3: TEXT SENTIMENT PLAYGROUND
# ==============================================================================
elif app_mode == "📝 Text Sentiment Playground":
    st.markdown('<div class="main-title">Text Sentiment Explorer (Phase 1)</div>', unsafe_allow_html=True)
    st.markdown("Inspect tokenization, negation handling, TF-IDF representations, and Logistic Regression decision boundaries.")

    test_text = st.text_area("Input sentence:", value="This is not a bad film; in fact, I enjoyed it quite a lot!", height=100)

    if st.button("Inspect Text Pipeline", type="primary"):
        res = predict_sentiment(test_text)
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("##### Preprocessed & Negation-Linked Tokens")
            st.code(res["cleaned_text"], language="text")
            st.markdown("##### Predicted Classification")
            st.metric("Sentiment", res["sentiment"].upper(), f"{res['confidence']*100:.1f}% Confidence")

        with col2:
            st.markdown("##### Class Probability Distribution")
            st.bar_chart(res["probabilities"])


# ==============================================================================
# MODE 4: FACIAL EXPRESSION STUDIO
# ==============================================================================
elif app_mode == "👤 Facial Expression Studio":
    st.markdown('<div class="main-title">Facial Expression Recognition Studio (Phase 2 & 3)</div>', unsafe_allow_html=True)
    st.markdown("Test face detection, landmark bounding boxes, 48x48 FER normalization, and expression classification.")

    uploaded_img = st.file_uploader("Upload Face Image for Expression Analysis:", type=["jpg", "jpeg", "png"])
    if uploaded_img is not None:
        raw_bytes = uploaded_img.getvalue()
        pil_img = Image.open(uploaded_img)
        img_np = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        res = predict_expression(raw_bytes)

        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("##### Original Image with Face Bounding Box")
            if res.get("face_detected") and res.get("bounding_box"):
                annotated = draw_bounding_box(img_np, res["bounding_box"])
                st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
            else:
                st.image(pil_img, use_container_width=True)
                st.warning("No face detected.")

        with col2:
            st.markdown("##### FER-2013 7-Class Probability Distribution")
            if res.get("face_detected"):
                st.bar_chart(res["probabilities"])
                st.write(f"**Top Expression:** `{res['expression'].capitalize()}` (`{res['confidence']*100:.1f}%`)")
                st.write(f"**Detector Used:** `{res['detector_used']}`")


# ==============================================================================
# MODE 5: MODEL METRICS & ETHICS
# ==============================================================================
elif app_mode == "📊 Model Metrics & Ethics":
    st.markdown('<div class="main-title">Model Metrics & Responsible AI Framework</div>', unsafe_allow_html=True)

    metrics_path = os.path.join(BASE_DIR, "models", "text_model_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)

        st.markdown("### 📈 Text Sentiment Model Performance (Phase 1)")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{metrics['evaluation']['accuracy']*100:.2f}%")
        col2.metric("Precision", f"{metrics['evaluation']['precision']*100:.2f}%")
        col3.metric("Recall", f"{metrics['evaluation']['recall']*100:.2f}%")
        col4.metric("F1-Score", f"{metrics['evaluation']['f1_score']*100:.2f}%")

        col_feat1, col_feat2 = st.columns(2)
        with col_feat1:
            st.markdown("##### Top Positive Words (TF-IDF Weights)")
            st.table(metrics["top_features"]["top_positive"])

        with col_feat2:
            st.markdown("##### Top Negative Words (TF-IDF Weights)")
            st.table(metrics["top_features"]["top_negative"])

    st.markdown("---")
    st.markdown("### 🛡️ Responsible AI Principles")
    st.markdown(
        """
        1. **Facial Expressions ≠ Psychological Truth**: Visible facial patterns (smiles, frowns) reflect muscular movements,
           not absolute proof of internal mood. Culture, context, social masking, and neurodiversity significantly alter expression.
        2. **Explicit Consent & Opt-In**: Facial analysis is strictly voluntary. Users can run the system in pure text-only mode anytime.
        3. **Zero Storage & In-Memory Processing**: Uploaded images and video frames are processed ephemeral in RAM and discarded immediately.
        4. **Separation of Results**: Modalities are maintained as distinct branches, avoiding arbitrary blending or misleading generalizations.
        """
    )
