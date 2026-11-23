# Multimodal Sentiment Analysis with Live Facial Recognition 🧠👁️🎥

An end-to-end AI/ML system combining **Natural Language Processing (NLP)**, **Computer Vision (CV)**, and **Multimodal Late Fusion** to analyze written text sentiment alongside **real-time live facial expression tracking** and **face identity recognition**.

---

## 🌟 Key Features

1. **Real-Time Live Facial Recognition & Expression Tracking**:
   - Accesses live webcam stream at 30-60 FPS.
   - Detects faces with smooth temporal exponential moving average (EMA) bounding box tracking.
   - Real-time classification across 7 FER-2013 expressions: `Happy`, `Sad`, `Angry`, `Surprise`, `Fear`, `Disgust`, `Neutral`.
   - Live cyber HUD overlay with dynamic confidence meters and emotion probability bar charts.

2. **Facial Identity Recognition**:
   - Allows users to register/enroll their facial signature with their name (e.g. *Utkarsh*).
   - Normalized spatial LBP & pyramid feature matching with cosine similarity.
   - Recognizes enrolled users vs unregistered guests in real time.

3. **Multimodal Late Fusion**:
   - Compares live facial expressions with written text sentiment in real time.
   - Evaluates congruence (`CONGRUENT`, `DISCREPANT`, `CONTEXT-DEPENDENT`).
   - Responsible AI safeguards strictly embedded.

---

## 📁 Repository Structure

```
Multimodal_Sentiment_Analysis/
├── datasets/
│   ├── text_reviews.csv              # Balanced sentiment training dataset
│   └── facial_expressions/           # FER-2013 dataset directory
├── models/
│   ├── text_model.pkl                # Trained Logistic Regression model
│   ├── tfidf_vectorizer.pkl          # Fitted TF-IDF vectorizer (unigrams + bigrams)
│   ├── text_model_metrics.json       # Test set evaluation metrics
│   ├── enrolled_faces.json           # Face identity recognition database
│   └── expression_model.keras        # Deep CNN weights for FER
├── src/
│   ├── text_analysis/
│   │   ├── preprocessing.py          # Contraction expansion, cleaning, negation tagging
│   │   ├── train.py                  # Training pipeline & evaluation (77.8% accuracy)
│   │   └── predict.py                # Text inference pipeline
│   ├── facial_analysis/
│   │   ├── face_detection.py         # MediaPipe / OpenCV Haar cascade face detector
│   │   ├── preprocessing.py          # Grayscale, CLAHE, 48x48 resize, [0, 1] normalization
│   │   ├── train_cnn.py              # FER-2013 CNN architecture (3 Conv Blocks + Dense Head)
│   │   ├── predict.py                # Facial expression inference pipeline
│   │   ├── face_recognizer.py        # Identity enrollment & cosine similarity recognition
│   │   ├── live_processor.py         # Real-time stream processor with EMA temporal smoothing & HUD
│   │   └── live_cam.py               # Standalone 60 FPS OpenCV desktop live camera app
│   └── fusion/
│       └── fusion_engine.py          # Late fusion & congruence evaluation engine
├── backend/
│   └── main.py                       # FastAPI REST API (/analyze/..., /recognize/...)
├── frontend/
│   └── app.py                        # Streamlit Interactive Dashboard
├── notebooks/
│   ├── text_exploration.ipynb        # Phase 1 exploratory data analysis & training
│   └── facial_exploration.ipynb      # Phase 2 FER-2013 & CNN architecture exploration
├── requirements.txt                  # Python dependencies
└── README.md                         # Project documentation
```

---

## 🚀 Running the Live Systems

### Option 1: Interactive Browser Dashboard (Streamlit)
```bash
streamlit run frontend/app.py
```
Open **`http://localhost:8501`** and navigate to:
- **`🎥 Live Face & Expression Recognition`** in the left sidebar.
- Toggle **▶️ Activate Continuous Live Camera Feed** to see your live facial expressions and identity in real time!
- Switch to the **👤 Enroll Face Identity** tab to register your face by name.

### Option 2: High-Speed 60 FPS Native Desktop App (OpenCV)
```bash
python3 -m src.facial_analysis.live_cam
```
**Desktop Hotkeys:**
- Press <kbd>q</kbd> or <kbd>ESC</kbd>: Exit camera window
- Press <kbd>r</kbd>: Enroll/register currently detected face with your name
- Press <kbd>s</kbd>: Capture snapshot and run live Multimodal consistency check against typed text

### Option 3: FastAPI Backend Server
```bash
python3 -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```
Interactive docs at: `http://127.0.0.1:8000/docs`
