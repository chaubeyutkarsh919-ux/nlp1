"""
Training Pipeline for Text Sentiment Analysis.

Pipeline:
1. Load dataset (CSV)
2. Clean & Preprocess reviews
3. Train/Test Stratified Split (80/20)
4. Fit TF-IDF Vectorizer (unigrams + bigrams)
5. Train Logistic Regression Classifier
6. Comprehensive Evaluation (Accuracy, Precision, Recall, F1, Confusion Matrix)
7. Serialize artifacts into models/
"""

import os
import json
# pyrefly: ignore[missing-import]
import joblib
import pandas as pd
# pyrefly: ignore[missing-import]
import numpy as np
from typing import Tuple, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from src.text_analysis.preprocessing import preprocess_text

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATASET_PATH = os.path.join(BASE_DIR, "datasets", "text_reviews.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_OUT_PATH = os.path.join(MODELS_DIR, "text_model.pkl")
VECTORIZER_OUT_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
METRICS_OUT_PATH = os.path.join(MODELS_DIR, "text_model_metrics.json")


def load_dataset(filepath: str = DATASET_PATH) -> pd.DataFrame:
    """Load sentiment dataset from CSV."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Dataset not found at {filepath}. Please ensure datasets/text_reviews.csv exists."
        )
    df = pd.read_csv(filepath)
    df = df.dropna(subset=["review", "sentiment"])
    df["sentiment"] = df["sentiment"].astype(str).str.strip().str.lower()
    # Normalize labels to binary {0: negative, 1: positive}
    label_map = {"positive": 1, "pos": 1, "negative": 0, "neg": 0}
    df["label"] = df["sentiment"].map(label_map)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    return df


def train_text_sentiment_model(
    dataset_path: str = DATASET_PATH,
    test_size: float = 0.20,
    random_state: int = 42,
    max_features: int = 5000,
    ngram_range: Tuple[int, int] = (1, 2),
    C: float = 2.5,
) -> Dict[str, Any]:
    """Train and evaluate the TF-IDF + Logistic Regression model."""
    print("=" * 60)
    print("  PHASE 1: TRAINING TEXT SENTIMENT ANALYSIS MODEL")
    print("=" * 60)

    os.makedirs(MODELS_DIR, exist_ok=True)

    # 1. Load Data
    print(f"\n[1/5] Loading dataset from: {dataset_path}")
    df = load_dataset(dataset_path)
    total_samples = len(df)
    pos_count = int((df["label"] == 1).sum())
    neg_count = int((df["label"] == 0).sum())
    print(f"      Total samples: {total_samples}")
    print(f"      Class balance: Positive={pos_count} ({pos_count/total_samples*100:.1f}%), Negative={neg_count} ({neg_count/total_samples*100:.1f}%)")

    # 2. Preprocess Text
    print("\n[2/5] Preprocessing text reviews (cleaning, contractions, stopwords)...")
    df["clean_review"] = df["review"].apply(preprocess_text)

    # 3. Train/Test Split
    print(f"\n[3/5] Splitting into train/test sets ({int((1-test_size)*100)}/{int(test_size*100)} split)...")
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        df["clean_review"].values,
        df["label"].values,
        test_size=test_size,
        random_state=random_state,
        stratify=df["label"].values,
    )
    print(f"      Train set: {len(X_train_raw)} samples")
    print(f"      Test set:  {len(X_test_raw)} samples")

    # 4. TF-IDF Vectorization
    print(f"\n[4/5] Fitting TF-IDF Vectorizer (ngram_range={ngram_range}, max_features={max_features})...")
    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        sublinear_tf=True,
        min_df=1,
    )
    X_train_vec = vectorizer.fit_transform(X_train_raw)
    X_test_vec = vectorizer.transform(X_test_raw)
    vocab_size = len(vectorizer.vocabulary_)
    print(f"      Vocabulary size: {vocab_size} features")

    # 5. Train Logistic Regression
    print(f"\n[5/5] Training Logistic Regression classifier (C={C}, class_weight='balanced')...")
    classifier = LogisticRegression(
        C=C,
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=random_state,
    )
    classifier.fit(X_train_vec, y_train)

    # 6. Evaluation
    y_pred = classifier.predict(X_test_vec)
    y_prob = classifier.predict_proba(X_test_vec)[:, 1]

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
    rec = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
    f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
    cm = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(
        y_test, y_pred, target_names=["negative", "positive"], output_dict=True
    )

    print("\n" + "-" * 60)
    print("  MODEL EVALUATION RESULTS (Test Set)")
    print("-" * 60)
    print(f"  Accuracy:  {acc * 100:.2f}%")
    print(f"  Precision: {prec * 100:.2f}%")
    print(f"  Recall:    {rec * 100:.2f}%")
    print(f"  F1-Score:  {f1 * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["negative", "positive"]))
    print("Confusion Matrix [[TN, FP], [FN, TP]]:")
    print(f"  TN: {cm[0][0]:<4}  FP: {cm[0][1]:<4}")
    print(f"  FN: {cm[1][0]:<4}  TP: {cm[1][1]:<4}")

    # Top Features Analysis
    feature_names = vectorizer.get_feature_names_out()
    coefs = classifier.coef_[0]
    top_pos_indices = np.argsort(coefs)[-10:][::-1]
    top_neg_indices = np.argsort(coefs)[:10]

    top_positive_features = [
        {"word": feature_names[i], "weight": round(float(coefs[i]), 4)}
        for i in top_pos_indices
    ]
    top_negative_features = [
        {"word": feature_names[i], "weight": round(float(coefs[i]), 4)}
        for i in top_neg_indices
    ]

    metrics_payload = {
        "dataset": {
            "total_samples": total_samples,
            "train_samples": len(X_train_raw),
            "test_samples": len(X_test_raw),
            "positive_samples": pos_count,
            "negative_samples": neg_count,
        },
        "hyperparameters": {
            "ngram_range": list(ngram_range),
            "max_features": max_features,
            "sublinear_tf": True,
            "C": C,
            "class_weight": "balanced",
            "solver": "lbfgs",
        },
        "evaluation": {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "confusion_matrix": cm,
            "classification_report": report,
        },
        "top_features": {
            "top_positive": top_positive_features,
            "top_negative": top_negative_features,
        },
    }

    # 7. Save Artifacts
    print("\nSaving artifacts...")
    joblib.dump(classifier, MODEL_OUT_PATH)
    joblib.dump(vectorizer, VECTORIZER_OUT_PATH)
    with open(METRICS_OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    print(f"  [✓] Model saved:      {MODEL_OUT_PATH}")
    print(f"  [✓] Vectorizer saved: {VECTORIZER_OUT_PATH}")
    print(f"  [✓] Metrics saved:    {METRICS_OUT_PATH}")
    print("=" * 60)
    print("  PHASE 1 TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 60 + "\n")

    return metrics_payload


if __name__ == "__main__":
    train_text_sentiment_model()
