"""
Inference Pipeline for Text Sentiment Analysis.

Exposes the TextSentimentClassifier class and standalone predict_sentiment function.
"""

import os
import joblib
from typing import Dict, Any, List, Optional
from src.text_analysis.preprocessing import preprocess_text

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL_PATH = os.path.join(BASE_DIR, "models", "text_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "models", "tfidf_vectorizer.pkl")


class TextSentimentClassifier:
    """
    Production-ready classifier for text sentiment inference.
    Loads serialized TF-IDF vectorizer and Logistic Regression model.
    """

    def __init__(
        self,
        model_path: str = MODEL_PATH,
        vectorizer_path: str = VECTORIZER_PATH,
        auto_train: bool = True,
    ):
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path
        self.model = None
        self.vectorizer = None

        if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
            if auto_train:
                print("Models not found. Initializing auto-train pipeline...")
                from src.text_analysis.train import train_text_sentiment_model
                train_text_sentiment_model()
            else:
                raise FileNotFoundError(
                    f"Model artifacts not found at {model_path} or {vectorizer_path}."
                )

        self.load_artifacts()

    def load_artifacts(self) -> None:
        """Load pickled artifacts into memory."""
        self.model = joblib.load(self.model_path)
        self.vectorizer = joblib.load(self.vectorizer_path)

    def predict(self, text: str) -> Dict[str, Any]:
        """
        Predict sentiment for a given single text input.

        Returns:
            dict matching the project specification:
            {
                "sentiment": "positive" | "negative",
                "confidence": 0.91,
                "probabilities": {
                    "positive": 0.91,
                    "negative": 0.09
                },
                "cleaned_text": "..."
            }
        """
        if not text or not str(text).strip():
            return {
                "sentiment": "neutral",
                "confidence": 0.50,
                "probabilities": {"positive": 0.50, "negative": 0.50},
                "cleaned_text": "",
                "note": "Empty or whitespace text provided",
            }

        cleaned = preprocess_text(text)
        if not cleaned:
            return {
                "sentiment": "neutral",
                "confidence": 0.50,
                "probabilities": {"positive": 0.50, "negative": 0.50},
                "cleaned_text": "",
                "note": "Text contained no recognizable tokens after preprocessing",
            }

        vec = self.vectorizer.transform([cleaned])
        probs = self.model.predict_proba(vec)[0]  # [prob_neg, prob_pos]
        prob_neg = float(probs[0])
        prob_pos = float(probs[1])

        # Classification decision with confidence
        if prob_pos >= prob_neg:
            sentiment = "positive"
            confidence = round(prob_pos, 4)
        else:
            sentiment = "negative"
            confidence = round(prob_neg, 4)

        return {
            "sentiment": sentiment,
            "confidence": confidence,
            "probabilities": {
                "positive": round(prob_pos, 4),
                "negative": round(prob_neg, 4),
            },
            "cleaned_text": cleaned,
        }

    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Predict sentiment for a batch of strings."""
        return [self.predict(t) for t in texts]


# Global cached instance for fast API inference
_global_classifier: Optional[TextSentimentClassifier] = None


def get_text_classifier() -> TextSentimentClassifier:
    """Singleton getter for the text sentiment classifier."""
    global _global_classifier
    if _global_classifier is None:
        _global_classifier = TextSentimentClassifier()
    return _global_classifier


def predict_sentiment(text: str) -> Dict[str, Any]:
    """Convenience function to classify a single text."""
    classifier = get_text_classifier()
    return classifier.predict(text)


if __name__ == "__main__":
    import sys
    test_sentences = [
        "I am having a wonderful day!",
        "A truly captivating journey that kept me glued to the screen.",
        "A terrible waste of time and money. Completely boring and devoid of any substance.",
        "This product is not good at all, very disappointing.",
        "This is not a bad film; in fact, I enjoyed it quite a lot!",
    ]
    classifier = TextSentimentClassifier()
    print("\n--- Running Sample Text Predictions ---")
    for sentence in test_sentences:
        res = classifier.predict(sentence)
        print(f"\nText:       \"{sentence}\"")
        print(f"Sentiment:  {res['sentiment'].upper()} (Confidence: {res['confidence'] * 100:.1f}%)")
        print(f"Probabilities: {res['probabilities']}")

