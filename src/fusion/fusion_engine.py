"""
Multimodal Fusion Engine (Late Fusion).

Combines independent predictions from:
1. NLP Text Sentiment Model (Positive / Negative / Neutral)
2. Computer Vision Facial Expression Model (Happy / Sad / Angry / Surprise / Fear / Disgust / Neutral)

Important Responsible AI Principle:
Facial expressions are visible physical patterns, NOT definitive proof of internal emotional states.
This engine keeps text and expression results separate, computing an experimental congruence comparison.
"""

from typing import Dict, Any, Optional

# Valence mapping for expression categories
EXPRESSION_VALENCE_MAP = {
    "happy": "positive",
    "sad": "negative",
    "angry": "negative",
    "disgust": "negative",
    "fear": "negative",
    "surprise": "ambiguous",
    "neutral": "neutral",
}

RESPONSIBLE_AI_DISCLAIMER = (
    "Responsible AI Notice: Facial expressions represent visible muscular patterns, "
    "not conclusive proof of an individual's actual emotional state. This comparison "
    "is an experimental multimodal prediction and should not be treated as psychological fact."
)


class MultimodalFusionEngine:
    """
    Late Fusion Engine that synthesizes text sentiment and facial expression predictions
    into a comparative alignment assessment with confidence calibrations and ethical boundaries.
    """

    def __init__(self):
        self.disclaimer = RESPONSIBLE_AI_DISCLAIMER

    def evaluate_consistency(
        self,
        text_sentiment: str,
        facial_expression: str,
        text_confidence: float,
        expression_confidence: float,
    ) -> Dict[str, Any]:
        """
        Evaluate alignment between text sentiment and facial expression.

        Categories:
        - "consistent": Both modalities align in valence (e.g. positive text + happy expression).
        - "discrepant": Modalities diverge (e.g. negative text + happy expression or positive text + angry expression).
        - "context_dependent": One or both modalities are neutral or ambiguous (surprise/neutral).
        """
        text_sentiment = (text_sentiment or "neutral").lower()
        facial_expression = (facial_expression or "neutral").lower()
        expr_valence = EXPRESSION_VALENCE_MAP.get(facial_expression, "neutral")

        # Case 1: Positive text + Happy face
        if text_sentiment == "positive" and expr_valence == "positive":
            status = "consistent"
            summary = "Predictions appear consistent (Congruent Positive)."
            details = (
                f"The text expresses positive sentiment (confidence: {text_confidence*100:.1f}%), "
                f"which aligns with the detected '{facial_expression}' facial expression "
                f"(confidence: {expression_confidence*100:.1f}%)."
            )
            alignment_score = min(text_confidence, expression_confidence)

        # Case 2: Negative text + Negative expression (sad, angry, fear, disgust)
        elif text_sentiment == "negative" and expr_valence == "negative":
            status = "consistent"
            summary = "Predictions appear consistent (Congruent Negative)."
            details = (
                f"The text conveys negative sentiment (confidence: {text_confidence*100:.1f}%), "
                f"which aligns with the detected '{facial_expression}' facial pattern "
                f"(confidence: {expression_confidence*100:.1f}%)."
            )
            alignment_score = min(text_confidence, expression_confidence)

        # Case 3: Positive text + Negative expression
        elif text_sentiment == "positive" and expr_valence == "negative":
            status = "discrepant"
            summary = "Predictions appear discrepant (Valence Mismatch)."
            details = (
                f"The text expresses positive sentiment (confidence: {text_confidence*100:.1f}%), "
                f"while the detected facial pattern indicates '{facial_expression}' "
                f"(confidence: {expression_confidence*100:.1f}%). Possible causes include irony, "
                "social masking, lighting artifacts, or nuanced emotional context."
            )
            alignment_score = max(0.0, 1.0 - abs(text_confidence - expression_confidence))

        # Case 4: Negative text + Positive expression (happy)
        elif text_sentiment == "negative" and expr_valence == "positive":
            status = "discrepant"
            summary = "Predictions appear discrepant (Valence Mismatch)."
            details = (
                f"The text conveys negative sentiment (confidence: {text_confidence*100:.1f}%), "
                f"while the detected facial expression shows '{facial_expression}' "
                f"(confidence: {expression_confidence*100:.1f}%). Possible causes include sarcasm, "
                "coping mechanisms, polite smiling, or mixed context."
            )
            alignment_score = max(0.0, 1.0 - abs(text_confidence - expression_confidence))

        # Case 5: Ambiguous (Surprise) or Neutral
        else:
            status = "context_dependent"
            summary = "Predictions show neutral or context-dependent alignment."
            details = (
                f"Text sentiment is '{text_sentiment}' while the detected expression is "
                f"'{facial_expression}'. Expressions like surprise and neutral depend heavily "
                "on situational cues and tone."
            )
            alignment_score = 0.50

        return {
            "status": status,
            "summary": summary,
            "details": details,
            "alignment_score": round(float(alignment_score), 4),
        }

    def fuse(
        self,
        text_result: Dict[str, Any],
        facial_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Execute multimodal fusion given the outputs of both independent models.

        Args:
            text_result: Dict from TextSentimentClassifier.predict()
            facial_result: Optional Dict from FacialExpressionClassifier.predict()

        Returns:
            Dict conforming to the required API design:
            {
                "text_analysis": { "sentiment": "...", "confidence": 0.91 },
                "facial_analysis": { "expression": "...", "confidence": 0.84 },
                "comparison": "consistent" | "discrepant" | "context_dependent" | "text_only",
                "comparison_details": { ... },
                "disclaimer": "..."
            }
        """
        text_summary = {
            "sentiment": text_result.get("sentiment", "unknown"),
            "confidence": text_result.get("confidence", 0.0),
            "probabilities": text_result.get("probabilities", {}),
        }

        # Check if facial input was provided and face detected
        if facial_result is None or not facial_result.get("face_detected", False):
            return {
                "text_analysis": text_summary,
                "facial_analysis": {
                    "expression": "not_available",
                    "confidence": 0.0,
                    "face_detected": False,
                    "message": "No facial image provided or no face detected.",
                },
                "comparison": "text_only",
                "comparison_summary": "Analysis based solely on text input.",
                "details": "Only the text modality was evaluated as no facial input was detected or provided.",
                "disclaimer": self.disclaimer,
            }

        facial_summary = {
            "expression": facial_result.get("expression", "unknown"),
            "confidence": facial_result.get("confidence", 0.0),
            "probabilities": facial_result.get("probabilities", {}),
            "face_detected": True,
            "detector_used": facial_result.get("detector_used", "unknown"),
        }

        eval_res = self.evaluate_consistency(
            text_sentiment=text_summary["sentiment"],
            facial_expression=facial_summary["expression"],
            text_confidence=text_summary["confidence"],
            expression_confidence=facial_summary["confidence"],
        )

        return {
            "text_analysis": text_summary,
            "facial_analysis": facial_summary,
            "comparison": eval_res["status"],
            "comparison_summary": eval_res["summary"],
            "details": eval_res["details"],
            "alignment_score": eval_res["alignment_score"],
            "disclaimer": self.disclaimer,
        }


# Singleton engine
_global_fusion_engine: Optional[MultimodalFusionEngine] = None


def get_fusion_engine() -> MultimodalFusionEngine:
    """Singleton getter for fusion engine."""
    global _global_fusion_engine
    if _global_fusion_engine is None:
        _global_fusion_engine = MultimodalFusionEngine()
    return _global_fusion_engine


def fuse_predictions(
    text_result: Dict[str, Any], facial_result: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Convenience function for multimodal prediction fusion."""
    engine = get_fusion_engine()
    return engine.fuse(text_result, facial_result)

