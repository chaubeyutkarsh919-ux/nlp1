"""
CNN Architecture and Training Pipeline for Facial Expression Recognition (FER-2013).

Categories:
0: Angry, 1: Disgust, 2: Fear, 3: Happy, 4: Sad, 5: Surprise, 6: Neutral
"""

import os
import json
from typing import Dict, Any, List

EMOTION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
NUM_CLASSES = len(EMOTION_LABELS)
INPUT_SHAPE = (48, 48, 1)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_OUT_PATH = os.path.join(MODELS_DIR, "expression_model.keras")
CONFIG_OUT_PATH = os.path.join(MODELS_DIR, "expression_model_config.json")


def build_fer_cnn_model():
    """
    Build a multi-block VGG-style CNN for 48x48 grayscale facial expression classification.
    Requires TensorFlow/Keras if executing full training.
    """
    try:
        import tensorflow as tf
        from tensorflow.keras import layers, models, regularizers

        model = models.Sequential([
            # Input Layer
            layers.Input(shape=INPUT_SHAPE),

            # Block 1
            layers.Conv2D(64, (3, 3), padding="same", kernel_regularizer=regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.Conv2D(64, (3, 3), padding="same"),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.25),

            # Block 2
            layers.Conv2D(128, (3, 3), padding="same", kernel_regularizer=regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.Conv2D(128, (3, 3), padding="same"),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.25),

            # Block 3
            layers.Conv2D(256, (3, 3), padding="same", kernel_regularizer=regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.Conv2D(256, (3, 3), padding="same"),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.3),

            # Dense Classification Head
            layers.Flatten(),
            layers.Dense(512, kernel_regularizer=regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.Dropout(0.5),

            layers.Dense(256),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.Dropout(0.5),

            layers.Dense(NUM_CLASSES, activation="softmax", name="expression_output")
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model
    except ImportError:
        return None


def export_model_metadata():
    """Export architecture metadata and label mapping for downstream inference."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    metadata = {
        "dataset": "FER-2013",
        "input_shape": list(INPUT_SHAPE),
        "num_classes": NUM_CLASSES,
        "classes": EMOTION_LABELS,
        "architecture": "VGG-style Deep CNN (3 Conv Blocks + Dense Head)",
        "preprocessing": {
            "grayscale": True,
            "clahe": True,
            "target_size": [48, 48],
            "pixel_normalization": "[0.0, 1.0]",
        },
    }
    with open(CONFIG_OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    return metadata


if __name__ == "__main__":
    meta = export_model_metadata()
    print("Exported FER-2013 Expression Model Metadata:")
    print(json.dumps(meta, indent=2))

