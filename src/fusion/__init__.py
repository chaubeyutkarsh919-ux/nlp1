"""
Multimodal Fusion Engine Package
"""
from src.fusion.fusion_engine import (
    MultimodalFusionEngine,
    get_fusion_engine,
    fuse_predictions,
    RESPONSIBLE_AI_DISCLAIMER,
)

__all__ = [
    "MultimodalFusionEngine",
    "get_fusion_engine",
    "fuse_predictions",
    "RESPONSIBLE_AI_DISCLAIMER",
]

