"""Tier 0 — Vision Encoder.

Converts camera frames into dense scene embeddings that downstream
models (Tier 1 scene reasoner, Tier 2 planner) consume.

Runs at 20Hz. Never blocks. Returns cached embedding if inference
would exceed deadline.

Supported backends:
    - SigLIP-SO400M (default): 384x384 → 768-dim, ~20ms on Jetson DLA
    - DINOv2-small (minimal):  224x224 → 384-dim, ~10ms on NEON
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np


@dataclass
class SceneEmbedding:
    embedding: np.ndarray
    timestamp: float
    latency_ms: float
    cached: bool = False


class VisionEncoder:
    def __init__(self, model_name: str = "siglip-so400m",
                 max_latency_ms: float = 25.0, device: str = "cpu"):
        self.model_name = model_name
        self.max_latency_ms = max_latency_ms
        self.device = device
        self._model = None
        self._processor = None
        dim = 768 if "siglip" in model_name else 384
        self._cached = SceneEmbedding(
            embedding=np.zeros(dim), timestamp=0.0, latency_ms=0.0, cached=True)
        self._avg_latency_ms = 0.0
        self._call_count = 0

    def _load_model(self):
        if self._model is not None:
            return
        try:
            if "siglip" in self.model_name:
                from transformers import SiglipModel, SiglipProcessor
                self._processor = SiglipProcessor.from_pretrained(
                    "google/siglip-so400m-patch14-384")
                self._model = SiglipModel.from_pretrained(
                    "google/siglip-so400m-patch14-384")
            elif "dinov2" in self.model_name:
                from transformers import AutoImageProcessor, AutoModel
                self._processor = AutoImageProcessor.from_pretrained(
                    "facebook/dinov2-small")
                self._model = AutoModel.from_pretrained("facebook/dinov2-small")
            if self._model and self.device != "cpu":
                self._model = self._model.to(self.device)
        except Exception:
            pass

    def encode(self, frame: np.ndarray) -> SceneEmbedding:
        if self._avg_latency_ms > self.max_latency_ms and self._call_count > 3:
            self._cached.cached = True
            return self._cached

        self._load_model()
        start = time.monotonic()

        if self._model is None or self._processor is None:
            dim = 768 if "siglip" in self.model_name else 384
            emb = SceneEmbedding(
                embedding=np.zeros(dim, dtype=np.float32),
                timestamp=time.time(), latency_ms=0.0)
            self._cached = emb
            return emb

        try:
            import torch
            from PIL import Image
            img = Image.fromarray(frame) if isinstance(frame, np.ndarray) else frame

            if "siglip" in self.model_name:
                inputs = self._processor(images=img, return_tensors="pt")
                if self.device != "cpu":
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = self._model.get_image_features(**inputs)
                embedding = outputs.cpu().numpy().flatten()
            else:
                inputs = self._processor(images=img, return_tensors="pt")
                if self.device != "cpu":
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = self._model(**inputs)
                embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy().flatten()

            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
        except Exception:
            dim = 768 if "siglip" in self.model_name else 384
            embedding = np.zeros(dim, dtype=np.float32)

        elapsed_ms = (time.monotonic() - start) * 1000
        self._call_count += 1
        self._avg_latency_ms = 0.2 * elapsed_ms + 0.8 * self._avg_latency_ms

        emb = SceneEmbedding(
            embedding=embedding, timestamp=time.time(), latency_ms=elapsed_ms)
        self._cached = emb
        return emb
