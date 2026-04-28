"""Monocular depth estimation for the cadenza stack.

Uses Depth-Anything-V2-Small (~25M params) — small enough to run on CPU/MPS
and produce a depth map in well under a second per frame. Returns a relative
depth map (larger values = farther) that the navigator samples to choose a
recovery direction.

Loaded lazily on the first ``predict()`` call so a stack that never gets
stuck never pays the model-load cost.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

_LOG = logging.getLogger("cadenza.stack.vision.depth")


class DepthEstimator:
    """Wraps a HuggingFace depth-estimation model behind a tiny RGB → depth API."""

    DEFAULT_MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"

    def __init__(self, model_id: str | None = None, device: str | None = None):
        self.model_id = model_id or self.DEFAULT_MODEL_ID
        self.device = device
        self._processor: Any = None
        self._model: Any = None
        self._loaded = False

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def load(self) -> None:
        if self._loaded:
            return
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "DepthEstimator requires `transformers torch pillow`. "
                "pip install 'transformers>=4.43' torch pillow"
            ) from e
        _LOG.info("DepthEstimator: loading %s ...", self.model_id)
        self._processor = AutoImageProcessor.from_pretrained(self.model_id)
        self._model = AutoModelForDepthEstimation.from_pretrained(self.model_id)
        if self.device is None:
            self.device = (
                "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
                else "cuda" if torch.cuda.is_available()
                else "cpu"
            )
        self._model.to(self.device).eval()
        self._loaded = True
        _LOG.info("DepthEstimator: ready on %s", self.device)

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    # ── Inference ────────────────────────────────────────────────────────────

    def predict(self, rgb: np.ndarray) -> np.ndarray:
        """Run depth estimation on a HxWx3 RGB frame.

        Args:
            rgb: uint8 HxWx3 array, or float HxWx3 in [0, 1] (auto-converted).

        Returns:
            HxW float32 depth map (relative depth, larger = farther).
        """
        if not self._loaded:
            self.load()

        import torch
        from PIL import Image

        if rgb.dtype != np.uint8:
            arr = rgb
            if arr.max() <= 1.0:
                arr = arr * 255.0
            rgb_u8 = arr.clip(0, 255).astype(np.uint8)
        else:
            rgb_u8 = rgb

        image = Image.fromarray(rgb_u8)
        inputs = self._processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self._model(**inputs)
        depth = outputs.predicted_depth.squeeze(0).cpu().numpy().astype(np.float32)
        return depth
