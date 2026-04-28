"""depth_anything_v2_small — client-owned multi-modal input model.

Wraps depth-anything/Depth-Anything-V2-Small-hf as a cadenza stack Modality.
The cadenza stack ships only the ``Modality`` interface; this file is the
"model" file the client points the stack at. Drop another file like this
next to it for any other multi-modal input model you want to plug in.

    from depth_anything_v2_small import DepthAnythingV2Small
    cadenza.stack.run(..., modalities=[DepthAnythingV2Small()])
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

from cadenza.stack import Modality, ModalityResult
from cadenza.stack.gym_adapter import Observation


_DEVICE = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)


class DepthAnythingV2Small(Modality):
    """Monocular depth from depth-anything/Depth-Anything-V2-Small-hf."""

    name = "depth_anything_v2_small"
    description = "Monocular depth from depth-anything/Depth-Anything-V2-Small-hf."
    MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"

    def __init__(self, device: str | None = None):
        self.device = device or _DEVICE
        self._processor = None
        self._model = None

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def setup(self) -> None:
        if self._model is not None:
            return
        print(f"  loading {self.MODEL_ID} on {self.device} ...")
        self._processor = AutoImageProcessor.from_pretrained(self.MODEL_ID)
        self._model = AutoModelForDepthEstimation.from_pretrained(self.MODEL_ID)
        self._model.to(self.device).eval()

    # ── Per-tick inference ───────────────────────────────────────────────────

    def compute(self, observation: Observation) -> ModalityResult:
        if observation.camera is None:
            return ModalityResult(keys={}, summary="depth: no camera")
        if self._model is None:
            self.setup()

        depth_map = self._predict(observation.camera)
        d_left, d_center, d_right = self._region_means(depth_map)

        return ModalityResult(
            keys={
                "depth_map": depth_map,
                "depth_left": d_left,
                "depth_center": d_center,
                "depth_right": d_right,
                "depth_min": float(depth_map.min()),
                "depth_max": float(depth_map.max()),
            },
            summary=f"depth: L={d_left:.2f} C={d_center:.2f} R={d_right:.2f}",
        )

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _predict(self, rgb: np.ndarray) -> np.ndarray:
        if rgb.dtype != np.uint8:
            rgb = (rgb * 255 if rgb.max() <= 1.0 else rgb).clip(0, 255).astype(np.uint8)
        image = Image.fromarray(rgb)
        inputs = self._processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self._model(**inputs)
        return out.predicted_depth.squeeze(0).cpu().numpy().astype(np.float32)

    @staticmethod
    def _region_means(depth_map: np.ndarray) -> tuple[float, float, float]:
        h, w = depth_map.shape[:2]
        ymin, ymax = int(h * 0.45), int(h * 0.85)
        third = w // 3
        return (
            float(depth_map[ymin:ymax, :third].mean()),
            float(depth_map[ymin:ymax, third:2 * third].mean()),
            float(depth_map[ymin:ymax, 2 * third:].mean()),
        )
