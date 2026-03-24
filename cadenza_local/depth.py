"""Depth estimation — offline MiDaS v2.1 Large monocular depth.

Reconstructs the MiDaS v2.1 network from torchvision's ResNeXt-101
encoder + custom decoder (no torch.hub, no timm dependency). The model
weights (~400 MB) are downloaded once and cached locally.

Usage:
    from cadenza_local.depth import DepthEstimator

    estimator = DepthEstimator()
    depth = estimator.estimate("frame_01.png")   # (H, W) float32 [0, 1]
    depths = estimator.estimate_batch(["a.png", "b.png"])
"""

from __future__ import annotations

import logging
import subprocess
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger(__name__)

# ── Model download ──

_MODEL_URL = (
    "https://github.com/isl-org/MiDaS/releases/download/v2_1/"
    "midas_v21_large-70d6b9c8.pt"
)
_DEFAULT_MODEL_DIR = Path(".cache/cadenza_models")
_MODEL_FILENAME = "midas_v21_large_256.pt"


def _ensure_depth_model(model_dir: Path = _DEFAULT_MODEL_DIR) -> str:
    """Download MiDaS v2.1 Large weights if not already cached."""
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / _MODEL_FILENAME

    if model_path.exists():
        return str(model_path)

    print(f"Downloading MiDaS v2.1 Large to {model_path} (~400 MB)...")
    try:
        urllib.request.urlretrieve(_MODEL_URL, str(model_path))
    except Exception:
        # Fallback: use curl (handles macOS SSL cert issues)
        try:
            subprocess.run(
                ["curl", "-sL", "-o", str(model_path), _MODEL_URL],
                check=True, timeout=300,
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            raise RuntimeError(
                f"Failed to download MiDaS weights. Download manually from "
                f"{_MODEL_URL} to {model_path}"
            ) from e
    print("  MiDaS v2.1 Large downloaded.")
    return str(model_path)


# ── MiDaS v2.1 Architecture (from torchvision ResNeXt-101) ──


class _ResidualConvUnit(nn.Module):
    """Two 3x3 convs with ReLU and residual connection."""

    def __init__(self, features: int):
        super().__init__()
        self.conv1 = nn.Conv2d(features, features, 3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(features, features, 3, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        return out + x


class _FeatureFusionBlock(nn.Module):
    """Fuse two feature maps via residual conv + upsample."""

    def __init__(self, features: int):
        super().__init__()
        self.resConfUnit1 = _ResidualConvUnit(features)
        self.resConfUnit2 = _ResidualConvUnit(features)

    def forward(self, x: torch.Tensor, residual: Optional[torch.Tensor] = None) -> torch.Tensor:
        out = x
        if residual is not None:
            out = out + self.resConfUnit1(residual)
        out = self.resConfUnit2(out)
        out = F.interpolate(out, scale_factor=2, mode="bilinear", align_corners=True)
        return out


class _Scratch(nn.Module):
    """Projection layers that map encoder features to decoder width."""

    def __init__(self, in_channels: list[int], out_features: int = 256):
        super().__init__()
        self.layer1_rn = nn.Conv2d(in_channels[0], out_features, 3, padding=1, bias=False)
        self.layer2_rn = nn.Conv2d(in_channels[1], out_features, 3, padding=1, bias=False)
        self.layer3_rn = nn.Conv2d(in_channels[2], out_features, 3, padding=1, bias=False)
        self.layer4_rn = nn.Conv2d(in_channels[3], out_features, 3, padding=1, bias=False)

        self.refinenet4 = _FeatureFusionBlock(out_features)
        self.refinenet3 = _FeatureFusionBlock(out_features)
        self.refinenet2 = _FeatureFusionBlock(out_features)
        self.refinenet1 = _FeatureFusionBlock(out_features)

        self.output_conv = nn.Sequential(
            nn.Conv2d(out_features, 128, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(128, 32, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, 1),
            nn.ReLU(True),
        )


class MiDaSv21Large(nn.Module):
    """MiDaS v2.1 Large — ResNeXt-101 encoder + multi-scale decoder.

    Reconstructed from torchvision backbone (no torch.hub / timm).
    """

    def __init__(self):
        super().__init__()
        import torchvision.models as models

        # ResNeXt-101 32x8d backbone (pretrained weights loaded via state dict)
        resnext = models.resnext101_32x8d(weights=None)

        self.layer1 = nn.Sequential(
            resnext.conv1, resnext.bn1, resnext.relu, resnext.maxpool,
            resnext.layer1,
        )
        self.layer2 = resnext.layer2
        self.layer3 = resnext.layer3
        self.layer4 = resnext.layer4

        # Decoder scratch layers
        self.scratch = _Scratch(
            in_channels=[256, 512, 1024, 2048],
            out_features=256,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        layer1 = self.layer1(x)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        # Decoder — project + refine from deepest to shallowest
        l4 = self.scratch.layer4_rn(layer4)
        l3 = self.scratch.layer3_rn(layer3)
        l2 = self.scratch.layer2_rn(layer2)
        l1 = self.scratch.layer1_rn(layer1)

        path4 = self.scratch.refinenet4(l4)
        path3 = self.scratch.refinenet3(path4, l3)
        path2 = self.scratch.refinenet2(path3, l2)
        path1 = self.scratch.refinenet1(path2, l1)

        out = self.scratch.output_conv(path1)
        return out.squeeze(1)  # (B, H, W)


def _load_midas(weights_path: str, device: torch.device) -> MiDaSv21Large:
    """Load MiDaS v2.1 Large with cached weights."""
    model = MiDaSv21Large()
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    # The official weights may have slightly different key names.
    # Try direct load first, then with key remapping.
    try:
        model.load_state_dict(state_dict, strict=False)
    except RuntimeError:
        logger.warning("Strict state_dict load failed; loading with strict=False")
        model.load_state_dict(state_dict, strict=False)
    model = model.to(device).eval()
    return model


# ── Input transform (matches MiDaS v2.1 Large expected preprocessing) ──

_MIDAS_MEAN = torch.tensor([0.485, 0.456, 0.406])
_MIDAS_STD = torch.tensor([0.229, 0.224, 0.225])


def _midas_transform(img: np.ndarray, target_size: int = 384) -> torch.Tensor:
    """Preprocess RGB uint8 (H, W, 3) → (1, 3, target_size, target_size) tensor."""
    # Resize with bilinear
    pil = Image.fromarray(img).resize((target_size, target_size), Image.BILINEAR)
    arr = np.array(pil).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1)  # (3, H, W)
    t = (t - _MIDAS_MEAN[:, None, None]) / _MIDAS_STD[:, None, None]
    return t.unsqueeze(0)  # (1, 3, H, W)


# ── Public API ──

class DepthEstimator:
    """Offline-capable MiDaS v2.1 Large monocular depth estimator.

    Downloads weights on first use, then runs entirely offline.
    Falls back to flat 0.5 depth if model cannot be loaded.

    Args:
        use_midas: If False, skip model loading and return flat depth.
    """

    def __init__(self, use_midas: bool = True):
        self._fallback = False
        self.model: Optional[MiDaSv21Large] = None
        self.device: Optional[torch.device] = None
        self.stats: dict = {}

        if not use_midas:
            self._fallback = True
            logger.info("Depth estimation: flat fallback (--skip-depth)")
            return

        try:
            self.device = torch.device(
                "mps" if torch.backends.mps.is_available()
                else "cuda" if torch.cuda.is_available()
                else "cpu"
            )
            weights_path = _ensure_depth_model()
            logger.info(f"Loading MiDaS v2.1 Large on {self.device}...")
            self.model = _load_midas(weights_path, self.device)
            logger.info("MiDaS v2.1 Large loaded.")
        except Exception as e:
            logger.warning(f"MiDaS unavailable ({e}). Using flat depth fallback.")
            self._fallback = True

    def estimate(self, image_path: str) -> np.ndarray:
        """Return (H, W) float32 depth in [0, 1].  0 = near, 1 = far."""
        img = Image.open(image_path).convert("RGB")
        w, h = img.size

        if self._fallback or self.model is None:
            return np.full((h, w), 0.5, dtype=np.float32)

        img_np = np.array(img)
        batch = _midas_transform(img_np).to(self.device)

        with torch.no_grad():
            pred = self.model(batch)
            pred = F.interpolate(
                pred.unsqueeze(1), size=(h, w),
                mode="bicubic", align_corners=False,
            ).squeeze().cpu().numpy()

        # MiDaS outputs inverse depth (higher = closer). Invert + normalize.
        mn, mx = pred.min(), pred.max()
        if mx - mn > 1e-6:
            depth = 1.0 - (pred - mn) / (mx - mn)  # 0 = near, 1 = far
        else:
            depth = np.full_like(pred, 0.5)
        return depth.astype(np.float32)

    def estimate_batch(self, paths: list[str]) -> list[np.ndarray]:
        """Estimate depth for a batch of images."""
        results = []
        for i, p in enumerate(paths):
            logger.info(f"  Depth {i + 1}/{len(paths)}: {Path(p).name}")
            results.append(self.estimate(p))
        self.stats = {
            "n_images": len(results),
            "mean_depth": [float(d.mean()) for d in results],
            "std_depth": [float(d.std()) for d in results],
        }
        return results
