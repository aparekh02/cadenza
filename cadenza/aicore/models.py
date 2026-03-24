"""AICore model registry — manages the SLM stack that governs robot reasoning.

Model Tiers (fastest → most capable):

    Tier 0 — Vision Encoder (20Hz, <25ms)
        SigLIP-SO400M or DINOv2-small
        Camera frame → 768-dim scene embedding
        Runs on-device via ONNX/CoreML/TensorRT

    Tier 1 — Scene Reasoner (10Hz, <50ms)
        Moondream2 (1.6B) or PaliGemma-3B-mix
        (image + text prompt) → structured scene description

    Tier 2 — Action Planner (2Hz, <500ms)
        Phi-3.5-mini (3.8B) or Gemma-2-2B
        (scene description + body state + goal) → action plan

    Tier 3 — Strategic Reasoner (0.5Hz, <2s, optional)
        Llama-3.2-3B or Qwen2.5-3B via Ollama
        Complex multi-step planning, goal decomposition, failure recovery
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class ModelTier(IntEnum):
    VISION = 0
    SCENE = 1
    PLANNER = 2
    STRATEGIC = 3


@dataclass
class ModelSpec:
    name: str
    tier: ModelTier
    hf_repo: str
    parameters: str
    max_latency_ms: int
    input_modalities: list[str]
    output_type: str
    quantization: str = "int4"
    context_length: int = 2048
    memory_mb: int = 0
    description: str = ""


VISION_MODELS = {
    "siglip-so400m": ModelSpec(
        name="siglip-so400m", tier=ModelTier.VISION,
        hf_repo="google/siglip-so400m-patch14-384", parameters="400M",
        max_latency_ms=25, input_modalities=["image"], output_type="embedding",
        quantization="fp16", memory_mb=800,
        description="Vision encoder. 384x384 image → 768-dim embedding.",
    ),
    "dinov2-small": ModelSpec(
        name="dinov2-small", tier=ModelTier.VISION,
        hf_repo="facebook/dinov2-small", parameters="22M",
        max_latency_ms=15, input_modalities=["image"], output_type="embedding",
        quantization="fp16", memory_mb=200,
        description="Lightweight vision encoder for terrain classification.",
    ),
}

SCENE_MODELS = {
    "moondream2": ModelSpec(
        name="moondream2", tier=ModelTier.SCENE,
        hf_repo="vikhyatk/moondream2", parameters="1.6B",
        max_latency_ms=50, input_modalities=["image", "text"], output_type="text",
        quantization="int4", context_length=2048, memory_mb=1200,
        description="Vision-language model for scene understanding.",
    ),
    "paligemma-3b": ModelSpec(
        name="paligemma-3b", tier=ModelTier.SCENE,
        hf_repo="google/paligemma-3b-mix-448", parameters="3B",
        max_latency_ms=80, input_modalities=["image", "text"], output_type="text",
        quantization="int4", context_length=512, memory_mb=2000,
        description="Multimodal model for detailed visual question answering.",
    ),
}

PLANNER_MODELS = {
    "phi-3.5-mini": ModelSpec(
        name="phi-3.5-mini", tier=ModelTier.PLANNER,
        hf_repo="microsoft/Phi-3.5-mini-instruct", parameters="3.8B",
        max_latency_ms=500, input_modalities=["text"], output_type="json",
        quantization="int4", context_length=4096, memory_mb=2500,
        description="Action planner. Scene + state + goal → JSON action plan.",
    ),
    "gemma-2-2b": ModelSpec(
        name="gemma-2-2b", tier=ModelTier.PLANNER,
        hf_repo="google/gemma-2-2b-it", parameters="2B",
        max_latency_ms=300, input_modalities=["text"], output_type="json",
        quantization="int4", context_length=2048, memory_mb=1500,
        description="Lighter action planner. Faster, less capable at multi-step.",
    ),
}

STRATEGIC_MODELS = {
    "llama-3.2-3b": ModelSpec(
        name="llama-3.2-3b", tier=ModelTier.STRATEGIC,
        hf_repo="meta-llama/Llama-3.2-3B-Instruct", parameters="3B",
        max_latency_ms=2000, input_modalities=["text"], output_type="text",
        quantization="int4", context_length=8192, memory_mb=2500,
        description="Strategic reasoner for complex goal decomposition via Ollama.",
    ),
    "qwen2.5-3b": ModelSpec(
        name="qwen2.5-3b", tier=ModelTier.STRATEGIC,
        hf_repo="Qwen/Qwen2.5-3B-Instruct", parameters="3B",
        max_latency_ms=2000, input_modalities=["text"], output_type="text",
        quantization="int4", context_length=8192, memory_mb=2500,
        description="Alternative strategic reasoner with strong structured output.",
    ),
}

STACK_JETSON_ORIN_NX = {
    "vision": "siglip-so400m", "scene": "moondream2",
    "planner": "phi-3.5-mini", "strategic": "llama-3.2-3b",
}
STACK_JETSON_ORIN_NANO = {
    "vision": "dinov2-small", "scene": "moondream2",
    "planner": "gemma-2-2b", "strategic": "llama-3.2-3b",
}
STACK_DEV_MACHINE = {
    "vision": "siglip-so400m", "scene": "moondream2",
    "planner": "phi-3.5-mini", "strategic": "llama-3.2-3b",
}
STACK_MINIMAL = {
    "vision": "dinov2-small", "scene": None,
    "planner": "gemma-2-2b", "strategic": None,
}


def get_model_spec(name: str) -> ModelSpec | None:
    for registry in [VISION_MODELS, SCENE_MODELS, PLANNER_MODELS, STRATEGIC_MODELS]:
        if name in registry:
            return registry[name]
    return None


def get_stack(hardware: str = "dev") -> dict[str, str | None]:
    stacks = {
        "jetson_orin_nx": STACK_JETSON_ORIN_NX,
        "jetson_orin_nano": STACK_JETSON_ORIN_NANO,
        "dev": STACK_DEV_MACHINE,
        "minimal": STACK_MINIMAL,
    }
    return stacks.get(hardware, STACK_DEV_MACHINE)
