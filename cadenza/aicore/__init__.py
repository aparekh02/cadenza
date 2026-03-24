"""cadenza.aicore — Multimodal intelligence kernel for Cadenza OS.

Tiered SLM stack that governs robot reasoning:

    Tier 0 — Vision Encoder (20Hz)
        SigLIP-SO400M or DINOv2-small
        Camera frame → scene embedding

    Tier 1 — Scene Reasoner (10Hz)
        Moondream2 (1.6B) or PaliGemma-3B
        (image + prompt) → "rough terrain ahead, stairs 2m away"

    Tier 2 — Action Planner (2Hz)
        Phi-3.5-mini (3.8B) or Gemma-2-2B via Ollama
        (scene + body state + goal) → [crawl 2m, climb step, walk 3m]

    Tier 3 — Strategic Reasoner (0.5Hz, optional)
        Llama-3.2-3B or Qwen2.5-3B via Ollama
        Complex goal decomposition when planner is uncertain

Usage::

    from cadenza.aicore import BehaviorEngine, ActionPlanner, VisionEncoder

    # Full stack
    engine = BehaviorEngine(
        "go1",
        slm=SLMBridge(),
        vision=VisionEncoder("siglip-so400m"),
        planner=ActionPlanner("phi3.5:3.8b-mini-instruct-q4_K_M"),
    )

    engine.set_goal("walk to the red cone and sit down")

    # In your sim loop:
    world = engine.observe_with_camera(qpos, qvel, camera_frame)
    decision = engine.decide(world)
    print(f"{decision.action} — {decision.reasoning}")

Minimal (no models, just behavior tree):

    engine = BehaviorEngine("go1")
    world = engine.observe(qpos, qvel)
    decision = engine.decide(world)  # rule-based only
"""

from cadenza.aicore.engine import BehaviorEngine, WorldState, ActionDecision
from cadenza.aicore.slm import SLMBridge, SLMConfig
from cadenza.aicore.vision import VisionEncoder, SceneEmbedding
from cadenza.aicore.planner import ActionPlanner, ActionPlan, ActionStep
from cadenza.aicore.models import (
    ModelSpec, ModelTier, get_model_spec, get_stack,
    VISION_MODELS, SCENE_MODELS, PLANNER_MODELS, STRATEGIC_MODELS,
)
