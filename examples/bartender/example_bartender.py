"""Bartender Negroni — cadenza gym example.

Dual-arm robot (2x SO-101) making a Negroni cocktail.
All scene composition, physics, compensation, and benchmarking
are handled by the cadenza Gym — this file just configures and runs it.

Usage (macOS needs mjpython for live viewer):
    mjpython examples/bartender/example_bartender.py
    mjpython examples/bartender/example_bartender.py --vla
    mjpython examples/bartender/example_bartender.py --episodes 5

    # plain python works too — viewer opens after episodes finish
    python examples/bartender/example_bartender.py
    python examples/bartender/example_bartender.py --no-render
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from cadenza_local import Gym

logging.basicConfig(level=logging.INFO, format="%(message)s")

TASK_FILE = str(Path(__file__).parent / "task.txt")
TASK_INSTRUCTION = "Pick up each bottle, pour into the mixing glass, stir, and serve."


# ── SmolVLA wiring (example-specific, not part of cadenza) ──

def _load_vla_policy():
    """Load SmolVLA base (open, runs locally on MPS/CPU)."""
    import torch
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
    policy = policy.to(device).eval()

    # Pre-tokenize the task instruction
    processor = policy.model.vlm_with_expert.processor
    tokens = processor.tokenizer(TASK_INSTRUCTION, return_tensors="pt")
    lang_tokens = tokens["input_ids"].to(device)
    lang_mask = tokens["attention_mask"].bool().to(device)

    return policy, device, lang_tokens, lang_mask


def _make_vla_action_fn(gym_instance, policy, device, lang_tokens, lang_mask):
    """Return an action_fn that feeds MuJoCo camera renders + state to SmolVLA."""
    import torch
    action_buf = []

    def action_fn(robot_state, microstep):
        nonlocal action_buf
        n = 6  # SmolVLA expects 6D state/action
        if not action_buf:
            # State: (batch=1, 6)
            state = torch.zeros(1, n, dtype=torch.float32, device=device)
            rs = robot_state[:n] if len(robot_state) >= n else robot_state
            state[0, :len(rs)] = torch.from_numpy(rs.astype(np.float32))

            # Camera images from MuJoCo scene: (batch=1, 3, 256, 256)
            cam_images = gym_instance.render_cameras(width=256, height=256)

            obs = {
                "observation.state": state,
                "observation.language.tokens": lang_tokens,
                "observation.language.attention_mask": lang_mask,
            }
            for cam_key, img_np in cam_images.items():
                obs[cam_key] = torch.from_numpy(img_np).unsqueeze(0).to(device)

            with torch.inference_mode():
                chunk = policy.select_action(obs)
            if isinstance(chunk, torch.Tensor):
                chunk = chunk.cpu().numpy()
            if chunk.ndim == 1:
                chunk = chunk.reshape(1, -1)
            action_buf.extend(list(chunk))

        act = action_buf.pop(0)
        return act[:n] if len(act) >= n else np.zeros(n)

    return action_fn


# ── Main ──

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bartender Negroni gym")
    parser.add_argument("--no-render", action="store_true", help="Headless")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--vla", action="store_true", help="Use SmolVLA")
    args = parser.parse_args()

    gym = Gym.from_task(TASK_FILE)

    action_fn = None
    if args.vla:
        print("Loading SmolVLA (lerobot/smolvla_base)...")
        policy, device, lang_tokens, lang_mask = _load_vla_policy()
        action_fn = _make_vla_action_fn(gym, policy, device, lang_tokens, lang_mask)

    gym.run_sync(
        episodes=args.episodes,
        render=not args.no_render,
        action_fn=action_fn,
    )
