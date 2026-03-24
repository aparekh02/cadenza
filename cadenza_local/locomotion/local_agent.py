"""local_agent.py — Local LLM steering agent via Ollama.

Drop-in replacement for EpisodeSteeringAgent (Groq) when you want to run
without an API key and without rate limits.

Requires Ollama running locally:
    brew install ollama          # macOS
    ollama pull qwen2.5:7b      # recommended model
    ollama serve                 # starts server at http://localhost:11434

Any Ollama model works. Recommended options by hardware:
    qwen2.5:7b    — best structured JSON output, ~4 GB RAM, good on CPU
    llama3.2:3b   — fastest, ~2 GB RAM, decent reasoning
    llama3.1:8b   — matches the Groq model, ~5 GB RAM
    mistral:7b    — reliable, well-tested, ~4 GB RAM
    qwen2.5:14b   — best quality, ~8 GB RAM (needs decent GPU/CPU)

Usage from visual_gym.py
-------------------------
    agent = LocalSteeringAgent(model="qwen2.5:7b")
    # same .suggest() interface as EpisodeSteeringAgent
    suggestion = agent.suggest(episode=1, distance_reached=2.3, ...)
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
#  System prompt (same logic as EpisodeSteeringAgent)
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are the steering brain for a quadruped robot locomotion memory system.
Your job: analyze each episode result and output SPECIFIC parameter values
for the next run. You replace reinforcement learning by reasoning about
failures and successes.

TASK: The robot starts at x=0 and must reach a GOAL at x=8 m.
      At x=4 m there is a HURDLE (10 cm tall) it must step over.

RULES — follow strictly:
  - Only use gait "trot" or "walk". Never "bound" or "crawl".
  - approach_speed: 0.10–0.40 m/s. Start at 0.15. Increase slowly only after success.
  - sprint_speed: 0.15–0.50 m/s. Used only near the hurdle. Keep ≤0.30 until hurdle cleared.
  - step_height: 0.08–0.20 m. Increase if robot clips the hurdle. Default 0.12.
  - body_height_factor: 0.85–1.00 (fraction of nominal standing height 0.28m). Default 0.95.
  - The robot walks stably at 0.10–0.20 m/s. Faster is NOT always better.
  - If robot fell before x=2 m: reduce approach_speed to 0.10-0.12 m/s.
  - If robot fell at hurdle: increase step_height, reduce sprint_speed.
  - Prioritise stability over speed. A slow completion beats a fast fall.

OUTPUT FORMAT — respond with ONLY valid JSON, no explanation outside it:
{
  "approach_speed": <float>,
  "sprint_speed": <float>,
  "step_height": <float>,
  "body_height_factor": <float>,
  "gait_to_use": "trot",
  "reasoning": "<one short sentence>"
}
"""


# ─────────────────────────────────────────────────────────────────────────────
#  LocalSteeringAgent
# ─────────────────────────────────────────────────────────────────────────────

class LocalSteeringAgent:
    """Episode steering agent backed by a local Ollama model.

    No API key, no rate limits, no internet required.

    Args
    ----
    model      : Ollama model name (default "qwen2.5:7b")
    host       : Ollama server URL (default "http://localhost:11434")
    timeout_s  : HTTP timeout in seconds (default 30 — local inference can be slow)
    """

    def __init__(
        self,
        model:     str = "qwen2.5:7b",
        host:      str = "http://localhost:11434",
        timeout_s: float = 30.0,
    ):
        self._model_requested = model
        self._model    = model        # may be updated to exact Ollama name
        self._host     = host.rstrip("/")
        self._timeout  = timeout_s
        self._history: list[dict] = []   # kept for multi-turn context
        self._available: Optional[bool] = None  # cached availability check

    # ── Public API (same signature as EpisodeSteeringAgent.suggest) ───────

    def suggest(
        self,
        episode:          int,
        distance_reached: float,
        fell:             bool,
        fall_x:           float | None,
        fall_speed:       float | None,
        max_phase:        str,
        jump_cleared:     bool,
        approach_pace:    float,
        prev_params:      dict,
        history_summary:  str,
        max_speed:        float,
    ) -> dict | None:
        """Analyse episode outcome, return parameter dict for next episode.

        Returns None on any error so the caller can fall back to statistical recall.
        """
        if not self._check_available():
            return None

        fall_str = (
            f"fell at x={fall_x:.1f} m going {fall_speed:.2f} m/s"
            if fell and fall_x is not None
            else "did not fall"
        )
        user_msg = (
            f"Episode {episode} result:\n"
            f"  distance_reached = {distance_reached:.2f} m\n"
            f"  hurdle_cleared   = {jump_cleared}\n"
            f"  furthest_phase   = {max_phase}\n"
            f"  {fall_str}\n"
            f"  approach_pace    = {approach_pace:.2f} m/s\n"
            f"  params_used      = approach={prev_params.get('approach_speed', 0):.2f}  "
            f"sprint={prev_params.get('sprint_speed', 0):.2f}  "
            f"step_h={prev_params.get('step_height', 0):.2f}\n"
            f"  robot_max_speed  = {max_speed:.2f} m/s\n"
            f"  history          = {history_summary}\n"
            f"Suggest parameters for episode {episode + 1}."
        )

        raw = self._call(user_msg)
        if raw is None:
            return None

        try:
            # Strip markdown code fences if the model adds them
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                lines = cleaned.split("\n")
                cleaned = "\n".join(
                    l for l in lines if not l.strip().startswith("```")
                )
            data = json.loads(cleaned)
            return {
                "approach_speed":     float(min(max(data.get("approach_speed", 0.15), 0.10), 0.60)),
                "sprint_speed":       float(min(max(data.get("sprint_speed",   0.25), 0.15), 0.80)),
                "step_height":        float(min(max(data.get("step_height",    0.12), 0.06), 0.22)),
                "body_height_factor": float(min(max(data.get("body_height_factor", 0.95), 0.85), 1.00)),
                "gait_to_use":        "trot",
                "reasoning":          str(data.get("reasoning", "")),
            }
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"  [LocalLLM] parse error: {e}  raw={raw[:120]}")
            return None

    # ── Availability check ────────────────────────────────────────────────

    def _check_available(self) -> bool:
        """Ping Ollama to see if it's running and the model exists."""
        if self._available is not None:
            return self._available
        try:
            req  = urllib.request.Request(f"{self._host}/api/tags", method="GET")
            resp = urllib.request.urlopen(req, timeout=3.0)
            body = json.loads(resp.read())
            names = [m.get("name", "") for m in body.get("models", [])]
            # Exact match first, then prefix match
            exact = next((n for n in names if n == self._model_requested), None)
            prefix_key = self._model_requested.split(":")[0]
            prefix = next((n for n in names if prefix_key in n), None)
            matched = exact or prefix
            if not matched:
                print(f"  [LocalLLM] model '{self._model_requested}' not found in Ollama.\n"
                      f"             Available: {', '.join(names) or 'none'}\n"
                      f"             Run: ollama pull {self._model_requested}")
                self._available = False
            else:
                self._model = matched   # use exact name Ollama knows
                print(f"  [LocalLLM] {self._model} ready at {self._host}")
                self._available = True
        except Exception as e:
            print(f"  [LocalLLM] Ollama not reachable at {self._host}: {e}\n"
                  f"             Start it with: ollama serve")
            self._available = False
        return self._available

    # ── HTTP call ─────────────────────────────────────────────────────────

    def _call(self, user_msg: str) -> str | None:
        """Send a chat message and return the assistant text, or None on error."""
        messages = [
            {"role": "system",    "content": _SYSTEM_PROMPT},
            *self._history[-4:],  # last 2 turns of context
            {"role": "user",      "content": user_msg},
        ]
        payload = json.dumps({
            "model":    self._model,
            "messages": messages,
            "format":   "json",    # tells Ollama to guarantee valid JSON output
            "stream":   False,
            "options":  {"temperature": 0.1, "num_predict": 300},
        }).encode()

        try:
            req  = urllib.request.Request(
                url     = f"{self._host}/api/chat",
                data    = payload,
                headers = {"Content-Type": "application/json"},
                method  = "POST",
            )
            t0   = time.monotonic()
            resp = urllib.request.urlopen(req, timeout=self._timeout)
            body = json.loads(resp.read())
            text = body["message"]["content"]
            elapsed = time.monotonic() - t0
            print(f"  [LocalLLM] {self._model} responded in {elapsed:.1f}s")

            # Store in history for multi-turn context
            self._history.append({"role": "user",      "content": user_msg})
            self._history.append({"role": "assistant",  "content": text})
            return text

        except urllib.error.URLError as e:
            print(f"  [LocalLLM] request failed: {e}")
            self._available = None  # reset so next call retries ping
            return None
        except (KeyError, json.JSONDecodeError) as e:
            print(f"  [LocalLLM] bad response: {e}")
            return None

    @property
    def model(self) -> str:
        return self._model

    @property
    def available(self) -> bool:
        return self._check_available()


# ─────────────────────────────────────────────────────────────────────────────
#  Convenience: check what models are available
# ─────────────────────────────────────────────────────────────────────────────

def list_ollama_models(host: str = "http://localhost:11434") -> list[str]:
    """Return list of model names available in the local Ollama instance."""
    try:
        req  = urllib.request.Request(f"{host}/api/tags", method="GET")
        resp = urllib.request.urlopen(req, timeout=3.0)
        body = json.loads(resp.read())
        return [m.get("name", "") for m in body.get("models", [])]
    except Exception:
        return []
