"""SLM Bridge — connects to a local small language model for complex reasoning.

When the behavior tree can't handle a decision (e.g. a natural language goal
like "find the red cone and sit next to it"), the SLM bridge sends the world
state to a local model and gets back a structured action decision.

Supports Ollama (default) and any OpenAI-compatible API endpoint.

The SLM is NOT in the critical path. The behavior tree handles safety
and reactive decisions at 20Hz. The SLM is only consulted for strategic
decisions, and only when the behavior tree explicitly defers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from cadenza.aicore.engine import WorldState


@dataclass
class SLMConfig:
    provider: str = "ollama"
    model: str = "llama3.2:3b"
    base_url: str = "http://localhost:11434"
    timeout_ms: int = 2000
    temperature: float = 0.1
    max_tokens: int = 200


class SLMBridge:
    def __init__(self, config: SLMConfig | None = None):
        self.config = config or SLMConfig()
        self._available = None

    def is_available(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            import urllib.request
            url = f"{self.config.base_url}/api/tags"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=1) as resp:
                self._available = resp.status == 200
        except Exception:
            self._available = False
        return self._available

    def reason(self, ws: WorldState) -> dict | None:
        if not self.is_available():
            return None
        prompt = self._build_prompt(ws)
        try:
            response = self._query(prompt)
            return self._parse_response(response)
        except Exception:
            return None

    def _build_prompt(self, ws: WorldState) -> str:
        actions = self._get_available_actions()
        return f"""You are the decision-making brain of a quadruped robot.

Current state:
- Body: roll={ws.roll:.2f} rad, pitch={ws.pitch:.2f} rad, height={ws.height:.2f}m
- Terrain: class={ws.terrain_class} (0=flat,1=rough,2=stairs,3=slope,4=gap), roughness={ws.roughness:.2f}, slope={ws.slope:.2f} rad
- Obstacle: distance={ws.obstacle_distance:.2f}m, height={ws.obstacle_height:.2f}m
- Foot contact: {ws.foot_contact.tolist()}
- Confidence: {ws.confidence:.2f}

Goal: "{ws.goal_text}"
Urgency: {ws.urgency:.1f}

Available actions: {actions}

Respond ONLY with valid JSON:
{{"action": "<action_name>", "speed": <0.1-2.0>, "confidence": <0.0-1.0>, "reasoning": "<brief explanation>"}}"""

    def _get_available_actions(self) -> str:
        try:
            from cadenza_local.actions import get_library
            lib = get_library("go1")
            return ", ".join(lib.list_actions())
        except Exception:
            return "stand, walk_forward, walk_backward, turn_left, turn_right, crawl_forward, jump, sit, climb_step"

    def _query(self, prompt: str) -> str:
        import urllib.request
        import json as json_mod

        if self.config.provider == "ollama":
            url = f"{self.config.base_url}/api/generate"
            payload = json_mod.dumps({
                "model": self.config.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.config.temperature,
                    "num_predict": self.config.max_tokens,
                },
            }).encode()
            req = urllib.request.Request(
                url, data=payload,
                headers={"Content-Type": "application/json"}, method="POST")
            timeout_s = self.config.timeout_ms / 1000.0
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                data = json_mod.loads(resp.read())
                return data.get("response", "")

        elif self.config.provider == "openai":
            url = f"{self.config.base_url}/v1/chat/completions"
            payload = json_mod.dumps({
                "model": self.config.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
            }).encode()
            req = urllib.request.Request(
                url, data=payload,
                headers={"Content-Type": "application/json"}, method="POST")
            timeout_s = self.config.timeout_ms / 1000.0
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                data = json_mod.loads(resp.read())
                return data["choices"][0]["message"]["content"]

        return ""

    def _parse_response(self, text: str) -> dict | None:
        text = text.strip()
        start = text.find("{")
        end = text.rfind("}") + 1
        if start < 0 or end <= start:
            return None
        try:
            data = json.loads(text[start:end])
            return {
                "action": str(data.get("action", "stand")),
                "speed": float(data.get("speed", 0.5)),
                "confidence": float(data.get("confidence", 0.5)),
                "reasoning": str(data.get("reasoning", "")),
            }
        except (json.JSONDecodeError, ValueError):
            return None
