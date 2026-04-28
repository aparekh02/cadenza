"""Model detector — finds a world model at the project root, in the
HuggingFace cache, or already loaded in the current Python process.

The client side stays trivial: ``cadenza.stack.run(robot="go1", goal=...)``
calls into here, which scans for a known world model and returns a
``WorldModelHandle`` bound to the right adapter.
"""

from __future__ import annotations

import importlib
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cadenza.stack.adapters.base import (
    WorldModelAdapter,
    get_adapter,
    list_adapters,
)


@dataclass
class WorldModelHandle:
    """Resolved world model: adapter class + checkpoint location + load state."""
    adapter_cls: type[WorldModelAdapter]
    checkpoint: Path | str | None
    source: str                       # "root" | "hf-cache" | "process" | "registered" | "fallback"
    loaded_model: Any = None          # already-loaded model object, if any
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.adapter_cls.name

    def build(self) -> WorldModelAdapter:
        """Instantiate the adapter (does not load weights)."""
        return self.adapter_cls(
            checkpoint=self.checkpoint,
            model=self.loaded_model,
        )


# ── Manual registration ───────────────────────────────────────────────────────

_REGISTERED: WorldModelHandle | None = None


def register_world_model(
    adapter: str | type[WorldModelAdapter],
    checkpoint: str | Path | None = None,
    model: Any = None,
    **metadata,
) -> WorldModelHandle:
    """Manually pin which world model the stack should use.

    Useful when the user has the model loaded in memory and does not want the
    stack to scan disk. Overrides auto-detection.
    """
    global _REGISTERED
    cls = adapter if isinstance(adapter, type) else get_adapter(adapter)
    _REGISTERED = WorldModelHandle(
        adapter_cls=cls,
        checkpoint=Path(checkpoint) if checkpoint else None,
        source="registered",
        loaded_model=model,
        metadata=metadata,
    )
    return _REGISTERED


def clear_registration() -> None:
    global _REGISTERED
    _REGISTERED = None


# ── Detection ─────────────────────────────────────────────────────────────────

def detect_world_model(
    root: str | Path = ".",
    *,
    use_registered: bool = True,
    use_hf_cache: bool = True,
    use_process: bool = True,
    fallback_adapter: str | None = "mock",
) -> WorldModelHandle:
    """Find a world model and return a handle bound to its adapter.

    Resolution order:
      1. A model registered via ``register_world_model``.
      2. A checkpoint folder at ``root`` (or any of its first-level subdirs).
      3. A HuggingFace cache hit.
      4. A model already imported in the current process.
      5. ``fallback_adapter`` (default: ``MockAdapter``) if nothing else hits.

    Returns a ``WorldModelHandle``. Raises ``RuntimeError`` only if no fallback
    is allowed and nothing was found.
    """
    if use_registered and _REGISTERED is not None:
        return _REGISTERED

    root_path = Path(root).resolve()

    # 2. Project root + first-level dirs.
    candidates = [root_path] + (
        [p for p in root_path.iterdir() if p.is_dir()]
        if root_path.exists() else []
    )
    for adapter_cls in list_adapters():
        for cand in candidates:
            try:
                hit = adapter_cls.detect(cand)
            except Exception:
                hit = None
            if hit:
                return WorldModelHandle(
                    adapter_cls=adapter_cls,
                    checkpoint=hit,
                    source="root",
                )

    # 3. HuggingFace cache (~/.cache/huggingface/hub).
    if use_hf_cache:
        hf_root = _hf_cache_dir()
        if hf_root and hf_root.exists():
            for adapter_cls in list_adapters():
                hit = _scan_hf_cache(adapter_cls, hf_root)
                if hit:
                    return WorldModelHandle(
                        adapter_cls=adapter_cls,
                        checkpoint=hit,
                        source="hf-cache",
                    )

    # 4. Already-imported modules in this process.
    if use_process:
        proc_hit = _scan_process()
        if proc_hit:
            return proc_hit

    # 5. Fallback.
    if fallback_adapter:
        cls = get_adapter(fallback_adapter)
        return WorldModelHandle(
            adapter_cls=cls,
            checkpoint=None,
            source="fallback",
        )

    raise RuntimeError(
        f"No world model detected at {root_path} and no fallback configured. "
        f"Drop a checkpoint in the project root, install one via the HF cache, "
        f"or call cadenza.stack.register_world_model(...)."
    )


# ── HF cache scan ─────────────────────────────────────────────────────────────

def _hf_cache_dir() -> Path | None:
    env = os.environ.get("HF_HOME") or os.environ.get("HUGGINGFACE_HUB_CACHE")
    if env:
        cache = Path(env)
        if cache.name != "hub":
            cache = cache / "hub"
        return cache
    home = Path.home()
    return home / ".cache" / "huggingface" / "hub"


def _scan_hf_cache(adapter_cls: type[WorldModelAdapter], hf_root: Path) -> Path | None:
    """Probe the HF hub cache for entries matching the adapter's keywords.

    The cache layout looks like ``models--<org>--<name>/snapshots/<sha>/...``.
    We let each adapter's ``detect`` method do the matching by feeding it the
    snapshot directories.
    """
    for entry in hf_root.glob("models--*/snapshots/*"):
        if not entry.is_dir():
            continue
        try:
            hit = adapter_cls.detect(entry)
        except Exception:
            hit = None
        if hit:
            return hit
    return None


# ── Process scan ──────────────────────────────────────────────────────────────

# Maps adapter name -> (module substring, attribute hint) used to spot a
# loaded model in sys.modules.
_PROCESS_HINTS: dict[str, list[str]] = {
    "pi_zero": ["pi_zero", "pi05", "physical_intelligence", "openpi"],
    "openvla": ["openvla", "prismatic"],
    "octo": ["octo"],
    "gr00t": ["gr00t", "groot"],
}


def _scan_process() -> WorldModelHandle | None:
    """Best-effort: see if a known WM module is already imported.

    We don't reach into module internals to grab the loaded weights — we just
    flag that the user is clearly running with this WM and let the adapter
    resolve the checkpoint itself when ``load()`` is called.

    Cadenza's own modules are excluded so the stack's adapter files don't
    self-trigger the process scan.
    """
    loaded = [
        m for m in sys.modules
        if m and not m.startswith("cadenza")
    ]
    for adapter_name, hints in _PROCESS_HINTS.items():
        if adapter_name not in {a.name for a in list_adapters()}:
            continue
        for hint in hints:
            if any(hint in mod for mod in loaded):
                try:
                    cls = get_adapter(adapter_name)
                except KeyError:
                    continue
                return WorldModelHandle(
                    adapter_cls=cls,
                    checkpoint=None,
                    source="process",
                    metadata={"hint": hint},
                )
    return None


# ── Filesystem helpers used by adapters ───────────────────────────────────────

def has_any(path: Path, names: list[str]) -> bool:
    """True if `path` is a dir containing any of the listed file names."""
    if not path.is_dir():
        return False
    contents = {p.name for p in path.iterdir()}
    return any(n in contents for n in names)


def has_keyword(path: Path, keywords: list[str]) -> bool:
    """True if any keyword appears in `path` or its immediate file names."""
    haystack = str(path).lower() + " " + " ".join(
        p.name.lower() for p in path.iterdir() if p.is_file()
    ) if path.is_dir() else str(path).lower()
    return any(k.lower() in haystack for k in keywords)


def import_optional(name: str):
    """Try to import a module; return None on failure."""
    try:
        return importlib.import_module(name)
    except Exception:
        return None
