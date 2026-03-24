"""Demo ingestion pipeline — encode demos and populate a MemoryBank.

Accepts 30 successful demos in a simple dict format and a VLA encode
function, then stores (embedding, action, task) in the bank.

Demo format:
    {
        "image":       np.ndarray | PIL.Image | bytes  (any VLA-accepted type)
        "state":       np.ndarray  (optional joint states)
        "instruction": str
        "action":      np.ndarray  shape (n_act,)
    }

Usage:
    from cadenza_local.vla_steer import MemoryBank, ingest

    def my_encode(image, instruction: str) -> np.ndarray:
        ...  # call your VLA encoder, return shape (dim,)

    bank = ingest(demos, encode_fn=my_encode)
    # or supply an existing bank to extend:
    bank = ingest(more_demos, encode_fn=my_encode, bank=bank)
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from cadenza_local.vla_steer.bank import MemoryBank

Demo = dict  # {"image": ..., "state": ..., "instruction": str, "action": np.ndarray}
EncodeFn = Callable[..., np.ndarray]  # (image, instruction) -> embedding


def ingest(
    demos: list[Demo],
    encode_fn: EncodeFn,
    bank: MemoryBank | None = None,
) -> MemoryBank:
    """Encode successful demos and store in a MemoryBank.

    Args:
        demos:     list of demo dicts (see module docstring)
        encode_fn: callable(image, instruction) -> np.ndarray embedding
        bank:      existing bank to extend, or None to create a new one
                   (dim is inferred from the first embedding)

    Returns:
        Populated MemoryBank ready for querying.

    Raises:
        ValueError: if demos is empty
    """
    if not demos:
        raise ValueError("demos list is empty — nothing to ingest")

    errors: list[str] = []
    embeddings: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    tasks: list[str] = []

    for i, demo in enumerate(demos):
        try:
            image = demo["image"]
            instruction = demo.get("instruction", "")
            action = np.array(demo["action"], dtype=np.float32)

            emb = np.array(encode_fn(image, instruction), dtype=np.float32)
            embeddings.append(emb)
            actions.append(action)
            tasks.append(instruction)
        except Exception as exc:
            errors.append(f"demo[{i}]: {exc}")

    if errors:
        import warnings
        warnings.warn(f"vla_steer.ingest: {len(errors)} demos skipped — {errors[:3]}")

    if not embeddings:
        raise RuntimeError("All demos failed to encode — bank is empty")

    if bank is None:
        dim = embeddings[0].shape[-1]
        bank = MemoryBank(dim)

    for emb, action, task in zip(embeddings, actions, tasks):
        bank.add(emb, action, task)

    return bank


def ingest_text_only(
    demos: list[Demo],
    embed_text_fn: Callable[[str], np.ndarray],
    bank: MemoryBank | None = None,
) -> MemoryBank:
    """Encode demos using instruction text only (no image encoder needed).

    Useful when the VLA task is primarily text-driven or for fast prototyping.

    Args:
        demos:          list of demo dicts
        embed_text_fn:  callable(instruction: str) -> np.ndarray embedding
        bank:           existing bank to extend

    Returns:
        Populated MemoryBank.
    """
    text_demos = [
        {**d, "image": None}
        for d in demos
    ]

    def wrap(image, instruction):
        return embed_text_fn(instruction)

    return ingest(text_demos, encode_fn=wrap, bank=bank)
