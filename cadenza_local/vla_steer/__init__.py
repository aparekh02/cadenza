"""vla_steer — zero-training VLA task specialisation via FAISS action memory.

Data flow:
    image + instruction → VLA encoder → embedding → FAISS query → top-k actions
                                                                ↓ blend (1-α)VLA + α·memory
                                    VLA action head → raw action ↗

Components:
    MemoryBank  — FAISS index of (embeddings, actions, task_labels)
    steer()     — blend VLA output with memory-retrieved actions
    ingest()    — encode successful demos and populate the bank

Quick start:
    from cadenza_local.vla_steer import MemoryBank, steer, ingest

    bank = ingest(demos, encode_fn=my_vla_encoder)          # 30 demos
    final = steer(vla_action, bank, embedding, alpha=0.3)   # 2ms overhead
"""

from cadenza_local.vla_steer.bank import MemoryBank
from cadenza_local.vla_steer.steer import steer
from cadenza_local.vla_steer.pipeline import ingest, ingest_text_only

__all__ = ["MemoryBank", "steer", "ingest", "ingest_text_only"]
