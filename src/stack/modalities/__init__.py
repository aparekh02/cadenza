"""cadenza.stack.modalities — plug-in interface for multi-modal sensing.

The stack ships only the framework: the abstract ``Modality`` base class,
the ``ModalityResult`` payload, and a registry. **No models are bundled.**
The client decides which multi-modal sensing models to add.

Define a modality in your script (or import one from a separate package),
then pass it to the stack::

    from cadenza.stack import Modality, ModalityResult
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    class DepthAnythingModality(Modality):
        name = "depth_anything_v2"
        def setup(self):
            self.proc = AutoImageProcessor.from_pretrained(...)
            self.model = AutoModelForDepthEstimation.from_pretrained(...)
        def compute(self, observation):
            depth = self.model(self.proc(images=observation.camera, ...))
            return ModalityResult(keys={"depth_map": depth}, summary="depth ok")

    cadenza.stack.run(..., modalities=[DepthAnythingModality()])

Each tick the stack calls ``compute`` on every modality and merges the
returned keys into the observation dict the world-model adapter sees.
"""

from cadenza.stack.modalities.base import (
    Modality,
    ModalityResult,
    register_modality,
    get_modality,
    list_modalities,
)

__all__ = [
    "Modality",
    "ModalityResult",
    "register_modality",
    "get_modality",
    "list_modalities",
]
