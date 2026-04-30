"""SmolVLA + Depth-Anything-V2-Small on the stairs course.

    pip install "lerobot[smolvla]" "transformers>=4.43" pillow
    mjpython test.py
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent / "examples" / "smolvla-test"))

import cadenza_lab as cadenza
import cadenza_lab.stack
from depth_anything_v2_small import DepthAnythingV2Small
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

cadenza.stack.register_world_model(
    "smolvla", model=SmolVLAPolicy.from_pretrained("lerobot/smolvla_base"),
)

cadenza.stack.run(
    robot="go1",
    goal="reach the green beacon at the top of the stairs and sit",
    target=(-5.5, 0.0),
    xml_path="examples/smolvla-test/stairs_scene.xml",
    modalities=[DepthAnythingV2Small()],
)
