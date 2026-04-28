"""Drive a Cadenza robot with HuggingFace's SmolVLA on the stairs course.

Same content as examples/smolvla-test/run_smolvla.py.

    pip install "lerobot[smolvla]"
    mjpython test.py
"""

import pathlib

import cadenza_lab as cadenza
import cadenza_lab.stack
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

SCENE = str(
    pathlib.Path(__file__).parent
    / "examples" / "smolvla-test" / "stairs_scene.xml"
)

policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")

cadenza.stack.register_world_model("smolvla", model=policy)
cadenza.stack.run(
    robot="go1",
    goal=(
        "walk_forward then walk_forward then walk_forward "
        "then climb_step then climb_step then climb_step then climb_step "
        "then walk_forward then sit"
    ),
    target=(-5.5, 0.0),     # green-beacon goal pad on the top deck
    xml_path=SCENE,
    max_iterations=40,
)
