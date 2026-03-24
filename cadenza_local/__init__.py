import os as _os
_os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

"""Cadenza — Action library + LoRA adaptation for quadruped VLA control.

Quick start::

    import cadenza
    cadenza.run("walk forward 2 meters then turn left then jump")

Or::

    sim = cadenza.Sim("go1")
    sim.run("shake hand then rear kick")
"""

from cadenza_local.vla_steer import MemoryBank, steer, ingest, ingest_text_only
from cadenza_local.actions import ActionSpec, ActionPhase, ActionLibrary, get_action, list_actions, get_library
from cadenza_local.sim import Sim, run
from cadenza_local.go1 import Go1, Step
from cadenza_local.g1 import G1
from cadenza_local.lora.translator import ActionCall


def go1(**kwargs) -> Go1:
    """Create a Go1 robot controller.

    Usage::

        import cadenza
        go1 = cadenza.go1()
        go1.run([go1.jump(), go1.walk_forward(speed=1.5)])
    """
    return Go1(**kwargs)


def g1(**kwargs) -> G1:
    """Create a G1 humanoid controller.

    Usage::

        import cadenza
        g1 = cadenza.g1()
        g1.run([g1.stand(), g1.walk_forward(), g1.lift_left_hand()])
    """
    return G1(**kwargs)

_LAZY: dict[str, tuple[str, str]] = {
    "LoRALayer":         ("cadenza.lora", "LoRALayer"),
    "LoRATranslator":    ("cadenza.lora", "LoRATranslator"),
    "LoRAOptimizer":     ("cadenza.lora", "LoRAOptimizer"),
    "ActionCall":        ("cadenza.lora", "ActionCall"),
    "ActionPlan":        ("cadenza.lora", "ActionPlan"),
    "SensorSnapshot":    ("cadenza.lora", "SensorSnapshot"),
    "EnvironmentState":  ("cadenza.lora", "EnvironmentState"),
    "Go1Sim":            ("cadenza.sim", "Go1Sim"),
    "STM":               ("cadenza.locomotion", "STM"),
    "STMFrame":          ("cadenza.locomotion", "STMFrame"),
    "MapMem":            ("cadenza.locomotion", "MapMem"),
    "TerrainCluster":    ("cadenza.locomotion", "TerrainCluster"),
    "stm_to_embedding":  ("cadenza.locomotion", "stm_to_embedding"),
    "SkillMem":          ("cadenza.locomotion", "SkillMem"),
    "Skill":             ("cadenza.locomotion", "Skill"),
    "goal_to_embedding": ("cadenza.locomotion", "goal_to_embedding"),
    "SafetyMem":         ("cadenza.locomotion", "SafetyMem"),
    "SafetyRule":        ("cadenza.locomotion", "SafetyRule"),
    "SafetyCheckResult": ("cadenza.locomotion", "SafetyCheckResult"),
    "UserMem":           ("cadenza.locomotion", "UserMem"),
    "UserPreference":    ("cadenza.locomotion", "UserPreference"),
    "LocoController":    ("cadenza.locomotion", "LocoController"),
    "LocoCommand":       ("cadenza.locomotion", "LocoCommand"),
    "ExperienceLogger":  ("cadenza.locomotion", "ExperienceLogger"),
    "UnitreeConfig":     ("cadenza.locomotion", "UnitreeConfig"),
    "load_config":       ("cadenza.locomotion", "load_config"),
    "RobotSetup":        ("cadenza.robot_setup", "RobotSetup"),
    "RobotSpec":         ("cadenza.robot_setup", "RobotSpec"),
    "RobotHints":        ("cadenza.robot_setup", "RobotHints"),
    "register_robot":    ("cadenza.robot_setup", "register_robot"),
    "list_robots":       ("cadenza.robot_setup", "list_robots"),
    "get_robot_hints":   ("cadenza.robot_setup", "get_robot_hints"),
    "SceneObject":       ("cadenza.task", "SceneObject"),
    "Interaction":       ("cadenza.task", "Interaction"),
    "SceneExtraction":   ("cadenza.task", "SceneExtraction"),
    "TaskPreset":        ("cadenza.task", "TaskPreset"),
    "SceneBuilder":      ("cadenza.task", "SceneBuilder"),
    "GraspManager":      ("cadenza.task", "GraspManager"),
    "load_motion_images":      ("cadenza.vision", "load_motion_images"),
    "extract_skeleton":        ("cadenza.vision", "extract_skeleton"),
    "extract_scene":           ("cadenza.vision", "extract_scene"),
    "compute_motion_sequence": ("cadenza.vision", "compute_motion_sequence"),
    "MotionImage":       ("cadenza.vision", "MotionImage"),
    "SkeletonPose":      ("cadenza.vision", "SkeletonPose"),
    "MotionSequence":    ("cadenza.vision", "MotionSequence"),
    "MotionMapper":      ("cadenza.motion_mapper", "MotionMapper"),
    "JointTrajectory":   ("cadenza.motion_mapper", "JointTrajectory"),
    "PoseDetector":      ("cadenza.pose", "PoseDetector"),
    "PoseResult":        ("cadenza.pose", "PoseResult"),
    "annotate_frame":    ("cadenza.annotate", "annotate_frame"),
    "annotate_sequence": ("cadenza.annotate", "annotate_sequence"),
    "PresetBuilder":     ("cadenza.presets", "PresetBuilder"),
    "BasisPreset":       ("cadenza.presets", "BasisPreset"),
    "MotionBlueprint":   ("cadenza.presets", "MotionBlueprint"),
    "RobotProfile":      ("cadenza.presets", "RobotProfile"),
    "ObjectProfile":     ("cadenza.presets", "ObjectProfile"),
    "TaskDirective":     ("cadenza.presets", "TaskDirective"),
    "SpatialRelation":   ("cadenza.presets", "SpatialRelation"),
    "FrameAnalysis":     ("cadenza.presets", "FrameAnalysis"),
    "DynamicsProfile":   ("cadenza.presets", "DynamicsProfile"),
    "MotorProfile":      ("cadenza.presets", "MotorProfile"),
    "Microstep":         ("cadenza.presets", "Microstep"),
    "MicrostepSequence": ("cadenza.presets", "MicrostepSequence"),
    "analyze_motion_images": ("cadenza.presets", "analyze_motion_images"),
    "annotate_images":       ("cadenza.presets", "annotate_images"),
    "extract_robot_profile": ("cadenza.presets", "extract_robot_profile"),
    "analyze_task_text":     ("cadenza.presets", "analyze_task_text"),
    "analyze_spatial_relations": ("cadenza.presets", "analyze_spatial_relations"),
    "compute_dynamics":      ("cadenza.presets", "compute_dynamics"),
    "compute_motor_profile": ("cadenza.presets", "compute_motor_profile"),
    "generate_microsteps":   ("cadenza.presets", "generate_microsteps"),
}


def __getattr__(name: str):
    if name in _LAZY:
        mod_path, attr = _LAZY[name]
        import importlib, sys
        mod = importlib.import_module(mod_path)
        val = getattr(mod, attr)
        sys.modules[__name__].__dict__[name] = val
        return val
    raise AttributeError(f"module 'cadenza' has no attribute {name!r}")
