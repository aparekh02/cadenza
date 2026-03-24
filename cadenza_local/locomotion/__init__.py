"""cadenza.locomotion — Memory-augmented locomotion for Unitree Go1/Go2.

Complete robot-specific layer. No RL. No fine-tuning required.

Usage
-----
    from cadenza_local.locomotion import (
        GO1, GO2, get_spec,           # robot specs (all physics hardcoded)
        GaitEngine,                    # analytical gait generation
        TerrainClassifier,             # online IMU-based terrain detection
        LocoController, LocoCommand,   # main control loop
        GroqAdvisor,                   # Groq AI guidance (requires GROQ_API_KEY)
        STM, MapMem, SkillMem, SafetyMem, UserMem,
        ExperienceLogger, load_config,
    )
"""

from cadenza_local.locomotion.robot_spec import (
    GO1, GO2, get_spec, RobotSpec,
    GAITS, GaitParams,
    JointLimits, MotorLimits, Kinematics,
    SafetyThresholds, Benchmarks,
    TerrainCapability,
    JOINT_NAMES, LEG_INDICES, FOOT_ORDER,
)
from cadenza_local.locomotion.kinematics import (
    foot_position_body, ik_leg,
    nominal_foot_positions,
    clip_joints, check_joint_margins,
    joint_vector_to_legs, legs_to_joint_vector,
)
from cadenza_local.locomotion.gait_engine       import GaitEngine
from cadenza_local.locomotion.terrain_classifier import TerrainClassifier, TerrainEstimate, BenchmarkResult
from cadenza_local.locomotion.groq_agent        import (
    GroqAdvisor, GroqMemoryLayer, EpisodeSteeringAgent,
    NavigatorAgent, MemoryAgent, BenchmarkAgent,
    NavigatorDecision, MemorySummary, BenchmarkAdvice,
)
from cadenza_local.locomotion.memory   import (
    STM, STMFrame,
    MapMem, TerrainCluster, stm_to_embedding,
    SkillMem, Skill, goal_to_embedding,
    SafetyMem, SafetyRule, SafetyCheckResult,
    UserMem, UserPreference,
)
from cadenza_local.locomotion.runtime       import LocoController, LocoCommand, ExperienceLogger
from cadenza_local.locomotion.config        import UnitreeConfig, load_config
from cadenza_local.locomotion.episode_guide import (
    EpisodeGuide,
    EpisodeRecord,
    PHASE_APPROACH, PHASE_SPRINT, PHASE_BOUND, PHASE_RECOVERY, PHASE_GOAL,
)
from cadenza_local.locomotion.local_agent import LocalSteeringAgent, list_ollama_models
from cadenza_local.locomotion.balance     import BalanceController

__all__ = [
    # Robot specs
    "GO1", "GO2", "get_spec", "RobotSpec",
    "GAITS", "GaitParams",
    "JointLimits", "MotorLimits", "Kinematics",
    "SafetyThresholds", "Benchmarks", "TerrainCapability",
    "JOINT_NAMES", "LEG_INDICES", "FOOT_ORDER",
    # Kinematics
    "foot_position_body", "ik_leg",
    "nominal_foot_positions", "clip_joints", "check_joint_margins",
    "joint_vector_to_legs", "legs_to_joint_vector",
    # Gait
    "GaitEngine",
    # Terrain
    "TerrainClassifier", "TerrainEstimate", "BenchmarkResult",
    # Groq agents
    "GroqAdvisor", "GroqMemoryLayer", "EpisodeSteeringAgent",
    "NavigatorAgent", "MemoryAgent", "BenchmarkAgent",
    "NavigatorDecision", "MemorySummary", "BenchmarkAdvice",
    # Memory compartments
    "STM", "STMFrame",
    "MapMem", "TerrainCluster", "stm_to_embedding",
    "SkillMem", "Skill", "goal_to_embedding",
    "SafetyMem", "SafetyRule", "SafetyCheckResult",
    "UserMem", "UserPreference",
    # Runtime
    "LocoController", "LocoCommand", "ExperienceLogger",
    # Config
    "UnitreeConfig", "load_config",
    # Episode planner
    "EpisodeGuide", "EpisodeRecord",
    "PHASE_APPROACH", "PHASE_SPRINT", "PHASE_BOUND", "PHASE_RECOVERY", "PHASE_GOAL",
    # Local LLM agent
    "LocalSteeringAgent", "list_ollama_models",
    # Balance / CoM stabiliser
    "BalanceController",
]
