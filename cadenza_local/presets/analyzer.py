"""Preset analyzer — orchestrates analysis of all inputs using Groq Vision.

Takes the raw inputs (images, robot spec, objects, text) and produces
structured knowledge for each basis category. Uses the existing Groq Vision
pipeline for image analysis, plus a lightweight text analysis pass for
task decomposition and spatial reasoning.
"""

from __future__ import annotations

import json
import os
import re
from typing import Optional

from dotenv import load_dotenv

from cadenza_local.presets.schemas import (
    ObjectProfile,
    TaskDirective,
    SpatialRelation,
    BasisPreset,
)


def analyze_task_text(
    description: str,
    objects: list[ObjectProfile],
    actions_hint: str = "",
) -> TaskDirective:
    """Analyze task text input into a structured directive.

    Uses Groq LLM for task decomposition when available, falls back
    to rule-based extraction.

    Args:
        description: What the robot will be doing.
        actions_hint: Additional text about specific actions.
        objects: Objects the robot interacts with.

    Returns:
        Structured TaskDirective.
    """
    obj_names = [o.name for o in objects]

    # Try LLM-based analysis
    try:
        return _llm_analyze_task(description, obj_names, actions_hint)
    except Exception as e:
        print(f"  LLM task analysis unavailable ({e}), using rule-based extraction")
        return _rule_based_task(description, obj_names, actions_hint)


def _llm_analyze_task(
    description: str,
    object_names: list[str],
    actions_hint: str,
) -> TaskDirective:
    """Use Groq LLM for task decomposition."""
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("No GROQ_API_KEY")

    from groq import Groq
    client = Groq(api_key=api_key)

    objects_str = ", ".join(object_names) if object_names else "none specified"
    prompt = f"""Analyze this robot task and decompose it into a structured plan.

Task: {description}
{f"Additional actions: {actions_hint}" if actions_hint else ""}
Objects involved: {objects_str}

Return ONLY a JSON object:
{{
  "action_sequence": ["step 1", "step 2", ...],
  "required_interactions": ["grasping", "pouring", ...],
  "speed_preference": 0.3,
  "behavioral_notes": "any notes about how to perform the task"
}}

Keep action_sequence specific and ordered. Speed: 0.1=very slow, 0.3=careful, 0.7=normal, 1.0=fast.
Return ONLY JSON."""

    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=500,
        response_format={"type": "json_object"},
    )

    reply = response.choices[0].message.content.strip()
    json_match = re.search(r"\{[\s\S]*\}", reply)
    if not json_match:
        raise ValueError("No JSON in response")

    data = json.loads(json_match.group())

    return TaskDirective(
        description=description,
        action_sequence=data.get("action_sequence", []),
        required_objects=object_names,
        required_interactions=data.get("required_interactions", []),
        speed_preference=float(data.get("speed_preference", 0.3)),
        behavioral_notes=data.get("behavioral_notes", ""),
    )


def _rule_based_task(
    description: str,
    object_names: list[str],
    actions_hint: str,
) -> TaskDirective:
    """Fallback rule-based task decomposition."""
    desc_lower = description.lower()
    actions_lower = actions_hint.lower()
    combined = f"{desc_lower} {actions_lower}"

    # Detect interaction types from keywords
    interactions = []
    keyword_map = {
        "pick": "grasping",
        "grab": "grasping",
        "grasp": "grasping",
        "hold": "holding",
        "pour": "pouring",
        "place": "placing",
        "set": "placing",
        "put": "placing",
        "lift": "lifting",
        "stir": "stirring",
        "mix": "stirring",
        "shake": "shaking",
        "reach": "reaching",
        "move": "moving",
    }
    for keyword, interaction in keyword_map.items():
        if keyword in combined and interaction not in interactions:
            interactions.append(interaction)

    # Build simple action sequence from object names + interactions
    actions = []
    if actions_hint:
        # Split on commas, semicolons, or "then"
        parts = re.split(r"[,;]|\bthen\b", actions_hint)
        actions = [p.strip() for p in parts if p.strip()]
    elif object_names:
        for obj in object_names:
            if "grasping" in interactions:
                actions.append(f"pick up {obj}")
            actions.append(f"interact with {obj}")

    # Speed heuristic
    speed = 0.3
    if "slow" in combined or "careful" in combined:
        speed = 0.1
    elif "fast" in combined or "quick" in combined:
        speed = 0.7

    return TaskDirective(
        description=description,
        action_sequence=actions,
        required_objects=object_names,
        required_interactions=interactions,
        speed_preference=speed,
        behavioral_notes=actions_hint,
    )


def analyze_spatial_relations(
    objects: list[ObjectProfile],
    robot_name: str = "robot",
) -> list[SpatialRelation]:
    """Infer spatial relations between objects based on their positions.

    Uses estimated positions from image analysis to determine left/right,
    near/far, and relative arrangement.
    """
    relations = []

    for i, obj_a in enumerate(objects):
        # Robot-to-object relation
        ax, ay, az = obj_a.estimated_position
        dist_to_robot = (ax**2 + ay**2 + az**2) ** 0.5
        relations.append(SpatialRelation(
            object_a=robot_name,
            object_b=obj_a.name,
            relation="reachable" if dist_to_robot < 0.5 else "distant",
            estimated_distance_m=round(dist_to_robot, 3),
        ))

        # Object-to-object relations
        for j in range(i + 1, len(objects)):
            obj_b = objects[j]
            bx, by, bz = obj_b.estimated_position

            dx = bx - ax
            dy = by - ay
            dist = ((dx**2) + (dy**2)) ** 0.5

            if abs(dx) > abs(dy):
                relation = "right_of" if dx > 0 else "left_of"
            else:
                relation = "behind" if dy > 0 else "in_front"

            relations.append(SpatialRelation(
                object_a=obj_a.name,
                object_b=obj_b.name,
                relation=relation,
                estimated_distance_m=round(dist, 3),
            ))

    return relations


def task_to_basis_records(
    directive: TaskDirective,
    user_id: str,
    preset_id: str,
) -> list[dict]:
    """Convert a TaskDirective into basis records."""
    records = []

    # Overall task directive
    actions_str = " -> ".join(directive.action_sequence) if directive.action_sequence else "unstructured"
    records.append({
        "user_id": user_id,
        "category": "task_directive",
        "content": (
            f"Task: {directive.description}. "
            f"Action sequence: {actions_str}. "
            f"Required interactions: {', '.join(directive.required_interactions)}. "
            f"Speed: {directive.speed_preference}."
        ),
        "data": {
            "description": directive.description,
            "action_sequence": directive.action_sequence,
            "required_objects": directive.required_objects,
            "required_interactions": directive.required_interactions,
            "speed_preference": directive.speed_preference,
            "behavioral_notes": directive.behavioral_notes,
        },
        "source": "text_input",
        "confidence": 0.9,
        "preset_id": preset_id,
    })

    # Per-action records
    for i, action in enumerate(directive.action_sequence):
        records.append({
            "user_id": user_id,
            "category": "task_directive",
            "content": f"Step {i + 1}: {action}",
            "data": {"step": i + 1, "action": action},
            "source": "text_input",
            "confidence": 0.85,
            "preset_id": preset_id,
        })

    return records


def objects_to_basis_records(
    objects: list[ObjectProfile],
    user_id: str,
    preset_id: str,
) -> list[dict]:
    """Convert ObjectProfiles into basis records."""
    records = []

    for obj in objects:
        interactions_str = ", ".join(obj.interaction_types) if obj.interaction_types else "none"
        records.append({
            "user_id": user_id,
            "category": "object_profile",
            "content": (
                f"Object '{obj.name}': shape={obj.shape}, "
                f"estimated mass={obj.estimated_mass_kg:.2f} kg, "
                f"interactions: {interactions_str}."
            ),
            "data": {
                "name": obj.name,
                "shape": obj.shape,
                "estimated_size": list(obj.estimated_size),
                "estimated_position": list(obj.estimated_position),
                "estimated_mass_kg": obj.estimated_mass_kg,
                "interaction_types": obj.interaction_types,
                "properties": obj.properties,
            },
            "source": "analysis",
            "confidence": 0.7,
            "preset_id": preset_id,
        })

    return records


def spatial_to_basis_records(
    relations: list[SpatialRelation],
    user_id: str,
    preset_id: str,
) -> list[dict]:
    """Convert SpatialRelations into basis records."""
    records = []

    for rel in relations:
        records.append({
            "user_id": user_id,
            "category": "spatial_relation",
            "content": (
                f"{rel.object_a} is {rel.relation} {rel.object_b} "
                f"(~{rel.estimated_distance_m:.2f}m apart)."
            ),
            "data": {
                "object_a": rel.object_a,
                "object_b": rel.object_b,
                "relation": rel.relation,
                "estimated_distance_m": rel.estimated_distance_m,
            },
            "source": "analysis",
            "confidence": 0.6,
            "preset_id": preset_id,
        })

    return records
