"""Planned skills catalog; NEVER registers a gym task or changes the active reward recipe.

Derive skill-specific configurations from the shared ManagerBasedRLEnv foundation only
once a new skill is actually implemented and tested. Avoid inheritance from historical experiments.
"""
from dataclasses import dataclass
from typing import Literal

Status = Literal["implemented", "planned"]

@dataclass(frozen=True)
class SkillDefinition:
    name: str
    status: Status
    description: str
    safe_for_hardware: bool = False

CAPABILITIES = (
    SkillDefinition("canonical_locomotion_acquisition", "implemented", "Current v10.3.1 bilateral acquisition recipe; host validation pending"),
    SkillDefinition("stand", "planned", "Independent stable neutral pose acquisition"),
    SkillDefinition("directional_walk", "planned", "Full planar commanded locomotion"),
    SkillDefinition("side_step", "planned", "Lateral locomotion command specialization"),
    SkillDefinition("crouch_walk", "planned", "Height-conditioned locomotion"),
    SkillDefinition("turn", "planned", "Commanded yaw tracking"),
    SkillDefinition("stairs", "planned", "Contact-aware step height progression"),
    SkillDefinition("obstacle", "planned", "Footpath clearance under terrain variation"),
    SkillDefinition("jump", "planned", "Ballistic-to-landing transition"),
    SkillDefinition("recovery", "planned", "Safe support-regaining behavior"),
)
