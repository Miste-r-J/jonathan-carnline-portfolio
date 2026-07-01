from __future__ import annotations

from dataclasses import replace
from typing import Optional

from .config import PlannerConfig, PlannerTargetConfig
from .core import PlanComputation, generate_plan
from .embed import EmbedField as PlanField
from .embed import PlanEmbed, PlanPayload, build_plan_payload as _compose_payload

__all__ = [
    "PlanField",
    "PlanEmbed",
    "PlanPayload",
    "build_plan_payload",
    "generate_plan_payload",
]


def build_plan_payload(
    config: PlannerConfig,
    target: Optional[PlannerTargetConfig] = None,
) -> PlanPayload:
    """
    Backwards-compatible wrapper that produces the plan payload.
    """
    effective = _apply_target_override(config, target)
    computation = generate_plan(effective)
    return _compose_payload(effective, computation)


def generate_plan_payload(
    config: PlannerConfig,
    target: Optional[PlannerTargetConfig] = None,
) -> PlanComputation:
    """
    Generate the raw computation payload for callers that need direct access.
    """
    effective = _apply_target_override(config, target)
    return generate_plan(effective)


def _apply_target_override(
    config: PlannerConfig,
    target: Optional[PlannerTargetConfig],
) -> PlannerConfig:
    if target is None:
        return config

    data_cfg = replace(config.data, csv_path=target.csv_path)
    instrument = target.instrument or config.instrument
    signals_path = target.signals_path or config.signals_path
    return replace(config, instrument=instrument, data=data_cfg, signals_path=signals_path)
