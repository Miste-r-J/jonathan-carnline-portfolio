"""
Premarket planner package - generates the early morning summary and Discord embed.
"""

from .config import PlannerConfig, load_config

__all__ = [
    "PlannerConfig",
    "PlanComputation",
    "PlanPayload",
    "build_plan_payload",
    "generate_plan",
    "load_config",
]


def __getattr__(name: str):
    if name in {"PlanComputation", "generate_plan"}:
        from .core import PlanComputation, generate_plan

        mapping = {
            "PlanComputation": PlanComputation,
            "generate_plan": generate_plan,
        }
        return mapping[name]
    if name in {"PlanPayload", "build_plan_payload"}:
        from .embed import PlanPayload, build_plan_payload

        mapping = {
            "PlanPayload": PlanPayload,
            "build_plan_payload": build_plan_payload,
        }
        return mapping[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
