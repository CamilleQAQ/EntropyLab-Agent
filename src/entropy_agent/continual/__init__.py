"""Continual learning: frozen configuration, growing batch history, champion/challenger."""

from __future__ import annotations

from .loop import Champion, run_continual
from .plan import BatchPlan, PlanError, save_plan

__all__ = ["BatchPlan", "Champion", "PlanError", "run_continual", "save_plan"]
