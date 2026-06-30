"""Cleaning Planner to separate AutoFixes from Warnings based on Trust Scores."""

from typing import Any


class CleaningPlanner:
    """Takes evaluated actions from the Trust Engine and separates them.

    Actions >= threshold become AutoFix commands.
    Actions < threshold become Warnings.
    """

    def __init__(self, threshold: float):
        """Initializes the PlanningEngine with a confidence threshold."""
        self.threshold = threshold

    def plan(
        self, evaluated_actions: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Separates actions into fixes and warnings.

        Args:
            evaluated_actions: List of potential actions with confidence scores.

        Returns:
            Tuple of (autofix_actions, warning_actions).
        """
        autofixes = []
        warnings = []

        for action in evaluated_actions:
            if action.get("confidence", 0.0) >= self.threshold:
                autofixes.append(action)
            else:
                warnings.append(action)

        return autofixes, warnings
