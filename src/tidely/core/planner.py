"""Cleaning Planner to separate AutoFixes from Warnings based on Trust Scores."""

from typing import Any, Dict, List, Tuple


class CleaningPlanner:
    """Takes evaluated actions from the Trust Engine and separates them.
    
    Actions >= threshold become AutoFix commands.
    Actions < threshold become Warnings.
    """
    
    def __init__(self, threshold: float):
        self.threshold = threshold
        
    def plan(self, evaluated_actions: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
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
