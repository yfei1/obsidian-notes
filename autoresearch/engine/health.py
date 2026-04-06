"""Re-export from autoresearch_core — real implementation lives there."""
from autoresearch_core.health import HealthReport, check_health

__all__ = ["HealthReport", "check_health"]
