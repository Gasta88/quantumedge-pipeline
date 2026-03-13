"""
QuantumEdge Pipeline - Quantum-Classical Hybrid Optimization System
"""

__version__ = "0.2.0"

from src.config import settings
from src.profile_loader import get_active_profile

__all__ = ["settings", "get_active_profile"]
