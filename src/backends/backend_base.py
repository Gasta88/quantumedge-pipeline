"""
Abstract base class for quantum hardware backends.

This module defines the QuantumBackend ABC, a high-level job-oriented interface
for quantum hardware. Unlike the circuit-level QuantumHardwareInterface in
src/solvers/quantum_hardware_interface.py, this interface operates at the
job/problem level and is driven by company profiles.

Implementations:
    - RotoniumMockBackend: Simulated photonic QPU (OAM encoding, edge-optimised)
    - QuiXCloudBackend: Real QuiX Quantum cloud API client
"""

from abc import ABC, abstractmethod
from typing import Any, Dict


class QuantumBackend(ABC):
    """
    Abstract interface for quantum hardware backends.

    Swap this out to connect to different QPU vendors. Each backend
    is instantiated by the profile-driven factory in src/backends/__init__.py.
    """

    @abstractmethod
    def submit_job(self, circuit: Any, shots: int = 1024) -> str:
        """
        Submit a quantum circuit for execution.

        Args:
            circuit: Quantum circuit or problem description to execute.
            shots: Number of measurement repetitions.

        Returns:
            A unique job identifier string.
        """

    @abstractmethod
    def get_job_result(self, job_id: str) -> dict:
        """
        Retrieve the result of a completed job.

        Args:
            job_id: Identifier returned by submit_job.

        Returns:
            Dictionary containing at minimum:
                - counts: Measurement outcome counts
                - success: bool indicating completion
                - execution_time_ms: Wall-clock time in milliseconds
        """

    @abstractmethod
    def get_hardware_specs(self) -> dict:
        """
        Return QPU capabilities and current status.

        Returns:
            Dictionary describing the hardware, e.g.:
                - provider: Vendor name
                - technology: Photonic / superconducting / etc.
                - max_qubits: Maximum supported qubits
                - status: online / offline / maintenance
        """

    @abstractmethod
    def estimate_job_cost(self, circuit: Any) -> dict:
        """
        Estimate execution cost before running a job.

        Args:
            circuit: Quantum circuit or problem description.

        Returns:
            Dictionary containing at minimum:
                - estimated_time_ms: Predicted wall-clock time
                - estimated_energy_mj: Predicted energy in millijoules
                - estimated_cost_usd: Predicted monetary cost (0 for mock)
        """
