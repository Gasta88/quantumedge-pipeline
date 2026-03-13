"""
Mock backend simulating Rotonium's photonic QPU characteristics.

This backend produces realistic simulated results for demonstration and
testing without requiring real quantum hardware. It models:
    - OAM (Orbital Angular Momentum) qudit encoding
    - Room-temperature photonic noise
    - Edge-optimised power consumption (SWaP constraints)
    - Compact form-factor execution profiles
"""

import hashlib
import random
import time
from typing import Any, Dict
from uuid import uuid4

from src.backends.backend_base import QuantumBackend


class RotoniumMockBackend(QuantumBackend):
    """
    Simulated Rotonium photonic QPU backend.

    Generates plausible quantum results by modelling photonic noise
    and OAM encoding characteristics. All execution happens locally —
    no network calls are made.
    """

    # Hardware specification constants
    MAX_QUBITS = 20
    OPERATING_TEMP_K = 293.15  # Room temperature (~20 C)
    PHOTON_LOSS_RATE = 0.02  # 2% per gate layer
    DETECTOR_EFFICIENCY = 0.95
    OAM_MODES = 4  # Number of OAM modes per photon

    def __init__(self) -> None:
        self._jobs: Dict[str, dict] = {}

    # ------------------------------------------------------------------
    # QuantumBackend interface
    # ------------------------------------------------------------------

    def submit_job(self, circuit: Any, shots: int = 1024) -> str:
        """Submit a simulated photonic job and return a job_id."""
        job_id = f"rot-{uuid4().hex[:12]}"
        start = time.perf_counter()

        # Derive a reproducible circuit "fingerprint" for deterministic results
        circuit_repr = str(circuit)
        seed = int(hashlib.md5(circuit_repr.encode()).hexdigest()[:8], 16)
        rng = random.Random(seed)

        # Simulate measurement outcomes with photonic noise
        num_qubits = self._estimate_qubits(circuit)
        counts = self._simulate_photonic_measurement(num_qubits, shots, rng)

        elapsed_ms = (time.perf_counter() - start) * 1000
        # Add a realistic latency component for QPU + classical control
        simulated_qpu_ms = num_qubits * 0.8 + shots * 0.005
        total_ms = elapsed_ms + simulated_qpu_ms

        self._jobs[job_id] = {
            "counts": counts,
            "success": True,
            "execution_time_ms": round(total_ms, 2),
            "num_qubits": num_qubits,
            "shots": shots,
            "energy_mj": self._estimate_energy(num_qubits, shots),
            "noise_model": "photonic_oam",
        }
        return job_id

    def get_job_result(self, job_id: str) -> dict:
        """Retrieve result of a previously submitted mock job."""
        if job_id not in self._jobs:
            raise KeyError(f"Job '{job_id}' not found in Rotonium mock backend")
        return self._jobs[job_id]

    def get_hardware_specs(self) -> dict:
        """Return Rotonium photonic QPU specifications."""
        return {
            "provider": "Rotonium",
            "technology": "Photonic (OAM encoding)",
            "max_qubits": self.MAX_QUBITS,
            "oam_modes": self.OAM_MODES,
            "operating_temperature_k": self.OPERATING_TEMP_K,
            "photon_loss_rate": self.PHOTON_LOSS_RATE,
            "detector_efficiency": self.DETECTOR_EFFICIENCY,
            "cryogenics_required": False,
            "status": "online",
            "deployment_targets": ["aerospace", "mobile", "ground_server"],
        }

    def estimate_job_cost(self, circuit: Any) -> dict:
        """Estimate execution cost for a Rotonium mock job."""
        num_qubits = self._estimate_qubits(circuit)
        shots = 1024  # default assumption
        return {
            "estimated_time_ms": round(num_qubits * 0.8 + shots * 0.005, 2),
            "estimated_energy_mj": self._estimate_energy(num_qubits, shots),
            "estimated_cost_usd": 0.0,  # mock — no monetary cost
            "swap_score": self._calculate_swap_score(num_qubits),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_qubits(circuit: Any) -> int:
        """Heuristically estimate qubit count from a circuit description."""
        if hasattr(circuit, "num_qubits"):
            return min(circuit.num_qubits, RotoniumMockBackend.MAX_QUBITS)
        if isinstance(circuit, dict) and "num_qubits" in circuit:
            return min(circuit["num_qubits"], RotoniumMockBackend.MAX_QUBITS)
        # Fallback: small default
        return 8

    def _simulate_photonic_measurement(
        self, num_qubits: int, shots: int, rng: random.Random
    ) -> Dict[str, int]:
        """
        Generate measurement counts with photonic noise characteristics.

        Models photon loss and detector inefficiency to produce realistic
        outcome distributions biased toward lower-energy states.
        """
        num_outcomes = min(2**num_qubits, 64)  # cap for performance
        outcomes: Dict[str, int] = {}
        remaining = shots

        for i in range(num_outcomes):
            bitstring = format(i, f"0{num_qubits}b")
            # Photonic bias: lower Hamming-weight states are more likely
            hamming = bitstring.count("1")
            weight = (1.0 - self.PHOTON_LOSS_RATE) ** hamming
            weight *= self.DETECTOR_EFFICIENCY
            count = max(1, int(rng.gauss(remaining / (num_outcomes - i), remaining * 0.05)))
            count = int(count * weight)
            count = max(0, min(count, remaining))
            if count > 0:
                outcomes[bitstring] = count
                remaining -= count
            if remaining <= 0:
                break

        # Distribute any remaining shots to the ground state
        if remaining > 0:
            ground = "0" * num_qubits
            outcomes[ground] = outcomes.get(ground, 0) + remaining

        return outcomes

    def _estimate_energy(self, num_qubits: int, shots: int) -> float:
        """Estimate energy consumption in millijoules (edge SWaP model)."""
        # Base: QPU idle + per-qubit + per-shot
        base_mj = 0.5
        per_qubit_mj = 0.15
        per_shot_mj = 0.0002
        return round(base_mj + num_qubits * per_qubit_mj + shots * per_shot_mj, 4)

    @staticmethod
    def _calculate_swap_score(num_qubits: int) -> float:
        """
        Calculate a Size-Weight-Power score (lower is better).

        This captures the edge-deployment suitability of the job.
        """
        # Simplified model: score increases with qubit count
        return round(0.1 * num_qubits + 0.5, 2)
