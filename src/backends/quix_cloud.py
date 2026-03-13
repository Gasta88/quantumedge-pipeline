"""
QuiX Quantum cloud API backend.

Connects to the QuiX Quantum cloud platform (cloud.quixquantum.com) to submit
jobs to real silicon-nitride photonic quantum hardware. When no API key is
configured the backend operates in **mock mode**, returning simulated results
that mirror the real API response format.

Environment variable:
    QUIX_API_KEY  — API key for authentication with the QuiX cloud.
"""

import hashlib
import logging
import os
import random
import time
from typing import Any, Dict, Optional
from uuid import uuid4

from src.backends.backend_base import QuantumBackend

logger = logging.getLogger(__name__)


class QuiXCloudBackend(QuantumBackend):
    """
    Backend for QuiX Quantum's cloud photonic processor.

    When a valid api_key is provided the backend will attempt to reach the
    real QuiX cloud API.  If no key is given (or the key is empty) it falls
    back to a local mock that produces realistic response shapes so the rest
    of the pipeline can run end-to-end without hardware access.
    """

    BASE_URL = "https://cloud.quixquantum.com/api"
    MAX_QUBITS = 20  # Current QuiX processor limit

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key = api_key or os.environ.get("QUIX_API_KEY", "")
        self.mock_mode = not bool(self.api_key)
        self._jobs: Dict[str, dict] = {}

        if self.mock_mode:
            logger.info("QuiXCloudBackend running in MOCK mode (no QUIX_API_KEY set)")
        else:
            logger.info("QuiXCloudBackend initialised with API key")

    # ------------------------------------------------------------------
    # QuantumBackend interface
    # ------------------------------------------------------------------

    def submit_job(self, circuit: Any, shots: int = 1024) -> str:
        """Submit a job to QuiX cloud (or mock)."""
        if not self.mock_mode:
            return self._submit_real(circuit, shots)
        return self._submit_mock(circuit, shots)

    def get_job_result(self, job_id: str) -> dict:
        """Retrieve job result from QuiX cloud (or mock)."""
        if not self.mock_mode:
            return self._get_result_real(job_id)
        if job_id not in self._jobs:
            raise KeyError(f"Job '{job_id}' not found in QuiX mock backend")
        return self._jobs[job_id]

    def get_hardware_specs(self) -> dict:
        """Return QuiX photonic processor specifications."""
        return {
            "provider": "QuiX Quantum",
            "technology": "Silicon Nitride Photonic",
            "max_qubits": self.MAX_QUBITS,
            "operating_temperature_k": 293.15,
            "clock_speed_mhz": 100,
            "optical_loss_db_per_cm": 0.1,
            "circuit_fidelity_pct": 99.0,
            "cryogenics_required": False,
            "status": "online" if not self.mock_mode else "mock",
            "deployment_targets": ["hpc_cluster", "datacenter_rack", "cloud_node"],
            "cloud_endpoint": self.BASE_URL,
        }

    def estimate_job_cost(self, circuit: Any) -> dict:
        """Estimate execution cost on QuiX hardware."""
        num_qubits = self._estimate_qubits(circuit)
        shots = 1024
        time_ms = num_qubits * 1.2 + shots * 0.008
        energy_mj = self._estimate_energy(num_qubits, shots)
        pue = self._calculate_pue(energy_mj)
        return {
            "estimated_time_ms": round(time_ms, 2),
            "estimated_energy_mj": energy_mj,
            "estimated_cost_usd": round(num_qubits * 0.002 + shots * 0.000005, 6),
            "pue_adjusted_energy_mj": pue,
        }

    # ------------------------------------------------------------------
    # Real API methods (stubs — require httpx at runtime)
    # ------------------------------------------------------------------

    def _submit_real(self, circuit: Any, shots: int) -> str:
        """Submit to the real QuiX cloud API."""
        try:
            import httpx
        except ImportError:
            raise ImportError("httpx is required for real QuiX cloud access: pip install httpx")

        payload = {
            "circuit": str(circuit),
            "shots": shots,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        response = httpx.post(
            f"{self.BASE_URL}/v1/jobs",
            json=payload,
            headers=headers,
            timeout=30.0,
        )
        response.raise_for_status()
        data = response.json()
        job_id = data.get("job_id", f"quix-{uuid4().hex[:12]}")
        return job_id

    def _get_result_real(self, job_id: str) -> dict:
        """Poll the real QuiX cloud API for a job result."""
        try:
            import httpx
        except ImportError:
            raise ImportError("httpx is required for real QuiX cloud access: pip install httpx")

        headers = {"Authorization": f"Bearer {self.api_key}"}
        response = httpx.get(
            f"{self.BASE_URL}/v1/jobs/{job_id}/result",
            headers=headers,
            timeout=60.0,
        )
        response.raise_for_status()
        return response.json()

    # ------------------------------------------------------------------
    # Mock helpers
    # ------------------------------------------------------------------

    def _submit_mock(self, circuit: Any, shots: int) -> str:
        """Simulate a QuiX cloud job submission locally."""
        job_id = f"quix-{uuid4().hex[:12]}"
        start = time.perf_counter()

        circuit_repr = str(circuit)
        seed = int(hashlib.md5(circuit_repr.encode()).hexdigest()[:8], 16)
        rng = random.Random(seed)

        num_qubits = self._estimate_qubits(circuit)
        counts = self._simulate_datacenter_measurement(num_qubits, shots, rng)

        elapsed_ms = (time.perf_counter() - start) * 1000
        simulated_qpu_ms = num_qubits * 1.2 + shots * 0.008
        total_ms = elapsed_ms + simulated_qpu_ms

        energy_mj = self._estimate_energy(num_qubits, shots)

        self._jobs[job_id] = {
            "counts": counts,
            "success": True,
            "execution_time_ms": round(total_ms, 2),
            "num_qubits": num_qubits,
            "shots": shots,
            "energy_mj": energy_mj,
            "pue_adjusted_energy_mj": self._calculate_pue(energy_mj),
            "noise_model": "silicon_nitride_photonic",
        }
        return job_id

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_qubits(circuit: Any) -> int:
        """Heuristically estimate qubit count from a circuit description."""
        if hasattr(circuit, "num_qubits"):
            return min(circuit.num_qubits, QuiXCloudBackend.MAX_QUBITS)
        if isinstance(circuit, dict) and "num_qubits" in circuit:
            return min(circuit["num_qubits"], QuiXCloudBackend.MAX_QUBITS)
        return 8

    @staticmethod
    def _simulate_datacenter_measurement(
        num_qubits: int, shots: int, rng: random.Random
    ) -> Dict[str, int]:
        """Generate measurement counts with silicon-nitride photonic noise."""
        num_outcomes = min(2**num_qubits, 64)
        outcomes: Dict[str, int] = {}
        remaining = shots

        for i in range(num_outcomes):
            bitstring = format(i, f"0{num_qubits}b")
            weight = 0.99 ** bitstring.count("1")  # High-fidelity chip
            count = max(1, int(rng.gauss(remaining / (num_outcomes - i), remaining * 0.04)))
            count = int(count * weight)
            count = max(0, min(count, remaining))
            if count > 0:
                outcomes[bitstring] = count
                remaining -= count
            if remaining <= 0:
                break

        if remaining > 0:
            ground = "0" * num_qubits
            outcomes[ground] = outcomes.get(ground, 0) + remaining

        return outcomes

    @staticmethod
    def _estimate_energy(num_qubits: int, shots: int) -> float:
        """Estimate energy in millijoules (datacenter model)."""
        base_mj = 1.0
        per_qubit_mj = 0.25
        per_shot_mj = 0.0003
        return round(base_mj + num_qubits * per_qubit_mj + shots * per_shot_mj, 4)

    @staticmethod
    def _calculate_pue(raw_energy_mj: float, pue_ratio: float = 1.3) -> float:
        """
        Apply Power Usage Effectiveness multiplier.

        PUE accounts for data-center overhead (cooling, networking, lighting).
        A PUE of 1.3 means 30% overhead on top of IT equipment power.
        """
        return round(raw_energy_mj * pue_ratio, 4)
