"""
Unit tests for quantum hardware backends.
"""

import pytest
from unittest.mock import patch, MagicMock

from src.backends.backend_base import QuantumBackend
from src.backends.rotonium_mock import RotoniumMockBackend
from src.backends.quix_cloud import QuiXCloudBackend
from src.backends import create_backend


# ============================================================================
# QuantumBackend ABC
# ============================================================================


class TestQuantumBackendABC:
    """Tests for the abstract base class."""

    def test_cannot_instantiate_abc(self):
        """QuantumBackend itself should not be instantiable."""
        with pytest.raises(TypeError):
            QuantumBackend()

    def test_subclass_must_implement_all_methods(self):
        """A subclass missing any abstract method should fail."""

        class IncompleteBackend(QuantumBackend):
            def submit_job(self, circuit, shots=1024):
                return "id"

            # Missing: get_job_result, get_hardware_specs, estimate_job_cost

        with pytest.raises(TypeError):
            IncompleteBackend()


# ============================================================================
# RotoniumMockBackend
# ============================================================================


class TestRotoniumMockBackend:
    """Tests for the Rotonium mock backend."""

    @pytest.fixture
    def backend(self):
        return RotoniumMockBackend()

    def test_submit_job_returns_id(self, backend):
        """submit_job should return a string job_id starting with 'rot-'."""
        job_id = backend.submit_job({"num_qubits": 5}, shots=100)
        assert isinstance(job_id, str)
        assert job_id.startswith("rot-")

    def test_get_job_result_structure(self, backend):
        """Result dict should contain expected keys."""
        job_id = backend.submit_job({"num_qubits": 4}, shots=200)
        result = backend.get_job_result(job_id)

        assert result["success"] is True
        assert "counts" in result
        assert isinstance(result["counts"], dict)
        assert "execution_time_ms" in result
        assert "energy_mj" in result
        assert result["shots"] == 200
        assert result["noise_model"] == "photonic_oam"

    def test_get_result_unknown_job(self, backend):
        """Requesting an unknown job_id should raise KeyError."""
        with pytest.raises(KeyError, match="not found"):
            backend.get_job_result("rot-nonexistent")

    def test_hardware_specs(self, backend):
        """Hardware specs should describe Rotonium photonic QPU."""
        specs = backend.get_hardware_specs()

        assert specs["provider"] == "Rotonium"
        assert specs["technology"] == "Photonic (OAM encoding)"
        assert specs["max_qubits"] == 20
        assert specs["cryogenics_required"] is False
        assert specs["status"] == "online"
        assert "aerospace" in specs["deployment_targets"]

    def test_estimate_job_cost(self, backend):
        """Cost estimate should include SWaP score."""
        estimate = backend.estimate_job_cost({"num_qubits": 10})

        assert "estimated_time_ms" in estimate
        assert "estimated_energy_mj" in estimate
        assert "estimated_cost_usd" in estimate
        assert estimate["estimated_cost_usd"] == 0.0  # mock
        assert "swap_score" in estimate
        assert estimate["swap_score"] > 0

    def test_shots_distribution_sums_correctly(self, backend):
        """Total counts across all bitstrings should equal shots."""
        shots = 500
        job_id = backend.submit_job({"num_qubits": 3}, shots=shots)
        result = backend.get_job_result(job_id)
        total = sum(result["counts"].values())
        assert total == shots

    def test_deterministic_for_same_circuit(self, backend):
        """Same circuit description should produce same counts."""
        circuit = {"num_qubits": 4, "desc": "test_circuit_abc"}
        id1 = backend.submit_job(circuit, shots=256)
        id2 = backend.submit_job(circuit, shots=256)
        r1 = backend.get_job_result(id1)
        r2 = backend.get_job_result(id2)
        assert r1["counts"] == r2["counts"]


# ============================================================================
# QuiXCloudBackend
# ============================================================================


class TestQuiXCloudBackend:
    """Tests for the QuiX cloud backend."""

    @pytest.fixture
    def mock_backend(self):
        """Backend in mock mode (no API key)."""
        return QuiXCloudBackend(api_key="")

    @pytest.fixture
    def keyed_backend(self):
        """Backend with an API key (will attempt real calls)."""
        return QuiXCloudBackend(api_key="test-key-123")

    def test_mock_mode_without_key(self, mock_backend):
        """Backend should be in mock mode when no key is provided."""
        assert mock_backend.mock_mode is True

    def test_real_mode_with_key(self, keyed_backend):
        """Backend should not be in mock mode when key is provided."""
        assert keyed_backend.mock_mode is False

    def test_mock_submit_returns_id(self, mock_backend):
        """Mock submit should return a 'quix-' prefixed job_id."""
        job_id = mock_backend.submit_job({"num_qubits": 6}, shots=128)
        assert isinstance(job_id, str)
        assert job_id.startswith("quix-")

    def test_mock_result_structure(self, mock_backend):
        """Mock result should include PUE-adjusted energy."""
        job_id = mock_backend.submit_job({"num_qubits": 4}, shots=100)
        result = mock_backend.get_job_result(job_id)

        assert result["success"] is True
        assert "counts" in result
        assert "energy_mj" in result
        assert "pue_adjusted_energy_mj" in result
        assert result["pue_adjusted_energy_mj"] >= result["energy_mj"]
        assert result["noise_model"] == "silicon_nitride_photonic"

    def test_mock_result_unknown_job(self, mock_backend):
        """Requesting unknown job in mock mode should raise KeyError."""
        with pytest.raises(KeyError, match="not found"):
            mock_backend.get_job_result("quix-nonexistent")

    def test_hardware_specs(self, mock_backend):
        """Specs should describe QuiX silicon-nitride processor."""
        specs = mock_backend.get_hardware_specs()

        assert specs["provider"] == "QuiX Quantum"
        assert specs["technology"] == "Silicon Nitride Photonic"
        assert specs["max_qubits"] == 20
        assert specs["cryogenics_required"] is False
        assert specs["clock_speed_mhz"] == 100
        assert specs["circuit_fidelity_pct"] == 99.0
        assert "hpc_cluster" in specs["deployment_targets"]

    def test_estimate_job_cost_includes_pue(self, mock_backend):
        """Cost estimate should include PUE-adjusted energy."""
        estimate = mock_backend.estimate_job_cost({"num_qubits": 8})

        assert "estimated_time_ms" in estimate
        assert "estimated_energy_mj" in estimate
        assert "pue_adjusted_energy_mj" in estimate
        assert estimate["pue_adjusted_energy_mj"] > estimate["estimated_energy_mj"]
        assert estimate["estimated_cost_usd"] > 0  # non-free for real API

    def test_real_submit_requires_httpx(self, keyed_backend):
        """Real submit should raise ImportError if httpx is missing."""
        with patch.dict("sys.modules", {"httpx": None}):
            with pytest.raises(ImportError, match="httpx"):
                keyed_backend._submit_real({"num_qubits": 4}, 100)

    def test_init_from_env_var(self):
        """Backend should read QUIX_API_KEY from environment."""
        with patch.dict("os.environ", {"QUIX_API_KEY": "env-key-456"}):
            backend = QuiXCloudBackend()
            assert backend.api_key == "env-key-456"
            assert backend.mock_mode is False


# ============================================================================
# create_backend factory
# ============================================================================


class TestCreateBackend:
    """Tests for the factory function."""

    def test_create_rotonium_mock(self):
        """Factory should return RotoniumMockBackend."""
        backend = create_backend("rotonium_mock")
        assert isinstance(backend, RotoniumMockBackend)

    def test_create_quix_cloud(self):
        """Factory should return QuiXCloudBackend."""
        backend = create_backend("quix_cloud")
        assert isinstance(backend, QuiXCloudBackend)

    def test_unknown_backend_raises(self):
        """Unknown backend name should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown backend"):
            create_backend("unknown_vendor")

    def test_kwargs_passed_to_constructor(self):
        """Factory should forward kwargs to the backend constructor."""
        backend = create_backend("quix_cloud", api_key="test-key")
        assert backend.api_key == "test-key"
        assert backend.mock_mode is False
