"""
Unit tests for datacenter environment simulator.
"""

import pytest

from src.router.datacenter_simulator import (
    DatacenterEnvironment,
    DatacenterProfile,
    DatacenterJobRequirements,
    DatacenterResourceTracker,
)


# ============================================================================
# DatacenterEnvironment
# ============================================================================


class TestDatacenterEnvironment:
    """Tests for DatacenterEnvironment profiles and constraint checks."""

    def test_hpc_cluster_profile(self):
        """HPC cluster should have high power, memory, and GPU count."""
        env = DatacenterEnvironment(DatacenterProfile.HPC_CLUSTER)

        assert env.power_budget_watts == 5000.0
        assert env.memory_gb == 512.0
        assert env.gpu_count == 8
        assert env.network_gbps == 100.0
        assert env.compute_timeout_seconds == 3600.0
        assert env.pue_ratio == 1.2

    def test_datacenter_rack_profile(self):
        """Datacenter rack should have moderate constraints."""
        env = DatacenterEnvironment(DatacenterProfile.DATACENTER_RACK)

        assert env.power_budget_watts == 2000.0
        assert env.memory_gb == 128.0
        assert env.gpu_count == 2
        assert env.network_gbps == 25.0
        assert env.compute_timeout_seconds == 600.0
        assert env.pue_ratio == 1.4

    def test_cloud_node_profile(self):
        """Cloud node should have the tightest datacenter constraints."""
        env = DatacenterEnvironment(DatacenterProfile.CLOUD_NODE)

        assert env.power_budget_watts == 500.0
        assert env.memory_gb == 64.0
        assert env.gpu_count == 0
        assert env.network_gbps == 10.0
        assert env.compute_timeout_seconds == 300.0
        assert env.pue_ratio == 1.3

    def test_can_execute_within_budget(self):
        """A small job should fit within HPC cluster constraints."""
        env = DatacenterEnvironment(DatacenterProfile.HPC_CLUSTER)
        job = DatacenterJobRequirements(
            power_watts=100.0,
            execution_time_seconds=60.0,
            memory_gb=16.0,
            storage_gb=50.0,
            gpu_count=1,
            network_gbps=1.0,
        )
        assert env.can_execute(job) is True

    def test_reject_over_power_budget(self):
        """A job exceeding power budget should be rejected."""
        env = DatacenterEnvironment(DatacenterProfile.CLOUD_NODE)
        job = DatacenterJobRequirements(
            power_watts=600.0,  # exceeds 500 W cloud node budget
            execution_time_seconds=60.0,
            memory_gb=8.0,
            storage_gb=10.0,
        )
        assert env.can_execute(job) is False

    def test_reject_over_memory(self):
        """A job exceeding memory should be rejected."""
        env = DatacenterEnvironment(DatacenterProfile.CLOUD_NODE)
        job = DatacenterJobRequirements(
            power_watts=100.0,
            execution_time_seconds=60.0,
            memory_gb=100.0,  # exceeds 64 GB
            storage_gb=10.0,
        )
        assert env.can_execute(job) is False

    def test_reject_over_timeout(self):
        """A job exceeding timeout should be rejected."""
        env = DatacenterEnvironment(DatacenterProfile.CLOUD_NODE)
        job = DatacenterJobRequirements(
            power_watts=100.0,
            execution_time_seconds=600.0,  # exceeds 300 s cloud limit
            memory_gb=8.0,
            storage_gb=10.0,
        )
        assert env.can_execute(job) is False

    def test_reject_over_gpu(self):
        """A job requiring GPUs on a GPU-less cloud node should be rejected."""
        env = DatacenterEnvironment(DatacenterProfile.CLOUD_NODE)
        job = DatacenterJobRequirements(
            power_watts=100.0,
            execution_time_seconds=60.0,
            memory_gb=8.0,
            storage_gb=10.0,
            gpu_count=1,  # cloud node has 0 GPUs
        )
        assert env.can_execute(job) is False

    def test_pue_calculation(self):
        """PUE-adjusted energy should equal raw * pue_ratio."""
        env = DatacenterEnvironment(DatacenterProfile.DATACENTER_RACK)
        raw = 100.0
        adjusted = env.calculate_pue_adjusted_energy(raw)
        assert adjusted == pytest.approx(raw * 1.4, rel=1e-3)

    def test_remaining_capacity(self):
        """Remaining capacity should be reduced by job requirements."""
        env = DatacenterEnvironment(DatacenterProfile.HPC_CLUSTER)
        job = DatacenterJobRequirements(
            power_watts=1000.0,
            execution_time_seconds=100.0,
            memory_gb=128.0,
            storage_gb=500.0,
            gpu_count=2,
            network_gbps=10.0,
        )
        remaining = env.estimate_remaining_capacity(job)

        assert remaining.power_watts == pytest.approx(4000.0)
        assert remaining.memory_gb == pytest.approx(384.0)
        assert remaining.gpu_count == 6
        assert remaining.network_gbps == pytest.approx(90.0)

    def test_get_profile_info(self):
        """Profile info dict should contain key fields."""
        env = DatacenterEnvironment(DatacenterProfile.HPC_CLUSTER)
        info = env.get_profile_info()

        assert info["profile"] == "hpc_cluster"
        assert "constraints" in info
        assert "pue_ratio" in info
        assert "quix_advantage" in info
        assert info["pue_ratio"] == 1.2

    def test_thermal_headroom(self):
        """Job exceeding thermal limit should be rejected."""
        env = DatacenterEnvironment(DatacenterProfile.CLOUD_NODE)
        # thermal_limit_watts is 400 for cloud node
        job = DatacenterJobRequirements(
            power_watts=450.0,  # exceeds 400 W thermal limit
            execution_time_seconds=60.0,
            memory_gb=8.0,
            storage_gb=10.0,
        )
        assert env.can_execute(job) is False


# ============================================================================
# DatacenterResourceTracker
# ============================================================================


class TestDatacenterResourceTracker:
    """Tests for resource tracking and allocation."""

    @pytest.fixture
    def tracker(self):
        env = DatacenterEnvironment(DatacenterProfile.HPC_CLUSTER)
        return DatacenterResourceTracker(env)

    def test_allocate_and_release(self, tracker):
        """Allocate then release should restore counters."""
        job = DatacenterJobRequirements(
            power_watts=500.0,
            execution_time_seconds=60.0,
            memory_gb=64.0,
            storage_gb=100.0,
            gpu_count=2,
            network_gbps=5.0,
        )
        assert tracker.allocate_resources(job, "j1") is True
        assert tracker.current_power_usage == pytest.approx(500.0)
        assert tracker.current_gpu_usage == 2

        assert tracker.release_resources("j1") is True
        assert tracker.current_power_usage == pytest.approx(0.0)
        assert tracker.current_gpu_usage == 0
        assert tracker.total_jobs_completed == 1

    def test_allocation_rejected_when_full(self, tracker):
        """Second allocation should fail if resources exhausted."""
        big_job = DatacenterJobRequirements(
            power_watts=3500.0,
            execution_time_seconds=60.0,
            memory_gb=400.0,
            storage_gb=5000.0,
            gpu_count=7,
            network_gbps=80.0,
        )
        assert tracker.allocate_resources(big_job, "j1") is True

        small_job = DatacenterJobRequirements(
            power_watts=1000.0,
            execution_time_seconds=60.0,
            memory_gb=200.0,
            storage_gb=1000.0,
            gpu_count=2,
            network_gbps=30.0,
        )
        # Fails because combined power (3500+1000=4500) exceeds thermal limit (4000W)
        assert tracker.allocate_resources(small_job, "j2") is False

    def test_release_unknown_job(self, tracker):
        """Releasing a non-existent job should return False."""
        assert tracker.release_resources("nonexistent") is False

    def test_utilization_stats(self, tracker):
        """Stats should reflect current usage."""
        job = DatacenterJobRequirements(
            power_watts=1000.0,
            execution_time_seconds=60.0,
            memory_gb=64.0,
            storage_gb=100.0,
        )
        tracker.allocate_resources(job, "j1")
        stats = tracker.get_utilization_stats()

        assert stats["active_jobs"] == 1
        assert stats["current_utilization"]["power_percent"] == pytest.approx(20.0)
        assert stats["pue_ratio"] == 1.2
        assert stats["job_statistics"]["total_scheduled"] == 1
