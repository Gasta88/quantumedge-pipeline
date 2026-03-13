"""
Data Center Environment Simulator for Quantum-Classical Hybrid Systems.

Analogous to edge_simulator.py (Rotonium edge profiles), this module
models resource constraints and energy accounting for QuiX Quantum
data-center deployments.

Key difference from edge deployments:
    - Power is measured in PUE-adjusted terms (overhead for cooling, networking)
    - Constraints are less about battery life and more about cost efficiency
    - Deployment profiles target HPC clusters, rack-mounted servers, and cloud nodes
    - Thermal management is handled by facility HVAC, not device-level cooling

Deployment Profiles:
    HPC_CLUSTER     — Dedicated HPC partition with high bandwidth interconnect
    DATACENTER_RACK — Standard 42U rack with mixed quantum/classical blades
    CLOUD_NODE      — Virtual cloud instance with shared infrastructure
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Deque
from enum import Enum
from collections import deque
from datetime import datetime
import statistics


class DatacenterProfile(Enum):
    """Predefined data-center deployment scenarios for QuiX Quantum."""

    HPC_CLUSTER = "hpc_cluster"
    DATACENTER_RACK = "datacenter_rack"
    CLOUD_NODE = "cloud_node"


@dataclass
class DatacenterJobRequirements:
    """
    Resource requirements for a data-center quantum/classical job.

    Attributes:
        power_watts: Peak IT-equipment power draw
        execution_time_seconds: Expected wall-clock runtime
        memory_gb: RAM needed (GB — datacenter scale)
        storage_gb: Persistent storage required
        gpu_count: Number of GPUs (for classical acceleration)
        network_gbps: Network bandwidth needed
    """

    power_watts: float
    execution_time_seconds: float
    memory_gb: float
    storage_gb: float
    gpu_count: int = 0
    network_gbps: float = 0.0


@dataclass
class DatacenterCapacity:
    """Current remaining capacity in the datacenter environment."""

    power_watts: float
    memory_gb: float
    storage_gb: float
    gpu_count: int
    network_gbps: float
    thermal_headroom_watts: float


@dataclass
class DatacenterJobAllocation:
    """Tracks allocated resources for a running datacenter job."""

    job: DatacenterJobRequirements
    job_id: str
    allocated_at: datetime
    power_allocated: float
    memory_allocated: float
    storage_allocated: float
    gpu_allocated: int
    network_allocated: float


class DatacenterEnvironment:
    """
    Simulates resource constraints of data-center deployment scenarios.

    Each profile represents a realistic data-center tier with different
    power budgets, memory, networking, and cost characteristics.

    QuiX Quantum advantage in data centers:
        - Room-temperature photonic processors integrate into standard racks
        - No cryogenic infrastructure — lower total cost of ownership
        - Silicon-nitride chips compatible with CMOS fabrication
        - High clock speed (100 MHz–1 GHz) vs superconducting alternatives
        - Modular architecture allows field upgrades
    """

    def __init__(self, profile: DatacenterProfile) -> None:
        self.profile = profile
        self._setup_profile()

    def _setup_profile(self) -> None:
        """Configure resource constraints based on deployment profile."""

        if self.profile == DatacenterProfile.HPC_CLUSTER:
            self.power_budget_watts = 5000.0  # Dedicated HPC partition
            self.thermal_limit_watts = 4000.0  # Cooling capacity
            self.compute_timeout_seconds = 3600.0  # 1 hour batch jobs
            self.memory_gb = 512.0  # High-memory nodes
            self.storage_gb = 10000.0  # Shared parallel FS
            self.gpu_count = 8  # GPU accelerators
            self.network_gbps = 100.0  # InfiniBand / RoCE
            self.pue_ratio = 1.2  # Efficient HPC facility
            self.deployment_context = "Dedicated HPC partition with QuiX photonic QPU co-processor"

        elif self.profile == DatacenterProfile.DATACENTER_RACK:
            self.power_budget_watts = 2000.0  # Single 42U rack budget
            self.thermal_limit_watts = 1500.0  # Rack cooling capacity
            self.compute_timeout_seconds = 600.0  # 10 minutes
            self.memory_gb = 128.0  # Standard server memory
            self.storage_gb = 4000.0  # Local NVMe + SAN
            self.gpu_count = 2  # Optional GPU
            self.network_gbps = 25.0  # 25 GbE
            self.pue_ratio = 1.4  # Average data center
            self.deployment_context = (
                "Standard rack-mounted QuiX quantum blade alongside classical servers"
            )

        elif self.profile == DatacenterProfile.CLOUD_NODE:
            self.power_budget_watts = 500.0  # Virtual instance power cap
            self.thermal_limit_watts = 400.0  # Shared cooling
            self.compute_timeout_seconds = 300.0  # 5 minutes
            self.memory_gb = 64.0  # Cloud instance memory
            self.storage_gb = 1000.0  # Cloud block storage
            self.gpu_count = 0  # CPU-only cloud tier
            self.network_gbps = 10.0  # Cloud VPC bandwidth
            self.pue_ratio = 1.3  # Hyperscaler average
            self.deployment_context = "Cloud-hosted QuiX quantum instance via cloud.quixquantum.com"

        else:
            raise ValueError(f"Unknown datacenter profile: {self.profile}")

    # ------------------------------------------------------------------
    # Constraint checks
    # ------------------------------------------------------------------

    def can_execute(self, job: DatacenterJobRequirements) -> bool:
        """Check if a job can execute within all datacenter constraints."""
        if job.power_watts > self.power_budget_watts:
            return False
        if job.execution_time_seconds > self.compute_timeout_seconds:
            return False
        if job.memory_gb > self.memory_gb:
            return False
        if job.storage_gb > self.storage_gb:
            return False
        if job.gpu_count > self.gpu_count:
            return False
        if job.network_gbps > self.network_gbps:
            return False
        if job.power_watts > self.thermal_limit_watts:
            return False
        return True

    def estimate_remaining_capacity(self, job: DatacenterJobRequirements) -> DatacenterCapacity:
        """Calculate remaining resources after a job is allocated."""
        return DatacenterCapacity(
            power_watts=max(0.0, self.power_budget_watts - job.power_watts),
            memory_gb=max(0.0, self.memory_gb - job.memory_gb),
            storage_gb=max(0.0, self.storage_gb - job.storage_gb),
            gpu_count=max(0, self.gpu_count - job.gpu_count),
            network_gbps=max(0.0, self.network_gbps - job.network_gbps),
            thermal_headroom_watts=max(0.0, self.thermal_limit_watts - job.power_watts),
        )

    def calculate_pue_adjusted_energy(self, raw_energy_mj: float) -> float:
        """
        Apply PUE multiplier to raw IT-equipment energy.

        PUE (Power Usage Effectiveness) = Total Facility Power / IT Equipment Power
        A PUE of 1.2 means 20 % overhead for cooling, lighting, and networking.
        """
        return round(raw_energy_mj * self.pue_ratio, 4)

    def get_profile_info(self) -> Dict[str, any]:
        """Get comprehensive information about the current datacenter profile."""
        return {
            "profile": self.profile.value,
            "deployment_context": self.deployment_context,
            "constraints": {
                "power_budget_watts": self.power_budget_watts,
                "thermal_limit_watts": self.thermal_limit_watts,
                "compute_timeout_seconds": self.compute_timeout_seconds,
                "memory_gb": self.memory_gb,
                "storage_gb": self.storage_gb,
                "gpu_count": self.gpu_count,
                "network_gbps": self.network_gbps,
            },
            "pue_ratio": self.pue_ratio,
            "quix_advantage": (
                "Room-temperature silicon-nitride photonic processor integrates "
                "into standard data-center racks with no cryogenic infrastructure, "
                "lowering TCO and enabling modular scaling."
            ),
        }


class DatacenterResourceTracker:
    """
    Tracks resource usage for intelligent job scheduling in data-center environments.

    Mirrors ResourceTracker from edge_simulator.py but with data-center-specific
    metrics (PUE, GPU count, GB-scale memory).
    """

    def __init__(self, environment: DatacenterEnvironment) -> None:
        self.environment = environment

        # Usage history
        self.power_usage_history: List[Tuple[datetime, float]] = []

        # Active allocations
        self.active_allocations: Dict[str, DatacenterJobAllocation] = {}

        # Job queue
        self.job_queue: Deque[Tuple[str, DatacenterJobRequirements]] = deque()

        # Current usage counters
        self.current_power_usage = 0.0
        self.current_memory_usage = 0.0
        self.current_storage_usage = 0.0
        self.current_gpu_usage = 0
        self.current_network_usage = 0.0

        # Statistics
        self.total_jobs_scheduled = 0
        self.total_jobs_completed = 0
        self.total_jobs_rejected = 0

    def allocate_resources(self, job: DatacenterJobRequirements, job_id: str = None) -> bool:
        """Try to allocate resources for a job. Returns True on success."""
        if job_id is None:
            job_id = f"dc_job_{self.total_jobs_scheduled}_{datetime.now().timestamp()}"

        available_power = self.environment.power_budget_watts - self.current_power_usage
        available_memory = self.environment.memory_gb - self.current_memory_usage
        available_storage = self.environment.storage_gb - self.current_storage_usage
        available_gpus = self.environment.gpu_count - self.current_gpu_usage
        available_network = self.environment.network_gbps - self.current_network_usage
        available_thermal = self.environment.thermal_limit_watts - self.current_power_usage

        if (
            job.power_watts > available_power
            or job.memory_gb > available_memory
            or job.storage_gb > available_storage
            or job.gpu_count > available_gpus
            or job.network_gbps > available_network
            or job.power_watts > available_thermal
        ):
            return False

        allocation = DatacenterJobAllocation(
            job=job,
            job_id=job_id,
            allocated_at=datetime.now(),
            power_allocated=job.power_watts,
            memory_allocated=job.memory_gb,
            storage_allocated=job.storage_gb,
            gpu_allocated=job.gpu_count,
            network_allocated=job.network_gbps,
        )

        self.active_allocations[job_id] = allocation
        self.current_power_usage += job.power_watts
        self.current_memory_usage += job.memory_gb
        self.current_storage_usage += job.storage_gb
        self.current_gpu_usage += job.gpu_count
        self.current_network_usage += job.network_gbps

        self.power_usage_history.append((datetime.now(), self.current_power_usage))
        self.total_jobs_scheduled += 1
        return True

    def release_resources(self, job_id: str) -> bool:
        """Free resources after job completes. Returns True on success."""
        if job_id not in self.active_allocations:
            return False

        alloc = self.active_allocations[job_id]
        self.current_power_usage = max(0.0, self.current_power_usage - alloc.power_allocated)
        self.current_memory_usage = max(0.0, self.current_memory_usage - alloc.memory_allocated)
        self.current_storage_usage = max(0.0, self.current_storage_usage - alloc.storage_allocated)
        self.current_gpu_usage = max(0, self.current_gpu_usage - alloc.gpu_allocated)
        self.current_network_usage = max(0.0, self.current_network_usage - alloc.network_allocated)

        self.power_usage_history.append((datetime.now(), self.current_power_usage))
        del self.active_allocations[job_id]
        self.total_jobs_completed += 1
        return True

    def get_utilization_stats(self) -> Dict:
        """Calculate comprehensive utilization statistics."""
        env = self.environment
        stats = {
            "current_utilization": {
                "power_watts": self.current_power_usage,
                "power_percent": (self.current_power_usage / env.power_budget_watts) * 100,
                "memory_gb": self.current_memory_usage,
                "memory_percent": (self.current_memory_usage / env.memory_gb) * 100,
                "gpu_count": self.current_gpu_usage,
                "gpu_percent": (self.current_gpu_usage / env.gpu_count * 100)
                if env.gpu_count > 0
                else 0.0,
                "network_gbps": self.current_network_usage,
                "network_percent": (self.current_network_usage / env.network_gbps) * 100,
            },
            "active_jobs": len(self.active_allocations),
            "queued_jobs": len(self.job_queue),
            "job_statistics": {
                "total_scheduled": self.total_jobs_scheduled,
                "total_completed": self.total_jobs_completed,
                "total_rejected": self.total_jobs_rejected,
                "completion_rate": (
                    self.total_jobs_completed / self.total_jobs_scheduled * 100
                    if self.total_jobs_scheduled > 0
                    else 0.0
                ),
            },
            "pue_ratio": env.pue_ratio,
        }

        if self.power_usage_history:
            values = [v for _, v in self.power_usage_history]
            stats["power_history"] = {
                "average_watts": statistics.mean(values),
                "peak_watts": max(values),
                "min_watts": min(values),
                "stddev_watts": statistics.stdev(values) if len(values) > 1 else 0.0,
            }
        else:
            stats["power_history"] = {
                "average_watts": 0.0,
                "peak_watts": 0.0,
                "min_watts": 0.0,
                "stddev_watts": 0.0,
            }

        return stats
