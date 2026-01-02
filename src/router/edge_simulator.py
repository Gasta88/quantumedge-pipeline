"""
Edge Computing Environment Simulator for Quantum-Classical Hybrid Systems

This module simulates resource constraints of different edge deployment scenarios
for quantum computing applications. It helps evaluate whether quantum or classical
jobs can be executed within the physical and operational constraints of edge devices.

Key advantage of Rotonium technology:
- Room-temperature quantum processing units (QPUs) eliminate cryogenic cooling
- Dramatically reduces power consumption and thermal management complexity
- Enables quantum computing in resource-constrained edge environments
- Makes aerospace and mobile quantum computing practical
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple, Deque
from enum import Enum
from collections import deque
from datetime import datetime
import statistics


class DeploymentProfile(Enum):
    """Predefined edge computing deployment scenarios"""
    AEROSPACE = "aerospace"
    MOBILE = "mobile"
    GROUND_SERVER = "ground_server"


@dataclass
class JobRequirements:
    """
    Resource requirements for a computational job.
    
    Attributes:
        power_watts: Peak power consumption during execution
        execution_time_seconds: Expected runtime duration
        memory_mb: RAM needed for computation
        storage_gb: Persistent storage required
        thermal_output_watts: Heat generation rate
        bandwidth_mbps: Network bandwidth needed
    """
    power_watts: float
    execution_time_seconds: float
    memory_mb: int
    storage_gb: float
    thermal_output_watts: float
    bandwidth_mbps: float = 0.0


@dataclass
class ResourceCapacity:
    """Current resource capacity state"""
    power_watts: float
    memory_mb: int
    storage_gb: float
    bandwidth_mbps: float
    thermal_headroom_celsius: float


@dataclass
class JobAllocation:
    """
    Tracks allocated resources for a running job.
    
    This dataclass maintains the resource reservation state for jobs
    that are currently executing on the edge environment.
    """
    job: JobRequirements
    job_id: str
    allocated_at: datetime
    power_allocated: float
    memory_allocated: int
    storage_allocated: float
    bandwidth_allocated: float
    thermal_allocated: float


class EdgeEnvironment:
    """
    Simulates resource constraints of edge deployment scenarios.
    
    This class models the physical and operational limitations of different
    edge computing environments where quantum-classical hybrid systems might
    be deployed. Each profile represents real-world constraints that must be
    considered when routing computational jobs.
    
    Why these constraints matter:
    
    1. POWER BUDGET: Edge devices run on limited power (battery, solar, or
       constrained grid access). Quantum computing traditionally requires
       massive power for cryogenic cooling. Rotonium's room-temperature QPU
       eliminates this bottleneck, making edge quantum computing viable.
    
    2. THERMAL LIMITS: Aerospace and mobile environments have strict thermal
       constraints. Traditional quantum computers need dilution refrigerators
       at ~15 millikelvin. Rotonium operates at room temperature, avoiding
       the 100+ watts of cooling power per qubit.
    
    3. COMPUTE TIMEOUT: Edge applications often need real-time or near-real-time
       responses. Long quantum computations must be balanced against latency
       requirements.
    
    4. MEMORY LIMITS: Edge devices have constrained RAM for storing quantum
       circuit descriptions, classical pre/post-processing, and results.
    
    5. STORAGE LIMITS: Limited space for quantum programs, calibration data,
       error mitigation tables, and result caching.
    
    6. NETWORK BANDWIDTH: Not all edge environments have reliable high-speed
       connectivity. Aerospace deployments might use low-bandwidth satellite
       links. This affects ability to offload to cloud quantum systems.
    """
    
    def __init__(self, profile: DeploymentProfile):
        """
        Initialize edge environment with predefined profile.
        
        Args:
            profile: Deployment scenario (AEROSPACE, MOBILE, or GROUND_SERVER)
        """
        self.profile = profile
        self._setup_profile()
    
    def _setup_profile(self) -> None:
        """Configure resource constraints based on deployment profile."""
        
        if self.profile == DeploymentProfile.AEROSPACE:
            # =================================================================
            # AEROSPACE PROFILE: UAV, Satellite, or High-altitude Platform
            # =================================================================
            # Context: Space-based or high-altitude quantum computing
            # 
            # Key challenges:
            # - Limited power from solar panels or batteries
            # - Strict thermal management (space has no air cooling!)
            # - Weight and volume constraints
            # - Intermittent satellite connectivity
            # - Radiation-hardened components needed
            # 
            # Rotonium advantage: Room-temperature QPU enables space-based
            # quantum computing without bulky cryogenic systems. A traditional
            # superconducting quantum computer would need ~1kW just for cooling,
            # making it impossible for aerospace deployment.
            # =================================================================
            
            self.power_budget_watts = 50.0
            # 50W is realistic for a small satellite or UAV payload
            # Compare to 1000+ watts for traditional quantum computer
            
            self.thermal_limit_celsius = 60.0
            # Space electronics typically rated to 60-85°C
            # Without air convection, thermal management is critical
            # Rotonium's room-temp operation is game-changing here
            
            self.compute_timeout_seconds = 10.0
            # Fast decisions needed for navigation, sensing, optimization
            # Satellite ground contact windows are limited
            
            self.memory_limit_mb = 4096
            # 4GB is typical for aerospace-grade radiation-hardened RAM
            # Sufficient for moderate-size quantum circuits and classical processing
            
            self.storage_limit_gb = 128.0
            # SSD storage for quantum programs, calibration data, results
            # Weight and radiation-hardening limit capacity
            
            self.network_bandwidth_mbps = 5.0
            # Satellite links: 1-10 Mbps typical (e.g., Iridium, low-earth orbit)
            # Too slow to stream quantum jobs to/from cloud
            # On-board quantum processing is essential
            
            self.deployment_context = "UAV or satellite deployment with Rotonium QPU"
        
        elif self.profile == DeploymentProfile.MOBILE:
            # =================================================================
            # MOBILE PROFILE: Vehicle-mounted or Portable Edge Device
            # =================================================================
            # Context: Autonomous vehicles, field sensors, tactical systems
            # 
            # Key challenges:
            # - Battery power constraints
            # - Vibration and shock resistance
            # - Compact form factor
            # - Variable environmental conditions
            # - Mobile network connectivity
            # 
            # Rotonium advantage: Enables quantum-enhanced navigation, sensing,
            # and optimization in vehicles. Traditional quantum computers would
            # require a truck just for the cryogenic system!
            # =================================================================
            
            self.power_budget_watts = 15.0
            # 15W is typical for edge AI accelerators in vehicles
            # Must share power budget with other vehicle systems
            # Rotonium's low power enables mobile quantum computing
            
            self.thermal_limit_celsius = 45.0
            # Consumer/automotive electronics comfort range
            # Vehicle cabin temperatures can vary widely
            # No cryogenics needed = major advantage
            
            self.compute_timeout_seconds = 5.0
            # Real-time requirements for autonomous systems
            # Route optimization, sensor fusion, decision-making
            
            self.memory_limit_mb = 2048
            # 2GB typical for embedded edge devices
            # Sufficient for small-to-medium quantum circuits
            
            self.storage_limit_gb = 64.0
            # eMMC or industrial SSD storage
            # Stores quantum programs, vehicle data, ML models
            
            self.network_bandwidth_mbps = 50.0
            # 4G/5G mobile network: 10-100 Mbps typical
            # Better than aerospace but still variable
            # Local quantum processing reduces latency and data transfer
            
            self.deployment_context = "Edge device in vehicle or portable unit"
        
        elif self.profile == DeploymentProfile.GROUND_SERVER:
            # =================================================================
            # GROUND_SERVER PROFILE: Edge Data Center or Ground Station
            # =================================================================
            # Context: Ground-based edge computing facility
            # 
            # Key challenges:
            # - Cost optimization (power costs money)
            # - Space constraints in facilities
            # - Cooling infrastructure
            # - Shared resources with other workloads
            # 
            # Rotonium advantage: Room-temperature QPU dramatically reduces
            # operational costs compared to cryogenic quantum systems. No need
            # for dilution refrigerators, liquid helium, or specialized HVAC.
            # Can be rack-mounted alongside classical servers.
            # =================================================================
            
            self.power_budget_watts = 200.0
            # Typical for a high-performance edge server
            # Still 5-10x less than traditional quantum computer + cooling
            
            self.thermal_limit_celsius = 80.0
            # Server-grade components rated to 80-90°C
            # Standard data center cooling is sufficient
            # Rotonium eliminates need for cryogenic infrastructure
            
            self.compute_timeout_seconds = 60.0
            # More relaxed timing for batch processing
            # Can handle longer optimization problems
            
            self.memory_limit_mb = 16384
            # 16GB typical for edge servers
            # Supports large quantum circuits and extensive classical processing
            
            self.storage_limit_gb = 1024.0
            # 1TB enterprise SSD
            # Ample space for quantum programs, datasets, results
            
            self.network_bandwidth_mbps = 1000.0
            # Gigabit ethernet or 10GbE
            # Could offload to cloud, but local quantum reduces latency
            
            self.deployment_context = "Edge data center or ground station"
        
        else:
            raise ValueError(f"Unknown deployment profile: {self.profile}")
    
    def can_execute_quantum(self, job: JobRequirements) -> bool:
        """
        Check if a quantum job can execute within environment constraints.
        
        This method validates whether a quantum computing job fits within the
        physical and operational limits of the edge environment. All constraints
        must be satisfied for successful execution.
        
        Key consideration: Rotonium's room-temperature QPU has fundamentally
        different power and thermal profiles than traditional quantum computers.
        A typical superconducting quantum computer needs:
        - 1000+ watts for cryogenic cooling
        - Dilution refrigerator to reach ~15 millikelvin
        - Bulky infrastructure (room-sized for 50+ qubits)
        
        Rotonium's molecular quantum system needs:
        - < 50 watts for the QPU itself
        - Room temperature operation (no cryogenics!)
        - Compact form factor suitable for edge deployment
        
        Args:
            job: Quantum job resource requirements
        
        Returns:
            bool: True if job can execute within all constraints
        """
        
        # Power constraint check
        # Critical for battery-powered and aerospace deployments
        if job.power_watts > self.power_budget_watts:
            return False
        
        # Thermal constraint check
        # Ensures device won't overheat during quantum computation
        # Rotonium advantage: No cryogenic cooling needed
        if job.thermal_output_watts > self._calculate_thermal_budget():
            return False
        
        # Timeout constraint check
        # Quantum jobs must complete within real-time requirements
        # Some quantum algorithms (QAOA, VQE) are iterative and time-consuming
        if job.execution_time_seconds > self.compute_timeout_seconds:
            return False
        
        # Memory constraint check
        # Must fit quantum circuit representation, state vectors (for simulation),
        # and classical pre/post-processing in available RAM
        if job.memory_mb > self.memory_limit_mb:
            return False
        
        # Storage constraint check
        # Need space for quantum programs, calibration data, results
        if job.storage_gb > self.storage_limit_gb:
            return False
        
        # Network bandwidth constraint check
        # If job needs to communicate with cloud services or other nodes
        if job.bandwidth_mbps > self.network_bandwidth_mbps:
            return False
        
        return True
    
    def can_execute_classical(self, job: JobRequirements) -> bool:
        """
        Check if a classical job can execute within environment constraints.
        
        Classical jobs (traditional CPU/GPU computation) have different resource
        profiles than quantum jobs. They typically:
        - Scale better with available memory
        - Have more predictable thermal characteristics
        - Can be interrupted and resumed more easily
        
        The quantum-classical hybrid router uses this method to determine if
        classical fallback is viable when quantum resources are constrained.
        
        Args:
            job: Classical job resource requirements
        
        Returns:
            bool: True if job can execute within all constraints
        """
        
        # Same constraint checks as quantum, but classical jobs may have
        # different characteristics (e.g., higher memory usage, different
        # power profiles, longer acceptable execution times)
        
        if job.power_watts > self.power_budget_watts:
            return False
        
        if job.thermal_output_watts > self._calculate_thermal_budget():
            return False
        
        # Classical jobs might be acceptable with longer timeouts
        # since they don't require maintaining quantum coherence
        if job.execution_time_seconds > self.compute_timeout_seconds:
            return False
        
        if job.memory_mb > self.memory_limit_mb:
            return False
        
        if job.storage_gb > self.storage_limit_gb:
            return False
        
        if job.bandwidth_mbps > self.network_bandwidth_mbps:
            return False
        
        return True
    
    def estimate_remaining_capacity(self, job: JobRequirements) -> ResourceCapacity:
        """
        Calculate remaining resources after job execution.
        
        This is crucial for resource scheduling in edge environments where
        multiple jobs might need to be queued or where system health must be
        monitored continuously.
        
        Args:
            job: Job that will consume resources
        
        Returns:
            ResourceCapacity: Remaining capacity after job execution
        """
        
        remaining_power = self.power_budget_watts - job.power_watts
        remaining_memory = self.memory_limit_mb - job.memory_mb
        remaining_storage = self.storage_limit_gb - job.storage_gb
        remaining_bandwidth = self.network_bandwidth_mbps - job.bandwidth_mbps
        
        # Calculate thermal headroom: how much temperature increase we can
        # tolerate before hitting thermal limits
        thermal_headroom = self._calculate_thermal_budget() - job.thermal_output_watts
        
        return ResourceCapacity(
            power_watts=max(0.0, remaining_power),
            memory_mb=max(0, remaining_memory),
            storage_gb=max(0.0, remaining_storage),
            bandwidth_mbps=max(0.0, remaining_bandwidth),
            thermal_headroom_celsius=max(0.0, thermal_headroom)
        )
    
    def simulate_thermal_impact(self, job: JobRequirements) -> float:
        """
        Estimate temperature increase from job execution.
        
        Thermal management is critical in edge environments, especially:
        - Aerospace: No air convection in space/high altitude
        - Mobile: Compact enclosures with limited cooling
        - Ground servers: Cooling costs money
        
        This simulation uses a simplified thermal model. In production, you'd
        want more sophisticated modeling based on:
        - Ambient temperature
        - Cooling system capacity (passive, active, liquid)
        - Thermal mass of the device
        - Duty cycle and transient thermal response
        
        Key insight: Rotonium's room-temperature operation means:
        - No need to maintain millikelvin temperatures
        - No cryogenic cooling power consumption
        - Thermal output is primarily from classical control electronics
        - Standard thermal management techniques apply
        
        Traditional quantum computers have inverse thermal challenges:
        - Must maintain extreme cold (millikelvin)
        - Heat leaks are the enemy
        - Require massive cooling systems (dilution refrigerators)
        
        Args:
            job: Job that generates thermal output
        
        Returns:
            float: Estimated temperature increase in degrees Celsius
        """
        
        # Simplified thermal model:
        # Temperature increase = (Power × Time) / (Thermal Mass × Specific Heat)
        # 
        # We use a rough approximation based on deployment profile:
        # - Aerospace: Poor heat dissipation (no air), high delta-T per watt
        # - Mobile: Moderate heat dissipation, medium delta-T per watt
        # - Ground: Good heat dissipation (data center cooling), low delta-T per watt
        
        if self.profile == DeploymentProfile.AEROSPACE:
            # Space/high-altitude: Limited cooling, high thermal impact
            # Relies on radiative cooling and thermal mass
            thermal_resistance = 2.0  # °C per watt (high resistance)
        elif self.profile == DeploymentProfile.MOBILE:
            # Mobile: Some air cooling, moderate thermal impact
            thermal_resistance = 1.0  # °C per watt
        else:  # GROUND_SERVER
            # Data center: Active cooling, low thermal impact
            thermal_resistance = 0.3  # °C per watt (good cooling)
        
        # Calculate steady-state temperature increase
        # This assumes thermal equilibrium; actual transient response more complex
        temperature_increase = job.thermal_output_watts * thermal_resistance
        
        return temperature_increase
    
    def _calculate_thermal_budget(self) -> float:
        """
        Calculate available thermal budget in watts.
        
        This represents how much heat the environment can dissipate before
        reaching thermal limits. Depends on cooling capacity and ambient
        temperature.
        
        Returns:
            float: Available thermal budget in watts
        """
        
        # Simplified model: thermal budget proportional to cooling capacity
        # In reality, this depends on ambient temperature, altitude, humidity, etc.
        
        if self.profile == DeploymentProfile.AEROSPACE:
            # Limited cooling in aerospace environment
            # Mainly radiative cooling, some conductive
            return 25.0  # watts
        elif self.profile == DeploymentProfile.MOBILE:
            # Moderate cooling in mobile environment
            # Air cooling with fans
            return 10.0  # watts
        else:  # GROUND_SERVER
            # Excellent cooling in data center
            # Active HVAC, cold aisle containment
            return 150.0  # watts
    
    def get_profile_info(self) -> Dict[str, any]:
        """
        Get comprehensive information about current deployment profile.
        
        Returns:
            Dict containing all profile parameters and context
        """
        return {
            "profile": self.profile.value,
            "deployment_context": self.deployment_context,
            "constraints": {
                "power_budget_watts": self.power_budget_watts,
                "thermal_limit_celsius": self.thermal_limit_celsius,
                "compute_timeout_seconds": self.compute_timeout_seconds,
                "memory_limit_mb": self.memory_limit_mb,
                "storage_limit_gb": self.storage_limit_gb,
                "network_bandwidth_mbps": self.network_bandwidth_mbps,
            },
            "thermal_budget_watts": self._calculate_thermal_budget(),
            "rotonium_advantage": (
                "Room-temperature quantum processing eliminates cryogenic "
                "cooling requirements, enabling quantum computing in resource-"
                "constrained edge environments."
            )
        }


class ResourceTracker:
    """
    Tracks resource usage over time for intelligent job scheduling.
    
    This class maintains historical resource utilization data and manages
    job queuing for edge environments. It enables:
    - Real-time resource tracking
    - Historical usage analysis
    - Intelligent job scheduling based on current load
    - Queue management for pending jobs
    
    Why this matters for edge quantum computing:
    
    1. RESOURCE CONTENTION: Edge devices run multiple workloads. The
       ResourceTracker ensures quantum jobs don't starve other critical
       tasks (navigation, sensing, communications).
    
    2. THERMAL MANAGEMENT: By tracking thermal history, we can implement
       thermal throttling to prevent damage. Rotonium's room-temperature
       operation helps, but we still need to manage heat from classical
       control electronics.
    
    3. POWER BUDGETING: Battery-powered edge devices need careful power
       management. Historical power data helps predict battery life and
       schedule jobs accordingly.
    
    4. QUALITY OF SERVICE: Queue management ensures high-priority quantum
       jobs (e.g., real-time navigation optimization) get resources before
       low-priority batch jobs (e.g., calibration, diagnostics).
    """
    
    def __init__(self, environment: EdgeEnvironment):
        """
        Initialize resource tracker for an edge environment.
        
        Args:
            environment: EdgeEnvironment instance to track
        """
        self.environment = environment
        
        # Resource usage history tracking
        # Format: List of (timestamp, value) tuples
        self.power_usage_history: List[Tuple[datetime, float]] = []
        self.thermal_history: List[Tuple[datetime, float]] = []
        
        # Currently allocated resources (running jobs)
        self.active_allocations: Dict[str, JobAllocation] = {}
        
        # Job queue for pending jobs
        # Deque allows efficient FIFO queue operations
        self.job_queue: Deque[Tuple[str, JobRequirements]] = deque()
        
        # Current resource usage counters
        self.current_power_usage = 0.0
        self.current_memory_usage = 0
        self.current_storage_usage = 0.0
        self.current_bandwidth_usage = 0.0
        self.current_thermal_output = 0.0
        
        # Statistics tracking
        self.total_jobs_scheduled = 0
        self.total_jobs_completed = 0
        self.total_jobs_rejected = 0
    
    def allocate_resources(self, job: JobRequirements, job_id: str = None) -> bool:
        """
        Try to allocate resources for a job.
        
        This method attempts to reserve resources for job execution. If
        sufficient resources are available, the allocation is recorded and
        True is returned. Otherwise, the job should be queued or rejected.
        
        Resource allocation in edge environments must be conservative:
        - Overcommitting can cause system instability
        - Battery-powered devices need power reserves for critical tasks
        - Thermal limits are hard constraints (hardware damage risk)
        
        For quantum jobs, allocation includes:
        - Power for quantum processor and classical control
        - Memory for circuit representation and classical processing
        - Storage for quantum program and results
        - Bandwidth for communication (if needed)
        - Thermal budget to prevent overheating
        
        Args:
            job: Job requiring resource allocation
            job_id: Optional unique identifier for the job
        
        Returns:
            bool: True if resources allocated successfully, False otherwise
        """
        
        # Generate job ID if not provided
        if job_id is None:
            job_id = f"job_{self.total_jobs_scheduled}_{datetime.now().timestamp()}"
        
        # Check if resources are available
        available_power = self.environment.power_budget_watts - self.current_power_usage
        available_memory = self.environment.memory_limit_mb - self.current_memory_usage
        available_storage = self.environment.storage_limit_gb - self.current_storage_usage
        available_bandwidth = self.environment.network_bandwidth_mbps - self.current_bandwidth_usage
        available_thermal = self.environment._calculate_thermal_budget() - self.current_thermal_output
        
        # Validate all resource constraints
        # ALL constraints must be satisfied for allocation to succeed
        if (job.power_watts > available_power or
            job.memory_mb > available_memory or
            job.storage_gb > available_storage or
            job.bandwidth_mbps > available_bandwidth or
            job.thermal_output_watts > available_thermal):
            return False
        
        # Allocate resources
        allocation = JobAllocation(
            job=job,
            job_id=job_id,
            allocated_at=datetime.now(),
            power_allocated=job.power_watts,
            memory_allocated=job.memory_mb,
            storage_allocated=job.storage_gb,
            bandwidth_allocated=job.bandwidth_mbps,
            thermal_allocated=job.thermal_output_watts
        )
        
        self.active_allocations[job_id] = allocation
        
        # Update current usage counters
        self.current_power_usage += job.power_watts
        self.current_memory_usage += job.memory_mb
        self.current_storage_usage += job.storage_gb
        self.current_bandwidth_usage += job.bandwidth_mbps
        self.current_thermal_output += job.thermal_output_watts
        
        # Record usage in history
        now = datetime.now()
        self.power_usage_history.append((now, self.current_power_usage))
        self.thermal_history.append((now, self.current_thermal_output))
        
        self.total_jobs_scheduled += 1
        
        return True
    
    def release_resources(self, job_id: str) -> bool:
        """
        Free up resources after job completes.
        
        When a job finishes execution, its allocated resources must be
        released so they can be used by other jobs. This method:
        - Removes the job from active allocations
        - Decrements resource usage counters
        - Records the release in usage history
        - Updates statistics
        
        Proper resource release is critical for:
        - Preventing resource leaks
        - Enabling accurate scheduling decisions
        - Maintaining system health monitoring
        
        Args:
            job_id: Unique identifier of the job to release
        
        Returns:
            bool: True if resources released successfully, False if job not found
        """
        
        if job_id not in self.active_allocations:
            return False
        
        allocation = self.active_allocations[job_id]
        
        # Release resources
        self.current_power_usage -= allocation.power_allocated
        self.current_memory_usage -= allocation.memory_allocated
        self.current_storage_usage -= allocation.storage_allocated
        self.current_bandwidth_usage -= allocation.bandwidth_allocated
        self.current_thermal_output -= allocation.thermal_allocated
        
        # Ensure values don't go negative due to floating point errors
        self.current_power_usage = max(0.0, self.current_power_usage)
        self.current_memory_usage = max(0, self.current_memory_usage)
        self.current_storage_usage = max(0.0, self.current_storage_usage)
        self.current_bandwidth_usage = max(0.0, self.current_bandwidth_usage)
        self.current_thermal_output = max(0.0, self.current_thermal_output)
        
        # Record usage in history
        now = datetime.now()
        self.power_usage_history.append((now, self.current_power_usage))
        self.thermal_history.append((now, self.current_thermal_output))
        
        # Remove allocation
        del self.active_allocations[job_id]
        
        self.total_jobs_completed += 1
        
        return True
    
    def get_utilization_stats(self) -> Dict:
        """
        Calculate comprehensive utilization statistics.
        
        This method provides insights into resource usage patterns over time,
        enabling:
        - Capacity planning (is the edge device over/under-utilized?)
        - Thermal management (are we approaching thermal limits?)
        - Power optimization (can we reduce idle power consumption?)
        - Queue management (do we need to reject jobs or add capacity?)
        
        Statistics are critical for edge quantum computing because:
        - Edge devices have no fallback resources (unlike cloud)
        - Resource exhaustion can cause mission failure (aerospace)
        - Battery life is precious (mobile deployments)
        - Cooling capacity is limited (all deployments)
        
        Returns:
            Dict: Comprehensive utilization statistics including:
                - Current utilization percentages
                - Historical averages
                - Peak usage values
                - Queue depth metrics
                - Job completion statistics
        """
        
        stats = {
            "current_utilization": {
                "power_watts": self.current_power_usage,
                "power_percent": (self.current_power_usage / self.environment.power_budget_watts) * 100,
                "memory_mb": self.current_memory_usage,
                "memory_percent": (self.current_memory_usage / self.environment.memory_limit_mb) * 100,
                "storage_gb": self.current_storage_usage,
                "storage_percent": (self.current_storage_usage / self.environment.storage_limit_gb) * 100,
                "bandwidth_mbps": self.current_bandwidth_usage,
                "bandwidth_percent": (self.current_bandwidth_usage / self.environment.network_bandwidth_mbps) * 100,
                "thermal_watts": self.current_thermal_output,
                "thermal_percent": (self.current_thermal_output / self.environment._calculate_thermal_budget()) * 100,
            },
            "active_jobs": len(self.active_allocations),
            "queued_jobs": len(self.job_queue),
            "job_statistics": {
                "total_scheduled": self.total_jobs_scheduled,
                "total_completed": self.total_jobs_completed,
                "total_rejected": self.total_jobs_rejected,
                "completion_rate": (self.total_jobs_completed / self.total_jobs_scheduled * 100) 
                                  if self.total_jobs_scheduled > 0 else 0.0,
            }
        }
        
        # Calculate historical statistics if we have data
        if len(self.power_usage_history) > 0:
            power_values = [usage for _, usage in self.power_usage_history]
            stats["power_history"] = {
                "average_watts": statistics.mean(power_values),
                "peak_watts": max(power_values),
                "min_watts": min(power_values),
                "stddev_watts": statistics.stdev(power_values) if len(power_values) > 1 else 0.0,
            }
        else:
            stats["power_history"] = {
                "average_watts": 0.0,
                "peak_watts": 0.0,
                "min_watts": 0.0,
                "stddev_watts": 0.0,
            }
        
        if len(self.thermal_history) > 0:
            thermal_values = [temp for _, temp in self.thermal_history]
            stats["thermal_profile"] = {
                "average_watts": statistics.mean(thermal_values),
                "peak_watts": max(thermal_values),
                "min_watts": min(thermal_values),
                "stddev_watts": statistics.stdev(thermal_values) if len(thermal_values) > 1 else 0.0,
            }
        else:
            stats["thermal_profile"] = {
                "average_watts": 0.0,
                "peak_watts": 0.0,
                "min_watts": 0.0,
                "stddev_watts": 0.0,
            }
        
        # Queue depth statistics
        stats["queue_statistics"] = {
            "current_depth": len(self.job_queue),
            "max_depth": len(self.job_queue),  # Would track historical max in production
        }
        
        return stats
    
    def can_schedule_job(self, job: JobRequirements, current_load: Dict = None) -> Tuple[bool, str]:
        """
        Determine if a job can be scheduled based on current resource state.
        
        This method implements intelligent scheduling logic that considers:
        - Current resource utilization
        - Available capacity headroom
        - Thermal constraints
        - Queue depth
        - Historical usage patterns
        
        The method returns both a decision (can/cannot schedule) and a
        human-readable reason explaining the decision. This is crucial for:
        - Debugging scheduling issues
        - Explaining decisions to operators
        - Logging and auditing
        - User feedback
        
        Scheduling logic for edge quantum computing must balance:
        - Utilization (maximize valuable computation)
        - Safety margins (don't overheat or drain battery)
        - Fairness (don't starve low-priority jobs)
        - Responsiveness (schedule time-critical jobs quickly)
        
        Args:
            job: Job to potentially schedule
            current_load: Optional dict with additional load information
        
        Returns:
            Tuple[bool, str]: (can_schedule, reason)
                - can_schedule: True if job can be scheduled now
                - reason: Human-readable explanation of the decision
        """
        
        # Calculate available resources
        available_power = self.environment.power_budget_watts - self.current_power_usage
        available_memory = self.environment.memory_limit_mb - self.current_memory_usage
        available_storage = self.environment.storage_limit_gb - self.current_storage_usage
        available_bandwidth = self.environment.network_bandwidth_mbps - self.current_bandwidth_usage
        available_thermal = self.environment._calculate_thermal_budget() - self.current_thermal_output
        
        # Check each resource constraint individually for detailed feedback
        
        # Power constraint
        if job.power_watts > available_power:
            utilization = (self.current_power_usage / self.environment.power_budget_watts) * 100
            return (False, 
                   f"Insufficient power: job requires {job.power_watts:.1f}W, "
                   f"only {available_power:.1f}W available ({utilization:.1f}% utilized). "
                   f"Rotonium's low-power QPU helps, but current load is too high.")
        
        # Memory constraint
        if job.memory_mb > available_memory:
            utilization = (self.current_memory_usage / self.environment.memory_limit_mb) * 100
            return (False,
                   f"Insufficient memory: job requires {job.memory_mb}MB, "
                   f"only {available_memory}MB available ({utilization:.1f}% utilized). "
                   f"Consider reducing quantum circuit size or freeing memory.")
        
        # Storage constraint
        if job.storage_gb > available_storage:
            utilization = (self.current_storage_usage / self.environment.storage_limit_gb) * 100
            return (False,
                   f"Insufficient storage: job requires {job.storage_gb:.1f}GB, "
                   f"only {available_storage:.1f}GB available ({utilization:.1f}% utilized). "
                   f"Clear old results or reduce data retention.")
        
        # Bandwidth constraint
        if job.bandwidth_mbps > available_bandwidth:
            utilization = (self.current_bandwidth_usage / self.environment.network_bandwidth_mbps) * 100
            return (False,
                   f"Insufficient bandwidth: job requires {job.bandwidth_mbps:.1f}Mbps, "
                   f"only {available_bandwidth:.1f}Mbps available ({utilization:.1f}% utilized). "
                   f"Edge quantum computing reduces cloud dependency!")
        
        # Thermal constraint (critical for edge devices!)
        if job.thermal_output_watts > available_thermal:
            utilization = (self.current_thermal_output / self.environment._calculate_thermal_budget()) * 100
            return (False,
                   f"Thermal limit exceeded: job generates {job.thermal_output_watts:.1f}W heat, "
                   f"only {available_thermal:.1f}W headroom available ({utilization:.1f}% utilized). "
                   f"Wait for cooling. Rotonium's room-temp operation eliminates cryogenic "
                   f"cooling, but we still need to manage classical electronics heat.")
        
        # Timeout constraint
        if job.execution_time_seconds > self.environment.compute_timeout_seconds:
            return (False,
                   f"Execution time too long: job needs {job.execution_time_seconds:.1f}s, "
                   f"max allowed is {self.environment.compute_timeout_seconds:.1f}s. "
                   f"Break into smaller quantum circuits or use classical approximation.")
        
        # All constraints satisfied!
        return (True,
               f"Job can be scheduled: sufficient resources available. "
               f"Power: {available_power:.1f}W free, Memory: {available_memory}MB free, "
               f"Thermal: {available_thermal:.1f}W headroom.")
    
    def add_to_queue(self, job: JobRequirements, job_id: str = None) -> str:
        """
        Add a job to the pending queue.
        
        Args:
            job: Job to queue
            job_id: Optional unique identifier for the job
        
        Returns:
            str: Job ID assigned to the queued job
        """
        if job_id is None:
            job_id = f"queued_job_{len(self.job_queue)}_{datetime.now().timestamp()}"
        
        self.job_queue.append((job_id, job))
        return job_id
    
    def process_queue(self) -> List[str]:
        """
        Attempt to schedule jobs from the queue.
        
        This method tries to allocate resources for queued jobs in FIFO order.
        It's useful for batch processing when resources become available.
        
        Returns:
            List[str]: Job IDs that were successfully scheduled from the queue
        """
        scheduled_jobs = []
        
        # Process queue until we can't schedule any more jobs
        while self.job_queue:
            job_id, job = self.job_queue[0]  # Peek at first job
            
            can_schedule, reason = self.can_schedule_job(job)
            
            if can_schedule:
                # Remove from queue and allocate
                self.job_queue.popleft()
                if self.allocate_resources(job, job_id):
                    scheduled_jobs.append(job_id)
            else:
                # Can't schedule this job, so we can't schedule any behind it either
                # (to maintain FIFO order)
                break
        
        return scheduled_jobs
    
    def get_current_state(self) -> Dict:
        """
        Get complete current state of the resource tracker.
        
        Returns:
            Dict: Complete state including allocations, queue, and usage
        """
        return {
            "active_allocations": {
                job_id: {
                    "power_allocated": alloc.power_allocated,
                    "memory_allocated": alloc.memory_allocated,
                    "storage_allocated": alloc.storage_allocated,
                    "bandwidth_allocated": alloc.bandwidth_allocated,
                    "thermal_allocated": alloc.thermal_allocated,
                    "allocated_at": alloc.allocated_at.isoformat(),
                }
                for job_id, alloc in self.active_allocations.items()
            },
            "queue_length": len(self.job_queue),
            "current_usage": {
                "power_watts": self.current_power_usage,
                "memory_mb": self.current_memory_usage,
                "storage_gb": self.current_storage_usage,
                "bandwidth_mbps": self.current_bandwidth_usage,
                "thermal_watts": self.current_thermal_output,
            },
            "utilization_stats": self.get_utilization_stats(),
        }


# Example usage and testing
if __name__ == "__main__":
    print("=" * 70)
    print("Edge Computing Environment Simulator - Rotonium Quantum Systems")
    print("=" * 70)
    print()
    
    # Test all deployment profiles
    profiles = [
        DeploymentProfile.AEROSPACE,
        DeploymentProfile.MOBILE,
        DeploymentProfile.GROUND_SERVER
    ]
    
    for profile in profiles:
        env = EdgeEnvironment(profile)
        info = env.get_profile_info()
        
        print(f"Profile: {info['profile'].upper()}")
        print(f"Context: {info['deployment_context']}")
        print(f"Constraints:")
        for key, value in info['constraints'].items():
            print(f"  {key}: {value}")
        print(f"Thermal Budget: {info['thermal_budget_watts']} watts")
        print()
        
        # Test with a sample quantum job
        quantum_job = JobRequirements(
            power_watts=30.0,
            execution_time_seconds=5.0,
            memory_mb=2048,
            storage_gb=10.0,
            thermal_output_watts=20.0,
            bandwidth_mbps=2.0
        )
        
        can_run_quantum = env.can_execute_quantum(quantum_job)
        print(f"Can execute quantum job: {can_run_quantum}")
        
        if can_run_quantum:
            remaining = env.estimate_remaining_capacity(quantum_job)
            temp_increase = env.simulate_thermal_impact(quantum_job)
            print(f"Remaining capacity after job:")
            print(f"  Power: {remaining.power_watts:.1f} watts")
            print(f"  Memory: {remaining.memory_mb} MB")
            print(f"  Storage: {remaining.storage_gb:.1f} GB")
            print(f"  Thermal headroom: {remaining.thermal_headroom_celsius:.1f}°C")
            print(f"Estimated temperature increase: {temp_increase:.1f}°C")
        
        print()
        print("-" * 70)
        print()
    
    # Test ResourceTracker functionality
    print("=" * 70)
    print("ResourceTracker Testing - Intelligent Job Scheduling")
    print("=" * 70)
    print()
    
    # Create an environment and tracker
    env = EdgeEnvironment(DeploymentProfile.GROUND_SERVER)
    tracker = ResourceTracker(env)
    
    print(f"Testing with {env.profile.value.upper()} profile")
    print(f"Power budget: {env.power_budget_watts}W")
    print(f"Memory limit: {env.memory_limit_mb}MB")
    print(f"Thermal budget: {env._calculate_thermal_budget()}W")
    print()
    
    # Create test jobs
    job1 = JobRequirements(
        power_watts=50.0,
        execution_time_seconds=10.0,
        memory_mb=4096,
        storage_gb=20.0,
        thermal_output_watts=40.0,
        bandwidth_mbps=100.0
    )
    
    job2 = JobRequirements(
        power_watts=80.0,
        execution_time_seconds=15.0,
        memory_mb=6144,
        storage_gb=30.0,
        thermal_output_watts=60.0,
        bandwidth_mbps=200.0
    )
    
    job3 = JobRequirements(
        power_watts=100.0,
        execution_time_seconds=20.0,
        memory_mb=8192,
        storage_gb=50.0,
        thermal_output_watts=80.0,
        bandwidth_mbps=300.0
    )
    
    # Try to schedule jobs
    print("Attempting to schedule Job 1...")
    can_schedule, reason = tracker.can_schedule_job(job1)
    print(f"Can schedule: {can_schedule}")
    print(f"Reason: {reason}")
    
    if can_schedule:
        success = tracker.allocate_resources(job1, "job_1")
        print(f"Allocation successful: {success}")
    print()
    
    print("Attempting to schedule Job 2...")
    can_schedule, reason = tracker.can_schedule_job(job2)
    print(f"Can schedule: {can_schedule}")
    print(f"Reason: {reason}")
    
    if can_schedule:
        success = tracker.allocate_resources(job2, "job_2")
        print(f"Allocation successful: {success}")
    print()
    
    print("Attempting to schedule Job 3...")
    can_schedule, reason = tracker.can_schedule_job(job3)
    print(f"Can schedule: {can_schedule}")
    print(f"Reason: {reason}")
    
    if not can_schedule:
        print("Job 3 queued for later execution")
        tracker.add_to_queue(job3, "job_3")
    print()
    
    # Show current utilization
    print("Current Utilization Statistics:")
    stats = tracker.get_utilization_stats()
    print(f"Active jobs: {stats['active_jobs']}")
    print(f"Queued jobs: {stats['queued_jobs']}")
    print(f"Power utilization: {stats['current_utilization']['power_percent']:.1f}%")
    print(f"Memory utilization: {stats['current_utilization']['memory_percent']:.1f}%")
    print(f"Thermal utilization: {stats['current_utilization']['thermal_percent']:.1f}%")
    print()
    
    # Release Job 1 and try to schedule from queue
    print("Releasing Job 1 resources...")
    tracker.release_resources("job_1")
    print("Attempting to process queue...")
    scheduled = tracker.process_queue()
    print(f"Jobs scheduled from queue: {scheduled}")
    print()
    
    # Final statistics
    print("Final Utilization Statistics:")
    stats = tracker.get_utilization_stats()
    print(f"Active jobs: {stats['active_jobs']}")
    print(f"Queued jobs: {stats['queued_jobs']}")
    print(f"Total scheduled: {stats['job_statistics']['total_scheduled']}")
    print(f"Total completed: {stats['job_statistics']['total_completed']}")
    print(f"Power utilization: {stats['current_utilization']['power_percent']:.1f}%")
    print(f"Memory utilization: {stats['current_utilization']['memory_percent']:.1f}%")
    print(f"Thermal utilization: {stats['current_utilization']['thermal_percent']:.1f}%")
    
    if stats['power_history']['peak_watts'] > 0:
        print(f"\nPower History:")
        print(f"  Average: {stats['power_history']['average_watts']:.1f}W")
        print(f"  Peak: {stats['power_history']['peak_watts']:.1f}W")
    
    if stats['thermal_profile']['peak_watts'] > 0:
        print(f"\nThermal Profile:")
        print(f"  Average: {stats['thermal_profile']['average_watts']:.1f}W")
        print(f"  Peak: {stats['thermal_profile']['peak_watts']:.1f}W")
    
    print()
    print("=" * 70)
    print("ResourceTracker enables intelligent quantum job scheduling on edge devices!")
    print("Rotonium's room-temperature QPU makes this practical for aerospace,")
    print("mobile, and ground server deployments.")
    print("=" * 70)
