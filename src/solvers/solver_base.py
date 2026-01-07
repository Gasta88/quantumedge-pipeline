"""
Abstract base class for all solvers in QuantumEdge Pipeline.

This module defines the standard interface that all solvers (classical, quantum,
and hybrid) must implement. It ensures consistent behavior, fair comparison, and
easy integration with the routing system.

Why Standardized Output Format?
--------------------------------
The pipeline needs to compare different solver types (classical vs quantum) fairly.
Without standardization:
- Can't compare solution quality across solvers
- Energy consumption metrics would be inconsistent
- Timing measurements would use different methodologies
- Debugging and analysis would be solver-specific

With standardized format:
✓ Fair apples-to-apples comparison
✓ Consistent metrics for router learning
✓ Easy to add new solvers without changing downstream code
✓ Unified monitoring and logging
✓ Reproducible benchmarking

Standardized Result Format
---------------------------
All solvers must return results in this exact format:
{
    'solution': List[Any],      # The actual solution (format varies by problem)
    'cost': float,              # Objective function value (lower is better)
    'time_ms': int,             # Wall-clock execution time in milliseconds
    'energy_mj': float,         # Estimated energy consumption in millijoules
    'iterations': int,          # Number of solver iterations/shots
    'metadata': Dict[str, Any]  # Solver-specific information
}

This enables:
- Performance comparison across solver types
- Energy efficiency analysis
- Cost-benefit analysis for routing decisions
- Historical performance tracking
- Algorithm selection based on resource constraints

Example Usage
-------------
```python
from src.solvers.classical_solver import ClassicalSolver
from src.problems.maxcut import MaxCutProblem

# Create problem
problem = MaxCutProblem(num_nodes=30)
problem.generate(edge_probability=0.3)

# Solve with classical solver
solver = ClassicalSolver(algorithm='goemans_williamson')
result = solver.solve(problem, max_iterations=1000)

# Access standardized results
print(f"Solution: {result['solution']}")
print(f"Cost: {result['cost']:.4f}")
print(f"Time: {result['time_ms']} ms")
print(f"Energy: {result['energy_mj']:.2f} mJ")

# Or use context manager for automatic resource cleanup
with ClassicalSolver(algorithm='greedy') as solver:
    result = solver.solve(problem)
```

Design Philosophy
-----------------
1. **Interface Segregation**: Abstract methods only for essential operations
2. **Template Method Pattern**: Concrete methods handle cross-cutting concerns
3. **Context Manager Support**: Automatic resource management
4. **Fail-Fast Validation**: Catch errors early with comprehensive checks
5. **Observable Behavior**: Extensive logging for debugging and monitoring
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import logging
import time
import psutil
import os

from src.problems.problem_base import ProblemBase


# Configure module logger
logger = logging.getLogger(__name__)


# ============================================================================
# Custom Exceptions
# ============================================================================

class SolverException(Exception):
    """Base exception for all solver-related errors."""
    pass


class SolverConfigurationError(SolverException):
    """Raised when solver is misconfigured or missing required parameters."""
    pass


class SolverTimeoutError(SolverException):
    """Raised when solver exceeds maximum allowed execution time."""
    pass


class SolverConvergenceError(SolverException):
    """Raised when solver fails to converge to a valid solution."""
    pass


class InvalidSolutionError(SolverException):
    """Raised when solver produces an invalid solution."""
    pass


# ============================================================================
# Energy Tracking Utilities
# ============================================================================

class EnergyTracker:
    """
    Track and estimate energy consumption during computation.
    
    This class provides utilities for estimating CPU energy consumption during
    solver execution. It's designed for edge computing scenarios where energy
    efficiency is critical for battery-powered or resource-constrained devices.
    
    Why Energy Tracking Matters for Edge Computing:
    ------------------------------------------------
    1. **Battery Life**: Edge devices often run on batteries
       - Every millijoule counts for extending device lifetime
       - Energy-efficient algorithms enable longer operation
    
    2. **Thermal Management**: Power consumption → heat generation
       - High power draw can trigger thermal throttling
       - Sustained high power reduces device lifespan
       - Cooling requirements increase cost and size
    
    3. **Cost Optimization**: Energy costs money
       - In datacenters: Power bills are significant operational cost
       - For IoT: Battery replacement is expensive at scale
       - Solar/battery systems: Limited energy budget
    
    4. **Sustainability**: Environmental impact
       - Lower energy = lower carbon footprint
       - Green computing initiatives
       - Regulatory requirements
    
    5. **Routing Decisions**: Energy vs Performance tradeoff
       - Classical vs Quantum: Different energy profiles
       - Fast but energy-hungry vs Slow but efficient
       - Enables energy-aware algorithm selection
    
    Energy Estimation Methodology:
    ------------------------------
    This class provides ESTIMATES, not exact measurements. Here's why:
    
    **What We Measure:**
    - CPU time (user + system time from psutil)
    - CPU utilization percentage
    - Wall-clock time
    
    **What We Estimate:**
    - Power draw (Watts) based on CPU model TDP
    - Energy consumption (Joules) = Power × Time
    
    **Why Estimation?**
    Precise energy measurement requires:
    - Hardware energy counters (Intel RAPL, ARM PMU)
    - Root/admin privileges to access counters
    - Platform-specific code (not portable)
    - Kernel support (not always available)
    
    Our approach:
    ✓ Works everywhere (no special hardware/permissions)
    ✓ Portable across platforms (Linux, Windows, macOS)
    ✓ Good enough for comparative analysis
    ✓ Captures relative differences between solvers
    
    **Accuracy Limitations:**
    - Assumes constant TDP (actual power varies with workload)
    - Doesn't account for memory/disk/network energy
    - Ignores CPU frequency scaling (turbo boost, throttling)
    - Utilization-based estimation is approximate
    - Different cores may have different power draws
    
    **When Accuracy Matters:**
    For research-grade measurements, use:
    - Intel RAPL (Running Average Power Limit) on Linux
    - ARM Performance Monitors
    - External power meters (Kill-A-Watt, etc.)
    - Specialized profiling tools (PowerAPI, PAPI)
    
    **When Our Estimates Are Sufficient:**
    - Comparative analysis (Solver A vs Solver B)
    - Algorithm selection and routing
    - Trend analysis over time
    - Energy-aware optimization
    - User feedback (energy cost indicators)
    
    CPU TDP (Thermal Design Power) Explained:
    -----------------------------------------
    TDP is the maximum power a CPU is designed to dissipate under load.
    
    Key Points:
    - **Not maximum power**: CPU can exceed TDP briefly (turbo boost)
    - **Not average power**: Typical workloads use 30-70% of TDP
    - **Design specification**: For cooling system design
    - **Marketing number**: May not reflect real-world power
    
    Typical TDP values:
    - Laptop CPUs: 15W - 45W
    - Desktop CPUs: 65W - 125W
    - Server CPUs: 150W - 300W
    - Mobile/Edge: 5W - 15W
    
    Our default: 65W (typical desktop CPU)
    You can calibrate for your specific CPU.
    
    Calibration for Specific CPU Types:
    -----------------------------------
    To get more accurate estimates for your hardware:
    
    1. **Identify your CPU**:
       - Linux: cat /proc/cpuinfo | grep "model name"
       - Windows: wmic cpu get name
       - macOS: sysctl -n machdep.cpu.brand_string
    
    2. **Look up TDP**:
       - Check CPU specifications online
       - Intel: ark.intel.com
       - AMD: amd.com/en/products/specifications
    
    3. **Calibrate**:
       ```python
       # For Intel Core i7-10700 (65W TDP)
       tracker = EnergyTracker(tdp_watts=65.0)
       
       # For Raspberry Pi 4 (~7W typical)
       tracker = EnergyTracker(tdp_watts=7.0, utilization_factor=0.8)
       
       # For server Xeon Gold 6258R (205W TDP)
       tracker = EnergyTracker(tdp_watts=205.0)
       ```
    
    4. **Adjust utilization factor**:
       - Default 0.6 = CPU runs at 60% of max power
       - Compute-heavy: 0.8-0.9
       - I/O-heavy: 0.3-0.5
       - Mixed workload: 0.5-0.7
    
    Example Usage:
    --------------
    ```python
    # Basic usage
    tracker = EnergyTracker()
    tracker.start_tracking()
    
    # ... do computation ...
    
    energy_mj = tracker.stop_tracking()
    print(f"Energy consumed: {energy_mj:.2f} mJ")
    
    # With calibration
    tracker = EnergyTracker(
        tdp_watts=45.0,          # Laptop CPU
        utilization_factor=0.7,   # Moderate workload
        efficiency=0.85           # 85% efficiency
    )
    
    # Real-time monitoring
    tracker.start_tracking()
    for iteration in range(1000):
        # ... computation ...
        current_power = tracker.estimate_power_draw()
        print(f"Current power draw: {current_power:.2f} W")
    
    energy_mj = tracker.stop_tracking()
    ```
    
    Thread Safety:
    --------------
    NOT thread-safe. Each thread should use its own EnergyTracker instance.
    """
    
    def __init__(
        self,
        tdp_watts: float = 65.0,
        utilization_factor: float = 0.6,
        efficiency: float = 0.8,
        process: Optional[psutil.Process] = None
    ):
        """
        Initialize energy tracker with CPU-specific parameters.
        
        Args:
            tdp_watts: CPU Thermal Design Power in watts
                      Default: 65W (typical desktop CPU)
                      Laptop: 15-45W, Desktop: 65-125W, Server: 150-300W
            
            utilization_factor: Fraction of TDP used during computation
                               Default: 0.6 (60% of max power)
                               Range: 0.3 (light) to 0.9 (heavy compute)
                               Accounts for: CPU not always at 100% turbo boost
            
            efficiency: Power conversion efficiency
                       Default: 0.8 (80% efficiency)
                       Accounts for: Not all power goes to computation
                       - Some lost as heat in voltage regulators
                       - Memory controller overhead
                       - Cache operations
            
            process: psutil.Process object (default: current process)
                    Allows tracking specific subprocess if needed
        
        Raises:
            ValueError: If parameters are out of valid ranges
        
        Example:
            >>> # Raspberry Pi 4
            >>> tracker = EnergyTracker(tdp_watts=7.0, utilization_factor=0.8)
            
            >>> # High-end desktop
            >>> tracker = EnergyTracker(tdp_watts=125.0, utilization_factor=0.7)
            
            >>> # Server workload
            >>> tracker = EnergyTracker(tdp_watts=200.0, utilization_factor=0.9)
        """
        # Validate parameters
        if tdp_watts <= 0 or tdp_watts > 500:
            raise ValueError(f"Invalid TDP: {tdp_watts}W. Expected 0-500W.")
        
        if utilization_factor <= 0 or utilization_factor > 1.0:
            raise ValueError(
                f"Invalid utilization_factor: {utilization_factor}. Expected 0-1."
            )
        
        if efficiency <= 0 or efficiency > 1.0:
            raise ValueError(f"Invalid efficiency: {efficiency}. Expected 0-1.")
        
        # Store calibration parameters
        self.tdp_watts = tdp_watts
        self.utilization_factor = utilization_factor
        self.efficiency = efficiency
        
        # Process to monitor
        self._process = process if process else psutil.Process(os.getpid())
        
        # Tracking state
        self._start_time: Optional[float] = None
        self._start_cpu_time: Optional[float] = None
        self._is_tracking: bool = False
        
        logger.debug(
            f"EnergyTracker initialized: TDP={tdp_watts}W, "
            f"utilization={utilization_factor}, efficiency={efficiency}"
        )
    
    def start_tracking(self) -> None:
        """
        Start tracking energy consumption.
        
        Records the initial state:
        - Wall-clock time (for duration calculation)
        - CPU time (user + system time)
        
        Must be called before stop_tracking().
        
        Example:
            >>> tracker = EnergyTracker()
            >>> tracker.start_tracking()
            >>> # ... computation ...
            >>> energy = tracker.stop_tracking()
        
        Raises:
            RuntimeError: If tracking is already active
        """
        if self._is_tracking:
            raise RuntimeError("Tracking already active. Call stop_tracking() first.")
        
        try:
            # Record wall-clock time
            self._start_time = time.perf_counter()
            
            # Record CPU time (user + system)
            # user time: time spent executing user code
            # system time: time spent in kernel on behalf of process
            cpu_times = self._process.cpu_times()
            self._start_cpu_time = cpu_times.user + cpu_times.system
            
            self._is_tracking = True
            
            logger.debug(
                f"Energy tracking started: time={self._start_time:.6f}s, "
                f"cpu_time={self._start_cpu_time:.6f}s"
            )
        
        except Exception as e:
            logger.error(f"Failed to start energy tracking: {e}")
            self._start_time = None
            self._start_cpu_time = None
            self._is_tracking = False
            raise
    
    def stop_tracking(self) -> float:
        """
        Stop tracking and calculate energy consumed.
        
        Calculation:
        1. Measure time delta: Δt = current_time - start_time
        2. Measure CPU time delta: Δcpu = current_cpu_time - start_cpu_time
        3. Calculate average power: P = TDP × utilization × efficiency
        4. Calculate energy: E = P × Δcpu (use CPU time, not wall time)
        5. Convert to millijoules: mJ = J × 1000
        
        Why use CPU time instead of wall time?
        - CPU time: actual time CPU spent on this process
        - Wall time: includes idle time, I/O waits, other processes
        - Energy consumed only when CPU is active
        - More accurate for compute-bound workloads
        
        Returns:
            Energy consumed in millijoules (mJ)
            Returns 0.0 if tracking wasn't started or failed
        
        Example:
            >>> tracker.start_tracking()
            >>> result = expensive_computation()
            >>> energy_mj = tracker.stop_tracking()
            >>> print(f"Consumed {energy_mj:.2f} mJ")
        
        Note:
            This is an ESTIMATE. For exact measurements, use hardware counters.
        """
        if not self._is_tracking:
            logger.warning("Tracking not active. Call start_tracking() first.")
            return 0.0
        
        try:
            # Record end times
            end_time = time.perf_counter()
            cpu_times = self._process.cpu_times()
            end_cpu_time = cpu_times.user + cpu_times.system
            
            # Calculate deltas
            wall_time_seconds = end_time - self._start_time
            cpu_time_seconds = end_cpu_time - self._start_cpu_time
            
            # Estimate average power draw (Watts)
            # Power = TDP × utilization_factor × efficiency
            # 
            # TDP: Maximum design power
            # utilization_factor: What fraction of TDP we actually use
            # efficiency: How much power actually does computation
            average_power_watts = (
                self.tdp_watts * 
                self.utilization_factor * 
                self.efficiency
            )
            
            # Calculate energy (Joules)
            # Energy = Power × Time
            # 
            # Use CPU time (not wall time) because:
            # - Only CPU time actually consumes power
            # - Wall time includes idle periods
            # - More accurate for compute workloads
            energy_joules = average_power_watts * cpu_time_seconds
            
            # Convert to millijoules (1 J = 1000 mJ)
            energy_mj = energy_joules * 1000.0
            
            # Reset tracking state
            self._is_tracking = False
            self._start_time = None
            self._start_cpu_time = None
            
            logger.debug(
                f"Energy tracking stopped: "
                f"wall_time={wall_time_seconds:.4f}s, "
                f"cpu_time={cpu_time_seconds:.4f}s, "
                f"power={average_power_watts:.2f}W, "
                f"energy={energy_mj:.2f}mJ"
            )
            
            return energy_mj
        
        except Exception as e:
            logger.error(f"Failed to stop energy tracking: {e}")
            self._is_tracking = False
            self._start_time = None
            self._start_cpu_time = None
            return 0.0
    
    def estimate_power_draw(self) -> float:
        """
        Estimate current power draw in watts.
        
        This provides a real-time estimate of power consumption based on
        current CPU utilization. Useful for monitoring during long computations.
        
        Calculation:
        1. Get current CPU utilization percentage (0-100%)
        2. Normalize to 0-1 scale
        3. Power = TDP × (utilization/100) × efficiency
        
        Note: This uses system-wide CPU utilization from psutil.cpu_percent(),
        not just this process. For per-process power, use the energy tracking
        methods which are based on per-process CPU time.
        
        Returns:
            Estimated power draw in watts
            Returns 0.0 if measurement fails
        
        Example:
            >>> tracker = EnergyTracker(tdp_watts=65.0)
            >>> power = tracker.estimate_power_draw()
            >>> print(f"Current power draw: {power:.2f} W")
            >>> # Output: Current power draw: 32.50 W (if CPU at 50%)
        
        Use Cases:
            - Real-time power monitoring
            - Detecting thermal throttling
            - Energy budget enforcement
            - Dynamic algorithm switching
        """
        try:
            # Get current CPU utilization (system-wide)
            # interval=0.1: measure over 100ms for accuracy
            # percpu=False: aggregate across all cores
            cpu_percent = psutil.cpu_percent(interval=0.1, percpu=False)
            
            # Normalize to 0-1 scale
            cpu_utilization = cpu_percent / 100.0
            
            # Estimate power
            # Power = TDP × utilization × efficiency
            # 
            # Example:
            # - TDP = 65W
            # - CPU at 50% utilization
            # - Efficiency = 0.8
            # - Power = 65 × 0.5 × 0.8 = 26W
            estimated_power = (
                self.tdp_watts * 
                cpu_utilization * 
                self.efficiency
            )
            
            logger.debug(
                f"Power estimate: {estimated_power:.2f}W "
                f"(CPU utilization: {cpu_percent:.1f}%)"
            )
            
            return estimated_power
        
        except Exception as e:
            logger.error(f"Failed to estimate power draw: {e}")
            return 0.0
    
    def get_calibration_info(self) -> Dict[str, Any]:
        """
        Get current calibration parameters.
        
        Returns:
            Dictionary with calibration settings and system info
        
        Example:
            >>> tracker = EnergyTracker(tdp_watts=45.0)
            >>> info = tracker.get_calibration_info()
            >>> print(info)
            {
                'tdp_watts': 45.0,
                'utilization_factor': 0.6,
                'efficiency': 0.8,
                'estimated_max_power_watts': 21.6,
                'cpu_count': 8,
                'cpu_freq_mhz': 2400.0
            }
        """
        try:
            cpu_info = {
                'cpu_count': psutil.cpu_count(logical=True),
                'cpu_freq_mhz': psutil.cpu_freq().current if psutil.cpu_freq() else None
            }
        except:
            cpu_info = {'cpu_count': None, 'cpu_freq_mhz': None}
        
        return {
            'tdp_watts': self.tdp_watts,
            'utilization_factor': self.utilization_factor,
            'efficiency': self.efficiency,
            'estimated_max_power_watts': (
                self.tdp_watts * self.utilization_factor * self.efficiency
            ),
            **cpu_info
        }
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"EnergyTracker(tdp={self.tdp_watts}W, "
            f"util={self.utilization_factor}, "
            f"eff={self.efficiency}, "
            f"tracking={self._is_tracking})"
        )


# ============================================================================
# Abstract Solver Base Class
# ============================================================================

class SolverBase(ABC):
    """
    Abstract base class for all optimization solvers.
    
    This class defines the standard interface that all solvers must implement,
    ensuring consistent behavior and enabling fair comparison across different
    solver types (classical, quantum, hybrid).
    
    All solvers must:
    1. Implement solve() to return standardized result format
    2. Implement get_solver_info() to describe capabilities
    3. Handle errors gracefully with specific exception types
    4. Support context manager protocol for resource cleanup
    
    Attributes:
        solver_type (str): Type of solver ('classical', 'quantum', 'hybrid')
        solver_name (str): Specific solver name (e.g., 'goemans_williamson', 'qaoa')
        _energy_start (float): CPU energy at measurement start
        _process (psutil.Process): Process object for energy monitoring
    
    Thread Safety:
        Solvers are generally NOT thread-safe. Create separate instances
        for concurrent execution.
    
    Resource Management:
        Use context manager (with statement) to ensure proper cleanup:
        ```python
        with MySolver() as solver:
            result = solver.solve(problem)
        # Resources automatically cleaned up
        ```
    """
    
    def __init__(self, solver_type: str, solver_name: str):
        """
        Initialize base solver.
        
        Args:
            solver_type: Type of solver ('classical', 'quantum', 'hybrid')
            solver_name: Specific solver name for identification
        
        Raises:
            SolverConfigurationError: If parameters are invalid
        """
        if not solver_type or not isinstance(solver_type, str):
            raise SolverConfigurationError("solver_type must be a non-empty string")
        
        if not solver_name or not isinstance(solver_name, str):
            raise SolverConfigurationError("solver_name must be a non-empty string")
        
        self.solver_type = solver_type.lower()
        self.solver_name = solver_name.lower()
        
        # Energy monitoring with enhanced EnergyTracker
        self._energy_tracker = EnergyTracker()
        
        # Backward compatibility: keep old attributes
        self._energy_start: Optional[float] = None
        self._process: psutil.Process = psutil.Process(os.getpid())
        
        logger.info(f"Initialized {self.solver_type} solver: {self.solver_name}")
    
    # ========================================================================
    # Abstract Methods - Must be implemented by subclasses
    # ========================================================================
    
    @abstractmethod
    def solve(self, problem: ProblemBase, **kwargs) -> Dict[str, Any]:
        """
        Solve the optimization problem and return standardized results.
        
        This is the main entry point for all solvers. It must:
        1. Validate the problem is generated and compatible
        2. Execute the solving algorithm
        3. Measure execution time and energy consumption
        4. Validate the solution is valid
        5. Return results in standardized format
        
        Standardized Result Format (REQUIRED):
        --------------------------------------
        {
            'solution': List[Any],      # Problem-specific solution format
                                        # - MaxCut: List[int] binary assignments
                                        # - TSP: List[int] city ordering
                                        # - Portfolio: List[float] asset weights
            
            'cost': float,              # Objective function value
                                        # - Lower is better (minimization)
                                        # - For maximization problems, negate the value
                                        # - Must match problem.calculate_cost(solution)
            
            'time_ms': int,             # Wall-clock execution time in milliseconds
                                        # - Measure from solve start to end
                                        # - Include setup but exclude validation
                                        # - Use time.perf_counter() for accuracy
            
            'energy_mj': float,         # Estimated energy consumption in millijoules
                                        # - Use measure_energy_start/end methods
                                        # - Based on CPU utilization
                                        # - Approximation, not exact measurement
            
            'iterations': int,          # Number of solver iterations/steps
                                        # - Classical: optimization iterations
                                        # - Quantum: number of shots/measurements
                                        # - Used for convergence analysis
            
            'metadata': Dict[str, Any]  # Solver-specific additional information
                                        # - Algorithm parameters used
                                        # - Convergence information
                                        # - Intermediate results
                                        # - Any debugging information
        }
        
        Why This Format?
        ----------------
        1. **Fair Comparison**: Same metrics across all solver types
        2. **Router Learning**: Consistent data for performance prediction
        3. **Resource Tracking**: Energy and time for cost-benefit analysis
        4. **Debugging**: Metadata helps diagnose solver issues
        5. **Benchmarking**: Standardized format for reproducible experiments
        
        Args:
            problem: Problem instance to solve (must be generated)
            **kwargs: Solver-specific parameters, may include:
                     - max_iterations: Maximum optimization iterations
                     - timeout_seconds: Maximum execution time
                     - tolerance: Convergence tolerance
                     - random_seed: For reproducibility
        
        Returns:
            Dictionary with standardized result format (see above)
        
        Raises:
            SolverConfigurationError: If problem is incompatible with solver
            SolverTimeoutError: If execution exceeds timeout
            SolverConvergenceError: If solver fails to find valid solution
            InvalidSolutionError: If solution violates problem constraints
        
        Example Implementation Pattern:
        -------------------------------
        ```python
        def solve(self, problem, **kwargs):
            # 1. Validate inputs
            if not problem.is_generated:
                raise SolverConfigurationError("Problem not generated")
            
            # 2. Start measurements
            start_time = time.perf_counter()
            self.measure_energy_start()
            
            # 3. Execute solving algorithm
            solution = self._run_algorithm(problem, **kwargs)
            
            # 4. End measurements
            end_time = time.perf_counter()
            energy_mj = self.measure_energy_end()
            
            # 5. Validate solution
            if not self.validate_result(problem, solution):
                raise InvalidSolutionError("Solution violates constraints")
            
            # 6. Calculate cost
            cost = problem.calculate_cost(solution)
            
            # 7. Return standardized format
            return {
                'solution': solution,
                'cost': cost,
                'time_ms': int((end_time - start_time) * 1000),
                'energy_mj': energy_mj,
                'iterations': self._iteration_count,
                'metadata': {'algorithm': self.solver_name, ...}
            }
        ```
        """
        pass
    
    @abstractmethod
    def get_solver_info(self) -> Dict[str, Any]:
        """
        Return information about solver capabilities and configuration.
        
        This method provides metadata about the solver for:
        - Router decision making (which problems can this solver handle?)
        - User interface (display solver options)
        - Logging and monitoring (track solver usage)
        - Configuration validation (check compatibility)
        
        Required Information:
        ---------------------
        {
            'solver_type': str,          # 'classical', 'quantum', 'hybrid'
            'solver_name': str,          # Specific algorithm name
            'version': str,              # Solver version
            'supported_problems': List[str],  # Problem types it can solve
            'capabilities': {
                'exact': bool,           # Can find optimal solution?
                'approximate': bool,     # Provides approximation guarantees?
                'anytime': bool,        # Can return best-so-far solution?
                'parallel': bool        # Supports parallel execution?
            },
            'parameters': {              # Configurable parameters
                'name': {
                    'type': str,         # Parameter type
                    'default': Any,      # Default value
                    'range': tuple,      # Valid range (if applicable)
                    'description': str   # Human-readable description
                }
            },
            'resource_requirements': {
                'memory_mb': int,        # Approximate memory usage
                'cpu_cores': int,        # CPU cores utilized
                'gpu_required': bool     # Requires GPU?
            }
        }
        
        Returns:
            Dictionary with solver information
        
        Example:
            >>> solver = ClassicalSolver('greedy')
            >>> info = solver.get_solver_info()
            >>> print(info['solver_name'])  # 'greedy'
            >>> print(info['supported_problems'])  # ['maxcut', 'tsp']
        """
        pass
    
    # ========================================================================
    # Concrete Methods - Provided for all solvers
    # ========================================================================
    
    def measure_energy_start(self) -> None:
        """
        Start measuring energy consumption using EnergyTracker.
        
        This method now uses the enhanced EnergyTracker class for more accurate
        and configurable energy estimation. The EnergyTracker provides:
        - Calibration for different CPU types
        - Better documentation of methodology
        - Real-time power monitoring capabilities
        
        See EnergyTracker class documentation for detailed methodology.
        
        Example:
            >>> solver.measure_energy_start()
            >>> # ... computation ...
            >>> energy = solver.measure_energy_end()
            >>> print(f"Energy consumed: {energy:.2f} mJ")
        
        Notes:
            - Uses EnergyTracker for improved accuracy
            - Call this immediately before computation starts
            - Must be paired with measure_energy_end()
            - Not thread-safe (don't call from multiple threads)
        """
        try:
            # Use new EnergyTracker (preferred method)
            self._energy_tracker.start_tracking()
            
            # Also maintain backward compatibility with old method
            cpu_times = self._process.cpu_times()
            self._energy_start = cpu_times.user + cpu_times.system
            
            logger.debug("Energy measurement started (using EnergyTracker)")
        
        except Exception as e:
            logger.warning(f"Failed to start energy measurement: {e}")
            self._energy_start = None
    
    def measure_energy_end(self) -> float:
        """
        End measuring energy consumption and return estimate using EnergyTracker.
        
        This method now uses the enhanced EnergyTracker for improved energy
        estimation with calibration support.
        
        Returns:
            Estimated energy consumption in millijoules (mJ)
            Returns 0.0 if measurement failed or wasn't started
        
        Example:
            >>> solver.measure_energy_start()
            >>> result = solver.solve(problem)
            >>> energy = solver.measure_energy_end()
            >>> result['energy_mj'] = energy
        
        Notes:
            - Uses EnergyTracker for improved accuracy
            - Results are estimates, not exact measurements
            - For comparative analysis, not absolute measurements
            - See EnergyTracker documentation for calibration options
        """
        try:
            # Use new EnergyTracker (preferred method)
            energy_mj = self._energy_tracker.stop_tracking()
            
            # Reset backward compatibility attribute
            self._energy_start = None
            
            logger.debug(f"Energy measurement ended: {energy_mj:.2f} mJ (using EnergyTracker)")
            
            return energy_mj
        
        except Exception as e:
            logger.warning(f"Failed to measure energy: {e}")
            self._energy_start = None
            
            # Fallback to old method if EnergyTracker fails
            return self._measure_energy_fallback()
    
    def _measure_energy_fallback(self) -> float:
        """
        Fallback energy measurement if EnergyTracker fails.
        
        This is the original simple estimation method, kept for backward
        compatibility and as a fallback.
        
        Returns:
            Estimated energy in millijoules
        """
        if self._energy_start is None:
            return 0.0
        
        try:
            cpu_times = self._process.cpu_times()
            energy_end = cpu_times.user + cpu_times.system
            cpu_time_seconds = energy_end - self._energy_start
            
            # Simple estimation (original method)
            tdp_watts = 65.0
            utilization_factor = 0.6
            efficiency = 0.8
            average_power = tdp_watts * utilization_factor * efficiency
            energy_joules = average_power * cpu_time_seconds
            energy_mj = energy_joules * 1000.0
            
            self._energy_start = None
            return energy_mj
        
        except Exception:
            self._energy_start = None
            return 0.0
    
    def validate_result(self, problem: ProblemBase, solution: List[Any]) -> bool:
        """
        Validate that a solution satisfies problem constraints.
        
        This method checks if the solution is valid according to the problem's
        constraints. It uses the problem's validate_solution() method but adds
        additional solver-level checks.
        
        Validation Checks:
        ------------------
        1. Solution is not None or empty
        2. Solution has correct length/format
        3. Problem's validate_solution() returns True
        4. Solution values are in valid range (problem-specific)
        
        Why Validate?
        -------------
        - Solvers can have bugs that produce invalid solutions
        - Heuristics might violate constraints
        - Quantum measurements might give invalid bitstrings
        - Early detection prevents downstream errors
        
        This is a **fail-fast** approach: better to catch errors immediately
        than to propagate invalid solutions through the pipeline.
        
        Args:
            problem: Problem instance the solution should satisfy
            solution: Candidate solution to validate
        
        Returns:
            True if solution is valid, False otherwise
        
        Example:
            >>> solution = solver._run_algorithm(problem)
            >>> if not solver.validate_result(problem, solution):
            ...     raise InvalidSolutionError("Solver produced invalid solution")
        
        Notes:
            - This method should be called in solve() before returning
            - Validation failures should raise InvalidSolutionError
            - For expensive validations, consider making this optional
        """
        # Check 1: Solution exists
        if solution is None:
            logger.error("Solution is None")
            return False
        
        # Check 2: Solution is not empty (unless problem size is 0)
        if len(solution) == 0 and problem.problem_size > 0:
            logger.error("Solution is empty but problem size > 0")
            return False
        
        # Check 3: Solution length matches problem size
        if len(solution) != problem.problem_size:
            logger.error(f"Solution length {len(solution)} != problem size {problem.problem_size}")
            return False
        
        # Check 4: Use problem's validation method
        try:
            is_valid = problem.validate_solution(solution)
            if not is_valid:
                logger.error("Problem's validate_solution() returned False")
            return is_valid
        
        except Exception as e:
            logger.error(f"Validation failed with exception: {e}")
            return False
    
    # ========================================================================
    # Context Manager Support
    # ========================================================================
    
    def __enter__(self):
        """
        Enter context manager.
        
        Enables usage like:
        ```python
        with ClassicalSolver() as solver:
            result = solver.solve(problem)
        # Automatic cleanup on exit
        ```
        
        Returns:
            self for use in 'as' clause
        """
        logger.debug(f"Entering context for {self.solver_name} solver")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit context manager, performing cleanup.
        
        This method is called automatically when exiting the 'with' block.
        It handles cleanup regardless of whether an exception occurred.
        
        Args:
            exc_type: Exception type (None if no exception)
            exc_val: Exception value (None if no exception)
            exc_tb: Exception traceback (None if no exception)
        
        Returns:
            False to propagate exceptions, True to suppress them
        """
        logger.debug(f"Exiting context for {self.solver_name} solver")
        
        # Perform cleanup
        self._cleanup()
        
        # Log exception if one occurred
        if exc_type is not None:
            logger.error(f"Exception in solver context: {exc_type.__name__}: {exc_val}")
        
        # Return False to propagate exceptions (don't suppress them)
        return False
    
    def _cleanup(self) -> None:
        """
        Perform cleanup operations.
        
        This method is called when exiting a context manager. Subclasses
        can override to add specific cleanup logic (e.g., releasing GPU memory,
        closing network connections, saving state).
        
        Default implementation does nothing. Override in subclasses as needed.
        
        Example Override:
        -----------------
        ```python
        def _cleanup(self):
            super()._cleanup()
            if self._quantum_circuit:
                self._quantum_circuit.clear()
            if self._gpu_allocated:
                self._release_gpu_memory()
        ```
        """
        logger.debug(f"Cleanup called for {self.solver_name}")
        # Default: no cleanup needed
        # Subclasses can override to add specific cleanup logic
    
    # ========================================================================
    # String Representations
    # ========================================================================
    
    def __repr__(self) -> str:
        """Developer-friendly string representation."""
        return (f"{self.__class__.__name__}("
                f"solver_type='{self.solver_type}', "
                f"solver_name='{self.solver_name}')")
    
    def __str__(self) -> str:
        """User-friendly string representation."""
        return f"{self.solver_type.title()} Solver: {self.solver_name}"
