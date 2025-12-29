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
from contextlib import contextmanager

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
        
        # Energy monitoring
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
        Start measuring energy consumption.
        
        This method records the initial CPU energy state. It should be called
        at the beginning of the solve() method, before any computation starts.
        
        Energy Measurement Methodology:
        -------------------------------
        We estimate energy consumption using CPU utilization and time:
        
        Energy (J) ≈ Power (W) × Time (s)
        Power (W) ≈ TDP × CPU_Utilization_Fraction
        
        Where:
        - TDP (Thermal Design Power): Maximum CPU power consumption
        - CPU_Utilization: Percentage of CPU used by this process
        - Time: Duration of computation
        
        This is an APPROXIMATION because:
        - Actual power varies with CPU frequency and workload
        - We don't account for memory/disk/network energy
        - Different CPUs have different power profiles
        - CPU governors affect actual power consumption
        
        Why Approximate?
        ----------------
        - Precise energy measurement requires hardware tools (e.g., Intel RAPL)
        - Not all platforms support hardware energy counters
        - Our goal is comparative analysis, not absolute accuracy
        - Relative differences between solvers are still meaningful
        
        For more accurate measurements:
        - Use Intel RAPL (Running Average Power Limit) on Linux
        - Use specialized power meters
        - Run on controlled hardware with power monitoring
        
        Example:
            >>> solver.measure_energy_start()
            >>> # ... computation ...
            >>> energy = solver.measure_energy_end()
            >>> print(f"Energy consumed: {energy:.2f} mJ")
        
        Notes:
            - Call this immediately before computation starts
            - Must be paired with measure_energy_end()
            - Not thread-safe (don't call from multiple threads)
        """
        try:
            # Get current CPU times for this process
            # cpu_times includes: user time, system time, children times
            cpu_times = self._process.cpu_times()
            
            # Store initial CPU time (user + system)
            # User time: time spent in user mode
            # System time: time spent in kernel mode
            self._energy_start = cpu_times.user + cpu_times.system
            
            logger.debug(f"Energy measurement started: CPU time = {self._energy_start:.4f}s")
        
        except Exception as e:
            logger.warning(f"Failed to start energy measurement: {e}")
            self._energy_start = None
    
    def measure_energy_end(self) -> float:
        """
        End measuring energy consumption and return estimate.
        
        This method calculates the energy consumed since measure_energy_start()
        was called. It uses CPU time and estimated TDP to approximate energy.
        
        Returns:
            Estimated energy consumption in millijoules (mJ)
            Returns 0.0 if measurement failed or wasn't started
        
        Calculation Details:
        --------------------
        1. Measure CPU time delta: Δt = current_cpu_time - start_cpu_time
        2. Estimate average CPU power: P ≈ TDP × utilization_factor
        3. Calculate energy: E = P × Δt
        4. Convert to millijoules: mJ = J × 1000
        
        Default Assumptions:
        - TDP: 65W (typical modern CPU)
        - Utilization factor: 0.6 (average during computation)
        - Efficiency: 0.8 (not all power goes to computation)
        
        These can be overridden in subclasses for specific hardware.
        
        Example:
            >>> solver.measure_energy_start()
            >>> result = solver.solve(problem)
            >>> energy = solver.measure_energy_end()
            >>> result['energy_mj'] = energy
        
        Notes:
            - Returns 0.0 if measure_energy_start() wasn't called
            - Results are approximate (see measure_energy_start() for details)
            - For comparative analysis only, not absolute measurements
        """
        if self._energy_start is None:
            logger.warning("Energy measurement not started, returning 0.0")
            return 0.0
        
        try:
            # Get final CPU times
            cpu_times = self._process.cpu_times()
            energy_end = cpu_times.user + cpu_times.system
            
            # Calculate CPU time used (in seconds)
            cpu_time_seconds = energy_end - self._energy_start
            
            # Estimate energy consumption
            # Assumptions:
            # - Average CPU TDP: 65W (adjustable per system)
            # - Utilization factor: 60% (not always at full power)
            # - Efficiency: 80% (not all power is computational)
            tdp_watts = 65.0
            utilization_factor = 0.6
            efficiency = 0.8
            
            # Calculate average power (Watts)
            average_power = tdp_watts * utilization_factor * efficiency
            
            # Energy = Power × Time
            energy_joules = average_power * cpu_time_seconds
            
            # Convert to millijoules
            energy_mj = energy_joules * 1000.0
            
            logger.debug(f"Energy measurement ended: {energy_mj:.2f} mJ "
                        f"(CPU time: {cpu_time_seconds:.4f}s)")
            
            # Reset for next measurement
            self._energy_start = None
            
            return energy_mj
        
        except Exception as e:
            logger.warning(f"Failed to measure energy: {e}")
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
