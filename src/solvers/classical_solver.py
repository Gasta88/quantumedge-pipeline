"""
Classical solver implementation for QuantumEdge Pipeline.

This module implements classical optimization algorithms for various problem types.
It inherits from SolverBase and provides concrete implementations of classical
solving methods, including greedy algorithms, simulated annealing, and exact methods.

Supported Problem Types:
------------------------
1. **MaxCut**: Maximum cut problem on weighted graphs
   - Greedy algorithm: Fast, ~0.5 approximation
   - Simulated annealing: Slower, higher quality solutions
   
2. **TSP**: Traveling Salesman Problem
   - Google OR-Tools: State-of-the-art constraint solver
   - Supports various local search strategies
   
3. **Portfolio Optimization**: Asset selection and allocation
   - scipy.optimize: Convex optimization for efficient frontier
   - Handles constraints on weights and risk

Algorithm Selection Strategy:
-----------------------------
The solver uses an 'auto' mode that intelligently selects the best classical
algorithm based on:
- Problem type (MaxCut, TSP, Portfolio)
- Problem size (small/medium/large)
- Time constraints (if specified)
- Quality requirements

Classical vs Quantum:
---------------------
Classical solvers are often preferred when:
- Problem size is small (n < 20): Exact methods work
- Problem has special structure (bipartite, planar)
- Time constraints are tight (classical is faster to start)
- Solution quality is less critical (approximation is acceptable)

Example Usage:
--------------
```python
from src.problems.maxcut import MaxCutProblem
from src.solvers.classical_solver import ClassicalSolver

# Create and generate problem
problem = MaxCutProblem(num_nodes=30)
problem.generate(edge_probability=0.3)

# Auto-select best method
solver = ClassicalSolver()
result = solver.solve(problem, method='auto')

print(f"Solution cost: {result['cost']:.4f}")
print(f"Time: {result['time_ms']} ms")
print(f"Energy: {result['energy_mj']:.2f} mJ")

# Or specify method explicitly
result = solver.solve(problem, method='simulated_annealing', 
                     max_iterations=10000, temperature=100.0)

# Use context manager for cleanup
with ClassicalSolver() as solver:
    result = solver.solve(problem)
```

Performance Characteristics:
----------------------------
Algorithm                | Time Complexity | Space Complexity | Quality
-------------------------|-----------------|------------------|----------
Greedy MaxCut            | O(m)           | O(n)            | ~0.5 approx
Simulated Annealing      | O(k×n)         | O(n)            | High (tunable)
OR-Tools TSP             | O(n²) to O(n!) | O(n²)           | High
scipy Portfolio          | O(n³)          | O(n²)           | Optimal

Where: n = problem size, m = edges, k = iterations
"""

import time
import logging
import random
from typing import Dict, Any, List, Optional
import numpy as np

# Import for type checking
from src.problems.problem_base import ProblemBase
from src.solvers.solver_base import (
    SolverBase, 
    SolverConfigurationError,
    SolverTimeoutError,
    InvalidSolutionError
)


# Configure module logger
logger = logging.getLogger(__name__)


class ClassicalSolver(SolverBase):
    """
    Classical optimization solver supporting multiple problem types.
    
    This solver implements various classical algorithms for combinatorial
    optimization. It automatically selects the best method based on problem
    characteristics or allows manual method selection.
    
    Supported Methods:
    ------------------
    - 'auto': Automatically select best method
    - 'greedy': Fast greedy algorithm (MaxCut)
    - 'simulated_annealing': Metaheuristic for high-quality solutions
    - 'ortools': Google OR-Tools (TSP)
    - 'scipy': scipy.optimize (Portfolio)
    
    Attributes:
        default_method (str): Default solving method if not specified
        max_time_seconds (float): Maximum execution time (None = unlimited)
    
    Thread Safety:
        Not thread-safe. Create separate instances for parallel execution.
    """
    
    def __init__(
        self, 
        default_method: str = 'auto',
        max_time_seconds: Optional[float] = None
    ):
        """
        Initialize classical solver.
        
        Args:
            default_method: Default method to use ('auto', 'greedy', 'simulated_annealing', etc.)
            max_time_seconds: Maximum execution time in seconds (None = no limit)
        
        Raises:
            SolverConfigurationError: If parameters are invalid
        """
        # Initialize base class
        super().__init__(solver_type='classical', solver_name='classical_multi')
        
        # Validate and store configuration
        valid_methods = ['auto', 'greedy', 'simulated_annealing', 'ortools', 'scipy']
        if default_method not in valid_methods:
            raise SolverConfigurationError(
                f"Invalid method '{default_method}'. Must be one of: {valid_methods}"
            )
        
        self.default_method = default_method
        self.max_time_seconds = max_time_seconds
        
        # Performance tracking
        self._last_iterations = 0
        self._last_method_used = None
        
        logger.info(f"ClassicalSolver initialized with default_method='{default_method}'")
    
    def solve(self, problem: ProblemBase, **kwargs) -> Dict[str, Any]:
        """
        Solve the problem using classical algorithms.
        
        This is the main entry point. It:
        1. Validates the problem
        2. Selects the best method (if auto mode)
        3. Executes the solving algorithm
        4. Measures time and energy
        5. Validates and returns results
        
        Args:
            problem: Problem instance to solve
            **kwargs: Solver parameters:
                     - method: Override default method
                     - max_iterations: Maximum iterations (for iterative methods)
                     - temperature: Initial temperature (for simulated annealing)
                     - cooling_rate: Cooling rate (for simulated annealing)
                     - timeout_seconds: Override max_time_seconds
        
        Returns:
            Standardized result dictionary (see SolverBase.solve docstring)
        
        Raises:
            SolverConfigurationError: If problem is incompatible
            SolverTimeoutError: If execution exceeds timeout
            InvalidSolutionError: If solution is invalid
        
        Example:
            >>> solver = ClassicalSolver()
            >>> result = solver.solve(problem, method='greedy')
            >>> print(f"Cost: {result['cost']}, Time: {result['time_ms']}ms")
        """
        # Validate problem
        if not problem.is_generated:
            raise SolverConfigurationError("Problem must be generated before solving")
        
        logger.info(f"Solving {problem.problem_type} problem (size={problem.problem_size})")
        
        # Get method to use
        method = kwargs.get('method', self.default_method)
        
        # Auto-select method if needed
        if method == 'auto':
            method = self._auto_select_method(problem)
            logger.info(f"Auto-selected method: {method}")
        
        self._last_method_used = method
        
        # Start timing and energy measurement
        start_time = time.perf_counter()
        self.measure_energy_start()
        
        # Get timeout
        timeout = kwargs.get('timeout_seconds', self.max_time_seconds)
        
        try:
            # Route to appropriate solver based on problem type and method
            problem_type = problem.problem_type.lower()
            
            if problem_type == 'maxcut':
                if method == 'greedy':
                    solution = self._solve_maxcut_greedy(problem)
                elif method == 'simulated_annealing':
                    solution = self._solve_maxcut_simulated_annealing(problem, **kwargs)
                else:
                    raise SolverConfigurationError(
                        f"Method '{method}' not supported for MaxCut. Use 'greedy' or 'simulated_annealing'."
                    )
            
            elif problem_type == 'tsp':
                if method in ['ortools', 'auto']:
                    solution = self._solve_tsp_ortools(problem, **kwargs)
                elif method == 'simulated_annealing':
                    solution = self._solve_tsp_simulated_annealing(problem, **kwargs)
                else:
                    raise SolverConfigurationError(
                        f"Method '{method}' not supported for TSP. Use 'ortools' or 'simulated_annealing'."
                    )
            
            elif problem_type == 'portfolio':
                if method in ['scipy', 'auto']:
                    solution = self._solve_portfolio_scipy(problem, **kwargs)
                else:
                    raise SolverConfigurationError(
                        f"Method '{method}' not supported for Portfolio. Use 'scipy'."
                    )
            
            else:
                # Fallback to simulated annealing for unknown problem types
                logger.warning(f"Unknown problem type '{problem_type}', using simulated annealing")
                solution = self._solve_generic_simulated_annealing(problem, **kwargs)
            
            # Check timeout
            elapsed = time.perf_counter() - start_time
            if timeout and elapsed > timeout:
                raise SolverTimeoutError(
                    f"Solver exceeded timeout of {timeout}s (elapsed: {elapsed:.2f}s)"
                )
            
            # End measurements
            end_time = time.perf_counter()
            energy_mj = self.measure_energy_end()
            
            # Validate solution
            if not self.validate_result(problem, solution):
                raise InvalidSolutionError(f"Solver produced invalid solution using method '{method}'")
            
            # Calculate cost
            cost = problem.calculate_cost(solution)
            
            # Build result
            result = {
                'solution': solution,
                'cost': cost,
                'time_ms': int((end_time - start_time) * 1000),
                'energy_mj': energy_mj,
                'iterations': self._last_iterations,
                'metadata': {
                    'solver_type': self.solver_type,
                    'solver_name': self.solver_name,
                    'method': method,
                    'problem_type': problem_type,
                    'problem_size': problem.problem_size
                }
            }
            
            logger.info(f"Solve complete: cost={cost:.4f}, time={result['time_ms']}ms, "
                       f"iterations={self._last_iterations}")
            
            return result
        
        except Exception as e:
            # Ensure energy measurement is ended
            self.measure_energy_end()
            
            # Re-raise if it's one of our exceptions
            if isinstance(e, (SolverConfigurationError, SolverTimeoutError, InvalidSolutionError)):
                raise
            
            # Wrap other exceptions
            logger.error(f"Solver failed with exception: {e}")
            raise SolverConfigurationError(f"Solver failed: {e}") from e
    
    def get_solver_info(self) -> Dict[str, Any]:
        """
        Return information about classical solver capabilities.
        
        Returns:
            Dictionary with solver information
        """
        return {
            'solver_type': 'classical',
            'solver_name': 'classical_multi',
            'version': '1.0.0',
            'supported_problems': ['maxcut', 'tsp', 'portfolio'],
            'capabilities': {
                'exact': False,          # Provides approximate solutions
                'approximate': True,     # Approximation guarantees for some methods
                'anytime': True,         # Can return best-so-far (simulated annealing)
                'parallel': False        # Single-threaded execution
            },
            'methods': {
                'greedy': {
                    'description': 'Fast greedy algorithm',
                    'problems': ['maxcut'],
                    'complexity': 'O(m)',
                    'quality': 'Low (~0.5 approximation for MaxCut)'
                },
                'simulated_annealing': {
                    'description': 'Metaheuristic optimization',
                    'problems': ['maxcut', 'tsp', 'portfolio'],
                    'complexity': 'O(k×n)',
                    'quality': 'High (tunable with iterations)'
                },
                'ortools': {
                    'description': 'Google OR-Tools constraint solver',
                    'problems': ['tsp'],
                    'complexity': 'O(n²) typical',
                    'quality': 'High'
                },
                'scipy': {
                    'description': 'scipy.optimize for continuous optimization',
                    'problems': ['portfolio'],
                    'complexity': 'O(n³)',
                    'quality': 'Optimal (for convex problems)'
                }
            },
            'parameters': {
                'method': {
                    'type': 'str',
                    'default': 'auto',
                    'options': ['auto', 'greedy', 'simulated_annealing', 'ortools', 'scipy'],
                    'description': 'Solving method to use'
                },
                'max_iterations': {
                    'type': 'int',
                    'default': 10000,
                    'range': (100, 100000),
                    'description': 'Maximum iterations for iterative methods'
                },
                'temperature': {
                    'type': 'float',
                    'default': 100.0,
                    'range': (1.0, 1000.0),
                    'description': 'Initial temperature for simulated annealing'
                },
                'cooling_rate': {
                    'type': 'float',
                    'default': 0.95,
                    'range': (0.8, 0.999),
                    'description': 'Cooling rate for simulated annealing'
                }
            },
            'resource_requirements': {
                'memory_mb': 100,        # Approximate memory usage
                'cpu_cores': 1,          # Single-threaded
                'gpu_required': False    # No GPU needed
            }
        }
    
    # ========================================================================
    # Private Helper Methods
    # ========================================================================
    
    def _auto_select_method(self, problem: ProblemBase) -> str:
        """
        Automatically select the best classical method for the problem.
        
        Selection criteria:
        - Problem type (different algorithms for different problems)
        - Problem size (exact methods for small, heuristics for large)
        - Problem structure (exploit special cases if detected)
        
        Args:
            problem: Problem to select method for
        
        Returns:
            Method name to use
        """
        problem_type = problem.problem_type.lower()
        size = problem.problem_size
        
        logger.debug(f"Auto-selecting method for {problem_type} (size={size})")
        
        if problem_type == 'maxcut':
            # For MaxCut:
            # - Small problems (n<20): Can use simulated annealing for quality
            # - Medium (20-50): Greedy is fast enough
            # - Large (>50): Greedy to avoid long runtime
            if size < 20:
                return 'simulated_annealing'  # Worth the extra time for quality
            else:
                return 'greedy'  # Fast for larger instances
        
        elif problem_type == 'tsp':
            # For TSP: OR-Tools is generally best
            # It has sophisticated local search and is well-optimized
            return 'ortools'
        
        elif problem_type == 'portfolio':
            # For Portfolio: scipy is the standard choice
            # Efficient convex optimization
            return 'scipy'
        
        else:
            # Unknown problem: use simulated annealing as general-purpose fallback
            logger.warning(f"Unknown problem type '{problem_type}', defaulting to simulated_annealing")
            return 'simulated_annealing'
    
    # ========================================================================
    # MaxCut Solvers
    # ========================================================================
    
    def _solve_maxcut_greedy(self, problem: ProblemBase) -> List[int]:
        """
        Solve MaxCut using a greedy algorithm.
        
        Algorithm Description:
        ----------------------
        The greedy algorithm iteratively assigns nodes to partitions to maximize
        the cut value at each step. It's fast but provides only approximate solutions.
        
        Procedure:
        1. Start with empty partitions A and B
        2. For each unassigned node v:
           - Calculate cut increase if v goes to A
           - Calculate cut increase if v goes to B
           - Assign v to the partition that increases cut the most
        3. Return the partition assignment
        
        Time Complexity: O(n × m) where n=nodes, m=edges
        - In practice, closer to O(m) if we're smart about bookkeeping
        
        Space Complexity: O(n)
        - Store node assignments
        
        Approximation Guarantee:
        - No worst-case guarantee (can be arbitrarily bad)
        - In practice: Usually finds 40-60% of optimal cut
        - For random graphs: Often near optimal
        
        Advantages:
        + Very fast (linear in edges)
        + Simple to implement
        + Works well on sparse graphs
        + No parameters to tune
        
        Disadvantages:
        - No approximation guarantee
        - Greedy choices can lead to local optima
        - Quality depends on node ordering
        
        Args:
            problem: MaxCut problem instance
        
        Returns:
            Binary assignment list [0, 1, 0, 1, ...] indicating partition membership
        """
        logger.debug("Solving MaxCut with greedy algorithm")
        
        # Get graph representation
        graph = problem.to_graph()
        n = graph.number_of_nodes()
        
        # Initialize solution: all nodes unassigned (-1), will be 0 or 1
        solution = [-1] * n
        
        # Track cut value contribution of each node to each partition
        # contribution[v][0] = cut increase if v goes to partition 0
        # contribution[v][1] = cut increase if v goes to partition 1
        contribution = {v: [0, 0] for v in range(n)}
        
        # Initial calculation: first node goes to partition 0 arbitrarily
        solution[0] = 0
        
        # Update contributions for neighbors of node 0
        for neighbor in graph.neighbors(0):
            weight = graph[0][neighbor].get('weight', 1.0)
            # Neighbor going to opposite partition increases cut
            contribution[neighbor][1] += weight  # If neighbor goes to 1, cut increases
            contribution[neighbor][0] -= weight  # If neighbor goes to 0, cut decreases
        
        # Greedily assign remaining nodes
        for _ in range(1, n):
            # Find best unassigned node and partition
            best_node = -1
            best_partition = -1
            best_increase = float('-inf')
            
            for v in range(n):
                if solution[v] != -1:
                    continue  # Already assigned
                
                # Check both partitions
                for partition in [0, 1]:
                    if contribution[v][partition] > best_increase:
                        best_increase = contribution[v][partition]
                        best_node = v
                        best_partition = partition
            
            # Assign best node to best partition
            solution[best_node] = best_partition
            
            # Update contributions for neighbors
            for neighbor in graph.neighbors(best_node):
                if solution[neighbor] != -1:
                    continue  # Already assigned
                
                weight = graph[best_node][neighbor].get('weight', 1.0)
                
                # Update contributions based on which partition best_node went to
                if best_partition == 0:
                    contribution[neighbor][1] += weight  # Going to 1 increases cut
                    contribution[neighbor][0] -= weight  # Going to 0 decreases cut
                else:
                    contribution[neighbor][0] += weight  # Going to 0 increases cut
                    contribution[neighbor][1] -= weight  # Going to 1 decreases cut
        
        # Track iterations (for greedy, it's just n)
        self._last_iterations = n
        
        logger.debug(f"Greedy completed in {n} iterations")
        
        return solution
    
    def _solve_maxcut_simulated_annealing(
        self, 
        problem: ProblemBase,
        max_iterations: int = 10000,
        temperature: float = 100.0,
        cooling_rate: float = 0.95,
        **kwargs
    ) -> List[int]:
        """
        Solve MaxCut using simulated annealing metaheuristic.
        
        Algorithm Description:
        ----------------------
        Simulated annealing is a probabilistic optimization technique inspired by
        metallurgical annealing. It allows occasional "uphill" moves to escape
        local optima, controlled by a temperature parameter that decreases over time.
        
        Procedure:
        1. Start with random solution
        2. Repeat until temperature is low:
           a. Generate neighbor by flipping random node's partition
           b. Calculate change in objective (Δ)
           c. If Δ improves solution: Accept
           d. If Δ worsens solution: Accept with probability exp(-Δ/T)
           e. Decrease temperature: T ← T × cooling_rate
        3. Return best solution found
        
        Time Complexity: O(k × m) where k=iterations, m=edges
        - Each iteration evaluates change in cut (O(degree) per node)
        - Typically k = 10,000 to 100,000
        
        Space Complexity: O(n)
        - Store current and best solutions
        
        Parameters:
        -----------
        max_iterations: int (default=10000)
            Number of iterations to run
            - More iterations → better quality but slower
            - Typical: 10,000 for n<50, 50,000 for n>50
        
        temperature: float (default=100.0)
            Initial temperature
            - Higher → more exploration early on
            - Should be set based on typical Δ values
            - Rule of thumb: temperature ≈ max_cost / 10
        
        cooling_rate: float (default=0.95)
            Temperature reduction factor per iteration
            - T_new = T_old × cooling_rate
            - Range: 0.8-0.999
            - Slower cooling (closer to 1.0) → better quality but slower
        
        Quality:
        --------
        - Typically finds solutions within 5-10% of optimal
        - Quality improves with more iterations
        - Can escape local optima (unlike greedy)
        - Results vary due to randomness (use seed for reproducibility)
        
        Advantages:
        + Often finds high-quality solutions
        + Simple to implement
        + Works for any problem (very general)
        + Can escape local optima
        
        Disadvantages:
        - Slower than greedy
        - Requires parameter tuning
        - No approximation guarantee
        - Results are non-deterministic
        
        Args:
            problem: MaxCut problem instance
            max_iterations: Maximum number of iterations
            temperature: Initial temperature
            cooling_rate: Temperature reduction factor (0.8-0.999)
        
        Returns:
            Binary assignment list indicating partition membership
        """
        logger.debug(f"Solving MaxCut with simulated annealing "
                    f"(iterations={max_iterations}, T={temperature}, cooling={cooling_rate})")
        
        # Get graph representation
        graph = problem.to_graph()
        n = graph.number_of_nodes()
        
        # Initialize with random solution
        current_solution = [random.randint(0, 1) for _ in range(n)]
        current_cost = problem.calculate_cost(current_solution)
        
        # Track best solution found
        best_solution = current_solution.copy()
        best_cost = current_cost
        
        # Simulated annealing loop
        T = temperature
        iterations = 0
        
        for iteration in range(max_iterations):
            iterations += 1
            
            # Generate neighbor: flip one random node
            neighbor = current_solution.copy()
            flip_node = random.randint(0, n - 1)
            neighbor[flip_node] = 1 - neighbor[flip_node]  # Flip 0→1 or 1→0
            
            # Calculate cost change (Δ)
            # For efficiency, only recalculate affected edges
            delta = 0
            for other in graph.neighbors(flip_node):
                weight = graph[flip_node][other].get('weight', 1.0)
                
                # Edge contributes to cut if endpoints in different partitions
                old_in_cut = (current_solution[flip_node] != current_solution[other])
                new_in_cut = (neighbor[flip_node] != neighbor[other])
                
                # MaxCut maximizes, but cost is negative, so delta sign is flipped
                if new_in_cut and not old_in_cut:
                    delta -= weight  # Cut increases, cost decreases (better)
                elif not new_in_cut and old_in_cut:
                    delta += weight  # Cut decreases, cost increases (worse)
            
            # Accept or reject move
            accept = False
            
            if delta <= 0:
                # Move improves or maintains cost (lower is better)
                accept = True
            else:
                # Move worsens cost, accept with probability exp(-Δ/T)
                # Higher temperature → more likely to accept bad moves
                acceptance_prob = np.exp(-delta / T)
                if random.random() < acceptance_prob:
                    accept = True
            
            if accept:
                current_solution = neighbor
                current_cost += delta
                
                # Update best if improved
                if current_cost < best_cost:
                    best_solution = current_solution.copy()
                    best_cost = current_cost
                    logger.debug(f"Iteration {iteration}: New best cost = {best_cost:.4f}")
            
            # Cool down temperature
            T *= cooling_rate
            
            # Stop if temperature is too low (frozen state)
            if T < 0.01:
                logger.debug(f"Temperature frozen at iteration {iteration}")
                break
        
        self._last_iterations = iterations
        
        logger.debug(f"Simulated annealing completed: {iterations} iterations, best_cost={best_cost:.4f}")
        
        return best_solution
    
    # ========================================================================
    # TSP Solvers
    # ========================================================================
    
    def _solve_tsp_ortools(
        self, 
        problem: ProblemBase,
        time_limit_seconds: int = 30,
        **kwargs
    ) -> List[int]:
        """
        Solve TSP using Google OR-Tools constraint solver.
        
        Algorithm Description:
        ----------------------
        OR-Tools is a sophisticated optimization toolkit from Google that includes
        state-of-the-art algorithms for TSP. It uses:
        - Constraint propagation
        - Local search metaheuristics (guided local search, tabu search)
        - Large neighborhood search
        
        The solver explores the search space intelligently, using domain-specific
        heuristics for TSP. It typically finds high-quality solutions quickly and
        continues improving if given more time.
        
        Time Complexity: Variable
        - Best case: O(n²) for easy instances
        - Worst case: O(n!) for hard instances (rarely reached due to time limit)
        - Typical: Finds good solution in O(n² log n) time
        
        Space Complexity: O(n²)
        - Distance matrix storage
        
        Parameters:
        -----------
        time_limit_seconds: int (default=30)
            Maximum time to search for solution
            - More time → potentially better solution
            - Returns best found within time limit
            - Typical: 5-60 seconds depending on problem size
        
        Quality:
        --------
        - Often finds optimal or near-optimal solutions
        - For n<100: Usually within 1-5% of optimal
        - For n>100: Quality depends on time limit
        
        Advantages:
        + State-of-the-art TSP solver
        + Finds high-quality solutions reliably
        + Anytime algorithm (improves with more time)
        + Well-optimized C++ implementation
        
        Disadvantages:
        - External dependency (requires OR-Tools installation)
        - Can be slow for very large instances (n>1000)
        - Less control over algorithm details
        
        Note:
        -----
        This is a placeholder implementation. To use OR-Tools:
        1. Install: pip install ortools
        2. Import: from ortools.constraint_solver import routing_enums_pb2, pywrapcp
        3. Implement routing model as shown in OR-Tools TSP examples
        
        For now, falls back to simulated annealing if OR-Tools not available.
        
        Args:
            problem: TSP problem instance
            time_limit_seconds: Time limit for search
        
        Returns:
            Tour as list of city indices [0, 3, 1, 2, ...]
        """
        logger.debug(f"Solving TSP with OR-Tools (time_limit={time_limit_seconds}s)")
        
        try:
            # Try to import OR-Tools
            from ortools.constraint_solver import routing_enums_pb2, pywrapcp
            
            # Get distance matrix from problem
            # TSP problems should provide distance matrix via to_graph() or custom method
            graph = problem.to_graph()
            n = graph.number_of_nodes()
            
            # Build distance matrix
            distance_matrix = []
            for i in range(n):
                row = []
                for j in range(n):
                    if i == j:
                        row.append(0)
                    elif graph.has_edge(i, j):
                        row.append(int(graph[i][j].get('weight', 1.0) * 1000))  # Scale for integer distances
                    else:
                        row.append(int(1e9))  # Large value for non-existent edges
                distance_matrix.append(row)
            
            # Create routing model
            manager = pywrapcp.RoutingIndexManager(n, 1, 0)  # n nodes, 1 vehicle, depot=0
            routing = pywrapcp.RoutingModel(manager)
            
            # Create distance callback
            def distance_callback(from_index, to_index):
                from_node = manager.IndexToNode(from_index)
                to_node = manager.IndexToNode(to_index)
                return distance_matrix[from_node][to_node]
            
            transit_callback_index = routing.RegisterTransitCallback(distance_callback)
            routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
            
            # Set search parameters
            search_parameters = pywrapcp.DefaultRoutingSearchParameters()
            search_parameters.first_solution_strategy = (
                routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
            )
            search_parameters.local_search_metaheuristic = (
                routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
            )
            search_parameters.time_limit.seconds = time_limit_seconds
            
            # Solve
            solution = routing.SolveWithParameters(search_parameters)
            
            if solution:
                # Extract tour
                tour = []
                index = routing.Start(0)
                while not routing.IsEnd(index):
                    tour.append(manager.IndexToNode(index))
                    index = solution.Value(routing.NextVar(index))
                
                self._last_iterations = 1  # OR-Tools doesn't expose iteration count
                
                logger.debug(f"OR-Tools found tour of length {len(tour)}")
                return tour
            else:
                logger.warning("OR-Tools failed to find solution, falling back to simulated annealing")
                return self._solve_tsp_simulated_annealing(problem, **kwargs)
        
        except ImportError:
            logger.warning("OR-Tools not installed, falling back to simulated annealing")
            return self._solve_tsp_simulated_annealing(problem, **kwargs)
        
        except Exception as e:
            logger.error(f"OR-Tools solver failed: {e}, falling back to simulated annealing")
            return self._solve_tsp_simulated_annealing(problem, **kwargs)
    
    def _solve_tsp_simulated_annealing(
        self,
        problem: ProblemBase,
        max_iterations: int = 10000,
        temperature: float = 100.0,
        cooling_rate: float = 0.95,
        **kwargs
    ) -> List[int]:
        """
        Solve TSP using simulated annealing (fallback method).
        
        Uses 2-opt neighborhood for local moves: reverse a segment of the tour.
        This is a standard and effective neighborhood for TSP.
        
        Args:
            problem: TSP problem instance
            max_iterations: Maximum iterations
            temperature: Initial temperature
            cooling_rate: Cooling rate
        
        Returns:
            Tour as list of city indices
        """
        logger.debug(f"Solving TSP with simulated annealing fallback")
        
        # Get problem size
        n = problem.problem_size
        
        # Initialize with random tour
        current_tour = list(range(n))
        random.shuffle(current_tour)
        current_cost = problem.calculate_cost(current_tour)
        
        best_tour = current_tour.copy()
        best_cost = current_cost
        
        T = temperature
        iterations = 0
        
        for iteration in range(max_iterations):
            iterations += 1
            
            # 2-opt move: reverse segment [i, j]
            i = random.randint(0, n - 2)
            j = random.randint(i + 1, n - 1)
            
            # Create neighbor by reversing segment
            neighbor_tour = current_tour.copy()
            neighbor_tour[i:j+1] = reversed(neighbor_tour[i:j+1])
            
            # Calculate cost
            neighbor_cost = problem.calculate_cost(neighbor_tour)
            delta = neighbor_cost - current_cost
            
            # Accept or reject
            if delta <= 0 or random.random() < np.exp(-delta / T):
                current_tour = neighbor_tour
                current_cost = neighbor_cost
                
                if current_cost < best_cost:
                    best_tour = current_tour.copy()
                    best_cost = current_cost
            
            T *= cooling_rate
            if T < 0.01:
                break
        
        self._last_iterations = iterations
        logger.debug(f"TSP simulated annealing completed: {iterations} iterations")
        
        return best_tour
    
    # ========================================================================
    # Portfolio Solver
    # ========================================================================
    
    def _solve_portfolio_scipy(
        self,
        problem: ProblemBase,
        solve_method: str = 'sharpe',
        target_return: Optional[float] = None,
        **kwargs
    ) -> List[float]:
        """
        Solve portfolio optimization using scipy.optimize.
        
        Algorithm Description:
        ----------------------
        Portfolio optimization typically involves finding the efficient frontier:
        the set of portfolios that maximize return for a given risk level, or
        minimize risk for a given return level.
        
        This implementation uses scipy's optimization routines to solve:
        
        Method 1 - Maximize Sharpe Ratio (default):
            Maximize: (w^T μ - r_f) / √(w^T Σ w)
            Subject to:
            - Σ w_i = 1 (weights sum to 100%)
            - w_i ≥ 0 (no short selling)
        
        Method 2 - Minimum Variance:
            Minimize: w^T Σ w (portfolio variance/risk)
            Subject to:
            - Σ w_i = 1 (weights sum to 100%)
            - w^T μ ≥ target_return (optional minimum return)
            - w_i ≥ 0 (no short selling)
        
        Where:
        - w: portfolio weights (decision variables)
        - Σ: covariance matrix (risk)
        - μ: expected returns vector
        - r_f: risk-free rate
        
        Time Complexity: O(n³)
        - Dominated by quadratic programming solver
        - Interior point methods: O(n³) per iteration
        - Typically converges in 10-50 iterations
        
        Space Complexity: O(n²)
        - Store covariance matrix
        
        Quality:
        --------
        - Finds optimal solution (for convex formulation)
        - Numerically stable
        - Fast even for large portfolios (n=1000+)
        
        Advantages:
        + Optimal solutions for convex problems
        + Fast and reliable
        + Well-tested implementation
        + Handles constraints naturally
        
        Disadvantages:
        - Assumes quadratic objective (mean-variance framework)
        - May not capture all real-world constraints
        - Requires good estimates of returns and covariances
        
        Args:
            problem: Portfolio optimization problem (PortfolioProblem instance)
            solve_method: Optimization method ('sharpe' or 'min_variance')
            target_return: Minimum required return (for min_variance method)
            **kwargs: Additional optimizer parameters
        
        Returns:
            Binary selection [0, 1, ...] if problem has num_selected (cardinality constraint)
            Otherwise, weight vector [w1, w2, ..., wn] summing to 1.0
            
        Raises:
            ImportError: If scipy not available
            ValueError: If problem is not PortfolioProblem
        """
        logger.debug(f"Solving portfolio optimization with scipy (method={solve_method})")
        
        try:
            from scipy.optimize import minimize
            
            # Extract portfolio problem data
            if not hasattr(problem, 'expected_returns') or not hasattr(problem, 'covariance_matrix'):
                raise ValueError(
                    "Problem must be PortfolioProblem with expected_returns and covariance_matrix"
                )
            
            n = problem.problem_size
            expected_returns = problem.expected_returns
            covariance_matrix = problem.covariance_matrix
            risk_free_rate = getattr(problem, 'risk_free_rate', 0.02)
            
            # Initial guess: equal weights
            initial_weights = np.ones(n) / n
            
            # Constraints
            constraints = []
            
            # Constraint 1: Weights sum to 1
            constraints.append({
                'type': 'eq',
                'fun': lambda w: np.sum(w) - 1.0
            })
            
            # Constraint 2: Minimum return (if specified and using min_variance)
            if solve_method == 'min_variance' and target_return is not None:
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda w: np.dot(w, expected_returns) - target_return
                })
            
            # Bounds: Each weight between 0 and 1 (no short selling, no leverage)
            bounds = tuple((0, 1) for _ in range(n))
            
            # Define objective functions
            if solve_method == 'sharpe':
                # Maximize Sharpe ratio = minimize negative Sharpe
                def objective(w):
                    portfolio_return = np.dot(w, expected_returns)
                    portfolio_volatility = np.sqrt(np.dot(w, np.dot(covariance_matrix, w)))
                    
                    # Avoid division by zero
                    if portfolio_volatility < 1e-10:
                        return 1e10  # Large penalty
                    
                    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
                    return -sharpe_ratio  # Minimize negative Sharpe = maximize Sharpe
                
            elif solve_method == 'min_variance':
                # Minimize portfolio variance
                def objective(w):
                    return np.dot(w, np.dot(covariance_matrix, w))
            
            else:
                raise ValueError(f"Unknown method: {solve_method}. Use 'sharpe' or 'min_variance'")
            
            # Optimize
            result = minimize(
                objective,
                initial_weights,
                method='SLSQP',  # Sequential Least Squares Programming
                bounds=bounds,
                constraints=constraints,
                options={
                    'ftol': 1e-9,
                    'maxiter': 1000
                }
            )
            
            # Check if optimization succeeded
            if not result.success:
                logger.warning(
                    f"Optimization did not converge: {result.message}. "
                    f"Using best result found."
                )
            
            # Extract optimal weights
            optimal_weights = result.x
            
            # Ensure weights sum to 1 (numerical precision)
            optimal_weights = optimal_weights / np.sum(optimal_weights)
            
            # Track iterations
            self._last_iterations = result.nit if hasattr(result, 'nit') else 1
            
            # Log results
            portfolio_return = np.dot(optimal_weights, expected_returns)
            portfolio_volatility = np.sqrt(np.dot(optimal_weights, np.dot(covariance_matrix, optimal_weights)))
            sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0
            
            logger.debug(
                f"Portfolio optimization complete: "
                f"return={portfolio_return:.4f}, "
                f"volatility={portfolio_volatility:.4f}, "
                f"sharpe={sharpe_ratio:.4f}, "
                f"iterations={self._last_iterations}"
            )

            # Convert to binary selection if problem requires cardinality constraint
            if hasattr(problem, 'num_selected'):
                k = problem.num_selected
                
                # Select top k assets by weight
                top_k_indices = np.argsort(optimal_weights)[-k:]
                
                # Create binary solution
                binary_solution = [0] * n
                for idx in top_k_indices:
                    binary_solution[idx] = 1
                
                logger.debug(
                    f"Converted continuous weights to binary selection: "
                    f"selected assets {sorted(top_k_indices.tolist())}"
                )
                
                return binary_solution
            
            return optimal_weights.tolist()
        
        except ImportError:
            logger.error("scipy not installed")
            # Return equal weights as fallback
            n = problem.problem_size
            weights = [1.0 / n] * n
            self._last_iterations = 0
            return weights
        
        except Exception as e:
            logger.error(f"scipy solver failed: {e}")
            # Return equal weights as fallback
            n = problem.problem_size
            weights = [1.0 / n] * n
            self._last_iterations = 0
            return weights
    
    def _solve_generic_simulated_annealing(
        self,
        problem: ProblemBase,
        **kwargs
    ) -> List[Any]:
        """
        Generic simulated annealing for unknown problem types.
        
        Uses random bit flips for binary problems as neighborhood function.
        This is a general-purpose fallback.
        
        Args:
            problem: Any problem instance
        
        Returns:
            Solution in problem-specific format
        """
        logger.warning("Using generic simulated annealing for unknown problem type")
        
        # Assume binary problem
        n = problem.problem_size
        
        # Use simulated annealing with bit flips
        current = [random.randint(0, 1) for _ in range(n)]
        current_cost = problem.calculate_cost(current)
        
        best = current.copy()
        best_cost = current_cost
        
        T = kwargs.get('temperature', 100.0)
        cooling = kwargs.get('cooling_rate', 0.95)
        max_iter = kwargs.get('max_iterations', 10000)
        
        for iteration in range(max_iter):
            # Flip random bit
            neighbor = current.copy()
            flip_idx = random.randint(0, n - 1)
            neighbor[flip_idx] = 1 - neighbor[flip_idx]
            
            neighbor_cost = problem.calculate_cost(neighbor)
            delta = neighbor_cost - current_cost
            
            if delta <= 0 or random.random() < np.exp(-delta / T):
                current = neighbor
                current_cost = neighbor_cost
                
                if current_cost < best_cost:
                    best = current.copy()
                    best_cost = current_cost
            
            T *= cooling
            if T < 0.01:
                break
        
        self._last_iterations = iteration + 1
        
        return best
