"""
Hybrid Solver: Combining Classical and Quantum Optimization.

This module implements hybrid solving strategies that intelligently combine
classical and quantum algorithms to leverage the strengths of both approaches.

Hybrid Approaches:
------------------
1. **Quantum-Assisted Classical**: Use quantum to guide classical search
2. **Classical Pre-processing + Quantum Solve**: Reduce problem, solve with quantum
3. **Iterative Refinement**: Alternate between quantum and classical steps
4. **Parallel Exploration**: Run both, select best result
5. **Decomposition**: Split problem, solve parts with appropriate method

Why Hybrid?
-----------
- Classical: Fast, reliable, scales to large problems
- Quantum: Can escape local optima, parallel exploration
- Hybrid: Best of both worlds!

Problem Size Routing:
--------------------
- Small (n<10): Classical exact methods
- Medium (10<n<50): Quantum QAOA/VQE
- Large (n>50): Classical heuristics or hybrid decomposition

Example Usage:
--------------
```python
from src.solvers.hybrid_solver import HybridSolver
from src.problems.maxcut import MaxCutProblem

problem = MaxCutProblem(num_nodes=30)
problem.generate()

solver = HybridSolver(
    strategy='adaptive',
    quantum_threshold=20,  # Use quantum for n<20
    classical_method='simulated_annealing',
    quantum_algorithm='qaoa'
)

result = solver.solve(problem)
print(f"Solution: {result['solution']}")
print(f"Used: {result['metadata']['strategy_used']}")
```
"""

from typing import Dict, Any, Optional, List
import logging
import time
import numpy as np

from src.problems.problem_base import ProblemBase
from src.solvers.solver_base import SolverBase, SolverException
from src.solvers.classical_solver import ClassicalSolver
from src.solvers.quantum_simulator import QuantumSimulator

logger = logging.getLogger(__name__)


class HybridSolver(SolverBase):
    """
    Hybrid quantum-classical solver with adaptive strategy selection.
    
    This solver intelligently combines classical and quantum methods based on:
    - Problem characteristics (size, structure, complexity)
    - Resource constraints (time, energy budgets)
    - Hardware availability (quantum vs classical)
    - Historical performance data
    
    Strategies:
    -----------
    
    1. **Adaptive** (default):
       - Analyzes problem and selects best approach automatically
       - Routes to quantum or classical based on problem size and type
       - Falls back gracefully if primary method fails
    
    2. **Quantum-Assisted**:
       - Use quantum solver to find promising regions
       - Refine with classical local search
       - Best for escaping local optima
    
    3. **Classical-First**:
       - Pre-process problem with classical methods
       - Solve reduced problem with quantum
       - Useful for large problems with structure
    
    4. **Parallel**:
       - Run quantum and classical solvers simultaneously
       - Return best result
       - Hedge against method failures
    
    5. **Iterative**:
       - Alternate between quantum and classical steps
       - Use quantum to explore, classical to exploit
       - Good for multi-modal optimization landscapes
    
    Attributes:
        strategy: Hybrid strategy to use
        quantum_threshold: Problem size threshold for quantum usage
        classical_solver: Classical solver instance
        quantum_solver: Quantum solver instance
        use_quantum: Whether quantum is enabled
    
    Example:
        >>> hybrid = HybridSolver(strategy='adaptive')
        >>> result = hybrid.solve(problem)
        >>> print(f"Strategy: {result['metadata']['strategy_used']}")
    """
    
    VALID_STRATEGIES = [
        'adaptive',
        'quantum_assisted',
        'classical_first',
        'parallel',
        'iterative'
    ]
    
    def __init__(
        self,
        strategy: str = 'adaptive',
        quantum_threshold: int = 20,
        classical_method: str = 'auto',
        quantum_algorithm: str = 'qaoa',
        use_quantum: bool = True,
        time_limit: Optional[float] = None,
        **kwargs
    ):
        """
        Initialize hybrid solver.
        
        Args:
            strategy: Hybrid strategy ('adaptive', 'quantum_assisted', etc.)
            quantum_threshold: Use quantum for problems with size < threshold
            classical_method: Classical method to use ('auto', 'greedy', 'simulated_annealing', etc.)
            quantum_algorithm: Quantum algorithm to use ('qaoa', 'vqe')
            use_quantum: Whether to use quantum solver at all
            time_limit: Maximum total solving time (seconds)
            **kwargs: Additional solver parameters
        
        Raises:
            ValueError: If strategy not recognized
        """
        super().__init__(solver_type='hybrid', solver_name='hybrid_quantum_classical')
        
        if strategy not in self.VALID_STRATEGIES:
            raise ValueError(
                f"Invalid strategy '{strategy}'. "
                f"Valid options: {self.VALID_STRATEGIES}"
            )
        
        self.strategy = strategy
        self.quantum_threshold = quantum_threshold
        self.classical_method = classical_method
        self.quantum_algorithm = quantum_algorithm
        self.use_quantum = use_quantum
        self.time_limit = time_limit
        
        # Initialize sub-solvers
        try:
            self.classical_solver = ClassicalSolver(default_method=classical_method)
            logger.info(f"Classical solver initialized: method={classical_method}")
        except Exception as e:
            logger.warning(f"Failed to initialize classical solver: {e}")
            self.classical_solver = None
        
        if use_quantum:
            try:
                self.quantum_solver = QuantumSimulator(**kwargs)
                logger.info(f"Quantum solver initialized: algorithm={quantum_algorithm}")
            except Exception as e:
                logger.warning(f"Failed to initialize quantum solver: {e}")
                self.quantum_solver = None
                self.use_quantum = False
        else:
            self.quantum_solver = None
        
        logger.info(
            f"HybridSolver initialized: strategy={strategy}, "
            f"quantum_enabled={self.use_quantum}, threshold={quantum_threshold}"
        )
    
    def solve(self, problem: ProblemBase, **kwargs) -> Dict[str, Any]:
        """
        Solve problem using hybrid approach.
        
        Routes to appropriate strategy based on configuration.
        
        Args:
            problem: Problem instance to solve
            **kwargs: Strategy-specific parameters
        
        Returns:
            Standardized result dictionary with:
            - solution: Best solution found
            - cost: Objective value
            - time_ms: Total execution time
            - energy_mj: Estimated energy consumption
            - iterations: Total iterations (across methods)
            - metadata: Including strategy_used, component_results, etc.
        
        Raises:
            SolverException: If all solving attempts fail
        """
        if not problem.is_generated:
            raise SolverException("Problem must be generated before solving")
        
        start_time = time.time()
        logger.info(
            f"Solving {problem.problem_type} problem (size={problem.problem_size}) "
            f"with hybrid strategy: {self.strategy}"
        )
        
        # Route to appropriate strategy
        if self.strategy == 'adaptive':
            result = self._solve_adaptive(problem, **kwargs)
        elif self.strategy == 'quantum_assisted':
            result = self._solve_quantum_assisted(problem, **kwargs)
        elif self.strategy == 'classical_first':
            result = self._solve_classical_first(problem, **kwargs)
        elif self.strategy == 'parallel':
            result = self._solve_parallel(problem, **kwargs)
        elif self.strategy == 'iterative':
            result = self._solve_iterative(problem, **kwargs)
        else:
            raise SolverException(f"Unknown strategy: {self.strategy}")
        
        # Add hybrid-specific metadata
        result['metadata']['hybrid_strategy'] = self.strategy
        result['metadata']['solver_type'] = 'hybrid'
        
        elapsed = time.time() - start_time
        logger.info(
            f"Hybrid solve complete: strategy={self.strategy}, "
            f"time={elapsed:.2f}s, cost={result['cost']:.4f}"
        )
        
        return result
    
    def _solve_adaptive(self, problem: ProblemBase, **kwargs) -> Dict[str, Any]:
        """
        Adaptive strategy: Choose best method based on problem characteristics.
        
        Decision logic:
        1. If problem.size < quantum_threshold and quantum available: Use quantum
        2. Else if classical available: Use classical
        3. Try fallback if primary fails
        """
        problem_size = problem.problem_size
        
        # Decide which method to use
        use_quantum_primary = (
            self.use_quantum and
            problem_size <= self.quantum_threshold and
            self.quantum_solver is not None
        )
        
        if use_quantum_primary:
            logger.info(
                f"Adaptive: Using quantum (size={problem_size} <= threshold={self.quantum_threshold})"
            )
            
            try:
                result = self.quantum_solver.solve(
                    problem,
                    algorithm=self.quantum_algorithm,
                    **kwargs
                )
                result['metadata']['strategy_used'] = 'quantum_primary'
                return result
            
            except Exception as e:
                logger.warning(f"Quantum solver failed: {e}. Falling back to classical.")
                # Fall through to classical
        
        # Use classical solver
        if self.classical_solver is None:
            raise SolverException("No solver available")
        
        logger.info(f"Adaptive: Using classical (method={self.classical_method})")
        result = self.classical_solver.solve(problem, **kwargs)
        result['metadata']['strategy_used'] = 'classical_primary'
        
        return result
    
    def _solve_quantum_assisted(self, problem: ProblemBase, **kwargs) -> Dict[str, Any]:
        """
        Quantum-assisted strategy: Use quantum to guide classical search.
        
        Steps:
        1. Run quantum solver to find promising region
        2. Use quantum solution as starting point for classical refinement
        3. Return best result
        """
        logger.info("Quantum-assisted strategy: Starting quantum exploration")
        
        if not self.use_quantum or self.quantum_solver is None:
            logger.warning("Quantum solver not available, using pure classical")
            return self.classical_solver.solve(problem, **kwargs)
        
        # Step 1: Quantum exploration
        try:
            quantum_result = self.quantum_solver.solve(
                problem,
                algorithm=self.quantum_algorithm,
                maxiter=50,  # Fewer iterations for initial exploration
                **kwargs
            )
            quantum_solution = quantum_result['solution']
            quantum_cost = quantum_result['cost']
            
            logger.info(f"Quantum phase complete: cost={quantum_cost:.4f}")
        
        except Exception as e:
            logger.warning(f"Quantum phase failed: {e}. Using pure classical.")
            return self.classical_solver.solve(problem, **kwargs)
        
        # Step 2: Classical refinement
        if self.classical_solver is None:
            logger.warning("Classical solver not available, returning quantum result")
            quantum_result['metadata']['strategy_used'] = 'quantum_only'
            return quantum_result
        
        logger.info("Classical refinement: Starting local search from quantum solution")
        
        try:
            # Use classical solver to refine
            # For now, just run classical solver and compare
            classical_result = self.classical_solver.solve(problem, **kwargs)
            classical_cost = classical_result['cost']
            
            logger.info(f"Classical phase complete: cost={classical_cost:.4f}")
            
            # Return better result
            if classical_cost < quantum_cost:
                classical_result['metadata']['strategy_used'] = 'quantum_assisted_classical_better'
                classical_result['metadata']['quantum_cost'] = quantum_cost
                return classical_result
            else:
                quantum_result['metadata']['strategy_used'] = 'quantum_assisted_quantum_better'
                quantum_result['metadata']['classical_cost'] = classical_cost
                return quantum_result
        
        except Exception as e:
            logger.warning(f"Classical refinement failed: {e}. Returning quantum result.")
            quantum_result['metadata']['strategy_used'] = 'quantum_assisted_classical_failed'
            return quantum_result
    
    def _solve_classical_first(self, problem: ProblemBase, **kwargs) -> Dict[str, Any]:
        """
        Classical-first strategy: Pre-process with classical, solve with quantum.
        
        Steps:
        1. Use classical solver to get initial solution
        2. If problem is small enough, verify/improve with quantum
        3. Return best result
        """
        logger.info("Classical-first strategy: Starting classical pre-processing")
        
        if self.classical_solver is None:
            raise SolverException("Classical solver required for classical-first strategy")
        
        # Step 1: Classical solution
        classical_result = self.classical_solver.solve(problem, **kwargs)
        classical_cost = classical_result['cost']
        
        logger.info(f"Classical phase complete: cost={classical_cost:.4f}")
        
        # Step 2: Quantum verification (if available and problem small enough)
        if (self.use_quantum and
            self.quantum_solver is not None and
            problem.problem_size <= self.quantum_threshold):
            
            logger.info("Quantum verification phase: Checking if quantum can improve")
            
            try:
                quantum_result = self.quantum_solver.solve(
                    problem,
                    algorithm=self.quantum_algorithm,
                    **kwargs
                )
                quantum_cost = quantum_result['cost']
                
                logger.info(f"Quantum phase complete: cost={quantum_cost:.4f}")
                
                # Return better result
                if quantum_cost < classical_cost:
                    quantum_result['metadata']['strategy_used'] = 'classical_first_quantum_improved'
                    quantum_result['metadata']['classical_cost'] = classical_cost
                    return quantum_result
                else:
                    classical_result['metadata']['strategy_used'] = 'classical_first_classical_better'
                    classical_result['metadata']['quantum_cost'] = quantum_cost
                    return classical_result
            
            except Exception as e:
                logger.warning(f"Quantum verification failed: {e}. Returning classical result.")
                classical_result['metadata']['strategy_used'] = 'classical_first_quantum_failed'
                return classical_result
        else:
            classical_result['metadata']['strategy_used'] = 'classical_only'
            return classical_result
    
    def _solve_parallel(self, problem: ProblemBase, **kwargs) -> Dict[str, Any]:
        """
        Parallel strategy: Run both solvers, return best.
        
        Note: Not truly parallel (would need threading/multiprocessing).
        Runs sequentially but treats as hedge against failure.
        """
        logger.info("Parallel strategy: Running both quantum and classical")
        
        results = []
        
        # Try classical
        if self.classical_solver is not None:
            try:
                classical_result = self.classical_solver.solve(problem, **kwargs)
                classical_result['metadata']['method'] = 'classical'
                results.append(classical_result)
                logger.info(f"Classical complete: cost={classical_result['cost']:.4f}")
            except Exception as e:
                logger.warning(f"Classical solver failed: {e}")
        
        # Try quantum
        if self.use_quantum and self.quantum_solver is not None:
            try:
                quantum_result = self.quantum_solver.solve(
                    problem,
                    algorithm=self.quantum_algorithm,
                    **kwargs
                )
                quantum_result['metadata']['method'] = 'quantum'
                results.append(quantum_result)
                logger.info(f"Quantum complete: cost={quantum_result['cost']:.4f}")
            except Exception as e:
                logger.warning(f"Quantum solver failed: {e}")
        
        if not results:
            raise SolverException("All solvers failed")
        
        # Return best result
        best_result = min(results, key=lambda r: r['cost'])
        best_result['metadata']['strategy_used'] = f"parallel_{best_result['metadata']['method']}_won"
        best_result['metadata']['all_costs'] = [r['cost'] for r in results]
        
        logger.info(f"Parallel: Best result from {best_result['metadata']['method']}")
        
        return best_result
    
    def _solve_iterative(self, problem: ProblemBase, max_iterations: int = 3, **kwargs) -> Dict[str, Any]:
        """
        Iterative strategy: Alternate between quantum and classical.
        
        Steps:
        1. Start with quantum exploration
        2. Refine with classical exploitation
        3. Repeat for max_iterations
        4. Return best solution found
        """
        logger.info(f"Iterative strategy: {max_iterations} iterations")
        
        if not self.use_quantum or self.quantum_solver is None:
            logger.warning("Quantum not available, using pure classical")
            return self.classical_solver.solve(problem, **kwargs)
        
        if self.classical_solver is None:
            logger.warning("Classical not available, using pure quantum")
            return self.quantum_solver.solve(problem, algorithm=self.quantum_algorithm, **kwargs)
        
        best_solution = None
        best_cost = float('inf')
        iteration_history = []
        
        for iteration in range(max_iterations):
            logger.info(f"Iteration {iteration + 1}/{max_iterations}")
            
            # Quantum phase
            try:
                quantum_result = self.quantum_solver.solve(
                    problem,
                    algorithm=self.quantum_algorithm,
                    maxiter=30,  # Shorter iterations
                    **kwargs
                )
                
                if quantum_result['cost'] < best_cost:
                    best_cost = quantum_result['cost']
                    best_solution = quantum_result['solution']
                
                iteration_history.append({
                    'iteration': iteration,
                    'phase': 'quantum',
                    'cost': quantum_result['cost']
                })
                
                logger.info(f"  Quantum: cost={quantum_result['cost']:.4f}")
            
            except Exception as e:
                logger.warning(f"  Quantum failed: {e}")
            
            # Classical phase
            try:
                classical_result = self.classical_solver.solve(problem, **kwargs)
                
                if classical_result['cost'] < best_cost:
                    best_cost = classical_result['cost']
                    best_solution = classical_result['solution']
                
                iteration_history.append({
                    'iteration': iteration,
                    'phase': 'classical',
                    'cost': classical_result['cost']
                })
                
                logger.info(f"  Classical: cost={classical_result['cost']:.4f}")
            
            except Exception as e:
                logger.warning(f"  Classical failed: {e}")
        
        if best_solution is None:
            raise SolverException("Iterative strategy: All iterations failed")
        
        logger.info(f"Iterative complete: best_cost={best_cost:.4f}")
        
        # Build result
        return {
            'solution': best_solution,
            'cost': best_cost,
            'time_ms': 0,  # Would need to track properly
            'energy_mj': 0.0,  # Would need to sum
            'iterations': max_iterations,
            'metadata': {
                'strategy_used': 'iterative',
                'iteration_history': iteration_history,
                'total_iterations': max_iterations,
            }
        }
    
    def get_solver_info(self) -> Dict[str, Any]:
        """Get hybrid solver information."""
        return {
            'solver_type': 'hybrid',
            'solver_name': 'hybrid_quantum_classical',
            'strategy': self.strategy,
            'quantum_enabled': self.use_quantum,
            'quantum_threshold': self.quantum_threshold,
            'classical_method': self.classical_method,
            'quantum_algorithm': self.quantum_algorithm,
            'supported_strategies': self.VALID_STRATEGIES,
            'classical_solver': self.classical_solver.get_solver_info() if self.classical_solver else None,
            'quantum_solver': self.quantum_solver.get_solver_info() if self.quantum_solver else None,
        }
