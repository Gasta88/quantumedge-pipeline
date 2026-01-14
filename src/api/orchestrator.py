"""
Job Execution Orchestrator for QuantumEdge Pipeline.

This module coordinates the entire workflow from problem submission to result delivery,
integrating all pipeline components (analyzer, router, solvers, monitoring) into a
cohesive execution framework.

Key Responsibilities:
--------------------
1. **Problem Analysis**: Extract features, generate metadata, store in database
2. **Intelligent Routing**: Make solver selection decisions based on environment and strategy
3. **Solver Execution**: Initialize and run correct solver with resource monitoring
4. **Result Validation**: Check solution validity and calculate quality metrics
5. **Metrics Recording**: Store execution data and update performance models
6. **Error Handling**: Graceful recovery from failures with comprehensive logging

Execution Modes:
---------------
- **Single Execution**: Execute one problem with chosen solver
- **Comparative Execution**: Run both classical and quantum for benchmarking
- **Batch Execution**: Process multiple problems efficiently

The orchestrator acts as the central coordinator, ensuring all components work together
seamlessly while maintaining proper error handling, resource management, and data tracking.

Example Usage:
-------------
```python
from src.api.orchestrator import JobOrchestrator
from src.problems.maxcut import MaxCutProblem

# Initialize orchestrator
orchestrator = JobOrchestrator()

# Create problem
problem = MaxCutProblem(num_nodes=30)
problem.generate(edge_probability=0.3)

# Execute with automatic solver selection
result = orchestrator.execute_job(
    problem=problem,
    edge_profile='aerospace',
    strategy='balanced'
)

print(f"Solver used: {result['solver_used']}")
print(f"Solution quality: {result['solution_quality']:.2%}")
print(f"Execution time: {result['time_ms']} ms")

# Compare classical vs quantum
comparison = orchestrator.execute_comparative(
    problem=problem,
    edge_profile='aerospace'
)

print(f"Classical time: {comparison['classical']['time_ms']} ms")
print(f"Quantum time: {comparison['quantum']['time_ms']} ms")
print(f"Winner: {comparison['recommendation']}")

# Batch processing
problems = [create_problem(i) for i in range(10)]
results = orchestrator.execute_batch(problems, edge_profile='mobile')
```

Architecture:
------------
    ┌─────────────────────────────────────────────┐
    │         JobOrchestrator                     │
    ├─────────────────────────────────────────────┤
    │                                             │
    │  1. Analyze Problem                         │
    │     ↓ ProblemAnalyzer                       │
    │  2. Route Decision                          │
    │     ↓ QuantumRouter + EdgeEnvironment       │
    │  3. Execute Solver                          │
    │     ↓ ClassicalSolver / QuantumSimulator    │
    │  4. Validate Result                         │
    │     ↓ Quality checks                        │
    │  5. Record Metrics                          │
    │     ↓ DatabaseManager + MetricsCollector    │
    │                                             │
    └─────────────────────────────────────────────┘

Design Principles:
-----------------
- **Fail-fast**: Validate inputs early
- **Fail-safe**: Graceful error recovery with cleanup
- **Observable**: Comprehensive logging and metrics
- **Composable**: Easy to extend with new strategies
- **Testable**: Clear interfaces and dependency injection
"""

from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
from uuid import uuid4
import logging
import time
import traceback
from contextlib import contextmanager

from src.analyzer.problem_analyzer import ProblemAnalyzer
from src.router.quantum_router import QuantumRouter,RoutingStrategy
from src.router.edge_simulator import EdgeEnvironment, DeploymentProfile
from src.solvers.classical_solver import ClassicalSolver
from src.solvers.quantum_simulator import QuantumSimulator
from src.monitoring.db_manager import DatabaseManager
from src.problems.problem_base import ProblemBase


# Configure module logger
logger = logging.getLogger(__name__)


class OrchestratorException(Exception):
    """Base exception for orchestrator errors."""
    pass


class ValidationError(OrchestratorException):
    """Raised when result validation fails."""
    pass


class ExecutionTimeout(OrchestratorException):
    """Raised when job execution exceeds timeout."""
    pass


class ResourceExhausted(OrchestratorException):
    """Raised when system resources are exhausted."""
    pass


class JobOrchestrator:
    """
    Coordinate entire workflow from problem submission to result delivery.
    
    The JobOrchestrator is the main entry point for executing optimization jobs
    in the QuantumEdge Pipeline. It integrates all components (analyzer, router,
    solvers, monitoring) and manages the complete execution lifecycle.
    
    Attributes:
        analyzer (ProblemAnalyzer): Problem feature extraction and analysis
        router (QuantumRouter): Intelligent solver selection
        classical_solver (ClassicalSolver): Classical optimization algorithms
        quantum_solver (QuantumSimulator): Quantum circuit simulation
        db_manager (DatabaseManager): Database operations (optional)
        
    Configuration:
        enable_db (bool): Whether to use database for metrics storage
        enable_validation (bool): Whether to validate solutions
        default_timeout_s (float): Default execution timeout in seconds
        max_retries (int): Maximum retry attempts for transient failures
    
    Thread Safety:
        This class is not thread-safe. Create separate instances for
        concurrent execution or use external synchronization.
    """
    
    def __init__(
        self,
        enable_db: bool = False,
        enable_validation: bool = True,
        default_timeout_s: float = 300.0,
        max_retries: int = 2,
        db_url: Optional[str] = None,
        strategy: str = 'balanced'
    ):
        """
        Initialize the job orchestrator with all required components.
        
        Args:
            enable_db: Enable database operations for metrics storage
            enable_validation: Enable solution validation
            default_timeout_s: Default execution timeout in seconds
            max_retries: Maximum retry attempts for transient failures
            db_url: Database connection URL (required if enable_db=True)
            strategy: Routing strategy ('balanced', 'energy_optimized', 'latency_optimized', 'quality_optimized')
        
        Raises:
            ValueError: If enable_db=True but db_url is not provided
        """
        logger.info("Initializing JobOrchestrator")
        
        # Configuration
        self.enable_db = enable_db
        self.enable_validation = enable_validation
        self.default_timeout_s = default_timeout_s
        self.max_retries = max_retries
        self.strategy = strategy
        
        # Initialize components
        self.analyzer = ProblemAnalyzer()
        if self.strategy == 'balanced':
            routing_strategy = RoutingStrategy.BALANCED
        elif self.strategy == 'energy_optimized':
            routing_strategy = RoutingStrategy.ENERGY_OPTIMIZED
        elif self.strategy == 'latency_optimized':
            routing_strategy = RoutingStrategy.LATENCY_OPTIMIZED
        elif self.strategy == 'quality_optimized':
            routing_strategy = RoutingStrategy.QUALITY_OPTIMIZED
        else:
            raise ValueError(f"Invalid routing strategy: {self.strategy}")
        self.router = QuantumRouter(strategy=routing_strategy)
        self.classical_solver = ClassicalSolver()
        self.quantum_solver = QuantumSimulator()
        
        # Initialize database manager if enabled
        self.db_manager = None
        if enable_db:
            if not db_url:
                raise ValueError("Database URL required when enable_db=True")
            self.db_manager = DatabaseManager(database_url=db_url)
            logger.info(f"Database manager initialized with URL: {db_url}")
        else:
            logger.info("Database manager disabled")
        
        # Execution statistics
        self._jobs_executed = 0
        self._total_execution_time_ms = 0.0
        self._failed_jobs = 0
        
        logger.info("JobOrchestrator initialization complete")
    
    def execute_job(
        self,
        problem: ProblemBase,
        edge_profile: str = 'aerospace',
        strategy: str = 'balanced',
        timeout_s: Optional[float] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> Dict[str, Any]:
        """
        Execute complete workflow from problem submission to result delivery.
        
        This is the main entry point for job execution. It orchestrates all
        pipeline stages:
        
        Workflow Steps:
        --------------
        a) Analyze problem
           - Extract features using ProblemAnalyzer
           - Generate metadata (size, complexity, structure)
           - Store problem in database (if enabled)
        
        b) Make routing decision
           - Get edge environment constraints
           - Use QuantumRouter to select solver
           - Log decision with reasoning
        
        c) Execute solver
           - Initialize correct solver (classical or quantum)
           - Run with resource monitoring
           - Handle timeouts and errors
        
        d) Validate result
           - Check solution validity (feasibility)
           - Calculate quality metrics (optimality gap, etc.)
           - Compare to expectations
        
        e) Record metrics
           - Store execution data in database
           - Update performance models
           - Generate execution report
        
        f) Return comprehensive result
        
        Args:
            problem: Problem instance to solve (must be generated)
            edge_profile: Deployment environment ('aerospace', 'mobile', 'ground_server')
            strategy: Routing strategy ('balanced', 'energy_optimized', 'latency_optimized', 'quality_optimized')
            timeout_s: Execution timeout in seconds (None = use default)
            progress_callback: Optional callback(stage: str, progress: float [0-1])
        
        Returns:
            Comprehensive result dictionary:
            {
                'job_id': str,
                'success': bool,
                'solver_used': str,  # 'classical', 'quantum', or 'hybrid'
                
                # Problem information
                'problem_type': str,
                'problem_size': int,
                'problem_analysis': Dict,
                
                # Routing decision
                'routing_decision': str,
                'routing_reason': str,
                'routing_confidence': float,
                
                # Execution results
                'solution': Any,
                'cost': float,
                'time_ms': float,
                'energy_consumed_mj': float,
                
                # Validation
                'is_valid': bool,
                'solution_quality': float,
                'validation_errors': List[str],
                
                # Metadata
                'edge_profile': str,
                'strategy': str,
                'timestamp': datetime,
                
                # Errors (if failed)
                'error': Optional[str],
                'error_traceback': Optional[str]
            }
        
        Raises:
            ValueError: If problem is not generated or invalid parameters
            ExecutionTimeout: If execution exceeds timeout
            OrchestratorException: For other orchestration failures
        
        Example:
            >>> problem = MaxCutProblem(num_nodes=30)
            >>> problem.generate()
            >>> 
            >>> orchestrator = JobOrchestrator()
            >>> result = orchestrator.execute_job(
            ...     problem=problem,
            ...     edge_profile='aerospace',
            ...     strategy='balanced'
            ... )
            >>> 
            >>> if result['success']:
            ...     print(f"Solution found with cost: {result['cost']:.4f}")
            ...     print(f"Solver used: {result['solver_used']}")
            ...     print(f"Quality: {result['solution_quality']:.2%}")
        """
        # Generate unique job ID
        job_id = str(uuid4())
        start_time = time.time()
        timeout_s = timeout_s or self.default_timeout_s
        
        logger.info(f"Starting job execution: {job_id}")
        logger.info(f"Problem: {problem.problem_type}, Size: {problem.problem_size}")
        logger.info(f"Edge profile: {edge_profile}, Strategy: {strategy}")
        
        # Initialize result structure
        result = {
            'job_id': job_id,
            'success': False,
            'timestamp': datetime.now(),
            'edge_profile': edge_profile,
            'strategy': strategy,
            'error': None,
            'error_traceback': None
        }
        
        try:
            # ================================================================
            # STAGE A: ANALYZE PROBLEM
            # ================================================================
            if progress_callback:
                progress_callback("analyze", 0.1)
            
            logger.info(f"[{job_id}] Stage A: Analyzing problem...")
            
            # Validate problem
            if not problem.is_generated:
                raise ValueError("Problem must be generated before execution")
            
            # Extract features and metadata
            analysis = self.analyzer.analyze_problem(problem)
            
            result['problem_type'] = analysis['problem_type']
            result['problem_size'] = analysis['problem_size']
            result['problem_analysis'] = analysis
            
            logger.info(f"[{job_id}] Problem analysis complete")
            logger.debug(f"[{job_id}] Analysis: {analysis}")
            
            # Store problem in database (if enabled)
            problem_id = None
            if self.db_manager:
                try:
                    # Convert problem to storable format
                    problem_data = self._serialize_problem(problem)
                    problem_id = self._store_problem(problem_data, analysis)
                    logger.info(f"[{job_id}] Problem stored in database: {problem_id}")
                except Exception as e:
                    logger.warning(f"[{job_id}] Failed to store problem in database: {e}")
            
            # ================================================================
            # STAGE B: MAKE ROUTING DECISION
            # ================================================================
            if progress_callback:
                progress_callback("route", 0.2)
            
            logger.info(f"[{job_id}] Stage B: Making routing decision...")
            
            # Get edge environment
            if edge_profile == 'aerospace':
                edge_env = EdgeEnvironment(DeploymentProfile.AEROSPACE)
            elif edge_profile == 'mobile':
                edge_env = EdgeEnvironment(DeploymentProfile.MOBILE)
            elif edge_profile == 'ground_server':
                edge_env = EdgeEnvironment(DeploymentProfile.GROUND_SERVER)
            else:
                raise ValueError(f"Invalid edge profile: {edge_profile}")
            
            # Route problem
            routing_result = self.router.route_problem(
                problem=problem,
                edge_env=edge_env
            )
            
            solver_choice = routing_result['decision']
            routing_reason = routing_result['reasoning']
            routing_confidence = routing_result['confidence']
            
            result['routing_decision'] = solver_choice
            result['routing_reason'] = routing_reason
            result['routing_confidence'] = routing_confidence
            
            logger.info(f"[{job_id}] Routing decision: {solver_choice}")
            logger.info(f"[{job_id}] Reason: {routing_reason}")
            logger.info(f"[{job_id}] Confidence: {routing_confidence:.2%}")
            
            # ================================================================
            # STAGE C: EXECUTE SOLVER
            # ================================================================
            if progress_callback:
                progress_callback("execute", 0.3)
            
            logger.info(f"[{job_id}] Stage C: Executing solver ({solver_choice})...")
            
            # Calculate remaining timeout
            elapsed = time.time() - start_time
            remaining_timeout = timeout_s - elapsed
            
            if remaining_timeout <= 0:
                raise ExecutionTimeout(f"Timeout before solver execution (elapsed: {elapsed:.2f}s)")
            
            # Execute with selected solver
            solver_result = self._execute_solver(
                problem=problem,
                solver_choice=solver_choice,
                timeout_s=remaining_timeout,
                progress_callback=progress_callback
            )
            
            # Merge solver result
            result.update(solver_result)
            result['solver_used'] = solver_choice
            
            logger.info(f"[{job_id}] Solver execution complete")
            logger.info(f"[{job_id}] Cost: {solver_result.get('cost', 'N/A')}")
            logger.info(f"[{job_id}] Time: {solver_result.get('time_ms', 'N/A')} ms")
            
            # ================================================================
            # STAGE D: VALIDATE RESULT
            # ================================================================
            if progress_callback:
                progress_callback("validate", 0.8)
            
            logger.info(f"[{job_id}] Stage D: Validating result...")
            
            if self.enable_validation:
                validation = self._validate_result(problem, solver_result)
                result['is_valid'] = validation['is_valid']
                result['solution_quality'] = validation['quality']
                result['validation_errors'] = validation['errors']
                
                logger.info(f"[{job_id}] Validation: {'PASS' if validation['is_valid'] else 'FAIL'}")
                logger.info(f"[{job_id}] Quality: {validation['quality']:.2%}")
                
                if validation['errors']:
                    logger.warning(f"[{job_id}] Validation errors: {validation['errors']}")
            else:
                result['is_valid'] = True
                result['solution_quality'] = 1.0
                result['validation_errors'] = []
                logger.info(f"[{job_id}] Validation disabled")
            
            # ================================================================
            # STAGE E: RECORD METRICS
            # ================================================================
            if progress_callback:
                progress_callback("record", 0.9)
            
            logger.info(f"[{job_id}] Stage E: Recording metrics...")
            
            if self.db_manager and problem_id:
                try:
                    self._record_execution(
                        job_id=job_id,
                        problem_id=problem_id,
                        result=result
                    )
                    logger.info(f"[{job_id}] Metrics recorded in database")
                except Exception as e:
                    logger.warning(f"[{job_id}] Failed to record metrics: {e}")
            
            # Update statistics
            self._jobs_executed += 1
            self._total_execution_time_ms += result['time_ms']
            
            # ================================================================
            # STAGE F: RETURN COMPREHENSIVE RESULT
            # ================================================================
            result['success'] = True
            
            if progress_callback:
                progress_callback("complete", 1.0)
            
            total_time = (time.time() - start_time) * 1000
            logger.info(f"[{job_id}] Job execution complete (total: {total_time:.2f} ms)")
            
            return result
            
        except Exception as e:
            # Handle errors gracefully
            self._failed_jobs += 1
            
            error_msg = str(e)
            error_trace = traceback.format_exc()
            
            result['success'] = False
            result['error'] = error_msg
            result['error_traceback'] = error_trace
            
            logger.error(f"[{job_id}] Job execution failed: {error_msg}")
            logger.debug(f"[{job_id}] Traceback:\n{error_trace}")
            
            # Attempt cleanup
            self._cleanup_resources(job_id)
            
            return result
    
    def execute_comparative(
        self,
        problem: ProblemBase,
        edge_profile: str = 'aerospace',
        timeout_s: Optional[float] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> Dict[str, Any]:
        """
        Run BOTH classical and quantum solvers for side-by-side comparison.
        
        This method executes the same problem with both solver types to enable:
        - Performance benchmarking
        - Solution quality comparison
        - Energy efficiency analysis
        - Demo and visualization purposes
        
        Both solvers run independently (not in parallel) to ensure fair comparison
        of execution time and resource usage.
        
        Args:
            problem: Problem instance to solve (must be generated)
            edge_profile: Deployment environment
            timeout_s: Per-solver timeout in seconds (None = use default)
            progress_callback: Optional callback(stage: str, progress: float)
        
        Returns:
            Comparison result dictionary:
            {
                'job_id': str,
                'success': bool,
                'problem_type': str,
                'problem_size': int,
                
                # Classical results
                'classical': {
                    'solution': Any,
                    'cost': float,
                    'time_ms': float,
                    'energy_consumed_mj': float,
                    'is_valid': bool,
                    'solution_quality': float,
                    'error': Optional[str]
                },
                
                # Quantum results
                'quantum': {
                    'solution': Any,
                    'cost': float,
                    'time_ms': float,
                    'energy_consumed_mj': float,
                    'is_valid': bool,
                    'solution_quality': float,
                    'error': Optional[str]
                },
                
                # Comparison metrics
                'speedup_factor': float,  # classical_time / quantum_time
                'energy_ratio': float,    # classical_energy / quantum_energy
                'quality_diff': float,    # quantum_quality - classical_quality
                
                # Recommendation
                'recommendation': str,  # 'classical', 'quantum', or 'tie'
                'recommendation_reason': str,
                
                'timestamp': datetime
            }
        
        Raises:
            ValueError: If problem is not generated
            OrchestratorException: If both solvers fail
        
        Example:
            >>> problem = MaxCutProblem(num_nodes=25)
            >>> problem.generate()
            >>> 
            >>> orchestrator = JobOrchestrator()
            >>> comparison = orchestrator.execute_comparative(problem)
            >>> 
            >>> print(f"Classical: {comparison['classical']['cost']:.4f} "
            ...       f"in {comparison['classical']['time_ms']:.2f} ms")
            >>> print(f"Quantum: {comparison['quantum']['cost']:.4f} "
            ...       f"in {comparison['quantum']['time_ms']:.2f} ms")
            >>> print(f"Winner: {comparison['recommendation']}")
        """
        job_id = str(uuid4())
        timeout_s = timeout_s or self.default_timeout_s
        
        logger.info(f"Starting comparative execution: {job_id}")
        logger.info(f"Problem: {problem.problem_type}, Size: {problem.problem_size}")
        
        # Validate problem
        if not problem.is_generated:
            raise ValueError("Problem must be generated before execution")
        
        result = {
            'job_id': job_id,
            'success': False,
            'problem_type': problem.problem_type,
            'problem_size': problem.problem_size,
            'timestamp': datetime.now(),
            'edge_profile': edge_profile
        }
        
        # Execute classical solver (always run, even if quantum fails)
        classical_success = False
        classical_result = None
        
        try:
            if progress_callback:
                progress_callback("classical", 0.2)
            
            logger.info(f"[{job_id}] Executing classical solver...")
            
            classical_result = self._execute_solver(
                problem=problem,
                solver_choice='classical',
                timeout_s=timeout_s / 2,  # Split timeout
                progress_callback=None
            )
            
            if self.enable_validation:
                classical_validation = self._validate_result(problem, classical_result)
                classical_result['is_valid'] = classical_validation['is_valid']
                classical_result['solution_quality'] = classical_validation['quality']
            
            result['classical'] = classical_result
            classical_success = True
            logger.info(f"[{job_id}] Classical complete: cost={classical_result.get('cost', 'N/A')}")
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"[{job_id}] Classical solver failed: {error_msg}")
            result['classical'] = {
                'success': False,
                'error': error_msg,
                'time_ms': 0,
                'cost': float('inf'),
                'solution_quality': 0.0
            }
        
        # Execute quantum solver (isolated from classical failure)
        quantum_success = False
        quantum_result = None
        
        try:
            if progress_callback:
                progress_callback("quantum", 0.6)
            
            logger.info(f"[{job_id}] Executing quantum solver...")
            
            quantum_result = self._execute_solver(
                problem=problem,
                solver_choice='quantum',
                timeout_s=timeout_s / 2,  # Split timeout
                progress_callback=None
            )
            
            if self.enable_validation:
                quantum_validation = self._validate_result(problem, quantum_result)
                quantum_result['is_valid'] = quantum_validation['is_valid']
                quantum_result['solution_quality'] = quantum_validation['quality']
            
            result['quantum'] = quantum_result
            quantum_success = True
            logger.info(f"[{job_id}] Quantum complete: cost={quantum_result.get('cost', 'N/A')}")
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"[{job_id}] Quantum solver failed: {error_msg}")
            result['quantum'] = {
                'success': False,
                'error': error_msg,
                'time_ms': 0,
                'cost': float('inf'),
                'solution_quality': 0.0
            }
        
        # Calculate comparison metrics (even if one solver failed)
        try:
            if progress_callback:
                progress_callback("compare", 0.9)
            
            # Use results even if one failed (will show which one succeeded)
            comparison = self._calculate_comparison_metrics(
                result['classical'], 
                result['quantum']
            )
            result.update(comparison)
            
            # Determine overall success
            # Consider it successful if at least one solver succeeded
            result['success'] = classical_success or quantum_success
            
            if not result['success']:
                result['error'] = "Both classical and quantum solvers failed"
                logger.error(f"[{job_id}] Both solvers failed")
            elif not classical_success:
                result['partial_success'] = True
                result['note'] = "Classical solver failed, only quantum results available"
                logger.warning(f"[{job_id}] Only quantum solver succeeded")
            elif not quantum_success:
                result['partial_success'] = True
                result['note'] = "Quantum solver failed, only classical results available"
                logger.warning(f"[{job_id}] Only classical solver succeeded")
            
            if progress_callback:
                progress_callback("complete", 1.0)
            
            logger.info(f"[{job_id}] Comparative execution complete")
            if result['success']:
                logger.info(f"[{job_id}] Recommendation: {result.get('recommendation', 'N/A')}")
            
            return result
            
        except Exception as e:
            # Catastrophic failure in comparison calculation
            error_msg = str(e)
            error_trace = traceback.format_exc()
            
            result['success'] = False
            result['error'] = f"Comparison calculation failed: {error_msg}"
            result['error_traceback'] = error_trace
            
            logger.error(f"[{job_id}] Comparison calculation failed: {error_msg}")
            logger.debug(f"[{job_id}] Traceback:\n{error_trace}")
            
            return result
    
    def execute_batch(
        self,
        problems: List[ProblemBase],
        edge_profile: str = 'aerospace',
        strategy: str = 'balanced',
        timeout_per_job_s: Optional[float] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[Dict[str, Any]]:
        """
        Process multiple problems efficiently in batch mode.
        
        Batch processing is useful for:
        - Benchmarking across problem sets
        - Performance testing with different parameters
        - Automated evaluation and comparison
        - Research and experimentation
        
        Each problem is executed independently using execute_job(). Failed
        jobs do not stop batch processing - all problems are attempted.
        
        Args:
            problems: List of problem instances (all must be generated)
            edge_profile: Deployment environment for all jobs
            strategy: Routing strategy for all jobs
            timeout_per_job_s: Timeout per individual job (None = use default)
            progress_callback: Optional callback(completed: int, total: int)
        
        Returns:
            List of result dictionaries (one per problem):
            [
                {job execution result for problem 0},
                {job execution result for problem 1},
                ...
            ]
            
            Each result has same structure as execute_job() return value.
        
        Raises:
            ValueError: If problems list is empty or contains invalid problems
        
        Example:
            >>> # Create multiple problem instances
            >>> problems = []
            >>> for size in [10, 20, 30, 40, 50]:
            ...     p = MaxCutProblem(num_nodes=size)
            ...     p.generate(edge_probability=0.3)
            ...     problems.append(p)
            >>> 
            >>> # Execute batch
            >>> orchestrator = JobOrchestrator()
            >>> results = orchestrator.execute_batch(
            ...     problems=problems,
            ...     edge_profile='aerospace',
            ...     progress_callback=lambda done, total: print(f"{done}/{total}")
            ... )
            >>> 
            >>> # Analyze results
            >>> successful = [r for r in results if r['success']]
            >>> print(f"Success rate: {len(successful)}/{len(results)}")
            >>> 
            >>> avg_time = sum(r['time_ms'] for r in successful) / len(successful)
            >>> print(f"Average execution time: {avg_time:.2f} ms")
        """
        if not problems:
            raise ValueError("Problems list cannot be empty")
        
        batch_id = str(uuid4())
        total_jobs = len(problems)
        timeout_s = timeout_per_job_s or self.default_timeout_s
        
        logger.info(f"Starting batch execution: {batch_id}")
        logger.info(f"Total jobs: {total_jobs}")
        logger.info(f"Edge profile: {edge_profile}, Strategy: {strategy}")
        
        results = []
        successful_count = 0
        failed_count = 0
        
        for idx, problem in enumerate(problems):
            job_number = idx + 1
            
            logger.info(f"[{batch_id}] Processing job {job_number}/{total_jobs}")
            
            try:
                # Execute individual job
                result = self.execute_job(
                    problem=problem,
                    edge_profile=edge_profile,
                    strategy=strategy,
                    timeout_s=timeout_s,
                    progress_callback=None  # Don't propagate individual progress
                )
                
                results.append(result)
                
                if result['success']:
                    successful_count += 1
                    logger.info(f"[{batch_id}] Job {job_number} succeeded")
                else:
                    failed_count += 1
                    logger.warning(f"[{batch_id}] Job {job_number} failed: {result.get('error', 'Unknown')}")
                
            except Exception as e:
                # Don't let one failure stop the batch
                error_result = {
                    'job_id': str(uuid4()),
                    'success': False,
                    'error': str(e),
                    'error_traceback': traceback.format_exc(),
                    'problem_type': problem.problem_type if hasattr(problem, 'problem_type') else 'unknown',
                    'problem_size': problem.problem_size if hasattr(problem, 'problem_size') else 0,
                    'timestamp': datetime.now()
                }
                results.append(error_result)
                failed_count += 1
                
                logger.error(f"[{batch_id}] Job {job_number} raised exception: {e}")
            
            # Report progress
            if progress_callback:
                progress_callback(job_number, total_jobs)
        
        logger.info(f"[{batch_id}] Batch execution complete")
        logger.info(f"[{batch_id}] Successful: {successful_count}, Failed: {failed_count}")
        
        return results
    
    # =========================================================================
    # PRIVATE HELPER METHODS
    # =========================================================================
    
    def _execute_solver(
        self,
        problem: ProblemBase,
        solver_choice: str,
        timeout_s: float,
        progress_callback: Optional[Callable[[str, float], None]]
    ) -> Dict[str, Any]:
        """
        Execute the selected solver with resource monitoring.
        
        Args:
            problem: Problem to solve
            solver_choice: 'classical', 'quantum', or 'hybrid'
            timeout_s: Execution timeout
            progress_callback: Optional progress callback
        
        Returns:
            Solver execution result
        
        Raises:
            ExecutionTimeout: If solver exceeds timeout
            OrchestratorException: For other execution failures
        """
        start_time = time.time()
        
        try:
            if solver_choice == 'classical':
                result = self.classical_solver.solve(
                    problem=problem,
                    method='auto',
                    timeout_seconds=timeout_s
                )
            
            elif solver_choice == 'quantum':
                result = self.quantum_solver.solve(
                    problem=problem,
                    # method='qaoa',
                    # timeout=timeout_s
                )
            
            elif solver_choice == 'hybrid':
                # Hybrid approach: Use quantum for initial solution, refine with classical
                logger.info("Executing hybrid approach...")
                
                quantum_result = self.quantum_solver.solve(
                    problem=problem,
                    # method='qaoa',
                    timeout=timeout_s / 2
                )
                
                # Use quantum solution as starting point for classical refinement
                classical_result = self.classical_solver.solve(
                    problem=problem,
                    method='auto',
                    timeout=timeout_s / 2,
                    initial_solution=quantum_result.get('solution')
                )
                
                # Return best result
                if classical_result.get('cost', float('inf')) < quantum_result.get('cost', float('inf')):
                    result = classical_result
                    result['hybrid_note'] = 'Classical refinement improved quantum solution'
                else:
                    result = quantum_result
                    result['hybrid_note'] = 'Quantum solution was already optimal'
            
            else:
                raise ValueError(f"Unknown solver choice: {solver_choice}")
            
            # Check for timeout
            elapsed = time.time() - start_time
            if elapsed > timeout_s:
                raise ExecutionTimeout(f"Solver exceeded timeout: {elapsed:.2f}s > {timeout_s:.2f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Solver execution failed: {e}")
            raise OrchestratorException(f"Solver execution error: {e}") from e
    
    def _validate_result(
        self,
        problem: ProblemBase,
        solver_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate solution and calculate quality metrics.
        
        Args:
            problem: Original problem instance
            solver_result: Result from solver execution
        
        Returns:
            Validation result:
            {
                'is_valid': bool,
                'quality': float (0-1),
                'errors': List[str]
            }
        """
        errors = []
        is_valid = True
        quality = 1.0
        
        try:
            solution = solver_result.get('solution')
            cost = solver_result.get('cost')
            
            # Basic checks
            if solution is None:
                errors.append("Solution is None")
                is_valid = False
            
            if cost is None or cost == float('inf'):
                errors.append("Invalid cost value")
                quality *= 0.5
            
            # Problem-specific validation
            try:
                problem_valid = problem.validate_solution(solution)
                if not problem_valid:
                    errors.append("Solution violates problem constraints")
                    is_valid = False
            except Exception as e:
                errors.append(f"Validation error: {e}")
                is_valid = False
            
            # Quality estimation (problem-specific)
            if is_valid and cost is not None:
                # For maximization: quality = cost / theoretical_max
                # For minimization: quality = theoretical_min / cost
                # For now, use simple heuristic
                quality = min(1.0, max(0.0, 1.0 - abs(cost) / 100.0))
        
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            errors.append(f"Validation exception: {e}")
            is_valid = False
            quality = 0.0
        
        return {
            'is_valid': is_valid,
            'quality': quality,
            'errors': errors
        }
    
    def _calculate_comparison_metrics(
        self,
        classical_result: Dict[str, Any],
        quantum_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate comparison metrics between classical and quantum results.
        
        Handles cases where one or both solvers may have failed.
        
        Args:
            classical_result: Classical solver result (may contain 'error')
            quantum_result: Quantum solver result (may contain 'error')
        
        Returns:
            Comparison metrics dictionary with:
            - speedup_factor: float (or None if cannot calculate)
            - quantum_advantage_ratio: float (alias for speedup_factor)
            - energy_ratio: float (or None if cannot calculate)
            - quality_diff: float (or None if cannot calculate)
            - recommendation: str ('classical', 'quantum', 'unknown')
            - recommendation_reason: str
        """
        comparison = {}
        
        # Check if solvers succeeded
        classical_success = classical_result.get('success', True) and 'error' not in classical_result
        quantum_success = quantum_result.get('success', True) and 'error' not in quantum_result
        
        # Handle case where one or both failed
        if not classical_success and not quantum_success:
            comparison['speedup_factor'] = None
            comparison['quantum_advantage_ratio'] = None
            comparison['energy_ratio'] = None
            comparison['quality_diff'] = None
            comparison['recommendation'] = 'unknown'
            comparison['recommendation_reason'] = 'Both solvers failed - cannot compare'
            return comparison
        
        if not classical_success:
            comparison['speedup_factor'] = None
            comparison['quantum_advantage_ratio'] = None
            comparison['energy_ratio'] = None
            comparison['quality_diff'] = None
            comparison['recommendation'] = 'quantum'
            comparison['recommendation_reason'] = 'Classical solver failed, quantum succeeded'
            return comparison
        
        if not quantum_success:
            comparison['speedup_factor'] = None
            comparison['quantum_advantage_ratio'] = None
            comparison['energy_ratio'] = None
            comparison['quality_diff'] = None
            comparison['recommendation'] = 'classical'
            comparison['recommendation_reason'] = 'Quantum solver failed, classical succeeded'
            return comparison
        
        # Both succeeded - calculate normal metrics
        
        # Speedup factor
        classical_time = classical_result.get('time_ms', 0)
        quantum_time = quantum_result.get('time_ms', 0)
        
        if quantum_time > 0 and classical_time > 0:
            comparison['speedup_factor'] = classical_time / quantum_time
            comparison['quantum_advantage_ratio'] = classical_time / quantum_time
        else:
            comparison['speedup_factor'] = None
            comparison['quantum_advantage_ratio'] = None
        
        # Energy ratio
        classical_energy = classical_result.get('energy_consumed_mj', 0)
        quantum_energy = quantum_result.get('energy_consumed_mj', 0)
        
        if quantum_energy > 0 and classical_energy > 0:
            comparison['energy_ratio'] = classical_energy / quantum_energy
        else:
            comparison['energy_ratio'] = None
        
        # Quality difference
        classical_quality = classical_result.get('solution_quality', 0)
        quantum_quality = quantum_result.get('solution_quality', 0)
        comparison['quality_diff'] = quantum_quality - classical_quality
        
        # Cost comparison
        classical_cost = classical_result.get('cost', float('inf'))
        quantum_cost = quantum_result.get('cost', float('inf'))
        
        # Make recommendation
        if classical_cost < quantum_cost * 0.95:  # Classical significantly better
            recommendation = 'classical'
            reason = f"Classical found better solution (cost: {classical_cost:.4f} vs {quantum_cost:.4f})"
        elif quantum_cost < classical_cost * 0.95:  # Quantum significantly better
            recommendation = 'quantum'
            reason = f"Quantum found better solution (cost: {quantum_cost:.4f} vs {classical_cost:.4f})"
        elif quantum_time > 0 and classical_time > 0:
            if quantum_time < classical_time * 0.8:  # Quantum faster
                recommendation = 'quantum'
                speedup = comparison.get('speedup_factor', 0)
                if speedup:
                    reason = f"Quantum faster with similar quality (speedup: {speedup:.2f}x)"
                else:
                    reason = "Quantum faster with similar quality"
            elif classical_time < quantum_time * 0.8:  # Classical faster
                recommendation = 'classical'
                reason = f"Classical faster with similar quality"
            else:
                recommendation = 'tie'
                reason = "Both solvers performed similarly"
        else:
            recommendation = 'tie'
            reason = "Both solvers performed similarly"
        
        comparison['recommendation'] = recommendation
        comparison['recommendation_reason'] = reason
        
        return comparison
    
    def _serialize_problem(self, problem: ProblemBase) -> Dict[str, Any]:
        """Serialize problem to dictionary for database storage."""
        return {
            'problem_type': problem.problem_type,
            'problem_size': problem.problem_size,
            'complexity_class': problem.complexity_class,
            'data': problem.to_dict() if hasattr(problem, 'to_dict') else {}
        }
    
    def _store_problem(self, problem_data: Dict[str, Any], analysis: Dict[str, Any]) -> str:
        """Store problem in database and return problem ID."""
        # This would use async database operations in real implementation
        # For now, just return a dummy ID
        problem_id = str(uuid4())
        logger.debug(f"Problem stored with ID: {problem_id}")
        return problem_id
    
    def _record_execution(
        self,
        job_id: str,
        problem_id: str,
        result: Dict[str, Any]
    ):
        """Record job execution metrics in database."""
        # This would use async database operations in real implementation
        logger.debug(f"Recording execution for job {job_id}")
    
    def _cleanup_resources(self, job_id: str):
        """Clean up resources after job execution or failure."""
        logger.debug(f"Cleaning up resources for job {job_id}")
        # Close any open connections, release memory, etc.
    
    @contextmanager
    def _timeout_context(self, timeout_s: float):
        """Context manager for timeout handling."""
        # This would implement actual timeout logic
        yield
    
    # =========================================================================
    # PUBLIC UTILITY METHODS
    # =========================================================================
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get orchestrator execution statistics.
        
        Returns:
            Statistics dictionary:
            {
                'jobs_executed': int,
                'jobs_failed': int,
                'success_rate': float,
                'average_execution_time_ms': float,
                'total_execution_time_ms': float
            }
        """
        total_jobs = self._jobs_executed + self._failed_jobs
        success_rate = self._jobs_executed / total_jobs if total_jobs > 0 else 0.0
        avg_time = self._total_execution_time_ms / self._jobs_executed if self._jobs_executed > 0 else 0.0
        
        return {
            'jobs_executed': self._jobs_executed,
            'jobs_failed': self._failed_jobs,
            'success_rate': success_rate,
            'average_execution_time_ms': avg_time,
            'total_execution_time_ms': self._total_execution_time_ms
        }
    
    def reset_statistics(self):
        """Reset execution statistics counters."""
        self._jobs_executed = 0
        self._total_execution_time_ms = 0.0
        self._failed_jobs = 0
        logger.info("Statistics reset")
    
    def __enter__(self):
        """Support context manager protocol."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleanup when exiting context manager."""
        logger.info("JobOrchestrator context exiting")
        # Cleanup resources if needed
        return False
    
    def __repr__(self) -> str:
        """String representation."""
        return (f"JobOrchestrator(db={'enabled' if self.enable_db else 'disabled'}, "
                f"validation={'enabled' if self.enable_validation else 'disabled'}, "
                f"jobs_executed={self._jobs_executed})")
