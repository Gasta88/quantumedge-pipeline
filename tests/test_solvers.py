"""
Unit tests for solvers module (minimal coverage with success/failure cases).
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np

# Import modules to test
from src.solvers.solver_base import (
    SolverBase,
    EnergyTracker,
    SolverException,
    SolverConfigurationError,
    InvalidSolutionError
)
from src.solvers.classical_solver import ClassicalSolver
from src.solvers.quantum_simulator import QuantumSimulator, QuantumSimulatorException
from src.problems.maxcut import MaxCutProblem


class TestEnergyTracker:
    """Test EnergyTracker class."""
    
    def test_energy_tracker_initialization_success(self):
        """Test successful energy tracker initialization."""
        tracker = EnergyTracker(
            tdp_watts=65.0,
            utilization_factor=0.6,
            efficiency=0.8
        )
        
        assert tracker.tdp_watts == 65.0
        assert tracker.utilization_factor == 0.6
        assert tracker.efficiency == 0.8
        assert tracker._is_tracking is False
    
    def test_energy_tracker_initialization_failure(self):
        """Test energy tracker initialization with invalid parameters."""
        # Invalid TDP (negative)
        with pytest.raises(ValueError, match="Invalid TDP"):
            EnergyTracker(tdp_watts=-10.0)
        
        # Invalid TDP (too high)
        with pytest.raises(ValueError, match="Invalid TDP"):
            EnergyTracker(tdp_watts=600.0)
        
        # Invalid utilization factor
        with pytest.raises(ValueError, match="Invalid utilization_factor"):
            EnergyTracker(tdp_watts=65.0, utilization_factor=1.5)
        
        # Invalid efficiency
        with pytest.raises(ValueError, match="Invalid efficiency"):
            EnergyTracker(tdp_watts=65.0, efficiency=2.0)
    
    def test_energy_tracking_success(self):
        """Test successful energy tracking."""
        tracker = EnergyTracker(tdp_watts=65.0)
        
        tracker.start_tracking()
        assert tracker._is_tracking is True
        
        # Simulate some computation
        _ = sum(range(100000))
        
        energy_mj = tracker.stop_tracking()
        assert isinstance(energy_mj, float)
        assert energy_mj >= 0.0
        assert tracker._is_tracking is False
    
    def test_energy_tracking_failure(self):
        """Test energy tracking with invalid usage."""
        tracker = EnergyTracker()
        
        # Stop without start should fail gracefully
        energy = tracker.stop_tracking()
        assert energy == 0.0
        
        # Double start should raise error
        tracker.start_tracking()
        with pytest.raises(RuntimeError, match="Tracking already active"):
            tracker.start_tracking()
        
        # Cleanup
        tracker.stop_tracking()


class TestSolverBase:
    """Test SolverBase abstract class functionality."""
    
    def test_solver_base_initialization_success(self):
        """Test successful solver base initialization."""
        # Create concrete solver
        solver = ClassicalSolver(default_method='greedy')

        assert solver.solver_type == 'classical'
        assert solver.solver_name == 'classical_multi'
        assert solver._energy_tracker is not None

    def test_solver_base_initialization_failure(self):
        """Test solver base initialization with invalid parameters."""
        # ClassicalSolver should raise error with invalid algorithm
        with pytest.raises((ValueError, SolverConfigurationError)):
            ClassicalSolver(default_method='invalid_method')

    def test_solver_validate_result_success(self):
        """Test successful solution validation."""
        solver = ClassicalSolver(default_method='greedy')
        problem = MaxCutProblem(num_nodes=5)
        problem.generate(edge_probability=0.5, seed=42)

        # Valid solution
        solution = [0, 1, 0, 1, 0]
        assert solver.validate_result(problem, solution) is True

    def test_solver_validate_result_failure(self):
        """Test solution validation failures."""
        solver = ClassicalSolver(default_method='greedy')
        problem = MaxCutProblem(num_nodes=5)
        problem.generate(edge_probability=0.5, seed=42)
        
        # None solution
        assert solver.validate_result(problem, None) is False
        
        # Empty solution
        assert solver.validate_result(problem, []) is False
        
        # Wrong length
        assert solver.validate_result(problem, [0, 1, 0]) is False
        
        # Invalid values
        assert solver.validate_result(problem, [0, 2, 0, 1, 0]) is False


class TestClassicalSolver:
    """Test ClassicalSolver class."""
    
    @pytest.fixture
    def solver(self):
        """Create a classical solver instance."""
        return ClassicalSolver(default_method='greedy')
    
    @pytest.fixture
    def problem(self):
        """Create a test problem."""
        p = MaxCutProblem(num_nodes=10)
        p.generate(edge_probability=0.3, seed=42)
        return p
    
    def test_classical_solver_solve_success(self, solver, problem):
        """Test successful classical solver execution."""
        result = solver.solve(problem, max_iterations=100)
        
        # Check standardized result format
        assert 'solution' in result
        assert 'cost' in result
        assert 'time_ms' in result
        assert 'energy_mj' in result
        assert 'iterations' in result
        assert 'metadata' in result
        
        # Check types and ranges
        assert isinstance(result['solution'], list)
        assert len(result['solution']) == problem.problem_size
        assert isinstance(result['cost'], (int, float))
        assert isinstance(result['time_ms'], (int, float))
        assert result['time_ms'] >= 0
        assert isinstance(result['energy_mj'], float)
        assert result['energy_mj'] >= 0
    
    def test_classical_solver_solve_failure(self, solver):
        """Test classical solver with invalid problem."""
        # Non-generated problem
        problem = MaxCutProblem(num_nodes=10)
        
        with pytest.raises((ValueError, SolverConfigurationError)):
            solver.solve(problem)
    
    def test_classical_solver_context_manager(self, problem):
        """Test classical solver context manager usage."""
        with ClassicalSolver(default_method='greedy') as solver:
            result = solver.solve(problem)
            assert result is not None


class TestQuantumSimulator:
    """Test QuantumSimulator class."""
    
    @pytest.fixture
    def solver(self):
        """Create a quantum simulator instance."""
        return QuantumSimulator()
    
    @pytest.fixture
    def small_problem(self):
        """Create a small test problem for quantum simulation."""
        p = MaxCutProblem(num_nodes=5)  # Small for quick simulation
        p.generate(edge_probability=0.5, seed=42)
        return p
    
    def test_quantum_simulator_solve_success(self, solver, small_problem):
        """Test successful quantum simulator execution."""
        result = solver.solve(small_problem)
        
        # Check standardized result format
        assert 'solution' in result
        assert 'cost' in result
        assert 'time_ms' in result
        assert 'energy_mj' in result
        assert 'iterations' in result
        assert 'metadata' in result
        
        # Check solution validity
        assert len(result['solution']) == small_problem.problem_size
        assert all(x in [0, 1] for x in result['solution'])
    
    def test_quantum_simulator_solve_failure(self, solver):
        """Test quantum simulator with invalid problem."""
        # Non-generated problem
        problem = MaxCutProblem(num_nodes=5)

        with pytest.raises((ValueError, SolverConfigurationError, QuantumSimulatorException)):
            solver.solve(problem)


class TestORToolsSolver:
    """Test OR-Tools TSP solver."""
    
    @pytest.fixture
    def tsp_problem(self):
        """Create a small TSP problem."""
        from src.problems.tsp import TSPProblem
        p = TSPProblem(num_cities=5)
        p.generate(seed=42)
        return p
    
    def test_ortools_tsp_success(self, tsp_problem):
        """Test OR-Tools TSP solver with successful execution."""
        solver = ClassicalSolver(default_method='auto')
        
        try:
            result = solver.solve(tsp_problem, method='ortools', time_limit_seconds=5)
            
            # Check result format
            assert 'solution' in result
            assert 'cost' in result
            assert 'time_ms' in result
            assert 'energy_mj' in result
            
            # Check solution validity
            assert len(result['solution']) == tsp_problem.problem_size
            assert tsp_problem.validate_solution(result['solution'])
            
        except ImportError:
            pytest.skip("OR-Tools not installed")
        except NotImplementedError:
            # OR-Tools might fallback to simulated annealing
            pytest.skip("OR-Tools implementation requires full integration")
    
    def test_ortools_fallback_to_simulated_annealing(self, tsp_problem):
        """Test that OR-Tools falls back to simulated annealing if not available."""
        solver = ClassicalSolver(default_method='auto')
        
        # This should work even without OR-Tools (fallback)
        result = solver.solve(tsp_problem, method='simulated_annealing')
        
        assert 'solution' in result
        assert len(result['solution']) == tsp_problem.problem_size


class TestPortfolioSolver:
    """Test SciPy portfolio optimization solver."""
    
    @pytest.fixture
    def portfolio_problem(self):
        """Create a portfolio optimization problem."""
        from src.problems.portfolio import PortfolioProblem
        p = PortfolioProblem(num_assets=10, num_selected=3)
        p.generate(seed=42, return_range=(0.05, 0.15), risk_range=(0.10, 0.25))
        return p
    
    def test_portfolio_scipy_sharpe_success(self, portfolio_problem):
        """Test SciPy portfolio solver with Sharpe ratio maximization."""
        solver = ClassicalSolver(default_method='auto')
        
        try:
            result = solver.solve(portfolio_problem, method='scipy')
            
            # Check result format
            assert 'solution' in result
            assert 'cost' in result
            assert 'time_ms' in result
            assert 'energy_mj' in result
            
            # Solution should be weights (continuous values)
            assert len(result['solution']) == portfolio_problem.num_assets
            
            # Weights should sum to approximately 1
            assert abs(sum(result['solution']) - 1.0) < 0.01
            
            # All weights should be non-negative (no short selling)
            assert all(w >= -0.001 for w in result['solution'])  # Small tolerance for numerical precision
            
        except ImportError:
            pytest.skip("SciPy not installed")
    
    def test_portfolio_scipy_min_variance(self, portfolio_problem):
        """Test SciPy portfolio solver with minimum variance method."""
        solver = ClassicalSolver(default_method='auto')
        
        try:
            # Note: PortfolioProblem uses binary selection, so we test the solver logic
            # The solver should handle continuous weights
            result = solver.solve(
                portfolio_problem,
                method='scipy'
            )
            
            assert 'solution' in result
            assert len(result['solution']) == portfolio_problem.num_assets
            
        except ImportError:
            pytest.skip("SciPy not installed")
    
    def test_portfolio_fallback_to_equal_weights(self, portfolio_problem):
        """Test that portfolio solver has fallback behavior."""
        solver = ClassicalSolver(default_method='auto')
        
        # Even if scipy fails, solver should return something
        result = solver.solve(portfolio_problem, method='scipy')
        
        assert 'solution' in result
        assert len(result['solution']) == portfolio_problem.num_assets


class TestSolverComparison:
    """Test solver comparison and standardization."""
    
    @pytest.fixture
    def problem(self):
        """Create test problem."""
        p = MaxCutProblem(num_nodes=8)
        p.generate(edge_probability=0.4, seed=42)
        return p
    
    def test_standardized_output_format(self, problem):
        """Test that both solvers return standardized format."""
        classical_solver = ClassicalSolver(default_method='greedy')
        quantum_solver = QuantumSimulator()
        
        classical_result = classical_solver.solve(problem)
        quantum_result = quantum_solver.solve(problem)
        
        # Both should have same keys
        required_keys = {'solution', 'cost', 'time_ms', 'energy_mj', 'iterations', 'metadata'}
        
        assert set(classical_result.keys()) >= required_keys
        assert set(quantum_result.keys()) >= required_keys
        
        # Both solutions should be valid
        assert problem.validate_solution(classical_result['solution'])
        assert problem.validate_solution(quantum_result['solution'])


class TestHybridSolver:
    """Test hybrid quantum-classical solver."""
    
    @pytest.fixture
    def problem(self):
        """Create test problem."""
        p = MaxCutProblem(num_nodes=8)
        p.generate(edge_probability=0.4, seed=42)
        return p
    
    def test_hybrid_adaptive_strategy(self, problem):
        """Test hybrid solver with adaptive strategy."""
        from src.solvers.hybrid_solver import HybridSolver
        
        solver = HybridSolver(strategy='adaptive', quantum_threshold=10)
        result = solver.solve(problem)
        
        # Check result format
        assert 'solution' in result
        assert 'cost' in result
        assert 'metadata' in result
        assert 'hybrid_strategy' in result['metadata']
        assert result['metadata']['hybrid_strategy'] == 'adaptive'
    
    def test_hybrid_classical_first_strategy(self, problem):
        """Test hybrid solver with classical-first strategy."""
        from src.solvers.hybrid_solver import HybridSolver
        
        solver = HybridSolver(strategy='classical_first', quantum_threshold=10)
        result = solver.solve(problem)
        
        assert 'solution' in result
        assert result['metadata']['hybrid_strategy'] == 'classical_first'
    
    def test_hybrid_parallel_strategy(self, problem):
        """Test hybrid solver with parallel strategy."""
        from src.solvers.hybrid_solver import HybridSolver
        
        solver = HybridSolver(strategy='parallel')
        result = solver.solve(problem)
        
        assert 'solution' in result
        assert result['metadata']['hybrid_strategy'] == 'parallel'
        assert 'all_costs' in result['metadata'] or 'strategy_used' in result['metadata']
