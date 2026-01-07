"""
Unit tests for problems module (minimal coverage with success/failure cases).
"""

import pytest
import numpy as np
import networkx as nx

# Import modules to test
from src.problems.problem_base import ProblemBase
from src.problems.maxcut import MaxCutProblem


class TestProblemBase:
    """Test ProblemBase abstract class properties."""
    
    def test_problem_base_properties_success(self):
        """Test successful problem base property access."""
        # Create a concrete problem instance
        problem = MaxCutProblem(num_nodes=10)
        
        # Test properties before generation
        assert problem.problem_type == 'maxcut'
        assert problem.problem_size == 10
        assert problem.complexity_class == 'NP-hard'
        assert problem.is_generated is False
    
    def test_problem_base_validation_failure(self):
        """Test problem validation fails when not generated."""
        problem = MaxCutProblem(num_nodes=10)
        
        # Should raise error when accessing methods before generation
        with pytest.raises(ValueError, match="Problem must be generated"):
            problem.validate_solution([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])


class TestMaxCutProblem:
    """Test MaxCutProblem class."""
    
    def test_maxcut_generation_success(self):
        """Test successful MaxCut problem generation."""
        problem = MaxCutProblem(num_nodes=10)
        problem.generate(edge_probability=0.3, seed=42)
        
        assert problem.is_generated is True
        assert problem.problem_size == 10
        assert problem.problem_type == 'maxcut'
        
        # Check graph was created
        graph = problem.to_graph()
        assert graph.number_of_nodes() == 10
        assert graph.number_of_edges() > 0
    
    def test_maxcut_generation_failure(self):
        """Test MaxCut problem generation with invalid parameters."""
        problem = MaxCutProblem(num_nodes=10)
        
        # Invalid edge probability (negative)
        with pytest.raises(ValueError):
            problem.generate(edge_probability=-0.1, seed=42)
        
        # Invalid edge probability (> 1)
        with pytest.raises(ValueError):
            problem.generate(edge_probability=1.5, seed=42)


class TestMaxCutSolution:
    """Test MaxCut solution validation and cost calculation."""
    
    @pytest.fixture
    def generated_problem(self):
        """Create a generated MaxCut problem for testing."""
        problem = MaxCutProblem(num_nodes=5)
        problem.generate(edge_probability=0.5, seed=42)
        return problem
    
    def test_solution_validation_success(self, generated_problem):
        """Test successful solution validation."""
        # Valid binary solution
        solution = [0, 1, 0, 1, 0]
        assert generated_problem.validate_solution(solution) is True
    
    def test_solution_validation_failure(self, generated_problem):
        """Test solution validation with invalid solutions."""
        # Wrong length
        wrong_length = [0, 1, 0]
        assert generated_problem.validate_solution(wrong_length) is False
        
        # Non-binary values
        non_binary = [0, 1, 2, 0, 1]
        assert generated_problem.validate_solution(non_binary) is False
        
        # None solution
        assert generated_problem.validate_solution(None) is False
    
    def test_cost_calculation_success(self, generated_problem):
        """Test successful cost calculation."""
        solution = [0, 1, 0, 1, 0]
        cost = generated_problem.calculate_cost(solution)
        
        # Cost should be a finite number (negative because MaxCut maximizes)
        assert isinstance(cost, (int, float))
        assert np.isfinite(cost)
    
    def test_cost_calculation_failure(self, generated_problem):
        """Test cost calculation with invalid solution."""
        # Invalid solution should raise error
        invalid_solution = [0, 2, 0, 1, 0]
        
        with pytest.raises((ValueError, AssertionError)):
            generated_problem.calculate_cost(invalid_solution)


class TestProblemRepresentations:
    """Test problem representation conversions."""
    
    @pytest.fixture
    def problem(self):
        """Create a small test problem."""
        p = MaxCutProblem(num_nodes=4)
        p.generate(edge_probability=0.5, seed=42)
        return p
    
    def test_to_graph_success(self, problem):
        """Test successful graph conversion."""
        graph = problem.to_graph()
        
        assert isinstance(graph, nx.Graph)
        assert graph.number_of_nodes() == 4
        assert graph.number_of_edges() >= 0
    
    def test_to_qubo_success(self, problem):
        """Test successful QUBO conversion."""
        qubo = problem.to_qubo()
        
        assert isinstance(qubo, np.ndarray)
        assert qubo.shape == (4, 4)
        # Check it's upper triangular
        assert np.allclose(qubo, np.triu(qubo))
    
    def test_get_metadata_success(self, problem):
        """Test successful metadata retrieval."""
        metadata = problem.get_metadata()
        
        assert isinstance(metadata, dict)
        assert 'problem_type' in metadata
        assert 'problem_size' in metadata
        assert 'complexity_class' in metadata
        assert metadata['problem_type'] == 'maxcut'
        assert metadata['problem_size'] == 4
    
    def test_representations_failure(self):
        """Test representation conversions fail without generation."""
        problem = MaxCutProblem(num_nodes=4)
        
        # Should fail when not generated
        with pytest.raises(ValueError):
            problem.to_graph()
        
        with pytest.raises(ValueError):
            problem.to_qubo()
        
        with pytest.raises(ValueError):
            problem.get_metadata()
