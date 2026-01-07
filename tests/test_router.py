"""
Unit tests for router module (minimal coverage with success/failure cases).
"""

import pytest
from unittest.mock import Mock, MagicMock, patch

# Import modules to test
from src.router.quantum_router import QuantumRouter, RoutingStrategy, RoutingPreferences
from src.router.edge_simulator import EdgeEnvironment, DeploymentProfile
from src.problems.maxcut import MaxCutProblem


class TestQuantumRouter:
    """Test QuantumRouter class."""
    
    def test_router_initialization_success(self):
        """Test successful router initialization."""
        router = QuantumRouter(
            strategy=RoutingStrategy.BALANCED,
            enable_learning=True
        )
        
        assert router.current_strategy == RoutingStrategy.BALANCED
        assert router.learning_enabled is True
        assert router.analyzer is not None
        assert router.execution_history == []
        assert router.performance_cache == {}
    
    def test_router_set_strategy_success(self):
        """Test successful strategy change."""
        router = QuantumRouter(strategy=RoutingStrategy.BALANCED)
        
        # Change strategy
        router.set_strategy(RoutingStrategy.ENERGY_OPTIMIZED)
        
        assert router.current_strategy == RoutingStrategy.ENERGY_OPTIMIZED
    
    def test_router_initialization_failure(self):
        """Test router with invalid strategy type."""
        # Should work with valid enum
        router = QuantumRouter(strategy=RoutingStrategy.BALANCED)
        assert router is not None
        
        # Invalid type would be caught by type system
        # But we can test that only valid enums are accepted
        with pytest.raises(AttributeError):
            # Trying to use non-existent strategy
            invalid = RoutingStrategy.INVALID_STRATEGY


class TestRoutingDecisions:
    """Test routing decision logic."""
    
    @pytest.fixture
    def router(self):
        """Create a router instance for testing."""
        return QuantumRouter(strategy=RoutingStrategy.BALANCED)
    
    @pytest.fixture
    def small_problem(self):
        """Create a small MaxCut problem."""
        problem = MaxCutProblem(num_nodes=5)
        problem.generate(edge_probability=0.3, seed=42)
        return problem
    
    @pytest.fixture
    def medium_problem(self):
        """Create a medium MaxCut problem."""
        problem = MaxCutProblem(num_nodes=30)
        problem.generate(edge_probability=0.3, seed=42)
        return problem
    
    @pytest.fixture
    def edge_env(self):
        """Create an edge environment."""
        return EdgeEnvironment(DeploymentProfile.AEROSPACE)
    
    def test_route_problem_success(self, router, medium_problem, edge_env):
        """Test successful problem routing."""
        result = router.route_problem(
            problem=medium_problem,
            edge_env=edge_env,
            preferences=None
        )
        
        # Check result structure
        assert 'decision' in result
        assert 'reasoning' in result
        assert 'confidence' in result
        assert 'estimated_time_ms' in result
        assert 'estimated_energy_mj' in result
        
        # Check decision is valid
        assert result['decision'] in ['classical', 'quantum', 'hybrid']
        assert 0.0 <= result['confidence'] <= 1.0
        assert result['estimated_time_ms'] > 0
        assert result['estimated_energy_mj'] > 0
    
    def test_route_problem_failure_not_generated(self, router, edge_env):
        """Test routing fails with non-generated problem."""
        problem = MaxCutProblem(num_nodes=10)
        # Don't generate the problem
        
        with pytest.raises(ValueError, match="Problem must be generated"):
            router.route_problem(problem, edge_env)
    
    def test_route_small_problem_prefers_classical(self, router, small_problem, edge_env):
        """Test that small problems are routed to classical."""
        result = router.route_problem(small_problem, edge_env)
        
        # Small problems should prefer classical
        assert result['decision'] == 'classical'
        assert result['confidence'] > 0.9  # High confidence


class TestStrategyScoring:
    """Test strategy-based scoring."""
    
    @pytest.fixture
    def router(self):
        """Create router for testing."""
        return QuantumRouter(strategy=RoutingStrategy.BALANCED)
    
    def test_calculate_strategy_score_success(self, router):
        """Test successful strategy score calculation."""
        classical_est = {
            'runtime_s': 5.0,
            'energy_mj': 150.0,
            'fits_resources': True
        }
        
        quantum_est = {
            'runtime_s': 3.0,
            'energy_mj': 120.0,
            'fits_resources': True,
            'quantum_advantage_prob': 0.65
        }
        
        decision, score_diff, reasoning = router.calculate_strategy_score(
            classical_est,
            quantum_est,
            RoutingStrategy.BALANCED
        )
        
        # Should return valid decision
        assert decision in ['classical', 'quantum']
        assert isinstance(score_diff, float)
        assert isinstance(reasoning, str)
        assert len(reasoning) > 0
    
    def test_calculate_strategy_score_resource_constrained(self, router):
        """Test strategy scoring with resource constraints."""
        classical_est = {
            'runtime_s': 5.0,
            'energy_mj': 150.0,
            'fits_resources': True
        }
        
        quantum_est = {
            'runtime_s': 3.0,
            'energy_mj': 120.0,
            'fits_resources': False,  # Doesn't fit resources
            'quantum_advantage_prob': 0.65
        }
        
        decision, score_diff, reasoning = router.calculate_strategy_score(
            classical_est,
            quantum_est
        )
        
        # Should choose classical when quantum doesn't fit
        assert decision == 'classical'
        assert 'resource constraints' in reasoning.lower()


class TestRoutingExplanation:
    """Test routing decision explanation generation."""
    
    @pytest.fixture
    def sample_routing_result(self):
        """Create a sample routing result."""
        return {
            'decision': 'quantum',
            'reasoning': 'Quantum solver predicted to have advantage',
            'confidence': 0.75,
            'estimated_time_ms': 2500,
            'estimated_energy_mj': 120.5,
            'strategy_used': 'balanced',
            'alternative_options': [
                {'option': 'classical', 'feasible': True, 'note': 'Would work but slower'}
            ],
            'problem_analysis': {
                'problem_type': 'maxcut',
                'problem_size': 30,
                'complexity': 'medium',
                'quantum_advantage_probability': 0.65,
                'suitability_scores': {
                    'classical_score': 0.6,
                    'quantum_score': 0.75
                }
            },
            'resource_constraints': {
                'environment_profile': 'aerospace',
                'classical_fits': True,
                'quantum_fits': True,
                'power_budget_watts': 50.0,
                'memory_limit_mb': 2048,
                'timeout_seconds': 60.0
            },
            'performance_predictions': {
                'classical_runtime_seconds': 5.0,
                'quantum_runtime_seconds': 2.5,
                'classical_energy_mj': 150.0,
                'quantum_energy_mj': 120.5,
                'quantum_faster': True,
                'quantum_more_efficient': True
            }
        }
    
    def test_explain_decision_success(self, sample_routing_result):
        """Test successful explanation generation."""
        router = QuantumRouter()
        
        explanation = router.explain_decision(sample_routing_result)
        
        # Check explanation contains key sections
        assert isinstance(explanation, str)
        assert len(explanation) > 0
        assert 'ROUTING DECISION EXPLANATION' in explanation
        assert 'DECISION: QUANTUM' in explanation
        assert 'CONFIDENCE:' in explanation
        assert 'REASONING:' in explanation
        assert 'PROBLEM ANALYSIS:' in explanation
    
    def test_suggest_alternatives_success(self, sample_routing_result):
        """Test successful alternative suggestions."""
        router = QuantumRouter()
        
        suggestions = router.suggest_alternatives(sample_routing_result)
        
        # Should return list of suggestions
        assert isinstance(suggestions, list)
        assert len(suggestions) > 0
        
        # Each suggestion should have required fields
        for suggestion in suggestions:
            assert 'suggestion_type' in suggestion
            assert 'description' in suggestion
            assert 'feasibility' in suggestion
            assert suggestion['feasibility'] in ['easy', 'moderate', 'difficult']
