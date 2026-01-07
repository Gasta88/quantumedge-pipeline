"""
Unit tests for analyzer module (minimal coverage with success/failure cases).
"""

import pytest
import numpy as np
import networkx as nx

# Import modules to test
from src.analyzer.problem_analyzer import ProblemAnalyzer
from src.problems.maxcut import MaxCutProblem
from src.problems.tsp import TSPProblem


class TestProblemAnalyzer:
    """Test ProblemAnalyzer class."""
    
    def test_analyzer_initialization_success(self):
        """Test successful analyzer initialization."""
        analyzer = ProblemAnalyzer()
        
        assert analyzer is not None
    
    def test_analyze_problem_success(self):
        """Test successful problem analysis."""
        analyzer = ProblemAnalyzer()
        problem = MaxCutProblem(num_nodes=20)
        problem.generate(edge_probability=0.3, seed=42)
        
        analysis = analyzer.analyze_problem(problem)
        
        # Check required fields
        assert 'problem_type' in analysis
        assert 'problem_size' in analysis
        assert 'complexity_estimate' in analysis
        assert 'graph_features' in analysis
        assert 'suitability_scores' in analysis
        assert 'estimated_classical_runtime' in analysis
        assert 'estimated_quantum_runtime' in analysis
        assert 'quantum_advantage_probability' in analysis
        
        # Check values
        assert analysis['problem_type'] == 'maxcut'
        assert analysis['problem_size'] == 20
        assert analysis['complexity_estimate'] in ['small', 'medium', 'large', 'very_large']
        assert isinstance(analysis['estimated_classical_runtime'], float)
        assert isinstance(analysis['estimated_quantum_runtime'], float)
        assert 0.0 <= analysis['quantum_advantage_probability'] <= 1.0
    
    def test_analyze_problem_failure(self):
        """Test problem analysis with non-generated problem."""
        analyzer = ProblemAnalyzer()
        problem = MaxCutProblem(num_nodes=20)
        # Don't generate the problem
        
        with pytest.raises(ValueError, match="Problem must be generated"):
            analyzer.analyze_problem(problem)


class TestRuntimeEstimation:
    """Test runtime estimation methods."""
    
    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance."""
        return ProblemAnalyzer()
    
    @pytest.fixture
    def small_problem(self):
        """Create small problem."""
        p = MaxCutProblem(num_nodes=10)
        p.generate(edge_probability=0.3, seed=42)
        return p
    
    @pytest.fixture
    def medium_problem(self):
        """Create medium problem."""
        p = MaxCutProblem(num_nodes=30)
        p.generate(edge_probability=0.3, seed=42)
        return p
    
    def test_estimate_classical_runtime_success(self, analyzer, medium_problem):
        """Test successful classical runtime estimation."""
        runtime = analyzer.estimate_classical_runtime(medium_problem)
        
        assert isinstance(runtime, float)
        assert runtime > 0
        assert runtime < 1000.0  # Should be reasonable
    
    def test_estimate_classical_runtime_failure(self, analyzer):
        """Test classical runtime estimation with non-generated problem."""
        problem = MaxCutProblem(num_nodes=20)
        
        with pytest.raises(ValueError, match="Problem must be generated"):
            analyzer.estimate_classical_runtime(problem)
    
    def test_estimate_quantum_runtime_success(self, analyzer, medium_problem):
        """Test successful quantum runtime estimation."""
        runtime = analyzer.estimate_quantum_runtime(medium_problem)
        
        assert isinstance(runtime, float)
        assert runtime > 0
    
    def test_estimate_quantum_runtime_failure(self, analyzer):
        """Test quantum runtime estimation with non-generated problem."""
        problem = MaxCutProblem(num_nodes=20)
        
        with pytest.raises(ValueError, match="Problem must be generated"):
            analyzer.estimate_quantum_runtime(problem)


class TestQuantumAdvantage:
    """Test quantum advantage prediction."""
    
    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance."""
        return ProblemAnalyzer()
    
    def test_predict_quantum_advantage_success(self, analyzer):
        """Test successful quantum advantage prediction."""
        # Medium-sized problem (sweet spot)
        problem = MaxCutProblem(num_nodes=25)
        problem.generate(edge_probability=0.3, seed=42)
        
        prob = analyzer.predict_quantum_advantage(problem)
        
        assert isinstance(prob, float)
        assert 0.0 <= prob <= 1.0
    
    def test_predict_quantum_advantage_small_problem(self, analyzer):
        """Test quantum advantage for small problem (should be low)."""
        # Small problem - classical should be preferred
        problem = MaxCutProblem(num_nodes=5)
        problem.generate(edge_probability=0.3, seed=42)
        
        prob = analyzer.predict_quantum_advantage(problem)
        
        # Small problems should have lower quantum advantage
        assert prob < 0.7
    
    def test_predict_quantum_advantage_failure(self, analyzer):
        """Test quantum advantage prediction with non-generated problem."""
        problem = MaxCutProblem(num_nodes=20)
        
        with pytest.raises(ValueError, match="Problem must be generated"):
            analyzer.predict_quantum_advantage(problem)


class TestProblemHardness:
    """Test problem hardness calculation."""
    
    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance."""
        return ProblemAnalyzer()
    
    def test_calculate_problem_hardness_success(self, analyzer):
        """Test successful hardness calculation."""
        problem = MaxCutProblem(num_nodes=15)
        problem.generate(edge_probability=0.3, seed=42)
        
        hardness = analyzer.calculate_problem_hardness(problem)
        
        assert hardness in ['easy', 'medium', 'hard']
    
    def test_calculate_problem_hardness_small_problem(self, analyzer):
        """Test hardness for small problem (should be easy)."""
        problem = MaxCutProblem(num_nodes=8)
        problem.generate(edge_probability=0.3, seed=42)
        
        hardness = analyzer.calculate_problem_hardness(problem)
        
        # Small problems should be easy
        assert hardness == 'easy'
    
    def test_calculate_problem_hardness_large_problem(self, analyzer):
        """Test hardness for large problem (should be hard)."""
        problem = MaxCutProblem(num_nodes=80)
        problem.generate(edge_probability=0.3, seed=42)
        
        hardness = analyzer.calculate_problem_hardness(problem)
        
        # Large problems should be medium or hard
        assert hardness in ['medium', 'hard']


class TestSolutionSpaceEstimation:
    """Test solution space size estimation."""
    
    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance."""
        return ProblemAnalyzer()
    
    def test_estimate_solution_space_size_success(self, analyzer):
        """Test successful solution space estimation."""
        problem = MaxCutProblem(num_nodes=10)
        problem.generate(edge_probability=0.3, seed=42)
        
        space_size = analyzer.estimate_solution_space_size(problem)
        
        # Binary problem: should be 2^10 = 1024
        assert space_size == 2**10
    
    def test_estimate_solution_space_size_tsp(self, analyzer):
        """Test solution space estimation for TSP."""
        problem = TSPProblem(num_cities=5)
        problem.generate(euclidean=True, seed=42)
        
        space_size = analyzer.estimate_solution_space_size(problem)
        
        # TSP: (n-1)!/2 for n cities
        # For 5 cities: 4!/2 = 24/2 = 12
        assert space_size == 12
    
    def test_estimate_solution_space_size_failure(self, analyzer):
        """Test solution space estimation with non-generated problem."""
        problem = MaxCutProblem(num_nodes=10)
        
        with pytest.raises(ValueError, match="Problem must be generated"):
            analyzer.estimate_solution_space_size(problem)


class TestStructureIdentification:
    """Test problem structure identification."""
    
    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance."""
        return ProblemAnalyzer()
    
    def test_identify_problem_structure_success(self, analyzer):
        """Test successful structure identification."""
        problem = MaxCutProblem(num_nodes=15)
        problem.generate(edge_probability=0.3, seed=42)
        
        structure = analyzer.identify_problem_structure(problem)
        
        # Check required fields
        assert 'is_planar' in structure
        assert 'is_bipartite' in structure
        assert 'is_tree' in structure
        assert 'is_regular' in structure
        assert 'has_communities' in structure
        assert 'num_components' in structure
        assert 'structure_notes' in structure
        
        # Check types
        assert isinstance(structure['is_planar'], bool)
        assert isinstance(structure['is_bipartite'], bool)
        assert isinstance(structure['num_components'], int)
        assert isinstance(structure['structure_notes'], list)
    
    def test_identify_problem_structure_failure(self, analyzer):
        """Test structure identification with non-generated problem."""
        problem = MaxCutProblem(num_nodes=15)
        
        with pytest.raises(ValueError, match="Problem must be generated"):
            analyzer.identify_problem_structure(problem)


class TestAnalysisReport:
    """Test analysis report generation."""
    
    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance."""
        return ProblemAnalyzer()
    
    @pytest.fixture
    def problem(self):
        """Create test problem."""
        p = MaxCutProblem(num_nodes=20)
        p.generate(edge_probability=0.3, seed=42)
        return p
    
    def test_generate_analysis_report_success(self, analyzer, problem):
        """Test successful report generation."""
        report = analyzer.generate_analysis_report(problem)
        
        # Check report is string and contains key sections
        assert isinstance(report, str)
        assert len(report) > 0
        assert 'PROBLEM ANALYSIS REPORT' in report
        assert 'PROBLEM OVERVIEW' in report
        assert 'SOLUTION SPACE ANALYSIS' in report
        assert 'RECOMMENDATIONS' in report
    
    def test_generate_analysis_report_failure(self, analyzer):
        """Test report generation with non-generated problem."""
        problem = MaxCutProblem(num_nodes=20)
        
        with pytest.raises(ValueError, match="Problem must be generated"):
            analyzer.generate_analysis_report(problem)
