"""
Problem Analyzer for QuantumEdge Pipeline.

This module provides comprehensive analysis of optimization problems to support
intelligent routing decisions between classical and quantum solvers. The analyzer
extracts problem features, estimates computational requirements, and predicts
quantum advantage potential.

Key Capabilities
----------------
1. **Feature Extraction**: Analyzes problem structure and characteristics
2. **Runtime Estimation**: Predicts classical and quantum solver performance
3. **Quantum Advantage Prediction**: Estimates likelihood of quantum speedup
4. **Graph Analysis**: Computes graph-theoretic properties for graph-based problems

The analyzer helps the router make informed decisions by quantifying:
- Problem complexity and size
- Structural properties (density, clustering, connectivity)
- Computational requirements for different solver types
- Expected performance gains from quantum approaches

Example Usage
-------------
```python
from src.problems.maxcut import MaxCutProblem
from src.analyzer.problem_analyzer import ProblemAnalyzer

# Create and generate problem
problem = MaxCutProblem(num_nodes=50)
problem.generate(edge_probability=0.2)

# Analyze problem characteristics
analyzer = ProblemAnalyzer()
analysis = analyzer.analyze_problem(problem)

# Check results
print(f"Problem size: {analysis['problem_size']}")
print(f"Graph density: {analysis['graph_features']['density']:.3f}")
print(f"Classical score: {analysis['suitability_scores']['classical_score']:.2f}")
print(f"Quantum score: {analysis['suitability_scores']['quantum_score']:.2f}")
print(f"Quantum advantage probability: {analyzer.predict_quantum_advantage(problem):.2%}")

# Use for routing decision
if analysis['suitability_scores']['quantum_score'] > 0.6:
    print("Route to quantum solver")
else:
    print("Route to classical solver")
```

Design Philosophy
-----------------
The analyzer uses heuristic-based rules for now, designed to be easily
upgraded to ML-based predictions later. All analysis methods are modular
and can be enhanced independently without breaking the interface.
"""

from typing import Dict, Any, Optional
import logging
import numpy as np
import networkx as nx

from src.problems.problem_base import ProblemBase


# Configure module logger
logger = logging.getLogger(__name__)


class ProblemAnalyzer:
    """
    Analyzes optimization problems to extract features and predict solver performance.
    
    This class provides comprehensive problem analysis to support intelligent routing
    decisions. It extracts structural features, estimates computational requirements,
    and predicts quantum advantage potential using heuristic rules.
    
    The analyzer is designed to work with any problem implementing the ProblemBase
    interface, making it problem-agnostic and easily extensible.
    
    Attributes:
        None (stateless design - all methods take problem as input)
    
    Thread Safety:
        This class is stateless and thread-safe. Multiple threads can share
        a single ProblemAnalyzer instance.
    
    Performance Notes:
        - Graph analysis (clustering, diameter) can be expensive for large graphs
        - Results are computed fresh on each call (no caching)
        - For repeated analysis, consider caching results externally
    
    Future Enhancements:
        - Add ML-based runtime prediction models
        - Incorporate historical solver performance data
        - Support custom feature extractors
        - Add problem similarity metrics
    """
    
    def __init__(self):
        """
        Initialize the problem analyzer.
        
        The analyzer is stateless by design - all analysis is performed
        based on the problem instance passed to analysis methods.
        """
        logger.info("ProblemAnalyzer initialized")
    
    def analyze_problem(self, problem: ProblemBase) -> Dict[str, Any]:
        """
        Extract comprehensive features from a problem instance.
        
        This is the main entry point for problem analysis. It extracts:
        - Basic problem properties (type, size, complexity)
        - Graph-theoretic features (for graph-based problems)
        - Suitability scores for classical vs quantum solvers
        - Runtime estimates
        
        The analysis results can be used by the router to make informed
        solver selection decisions.
        
        Args:
            problem: A problem instance implementing ProblemBase interface.
                    Must be generated (problem.is_generated == True).
        
        Returns:
            Dictionary containing:
            {
                'problem_type': str,           # 'maxcut', 'tsp', 'portfolio'
                'problem_size': int,           # Number of variables
                'complexity_estimate': str,    # 'small', 'medium', 'large', 'very_large'
                'graph_features': {            # Only for graph-based problems
                    'density': float,          # 0-1, edge density
                    'clustering_coefficient': float,  # 0-1, transitivity
                    'average_degree': float,   # Average node degree
                    'diameter': int or None    # Graph diameter (None if too expensive)
                },
                'suitability_scores': {
                    'classical_score': float,  # 0-1, classical solver suitability
                    'quantum_score': float     # 0-1, quantum solver suitability
                },
                'estimated_classical_runtime': float,  # seconds
                'estimated_quantum_runtime': float,    # seconds
                'quantum_advantage_probability': float  # 0-1, P(quantum faster)
            }
        
        Raises:
            ValueError: If problem is not generated
            TypeError: If problem doesn't implement ProblemBase
        
        Example:
            >>> from src.problems.maxcut import MaxCutProblem
            >>> problem = MaxCutProblem(num_nodes=30)
            >>> problem.generate(edge_probability=0.3)
            >>> 
            >>> analyzer = ProblemAnalyzer()
            >>> analysis = analyzer.analyze_problem(problem)
            >>> 
            >>> print(f"Type: {analysis['problem_type']}")
            >>> print(f"Size: {analysis['problem_size']}")
            >>> print(f"Density: {analysis['graph_features']['density']:.2f}")
            >>> print(f"Classical score: {analysis['suitability_scores']['classical_score']:.2f}")
        
        Notes:
            - Graph diameter computation is skipped for graphs with >100 nodes (too expensive)
            - All scores are normalized to [0, 1] range
            - Runtime estimates are rough approximations, not precise predictions
        """
        # Validate input
        if not isinstance(problem, ProblemBase):
            raise TypeError(f"Problem must implement ProblemBase, got {type(problem)}")
        
        if not problem.is_generated:
            raise ValueError("Problem must be generated before analysis. Call problem.generate() first.")
        
        logger.info(f"Analyzing problem: {problem}")
        
        # Extract basic properties
        problem_type = problem.problem_type
        problem_size = problem.problem_size
        complexity_class = problem.complexity_class
        
        # Categorize problem size for routing decisions
        complexity_estimate = self._estimate_complexity_category(problem_size)
        
        logger.debug(f"Basic properties - type: {problem_type}, size: {problem_size}, "
                    f"complexity: {complexity_estimate}")
        
        # Extract graph features (if applicable)
        graph_features = {}
        try:
            graph = problem.to_graph()
            graph_features = self._extract_graph_features(graph)
            logger.debug(f"Graph features extracted: density={graph_features.get('density', 0):.3f}")
        except Exception as e:
            logger.warning(f"Could not extract graph features: {e}")
            graph_features = None
        
        # Calculate suitability scores
        suitability_scores = self._calculate_suitability_scores(
            problem_type=problem_type,
            problem_size=problem_size,
            complexity_class=complexity_class,
            graph_features=graph_features
        )
        
        logger.debug(f"Suitability scores - classical: {suitability_scores['classical_score']:.2f}, "
                    f"quantum: {suitability_scores['quantum_score']:.2f}")
        
        # Estimate runtimes
        classical_runtime = self.estimate_classical_runtime(problem)
        quantum_runtime = self.estimate_quantum_runtime(problem)
        
        logger.debug(f"Runtime estimates - classical: {classical_runtime:.3f}s, "
                    f"quantum: {quantum_runtime:.3f}s")
        
        # Predict quantum advantage
        quantum_advantage_prob = self.predict_quantum_advantage(problem)
        
        logger.info(f"Analysis complete - quantum advantage probability: {quantum_advantage_prob:.2%}")
        
        # Compile results
        analysis = {
            'problem_type': problem_type,
            'problem_size': problem_size,
            'complexity_estimate': complexity_estimate,
            'graph_features': graph_features,
            'suitability_scores': suitability_scores,
            'estimated_classical_runtime': classical_runtime,
            'estimated_quantum_runtime': quantum_runtime,
            'quantum_advantage_probability': quantum_advantage_prob
        }
        
        return analysis
    
    def estimate_classical_runtime(self, problem: ProblemBase) -> float:
        """
        Estimate classical solver runtime based on problem characteristics.
        
        Uses heuristics based on:
        - Problem type (different algorithms have different complexities)
        - Problem size (scaling behavior)
        - Historical benchmarks (rough averages)
        
        These estimates are intentionally conservative (upper bounds) to
        avoid underestimating runtime and making poor routing decisions.
        
        Complexity Assumptions:
        -----------------------
        MaxCut:
            - Small (n≤20): Exact solver possible, O(2^n) ~ 0.01-1s
            - Medium (20<n≤50): Approximation algorithms, O(n²) ~ 0.1-10s
            - Large (n>50): Heuristics required, O(n² log n) ~ 1-100s
        
        TSP:
            - Small (n≤15): Exact DP, O(n² 2^n) ~ 0.1-10s
            - Medium (15<n≤50): Branch & bound, O(n!) ~ 10-300s
            - Large (n>50): Heuristics (Christofides), O(n³) ~ 1-100s
        
        Portfolio Optimization:
            - Generally efficient: Convex optimization O(n³) ~ 0.001-1s
            - With constraints: May need MILP, can be slower
        
        Args:
            problem: Problem instance to estimate runtime for
        
        Returns:
            Estimated runtime in seconds (float)
        
        Raises:
            ValueError: If problem is not generated
        
        Example:
            >>> problem = MaxCutProblem(num_nodes=40)
            >>> problem.generate()
            >>> analyzer = ProblemAnalyzer()
            >>> runtime = analyzer.estimate_classical_runtime(problem)
            >>> print(f"Estimated classical runtime: {runtime:.2f}s")
        
        Notes:
            - These are rough estimates, actual runtime can vary significantly
            - Assumes modern hardware (multi-core CPU, sufficient RAM)
            - Does not account for problem structure (all instances treated similarly)
            - Conservative estimates to avoid underestimation
        """
        if not problem.is_generated:
            raise ValueError("Problem must be generated before runtime estimation")
        
        problem_type = problem.problem_type.lower()
        size = problem.problem_size
        
        logger.debug(f"Estimating classical runtime for {problem_type} problem of size {size}")
        
        # Type-specific runtime estimation
        if problem_type == 'maxcut':
            if size <= 20:
                # Small: can solve exactly with branch & bound or SDP
                runtime = 0.01 * (2 ** (size / 10))  # Exponential scaling
            elif size <= 50:
                # Medium: use approximation algorithms (SDP relaxation)
                runtime = 0.001 * (size ** 2)  # Quadratic scaling
            else:
                # Large: use fast heuristics (greedy, local search)
                runtime = 0.001 * size * np.log(size)  # O(n log n)
        
        elif problem_type == 'tsp':
            if size <= 15:
                # Small: exact dynamic programming (Held-Karp)
                runtime = 0.001 * (size ** 2) * (2 ** size)  # O(n² 2^n)
            elif size <= 50:
                # Medium: branch & bound with pruning
                runtime = 1.0 * (size ** 3)  # Pessimistic estimate
            else:
                # Large: heuristics (Christofides, Lin-Kernighan)
                runtime = 0.01 * (size ** 2)  # O(n²) for good heuristics
        
        elif problem_type == 'portfolio':
            # Generally efficient: quadratic programming
            # Scales well even for large portfolios
            runtime = 0.0001 * (size ** 3)  # O(n³) for interior point methods
            
            # Add penalty for highly constrained problems
            if size > 100:
                runtime *= 1.5  # Additional overhead for large-scale problems
        
        else:
            # Unknown problem type: use generic scaling
            logger.warning(f"Unknown problem type '{problem_type}', using generic estimate")
            runtime = 0.01 * (size ** 2)
        
        # Apply minimum runtime (overhead for problem loading, validation)
        runtime = max(runtime, 0.001)
        
        logger.debug(f"Classical runtime estimate: {runtime:.4f}s")
        
        return runtime
    
    def estimate_quantum_runtime(self, problem: ProblemBase) -> float:
        """
        Estimate quantum simulator runtime.
        
        Quantum runtime depends on:
        - Number of qubits required (determines state space size: 2^n)
        - Circuit depth (number of gate layers)
        - Number of shots (measurements for statistics)
        - Simulator overhead (exponential in qubits)
        
        NOTE: This estimates SIMULATOR runtime, not real quantum hardware.
        Real quantum computers would be much faster but have limited qubit counts
        and high error rates currently.
        
        Simulator Complexity:
        ---------------------
        - State vector simulation: O(2^n) memory, O(2^n × depth) time
        - Practical limit: ~25-30 qubits on standard workstation
        - Each additional qubit doubles memory and runtime
        
        QAOA Circuit Structure (typical):
        ----------------------------------
        - Qubits needed: n (one per problem variable)
        - Circuit depth: p × (problem_layers + mixer_layers)
          * p = QAOA parameter layers (typically 1-5)
          * problem_layers ≈ number of QUBO terms / n
          * mixer_layers = n (X rotations)
        - Shots: 1000-10000 for statistical accuracy
        
        Args:
            problem: Problem instance to estimate runtime for
        
        Returns:
            Estimated simulator runtime in seconds (float)
        
        Raises:
            ValueError: If problem is not generated
        
        Example:
            >>> problem = MaxCutProblem(num_nodes=20)
            >>> problem.generate()
            >>> analyzer = ProblemAnalyzer()
            >>> runtime = analyzer.estimate_quantum_runtime(problem)
            >>> print(f"Estimated quantum simulator runtime: {runtime:.2f}s")
        
        Notes:
            - These estimates are for SIMULATION, not real quantum hardware
            - Real quantum computers would be faster but are limited by:
              * Qubit count (current systems: 50-1000 qubits)
              * Error rates (requiring error correction overhead)
              * Queue times and calibration
            - Simulator runtime grows exponentially with qubit count
            - For >30 qubits, simulation becomes impractical on classical hardware
        """
        if not problem.is_generated:
            raise ValueError("Problem must be generated before runtime estimation")
        
        size = problem.problem_size
        
        logger.debug(f"Estimating quantum runtime for problem of size {size}")
        
        # Estimate circuit parameters
        num_qubits = size  # Typically 1 qubit per variable
        
        # Estimate circuit depth based on problem structure
        # QAOA: depth = p × (C + M), where C=problem, M=mixer
        # Assume p=3 layers, C≈n, M=n → depth ≈ 6n
        qaoa_layers = 3
        circuit_depth = qaoa_layers * size * 2
        
        # Number of measurements for statistical accuracy
        num_shots = 1000
        
        logger.debug(f"Circuit parameters - qubits: {num_qubits}, depth: {circuit_depth}, "
                    f"shots: {num_shots}")
        
        # Simulator time estimation
        # Base time per shot: exponential in qubits
        time_per_shot = 0.0001 * (2 ** (num_qubits / 10))  # Grows exponentially
        
        # Total execution time
        execution_time = time_per_shot * circuit_depth * num_shots
        
        # Add compilation overhead (circuit optimization, transpilation)
        compilation_time = 1.0 + (0.1 * num_qubits)
        
        # Total runtime
        runtime = execution_time + compilation_time
        
        # For very large problems, simulation becomes impractical
        if num_qubits > 25:
            # Add penalty to reflect difficulty of simulation
            penalty = 2 ** (num_qubits - 25)
            runtime *= penalty
            logger.warning(f"Large qubit count ({num_qubits}) - simulation may be impractical")
        
        logger.debug(f"Quantum runtime estimate: {runtime:.4f}s "
                    f"(execution: {execution_time:.4f}s, compilation: {compilation_time:.4f}s)")
        
        return runtime
    
    def predict_quantum_advantage(self, problem: ProblemBase) -> float:
        """
        Predict probability that quantum solver will be faster than classical.
        
        Uses heuristic rules based on:
        - Problem type (some problems benefit more from quantum)
        - Problem size (quantum advantage appears in certain size ranges)
        - Problem structure (graph properties affect quantum performance)
        - Historical benchmarks (literature results)
        
        Heuristic Rules:
        ----------------
        1. **Problem Type**:
           - MaxCut, Graph Coloring: High potential (proven QAOA advantage)
           - TSP: Medium potential (mixed results)
           - Portfolio: Low potential (classical convex opt is very efficient)
        
        2. **Problem Size**:
           - Too small (n<10): Classical faster (quantum overhead dominates)
           - Sweet spot (10≤n≤50): Potential quantum advantage
           - Too large (n>50): Simulation impractical, but real QC could help
        
        3. **Graph Structure** (for graph problems):
           - Sparse graphs (density<0.3): Better for quantum (fewer interactions)
           - High clustering: Classical algorithms exploit locality better
           - Regular structure: Quantum can find patterns efficiently
        
        4. **Complexity Class**:
           - NP-hard: Higher quantum potential
           - P: Classical algorithms are optimal
        
        Score Calculation:
        ------------------
        Start with base score, then apply multipliers for each factor.
        Final score is probability quantum will be faster (0-1 scale).
        
        Args:
            problem: Problem instance to predict advantage for
        
        Returns:
            Probability quantum will be faster (0.0 - 1.0)
            - 0.0-0.3: Classical strongly preferred
            - 0.3-0.5: Classical slightly preferred
            - 0.5-0.7: Quantum slightly preferred
            - 0.7-1.0: Quantum strongly preferred
        
        Raises:
            ValueError: If problem is not generated
        
        Example:
            >>> problem = MaxCutProblem(num_nodes=30)
            >>> problem.generate(edge_probability=0.2)
            >>> analyzer = ProblemAnalyzer()
            >>> prob = analyzer.predict_quantum_advantage(problem)
            >>> print(f"Quantum advantage probability: {prob:.2%}")
            >>> 
            >>> if prob > 0.5:
            ...     print("Recommend quantum solver")
            ... else:
            ...     print("Recommend classical solver")
        
        Notes:
            - This is a heuristic prediction, not a guarantee
            - Based on theoretical analysis and empirical benchmarks
            - Real quantum hardware performance may differ from simulation
            - Consider this as one factor in routing decisions, not the only factor
            - Can be upgraded to ML-based prediction with historical data
        """
        if not problem.is_generated:
            raise ValueError("Problem must be generated before prediction")
        
        problem_type = problem.problem_type.lower()
        size = problem.problem_size
        complexity_class = problem.complexity_class.lower()
        
        logger.debug(f"Predicting quantum advantage for {problem_type} problem (size={size})")
        
        # Start with base score
        score = 0.5  # Neutral starting point
        
        # Factor 1: Problem type
        # Some problems are known to benefit more from quantum approaches
        problem_type_scores = {
            'maxcut': 0.7,      # Strong quantum advantage demonstrated
            'tsp': 0.5,         # Mixed results, problem-dependent
            'portfolio': 0.3,   # Classical convex optimization is very efficient
            'graph_coloring': 0.6,  # Quantum annealing shows promise
            'sat': 0.6,         # QAOA can find satisfying assignments efficiently
        }
        
        type_bonus = problem_type_scores.get(problem_type, 0.5)
        score = score * 0.4 + type_bonus * 0.6  # Weight toward problem type
        
        logger.debug(f"After problem type factor: {score:.3f}")
        
        # Factor 2: Problem size
        # Quantum advantage appears in specific size ranges
        if size < 10:
            # Too small: quantum overhead dominates
            size_multiplier = 0.5
        elif 10 <= size <= 30:
            # Sweet spot: quantum can show advantage
            size_multiplier = 1.3
        elif 30 < size <= 50:
            # Still good, but classical algorithms scale well too
            size_multiplier = 1.1
        else:
            # Large: simulation impractical, but real QC could help
            size_multiplier = 0.8
        
        score *= size_multiplier
        logger.debug(f"After size factor: {score:.3f}")
        
        # Factor 3: Graph structure (if applicable)
        try:
            graph = problem.to_graph()
            density = nx.density(graph)
            
            # Sparse graphs favor quantum (fewer entangling gates needed)
            if density < 0.2:
                structure_multiplier = 1.2
            elif density < 0.5:
                structure_multiplier = 1.0
            else:
                # Dense graphs: classical algorithms can be more efficient
                structure_multiplier = 0.8
            
            score *= structure_multiplier
            logger.debug(f"After structure factor (density={density:.3f}): {score:.3f}")
            
            # High clustering: classical algorithms exploit locality
            try:
                clustering = nx.average_clustering(graph)
                if clustering > 0.5:
                    score *= 0.9  # Slight penalty
                logger.debug(f"After clustering factor (clustering={clustering:.3f}): {score:.3f}")
            except:
                pass  # Skip if clustering computation fails
        
        except Exception as e:
            logger.debug(f"Could not analyze graph structure: {e}")
        
        # Factor 4: Complexity class
        # NP-hard problems have higher quantum potential
        if 'np' in complexity_class and 'hard' in complexity_class:
            score *= 1.1
            logger.debug(f"After complexity factor (NP-hard): {score:.3f}")
        elif complexity_class == 'p':
            score *= 0.8
            logger.debug(f"After complexity factor (P): {score:.3f}")
        
        # Clamp to [0, 1] range
        score = max(0.0, min(1.0, score))
        
        logger.info(f"Quantum advantage probability: {score:.3f}")
        
        return score
    
    # =========================================================================
    # Private Helper Methods
    # =========================================================================
    
    def _estimate_complexity_category(self, size: int) -> str:
        """
        Categorize problem size into complexity categories.
        
        These categories are used for routing decisions and resource allocation:
        - 'small': Can be solved exactly, fast execution
        - 'medium': Requires approximation, moderate resources
        - 'large': Heuristics needed, significant resources
        - 'very_large': May need distributed computing or specialized hardware
        
        Args:
            size: Problem size (number of variables)
        
        Returns:
            Complexity category string
        """
        if size <= 20:
            return 'small'
        elif size <= 50:
            return 'medium'
        elif size <= 200:
            return 'large'
        else:
            return 'very_large'
    
    def _extract_graph_features(self, graph: nx.Graph) -> Dict[str, Any]:
        """
        Extract graph-theoretic features from a NetworkX graph.
        
        Computes:
        - density: Ratio of actual edges to possible edges
        - clustering_coefficient: Measure of local connectivity (transitivity)
        - average_degree: Average number of connections per node
        - diameter: Longest shortest path (skipped for large graphs)
        
        These features help characterize problem structure and predict
        solver performance.
        
        Args:
            graph: NetworkX graph to analyze
        
        Returns:
            Dictionary of graph features
        """
        features = {}
        
        # Density: How many edges exist vs how many possible
        # density = 0: no edges (disconnected)
        # density = 1: complete graph (all possible edges)
        density = nx.density(graph)
        features['density'] = density
        
        # Clustering coefficient: Measure of transitivity
        # High clustering → nodes form tight communities
        # Low clustering → sparse, tree-like structure
        try:
            clustering = nx.average_clustering(graph)
            features['clustering_coefficient'] = clustering
        except:
            features['clustering_coefficient'] = 0.0
        
        # Average degree: Average number of neighbors per node
        # Useful for estimating problem difficulty
        if graph.number_of_nodes() > 0:
            degrees = [d for n, d in graph.degree()]
            average_degree = sum(degrees) / len(degrees)
            features['average_degree'] = average_degree
        else:
            features['average_degree'] = 0.0
        
        # Diameter: Longest shortest path in graph
        # Expensive to compute for large graphs, so we skip it
        # High diameter → elongated structure
        # Low diameter → compact, well-connected
        if graph.number_of_nodes() <= 100 and nx.is_connected(graph):
            try:
                diameter = nx.diameter(graph)
                features['diameter'] = diameter
            except:
                features['diameter'] = None
        else:
            features['diameter'] = None  # Too expensive or disconnected
        
        return features
    
    def _calculate_suitability_scores(
        self,
        problem_type: str,
        problem_size: int,
        complexity_class: str,
        graph_features: Optional[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Calculate suitability scores for classical and quantum solvers.
        
        Scores range from 0 (not suitable) to 1 (highly suitable).
        These scores help the router decide which solver type to use.
        
        Classical Score Factors:
        - Small problems: Very suitable (exact algorithms)
        - Polynomial complexity: High suitability
        - Dense graphs: Classical algorithms handle well
        - High clustering: Can exploit locality
        
        Quantum Score Factors:
        - Medium size (10-50): Sweet spot for quantum
        - NP-hard: Potential quantum advantage
        - Sparse graphs: Fewer quantum gates needed
        - Certain problem types (MaxCut, etc.)
        
        Args:
            problem_type: Problem type identifier
            problem_size: Number of variables
            complexity_class: Computational complexity
            graph_features: Graph analysis results (or None)
        
        Returns:
            Dictionary with 'classical_score' and 'quantum_score'
        """
        classical_score = 0.5
        quantum_score = 0.5
        
        # Size-based scoring
        if problem_size <= 20:
            # Small: classical excels
            classical_score += 0.3
            quantum_score -= 0.2
        elif problem_size <= 50:
            # Medium: both can work
            classical_score += 0.1
            quantum_score += 0.2
        else:
            # Large: depends on problem structure
            classical_score += 0.0
            quantum_score -= 0.1
        
        # Complexity class
        if 'np' in complexity_class.lower():
            quantum_score += 0.2
        if complexity_class.lower() == 'p':
            classical_score += 0.2
        
        # Problem type
        quantum_friendly = ['maxcut', 'graph_coloring', 'sat']
        if problem_type.lower() in quantum_friendly:
            quantum_score += 0.2
        else:
            classical_score += 0.1
        
        # Graph structure (if available)
        if graph_features:
            density = graph_features.get('density', 0.5)
            clustering = graph_features.get('clustering_coefficient', 0.5)
            
            # Sparse graphs favor quantum
            if density < 0.3:
                quantum_score += 0.1
            elif density > 0.7:
                classical_score += 0.1
            
            # High clustering favors classical
            if clustering > 0.5:
                classical_score += 0.1
        
        # Normalize to [0, 1]
        classical_score = max(0.0, min(1.0, classical_score))
        quantum_score = max(0.0, min(1.0, quantum_score))
        
        return {
            'classical_score': classical_score,
            'quantum_score': quantum_score
        }
